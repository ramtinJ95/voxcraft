from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from .config import PipelineConfig
from .manifest import initialize_workspace, resolve_artifact_paths
from .models import (
    ChunkManifestEntry,
    ChunkSummaryEntry,
    ProcessResult,
    SummaryManifest,
    SummaryPayload,
    VideoMetadata,
)
from .utils import append_log, path_string, read_json, write_json, write_text


def summarize_video(
    video_id: str,
    config: PipelineConfig,
    model: str | None = None,
    force: bool = False,
) -> ProcessResult:
    paths = initialize_workspace(resolve_artifact_paths(config.base_data_dir, video_id))
    metadata = VideoMetadata.model_validate(read_json(paths.metadata_path))
    chunk_manifest = [ChunkManifestEntry.model_validate(item) for item in read_json(paths.chunk_index_path)]

    if not chunk_manifest:
        raise RuntimeError("No chunk files are available to summarize.")

    manifest = _load_summary_manifest(paths.summary_manifest_path)
    if manifest is not None and not force and paths.summary_final_path.exists():
        if len(manifest.chunk_summaries) == len(chunk_manifest) and all(
            (paths.root_dir / entry.output_path).exists() for entry in manifest.chunk_summaries
        ):
            summary_payload = SummaryPayload.model_validate(read_json(paths.summary_payload_path))
            return ProcessResult(
                metadata=metadata,
                source_kind=summary_payload.source_kind,
                artifact_root=paths.root_dir,
                chunk_count=len(chunk_manifest),
                notes=["Reused cached Codex summaries."],
                summary_manifest_path=path_string(paths.summary_manifest_path, paths.root_dir),
                final_summary_path=path_string(paths.summary_final_path, paths.root_dir),
            )

    append_log(paths.pipeline_log_path, f"Starting Codex summarization for {video_id}")
    codex_model = model or config.codex_summary_model or "gpt-5.4"
    codex_reasoning_effort = _codex_reasoning_effort_for_model(codex_model)
    summary_entries: list[ChunkSummaryEntry] = []
    rendered_chunk_summaries: list[str] = []

    for chunk in chunk_manifest:
        chunk_path = paths.root_dir / chunk.path
        chunk_text = chunk_path.read_text(encoding="utf-8").strip()
        prompt_path = paths.summary_prompts_dir / f"chunk-{chunk.index:03}.prompt.txt"
        output_path = paths.summary_dir / f"chunk-{chunk.index:03}.md"
        prompt = build_chunk_summary_prompt(metadata=metadata, chunk=chunk, chunk_text=chunk_text)
        write_text(prompt_path, prompt)

        if force or not output_path.exists():
            rendered = run_codex_exec(
                prompt=prompt,
                output_path=output_path,
                workdir=paths.root_dir,
                codex_command=config.codex_command,
                model=codex_model,
                reasoning_effort=codex_reasoning_effort,
            )
            append_log(paths.pipeline_log_path, f"Summarized chunk {chunk.index}")
        else:
            rendered = output_path.read_text(encoding="utf-8").strip()

        rendered_chunk_summaries.append(rendered)
        summary_entries.append(
            ChunkSummaryEntry(
                index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                source_chunk_path=chunk.path,
                prompt_path=path_string(prompt_path, paths.root_dir),
                output_path=path_string(output_path, paths.root_dir),
            )
        )

    final_prompt = build_final_summary_prompt(
        metadata=metadata,
        chunk_summaries=rendered_chunk_summaries,
        chunk_manifest=chunk_manifest,
    )
    write_text(paths.summary_final_prompt_path, final_prompt)
    if force or not paths.summary_final_path.exists():
        run_codex_exec(
            prompt=final_prompt,
            output_path=paths.summary_final_path,
            workdir=paths.root_dir,
            codex_command=config.codex_command,
            model=codex_model,
            reasoning_effort=codex_reasoning_effort,
        )
        append_log(paths.pipeline_log_path, "Wrote final Codex summary")

    summary_manifest = SummaryManifest(
        video_id=video_id,
        codex_command=config.codex_command,
        codex_model=codex_model,
        codex_reasoning_effort=codex_reasoning_effort,
        chunk_summaries=summary_entries,
        final_prompt_path=path_string(paths.summary_final_prompt_path, paths.root_dir),
        final_summary_path=path_string(paths.summary_final_path, paths.root_dir),
    )
    write_json(paths.summary_manifest_path, summary_manifest.model_dump(mode="json"))

    summary_payload = SummaryPayload.model_validate(read_json(paths.summary_payload_path))
    return ProcessResult(
        metadata=metadata,
        source_kind=summary_payload.source_kind,
        artifact_root=paths.root_dir,
        chunk_count=len(chunk_manifest),
        notes=["Codex chunk summaries and final summary created."],
        summary_manifest_path=path_string(paths.summary_manifest_path, paths.root_dir),
        final_summary_path=path_string(paths.summary_final_path, paths.root_dir),
    )


def build_chunk_summary_prompt(
    metadata: VideoMetadata,
    chunk: ChunkManifestEntry,
    chunk_text: str,
) -> str:
    return f"""Summarize this single transcript chunk from a YouTube video as detailed technical study notes.

Do not inspect files or run tools. Use only the content in this prompt.
Prefer preserving important information over being brief.
Do not throw away mechanisms, definitions, examples, edge cases, caveats, or strong claims just to make the output shorter.
If the transcript is noisy or ambiguous, say so explicitly instead of inventing missing details.
Write markdown with exactly these sections:

## Chunk Summary
Write 2-5 substantial paragraphs covering only this chunk.
Preserve technical detail, named concepts, causal relationships, and the speaker's reasoning.

## Key Points
Write 6-12 bullet points.
Include mechanisms, definitions, claims, examples, constraints, and practical implications.

## Notable Details
Write 4-10 bullet points capturing specific terminology, formal conditions, illustrative examples,
surprising observations, caveats, transcript artifacts, or anything easy to lose in compression.

Video title: {metadata.title or metadata.video_id}
Channel: {metadata.channel or "unknown"}
Chunk index: {chunk.index}
Chunk time range: {chunk.start_sec:.2f}s to {chunk.end_sec:.2f}s

Transcript chunk:
<chunk>
{chunk_text}
</chunk>
"""


def build_final_summary_prompt(
    metadata: VideoMetadata,
    chunk_summaries: list[str],
    chunk_manifest: list[ChunkManifestEntry],
) -> str:
    sections = []
    for chunk, summary in zip(chunk_manifest, chunk_summaries, strict=True):
        sections.append(
            f"Chunk {chunk.index} ({chunk.start_sec:.2f}s-{chunk.end_sec:.2f}s):\n{summary.strip()}"
        )
    joined = "\n\n".join(sections)
    return f"""Combine these chunk summaries into one final technical, information-preserving summary of the full YouTube video.

Do not inspect files or run tools. Use only the content in this prompt.
Prefer fidelity and recall over brevity.
Do not collapse away important technical distinctions, examples, or caveats.
If a claim seems strong or the transcript looks noisy, keep that uncertainty visible instead of smoothing it over.
Write markdown with exactly these sections:

# Final Summary
Write 4-8 substantive paragraphs that preserve the structure of the talk, the main mechanisms,
and the technical reasoning.

## Main Takeaways
Write 10-18 bullet points.
Include concepts, mechanisms, definitions, examples, limitations, and conclusions that matter for later reference.

## Timeline
Write 8-16 bullet points that map the major parts of the video in order.
Keep the bullets specific enough that a reader can reconstruct the flow of the video.

## Open Questions Or Uncertainties
Write bullet points for unclear, noisy, ambiguous, or potentially overstated material.
If none, write "- None."

Video title: {metadata.title or metadata.video_id}
Channel: {metadata.channel or "unknown"}
Duration seconds: {metadata.duration_sec or "unknown"}

Chunk summaries:
<chunk_summaries>
{joined}
</chunk_summaries>
"""


def run_codex_exec(
    prompt: str,
    output_path: Path,
    workdir: Path,
    codex_command: str,
    model: str | None = None,
    reasoning_effort: str | None = None,
) -> str:
    if shutil.which(codex_command) is None:
        raise RuntimeError(f"{codex_command} is not available on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        codex_command,
        "exec",
        "--skip-git-repo-check",
        "--sandbox",
        "read-only",
        "--ephemeral",
        "-C",
        str(workdir),
        "-o",
        str(output_path),
    ]
    if model:
        command.extend(["--model", model])
    if reasoning_effort:
        command.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])

    completed = subprocess.run(
        command,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        error_text = completed.stderr.strip() or completed.stdout.strip() or "codex exec exited with a non-zero status."
        raise RuntimeError(error_text)
    if not output_path.exists():
        raise RuntimeError(f"codex exec completed without writing {output_path}")
    return output_path.read_text(encoding="utf-8").strip()


def _load_summary_manifest(path: Path) -> SummaryManifest | None:
    if not path.exists():
        return None
    return SummaryManifest.model_validate(read_json(path))


def _codex_reasoning_effort_for_model(model: str | None) -> str | None:
    if model == "gpt-5.4":
        return "high"
    return None
