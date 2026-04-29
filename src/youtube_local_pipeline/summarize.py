from __future__ import annotations

import re
import shutil
import subprocess
import textwrap
from pathlib import Path

from .config import PipelineConfig, normalize_summary_provider
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

FINAL_SUMMARY_WRAP_WIDTH = 80
SUMMARY_STDIN_DIRECTIVE = "Follow the piped prompt exactly and output only the requested markdown. Do not use tools."
FENCE_PATTERN = re.compile(r"^\s*(```|~~~)")
LIST_ITEM_PATTERN = re.compile(r"^(\s*)([-*+]|\d+\.)\s+(.*)$")
BLOCKQUOTE_PATTERN = re.compile(r"^(\s*>\s?)(.*)$")


def summarize_video(
    video_id: str,
    config: PipelineConfig,
    model: str | None = None,
    thinking_level: str | None = None,
    force: bool = False,
) -> ProcessResult:
    paths = initialize_workspace(resolve_artifact_paths(config.base_data_dir, video_id))
    metadata = VideoMetadata.model_validate(read_json(paths.metadata_path))
    chunk_manifest = [ChunkManifestEntry.model_validate(item) for item in read_json(paths.chunk_index_path)]
    summary_provider = normalize_summary_provider(config.summary_provider)
    summary_command = config.summary_command or summary_provider
    summary_model = model if model is not None else config.summary_model
    summary_thinking_level = thinking_level if thinking_level is not None else config.summary_thinking_level

    if not chunk_manifest:
        raise RuntimeError("No chunk files are available to summarize.")

    manifest = _load_summary_manifest(paths.summary_manifest_path)
    manifest = _migrate_legacy_final_summary(paths, manifest)
    if manifest is not None and not force and paths.summary_final_path.exists():
        if (
            manifest.summary_provider == summary_provider
            and manifest.summary_command == summary_command
            and manifest.summary_model == summary_model
            and manifest.summary_thinking_level == summary_thinking_level
            and len(manifest.chunk_summaries) == len(chunk_manifest)
            and all((paths.root_dir / entry.output_path).exists() for entry in manifest.chunk_summaries)
        ):
            wrap_markdown_file(paths.summary_final_path, width=FINAL_SUMMARY_WRAP_WIDTH)
            summary_payload = SummaryPayload.model_validate(read_json(paths.summary_payload_path))
            return ProcessResult(
                metadata=metadata,
                source_kind=summary_payload.source_kind,
                artifact_root=paths.root_dir,
                chunk_count=len(chunk_manifest),
                notes=["Reused cached summaries."],
                summary_manifest_path=path_string(paths.summary_manifest_path, paths.root_dir),
                final_summary_path=path_string(paths.summary_final_path, paths.root_dir),
            )

    provider_label = _summary_provider_label(summary_provider)
    append_log(paths.pipeline_log_path, f"Starting {provider_label} summarization for {video_id}")
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
            rendered = run_summary_cli(
                prompt=prompt,
                output_path=output_path,
                workdir=paths.root_dir,
                provider=summary_provider,
                command=summary_command,
                model=summary_model,
                thinking_level=summary_thinking_level,
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
        run_summary_cli(
            prompt=final_prompt,
            output_path=paths.summary_final_path,
            workdir=paths.root_dir,
            provider=summary_provider,
            command=summary_command,
            model=summary_model,
            thinking_level=summary_thinking_level,
        )
        append_log(paths.pipeline_log_path, f"Wrote final {provider_label} summary")

    summary_manifest = SummaryManifest(
        video_id=video_id,
        summary_provider=summary_provider,
        summary_command=summary_command,
        summary_model=summary_model,
        summary_thinking_level=summary_thinking_level,
        chunk_summaries=summary_entries,
        final_prompt_path=path_string(paths.summary_final_prompt_path, paths.root_dir),
        final_summary_path=path_string(paths.summary_final_path, paths.root_dir),
    )
    write_json(paths.summary_manifest_path, summary_manifest.model_dump(mode="json"))
    wrap_markdown_file(paths.summary_final_path, width=FINAL_SUMMARY_WRAP_WIDTH)

    summary_payload = SummaryPayload.model_validate(read_json(paths.summary_payload_path))
    return ProcessResult(
        metadata=metadata,
        source_kind=summary_payload.source_kind,
        artifact_root=paths.root_dir,
        chunk_count=len(chunk_manifest),
        notes=[f"{provider_label} chunk summaries and final summary created."],
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
Write 4-7 substantial paragraphs covering only this chunk.
Preserve technical detail, named concepts, causal relationships, examples, caveats,
and the speaker's reasoning. Prefer adding another paragraph over dropping a useful distinction.

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
Write 8-12 substantive paragraphs. Preserve the structure of the talk and include
specific examples, caveats, mechanisms, tradeoffs, and named concepts when they help
reconstruct the speaker's reasoning. Do not merge distinct arguments into a single
broad takeaway when the distinction would matter for later reference.

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


def run_summary_cli(
    prompt: str,
    output_path: Path,
    workdir: Path,
    provider: str,
    command: str,
    model: str | None = None,
    thinking_level: str | None = None,
) -> str:
    normalized_provider = normalize_summary_provider(provider)
    if shutil.which(command) is None:
        raise RuntimeError(f"{command} is not available on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cli_command, writes_to_stdout = _build_summary_command(
        provider=normalized_provider,
        command=command,
        model=model,
        thinking_level=thinking_level,
        workdir=workdir,
        output_path=output_path,
    )

    completed = subprocess.run(
        cli_command,
        cwd=workdir,
        input=prompt,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        provider_label = _summary_provider_label(normalized_provider)
        error_text = (
            completed.stderr.strip()
            or completed.stdout.strip()
            or f"{provider_label} exited with a non-zero status."
        )
        raise RuntimeError(error_text)
    if writes_to_stdout:
        rendered = completed.stdout.strip()
        if not rendered:
            raise RuntimeError(
                completed.stderr.strip() or f"{_summary_provider_label(normalized_provider)} did not produce any text."
            )
        write_text(output_path, rendered + "\n")
    if not output_path.exists():
        raise RuntimeError(f"{_summary_provider_label(normalized_provider)} completed without writing {output_path}")
    return output_path.read_text(encoding="utf-8").strip()


def _build_summary_command(
    *,
    provider: str,
    command: str,
    model: str | None,
    thinking_level: str | None,
    workdir: Path,
    output_path: Path,
) -> tuple[list[str], bool]:
    if provider == "codex":
        resolved_workdir = workdir.resolve()
        resolved_output_path = output_path.resolve()
        command_parts = [
            command,
            "exec",
            "--skip-git-repo-check",
            "--sandbox",
            "read-only",
            "--ephemeral",
            "-C",
            str(resolved_workdir),
            "-o",
            str(resolved_output_path),
        ]
        if model:
            command_parts.extend(["--model", model])
        if thinking_level:
            command_parts.extend(["-c", f'model_reasoning_effort="{thinking_level}"'])
        return command_parts, False

    if provider == "claude":
        command_parts = [
            command,
            "--print",
            SUMMARY_STDIN_DIRECTIVE,
            "--output-format",
            "text",
            "--no-session-persistence",
            "--permission-mode",
            "default",
            "--bare",
            "--max-turns",
            "1",
            "--tools",
            "",
        ]
        if model:
            command_parts.extend(["--model", model])
        return command_parts, True

    if provider == "gemini":
        command_parts = [
            command,
            "--prompt",
            SUMMARY_STDIN_DIRECTIVE,
            "--output-format",
            "text",
            "--approval-mode",
            "default",
        ]
        if model:
            command_parts.extend(["--model", model])
        return command_parts, True

    if provider == "pi":
        command_parts = [
            command,
            "-p",
            SUMMARY_STDIN_DIRECTIVE,
            "--no-session",
            "--no-tools",
        ]
        if model:
            command_parts.extend(["--model", model])
        if thinking_level:
            command_parts.extend(["--thinking", thinking_level])
        return command_parts, True

    raise ValueError(f"Unsupported summary provider: {provider}")


def _load_summary_manifest(path: Path) -> SummaryManifest | None:
    if not path.exists():
        return None
    return SummaryManifest.model_validate(read_json(path))


def _migrate_legacy_final_summary(
    paths,
    manifest: SummaryManifest | None,
) -> SummaryManifest | None:
    legacy_final_path = paths.summary_dir / "final.md"
    if legacy_final_path == paths.summary_final_path:
        return manifest

    desired_final_summary_path = path_string(paths.summary_final_path, paths.root_dir)
    if paths.summary_final_path.exists():
        if manifest is None or manifest.final_summary_path == desired_final_summary_path:
            return manifest
        updated_manifest = manifest.model_copy(update={"final_summary_path": desired_final_summary_path})
        write_json(paths.summary_manifest_path, updated_manifest.model_dump(mode="json"))
        return updated_manifest

    if not legacy_final_path.exists():
        return manifest

    write_text(paths.summary_final_path, legacy_final_path.read_text(encoding="utf-8"))
    legacy_final_path.unlink()
    append_log(paths.pipeline_log_path, "Moved legacy summary/final.md to final.md")

    if manifest is None:
        return None

    updated_manifest = manifest.model_copy(update={"final_summary_path": desired_final_summary_path})
    write_json(paths.summary_manifest_path, updated_manifest.model_dump(mode="json"))
    return updated_manifest


def _summary_provider_label(provider: str) -> str:
    labels = {
        "claude": "Claude Code",
        "codex": "Codex",
        "gemini": "Gemini CLI",
        "pi": "Pi",
    }
    return labels.get(provider, provider)


def wrap_markdown_file(path: Path, width: int = FINAL_SUMMARY_WRAP_WIDTH) -> bool:
    content = path.read_text(encoding="utf-8")
    wrapped = wrap_markdown_text(content, width=width)
    if wrapped == content:
        return False
    write_text(path, wrapped)
    return True


def wrap_markdown_text(content: str, width: int = FINAL_SUMMARY_WRAP_WIDTH) -> str:
    lines = content.splitlines()
    wrapped_lines: list[str] = []
    paragraph_lines: list[str] = []
    in_fence = False

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        paragraph = " ".join(line.strip() for line in paragraph_lines if line.strip())
        wrapped_lines.extend(
            textwrap.fill(
                paragraph,
                width=width,
                break_long_words=False,
                break_on_hyphens=False,
            ).splitlines()
        )
        paragraph_lines.clear()

    for line in lines:
        stripped = line.strip()
        if FENCE_PATTERN.match(line):
            flush_paragraph()
            wrapped_lines.append(line.rstrip())
            in_fence = not in_fence
            continue

        if in_fence:
            wrapped_lines.append(line.rstrip())
            continue

        if not stripped:
            flush_paragraph()
            wrapped_lines.append("")
            continue

        if _is_passthrough_markdown_line(stripped):
            flush_paragraph()
            wrapped_lines.append(line.rstrip())
            continue

        wrapped_special = _wrap_special_markdown_line(line, width=width)
        if wrapped_special is not None:
            flush_paragraph()
            wrapped_lines.extend(wrapped_special)
            continue

        paragraph_lines.append(line)

    flush_paragraph()
    return "\n".join(wrapped_lines).rstrip() + "\n"


def _is_passthrough_markdown_line(stripped: str) -> bool:
    return (
        stripped.startswith("#")
        or stripped.startswith("|")
        or stripped.startswith("<")
        or _is_horizontal_rule(stripped)
    )


def _is_horizontal_rule(stripped: str) -> bool:
    return stripped in {"---", "***", "___"}


def _wrap_special_markdown_line(line: str, *, width: int) -> list[str] | None:
    list_match = LIST_ITEM_PATTERN.match(line)
    if list_match is not None:
        indent, marker, body = list_match.groups()
        body = body.strip()
        if not body:
            return [line.rstrip()]
        item_prefix = f"{indent}{marker} "
        return textwrap.fill(
            body,
            width=width,
            initial_indent=item_prefix,
            subsequent_indent=" " * len(item_prefix),
            break_long_words=False,
            break_on_hyphens=False,
        ).splitlines()

    blockquote_match = BLOCKQUOTE_PATTERN.match(line)
    if blockquote_match is not None:
        prefix, body = blockquote_match.groups()
        body = body.strip()
        if not body:
            return [line.rstrip()]
        return textwrap.fill(
            body,
            width=width,
            initial_indent=prefix,
            subsequent_indent=" " * len(prefix),
            break_long_words=False,
            break_on_hyphens=False,
        ).splitlines()
    return None
