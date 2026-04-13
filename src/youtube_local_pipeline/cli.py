from __future__ import annotations

import os
import platform
import shutil
import sys
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from .config import PipelineConfig, normalize_asr_backend
from .pipeline import prepare_summary_input, process_video, rechunk_video
from .summarize import summarize_video
from .transcribe import describe_qwen_command

app = typer.Typer(help="Local YouTube transcript preparation pipeline.")
console = Console()


@app.command()
def doctor() -> None:
    config = PipelineConfig()
    table = Table(title="Environment Check")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Details")

    resolved_qwen_command = describe_qwen_command(config.qwen_command)
    command_rows = [
        ("ffmpeg", shutil.which("ffmpeg"), "required for audio normalization and local ASR"),
        (
            "yt-transcriber-qwen",
            resolved_qwen_command or f"{Path(sys.executable).resolve()} -m youtube_local_pipeline.qwen_cli",
            "default Qwen entrypoint",
        ),
        ("python3.11 on PATH", shutil.which("python3.11"), "optional after the environment is created"),
        ("yt-dlp command", shutil.which("yt-dlp"), "optional; runtime uses the installed yt_dlp Python package"),
        ("mlx-qwen3-asr command", shutil.which("mlx-qwen3-asr"), "optional; the wrapper is the default path"),
        ("whisper-cli", shutil.which("whisper-cli"), "optional; only needed for the whisper.cpp fallback"),
        ("codex", shutil.which("codex"), "optional; only needed for summarize"),
        ("uv", shutil.which("uv"), "optional; setup helper"),
        ("brew", shutil.which("brew"), "optional; setup helper on macOS"),
    ]
    for name, location, description in command_rows:
        status = "ok" if location else "optional"
        detail = location or description
        table.add_row(name, status, detail)

    distributions = [
        "mlx-qwen3-asr",
        "pydantic",
        "pyannote.audio",
        "rich",
        "typer",
        "webvtt-py",
        "yt-dlp",
    ]
    for package_name in distributions:
        try:
            installed_version = version(package_name)
            table.add_row(package_name, "ok", installed_version)
        except PackageNotFoundError:
            table.add_row(package_name, "missing", "not installed in current interpreter")

    pyannote_token = (
        os.getenv("PYANNOTE_AUTH_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or ""
    ).strip()
    table.add_row(
        "pyannote auth token",
        "ok" if pyannote_token else "optional",
        "set" if pyannote_token else "needed only for --diarize with gated pyannote models",
    )

    console.print(table)
    console.print(f"Interpreter: {platform.python_version()}")
    console.print(f"Data root: {config.base_data_dir}")
    console.print("Notes:")
    console.print("- The transcript pipeline is local; Codex is only required for summarization.")
    console.print("- The yt_dlp Python package is what the runtime uses; the yt-dlp shell command is optional.")
    console.print("- The pyannote token is only required for diarization.")


@app.command()
def process(
    url: str,
    language: str = typer.Option(
        "en",
        help="Preferred subtitle/transcription language. Defaults to explicit English for the qwen3-asr path; use 'auto' only if you want ASR language detection.",
    ),
    high_quality: bool = typer.Option(False, help="Use the highest-accuracy transcription profile for the selected backend."),
    force: bool = typer.Option(False, help="Ignore cached artifacts and recompute the pipeline."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Probe metadata and print the planned branch without downloading media."),
    asr_backend: str = typer.Option(
        "qwen3-asr",
        help="ASR backend to use when creator subtitles are unavailable. qwen3-asr is the default path; whisper-cpp remains the fallback/safe option.",
    ),
    model: str | None = typer.Option(
        None,
        help="Override the ASR model alias or path for the selected backend.",
    ),
    threads: int | None = typer.Option(
        None,
        min=1,
        help="Override whisper.cpp thread count. Ignored by qwen3-asr.",
    ),
    diarize: bool = typer.Option(
        False,
        help="Enable pyannote speaker diarization for the qwen3-asr backend.",
    ),
    num_speakers: int | None = typer.Option(
        None,
        min=1,
        help="Fixed speaker count for qwen3-asr diarization.",
    ),
    min_speakers: int = typer.Option(
        1,
        min=1,
        help="Minimum speaker count for qwen3-asr diarization auto mode.",
    ),
    max_speakers: int = typer.Option(
        8,
        min=1,
        help="Maximum speaker count for qwen3-asr diarization auto mode.",
    ),
    summarize: bool = typer.Option(
        False,
        help="Run Codex headless summarization after the local transcript pipeline finishes.",
    ),
    summary_model: str | None = typer.Option(
        None,
        help="Override the Codex model used for chunk summaries and the final synthesis pass.",
    ),
    whisper_cpp_model: Path | None = typer.Option(
        None,
        help="Path to an exact whisper.cpp model file. Overrides model alias resolution and WHISPER_CPP_MODEL.",
    ),
    data_dir: Path = typer.Option(
        Path("data/videos"),
        help="Artifact root directory.",
    ),
    ) -> None:
    if dry_run and summarize:
        raise typer.BadParameter("--summarize cannot be used with --dry-run.")
    if max_speakers < min_speakers:
        raise typer.BadParameter("--max-speakers must be >= --min-speakers.")
    try:
        normalized_asr_backend = normalize_asr_backend(asr_backend)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc
    if normalized_asr_backend == "whisper-cpp" and diarize:
        raise typer.BadParameter("--diarize is only supported with --asr-backend qwen3-asr.")

    base_config = PipelineConfig()
    config = base_config.model_copy(
        update={
            "base_data_dir": data_dir,
            "whisper_cpp_model_path": whisper_cpp_model or base_config.whisper_cpp_model_path,
            "whisper_cpp_threads": threads or base_config.whisper_cpp_threads,
            "codex_summary_model": summary_model or base_config.codex_summary_model,
        }
    )
    result = process_video(
        url=url,
        config=config,
        language=language,
        high_quality=high_quality,
        force=force,
        asr_backend=normalized_asr_backend,
        model=model,
        diarize=diarize,
        diarization_num_speakers=num_speakers,
        diarization_min_speakers=min_speakers,
        diarization_max_speakers=max_speakers,
        dry_run=dry_run,
    )

    if summarize:
        summary_result = summarize_video(
            video_id=result.metadata.video_id,
            config=config,
            model=summary_model,
            force=force,
        )
        result = result.model_copy(
            update={
                "summary_manifest_path": summary_result.summary_manifest_path,
                "final_summary_path": summary_result.final_summary_path,
                "notes": [*result.notes, *summary_result.notes],
            }
        )

    summary = Table(title="Process Result")
    summary.add_column("Field")
    summary.add_column("Value")
    summary.add_row("video_id", result.metadata.video_id)
    summary.add_row("source_kind", result.source_kind.value)
    summary.add_row("workspace", str(result.artifact_root))
    summary.add_row("subtitle_policy", "creator-only")
    summary.add_row("force", str(force))
    summary.add_row("cached", str(result.used_cache))
    summary.add_row("chunk_count", str(result.chunk_count))
    if result.transcription is not None:
        summary.add_row("transcription_backend", result.transcription.backend or "n/a")
        summary.add_row("transcription_model", result.transcription.model or "n/a")
        summary.add_row("threads", str(result.transcription.threads or "n/a"))
        summary.add_row("language", result.transcription.language or "auto")
        summary.add_row("diarized", str(result.transcription.diarized or False))
        summary.add_row("speaker_count", str(result.transcription.speaker_count or "n/a"))
        if result.transcription.model_path:
            summary.add_row("model_path", result.transcription.model_path)
    console.print(summary)

    if result.subtitle_path:
        console.print(f"Subtitle source: {result.subtitle_path}")
    if result.audio_source_path:
        console.print(f"Audio source: {result.audio_source_path}")
    if result.normalized_audio_path:
        console.print(f"Normalized audio: {result.normalized_audio_path}")
    if result.summary_manifest_path:
        console.print(f"Summary manifest: {result.summary_manifest_path}")
    if result.final_summary_path:
        console.print(f"Final summary: {result.final_summary_path}")
    if result.notes:
        console.print("Notes:")
        for note in result.notes:
            console.print(f"- {note}")


@app.command()
def rechunk(
    video_id: str,
    data_dir: Path = typer.Option(Path("data/videos"), help="Artifact root directory."),
) -> None:
    config = PipelineConfig(base_data_dir=data_dir)
    result = rechunk_video(video_id=video_id, config=config)
    console.print(f"Rechunked {result.metadata.video_id} into {result.chunk_count} chunks at {result.artifact_root}.")


@app.command("prepare-summary")
def prepare_summary(
    video_id: str,
    data_dir: Path = typer.Option(Path("data/videos"), help="Artifact root directory."),
) -> None:
    config = PipelineConfig(base_data_dir=data_dir)
    result = prepare_summary_input(video_id=video_id, config=config)
    console.print(
        f"Prepared summary payload for {result.metadata.video_id} with {result.chunk_count} chunks at {result.artifact_root}."
    )


@app.command()
def summarize(
    video_id: str,
    force: bool = typer.Option(False, help="Regenerate chunk summaries and the final summary."),
    model: str | None = typer.Option(None, help="Override the Codex model used for summarization."),
    data_dir: Path = typer.Option(Path("data/videos"), help="Artifact root directory."),
) -> None:
    config = PipelineConfig(base_data_dir=data_dir, codex_summary_model=model)
    result = summarize_video(video_id=video_id, config=config, model=model, force=force)
    console.print(f"Summarized {result.metadata.video_id} into {result.final_summary_path}.")


def main() -> None:
    app()
