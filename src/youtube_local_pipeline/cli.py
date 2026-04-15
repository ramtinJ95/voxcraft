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

from .config import (
    CONFIG_ENV_VAR,
    PipelineConfig,
    default_config_path,
    load_pipeline_config,
    normalize_asr_backend,
    normalize_summary_provider,
)
from .pipeline import prepare_summary_input, process_video, rechunk_video
from .summarize import summarize_video
from .transcribe import describe_qwen_command

app = typer.Typer(help="Local YouTube transcript preparation pipeline.")
console = Console()


@app.callback()
def main_callback(
    ctx: typer.Context,
    config: Path | None = typer.Option(
        None,
        "--config",
        help=(
            f"Path to runtime config.json. Defaults to ${CONFIG_ENV_VAR} or "
            f"{default_config_path()} when present."
        ),
    ),
) -> None:
    ctx.obj = {"config_path": config}


def _find_command_location(command: str | None) -> str | None:
    if not command:
        return None
    resolved = shutil.which(command)
    if resolved:
        return resolved
    candidate = Path(command).expanduser()
    if candidate.exists():
        return str(candidate)
    return None


def _load_runtime_config(
    ctx: typer.Context,
    *,
    overrides: dict[str, object] | None = None,
) -> tuple[PipelineConfig, Path | None]:
    config_path = ctx.obj.get("config_path") if ctx.obj else None
    try:
        return load_pipeline_config(config_path=config_path, overrides=overrides)
    except (FileNotFoundError, ValueError) as exc:
        if config_path is not None:
            raise typer.BadParameter(str(exc), param_hint="--config") from exc
        raise typer.BadParameter(str(exc)) from exc


@app.command()
def doctor(ctx: typer.Context) -> None:
    config, resolved_config_path = _load_runtime_config(ctx)
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
        ("uv", shutil.which("uv"), "optional; setup helper"),
        ("brew", shutil.which("brew"), "optional; setup helper on macOS"),
    ]
    for provider, profile in config.summary_profiles.items():
        location = _find_command_location(profile.command)
        command_rows.append(
            (
                f"summary CLI ({provider})",
                location,
                f"configured as {profile.command}",
            )
        )
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
    console.print(f"Config: {resolved_config_path or 'built-in defaults'}")
    console.print(f"Data root: {config.base_data_dir}")
    console.print(f"Default summary provider: {config.summary_provider}")
    console.print("Notes:")
    console.print("- The transcript pipeline is local; summarization requires one supported summary CLI.")
    console.print("- The yt_dlp Python package is what the runtime uses; the yt-dlp shell command is optional.")
    console.print("- The pyannote token is only required for diarization.")


@app.command()
def process(
    ctx: typer.Context,
    url: str,
    language: str | None = typer.Option(
        None,
        help="Override the preferred subtitle/transcription language for this run. Defaults to the config value.",
    ),
    high_quality: bool = typer.Option(False, help="Use the highest-accuracy transcription profile for the selected backend."),
    force: bool = typer.Option(False, help="Ignore cached artifacts and recompute the pipeline."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Probe metadata and print the planned branch without downloading media."),
    asr_backend: str | None = typer.Option(
        None,
        help="Override the ASR backend for this run. Defaults to the config value.",
    ),
    model: str | None = typer.Option(
        None,
        help="Override the ASR model alias or path for this run.",
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
        help="Run headless summarization after the local transcript pipeline finishes.",
    ),
    summary_provider: str | None = typer.Option(
        None,
        help="Override the summary CLI for this run: codex, claude, gemini, or pi.",
    ),
    summary_command: str | None = typer.Option(
        None,
        help="Override the summary CLI executable name or path for this run.",
    ),
    summary_model: str | None = typer.Option(
        None,
        help="Override the model passed to the selected summary CLI for this run.",
    ),
    summary_thinking_level: str | None = typer.Option(
        None,
        help="Override the thinking or reasoning level passed to the selected summary CLI for this run when supported.",
    ),
    whisper_cpp_model: Path | None = typer.Option(
        None,
        help="Path to an exact whisper.cpp model file. Overrides model alias resolution and WHISPER_CPP_MODEL.",
    ),
    data_dir: Path | None = typer.Option(
        None,
        help="Override the artifact root directory for this run.",
    ),
) -> None:
    if dry_run and summarize:
        raise typer.BadParameter("--summarize cannot be used with --dry-run.")
    if max_speakers < min_speakers:
        raise typer.BadParameter("--max-speakers must be >= --min-speakers.")
    normalized_asr_backend: str | None = None
    if asr_backend is not None:
        try:
            normalized_asr_backend = normalize_asr_backend(asr_backend)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    normalized_summary_provider: str | None = None
    if summary_provider is not None:
        try:
            normalized_summary_provider = normalize_summary_provider(summary_provider)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    if normalized_asr_backend == "whisper-cpp" and diarize:
        raise typer.BadParameter("--diarize is only supported with --asr-backend qwen3-asr.")
    config_overrides: dict[str, object] = {}
    if data_dir is not None:
        config_overrides["base_data_dir"] = data_dir
    if whisper_cpp_model is not None:
        config_overrides["whisper_cpp_model_path"] = whisper_cpp_model
    if threads is not None:
        config_overrides["whisper_cpp_threads"] = threads
    if normalized_summary_provider is not None:
        config_overrides["summary_provider"] = normalized_summary_provider
    if summary_command is not None:
        config_overrides["summary_command"] = summary_command
    if summary_model is not None:
        config_overrides["summary_model"] = summary_model
    if summary_thinking_level is not None:
        config_overrides["summary_thinking_level"] = summary_thinking_level
    config, resolved_config_path = _load_runtime_config(ctx, overrides=config_overrides)
    effective_asr_backend = normalized_asr_backend or config.default_asr_backend
    effective_summary_provider = config.summary_provider
    if effective_asr_backend == "whisper-cpp" and diarize:
        raise typer.BadParameter("--diarize is only supported with the qwen3-asr backend.")
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
            thinking_level=summary_thinking_level,
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
    summary.add_row("config", str(resolved_config_path or "built-in defaults"))
    summary.add_row("asr_backend", effective_asr_backend)
    if summarize:
        summary.add_row("summary_provider", effective_summary_provider)
        summary.add_row("summary_model", config.summary_model or "default")
        summary.add_row("summary_thinking_level", config.summary_thinking_level or "default")
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
    ctx: typer.Context,
    video_id: str,
    data_dir: Path | None = typer.Option(None, help="Override the artifact root directory for this run."),
) -> None:
    overrides = {"base_data_dir": data_dir} if data_dir is not None else None
    config, _ = _load_runtime_config(ctx, overrides=overrides)
    result = rechunk_video(video_id=video_id, config=config)
    console.print(f"Rechunked {result.metadata.video_id} into {result.chunk_count} chunks at {result.artifact_root}.")


@app.command("prepare-summary")
def prepare_summary(
    ctx: typer.Context,
    video_id: str,
    data_dir: Path | None = typer.Option(None, help="Override the artifact root directory for this run."),
) -> None:
    overrides = {"base_data_dir": data_dir} if data_dir is not None else None
    config, _ = _load_runtime_config(ctx, overrides=overrides)
    result = prepare_summary_input(video_id=video_id, config=config)
    console.print(
        f"Prepared summary payload for {result.metadata.video_id} with {result.chunk_count} chunks at {result.artifact_root}."
    )


@app.command()
def summarize(
    ctx: typer.Context,
    video_id: str,
    force: bool = typer.Option(False, help="Regenerate chunk summaries and the final summary."),
    provider: str | None = typer.Option(None, help="Override the summary CLI for this run: codex, claude, gemini, or pi."),
    summary_command: str | None = typer.Option(None, help="Override the summary CLI executable name or path for this run."),
    model: str | None = typer.Option(None, help="Override the model passed to the selected summary CLI for this run."),
    thinking_level: str | None = typer.Option(
        None,
        help="Override the thinking or reasoning level passed to the selected summary CLI for this run when supported.",
    ),
    data_dir: Path | None = typer.Option(None, help="Override the artifact root directory for this run."),
) -> None:
    normalized_provider: str | None = None
    if provider is not None:
        try:
            normalized_provider = normalize_summary_provider(provider)
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
    config_overrides: dict[str, object] = {}
    if data_dir is not None:
        config_overrides["base_data_dir"] = data_dir
    if normalized_provider is not None:
        config_overrides["summary_provider"] = normalized_provider
    if summary_command is not None:
        config_overrides["summary_command"] = summary_command
    if model is not None:
        config_overrides["summary_model"] = model
    if thinking_level is not None:
        config_overrides["summary_thinking_level"] = thinking_level
    config, _ = _load_runtime_config(ctx, overrides=config_overrides)
    result = summarize_video(video_id=video_id, config=config, model=model, thinking_level=thinking_level, force=force)
    console.print(f"Summarized {result.metadata.video_id} into {result.final_summary_path}.")


def main() -> None:
    app()
