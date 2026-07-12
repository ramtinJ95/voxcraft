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
from .client import (
    DEFAULT_SERVER_URL,
    SERVER_TOKEN_ENV_VAR,
    SERVER_URL_ENV_VAR,
    ServerClientError,
    ServerJobResponse,
    VoxcraftServerClient,
)
from .jobs import JobOptions
from .pipeline import process_video, rechunk_video
from .summarize import summarize_video
from .transcribe import describe_qwen_command
from .utils import write_text

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


def _command_status(location: str | None, *, required: bool) -> str:
    if location:
        return "ok"
    return "missing" if required else "optional"


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


def _server_client(
    *,
    server_url: str | None,
    token: str | None,
    timeout: float = 30.0,
) -> VoxcraftServerClient:
    resolved_token = token or os.getenv(SERVER_TOKEN_ENV_VAR)
    if not resolved_token:
        raise typer.BadParameter(f"Set --token or ${SERVER_TOKEN_ENV_VAR}.")
    return VoxcraftServerClient(
        base_url=server_url or os.getenv(SERVER_URL_ENV_VAR) or DEFAULT_SERVER_URL,
        token=resolved_token,
        timeout=timeout,
    )


def _job_payload(
    *,
    url: str,
    language: str | None,
    high_quality: bool,
    force: bool,
    asr_backend: str | None,
    model: str | None,
    diarize: bool,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
) -> dict[str, object]:
    payload = JobOptions(
        language=language,
        high_quality=high_quality,
        force=force,
        asr_backend=asr_backend,
        model=model,
        diarize=diarize,
        num_speakers=num_speakers,
        min_speakers=min_speakers,
        max_speakers=max_speakers,
    ).model_dump(exclude_none=True)
    return {"url": url, **payload}


def _print_job_response(response: ServerJobResponse) -> None:
    job = response.job
    table = Table(title="Voxcraft Job")
    table.add_column("Field")
    table.add_column("Value")
    table.add_row("job_id", job.id)
    table.add_row("status", job.status)
    table.add_row("url", job.url)
    table.add_row("message", job.message or "")
    table.add_row("created_at", job.created_at)
    table.add_row("updated_at", job.updated_at)
    if job.video_id:
        table.add_row("video_id", job.video_id)
    if job.workspace_path:
        table.add_row("workspace", job.workspace_path)
    if job.final_md_path:
        table.add_row("final.md", job.final_md_path)
    if job.log_path:
        table.add_row("log", job.log_path)
    if job.error:
        table.add_row("error", job.error)
    if response.final_md_url:
        table.add_row("final_md_url", response.final_md_url)
    table.add_row("log_url", response.log_url)
    console.print(table)


def _handle_client_error(exc: ServerClientError) -> None:
    prefix = f"HTTP {exc.status_code}: " if exc.status_code is not None else ""
    raise typer.BadParameter(f"{prefix}{exc}") from exc


def _save_text_output(content: str, output_path: Path, *, default_name: str) -> Path:
    resolved_output = output_path.expanduser()
    if resolved_output.exists() and resolved_output.is_dir():
        resolved_output = resolved_output / default_name
    write_text(resolved_output, content)
    return resolved_output


def _handle_final_markdown(
    client: VoxcraftServerClient,
    job_id: str,
    *,
    print_final: bool,
    output: Path | None,
) -> None:
    final_markdown = client.get_final_markdown(job_id)
    saved_path: Path | None = None
    if output is not None:
        saved_path = _save_text_output(final_markdown, output, default_name="final.md")

    if print_final or output is None:
        sys.stdout.write(final_markdown)
        return

    console.print(f"Saved final.md: {saved_path}")


@app.command()
def doctor(ctx: typer.Context) -> None:
    config, resolved_config_path = _load_runtime_config(ctx)
    table = Table(title="Environment Check")
    table.add_column("Item")
    table.add_column("Status")
    table.add_column("Details")

    resolved_qwen_command = describe_qwen_command(config.qwen_command)
    command_rows = [
        ("ffmpeg", shutil.which("ffmpeg"), "required for audio normalization and local ASR", True),
        (
            "voxcraft-qwen",
            resolved_qwen_command or f"{Path(sys.executable).resolve()} -m voxcraft.qwen_cli",
            "default Qwen entrypoint",
            config.default_asr_backend == "qwen3-asr",
        ),
        ("python3.11 on PATH", shutil.which("python3.11"), "optional after the environment is created", False),
        ("yt-dlp command", shutil.which("yt-dlp"), "optional; runtime uses the installed yt_dlp Python package", False),
        ("mlx-qwen3-asr command", shutil.which("mlx-qwen3-asr"), "optional; the wrapper is the default path", False),
        ("whisper-cli", shutil.which("whisper-cli"), "optional; only needed for the whisper.cpp fallback", config.default_asr_backend == "whisper-cpp"),
        ("uv", shutil.which("uv"), "optional; setup helper", False),
        ("brew", shutil.which("brew"), "optional; setup helper on macOS", False),
    ]
    for provider, profile in config.summary_profiles.items():
        location = _find_command_location(profile.command)
        command_rows.append(
            (
                f"summary CLI ({provider})",
                location,
                f"configured as {profile.command}",
                False,
            )
        )
    for name, location, description, required in command_rows:
        status = _command_status(location, required=required)
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
    config, resolved_config_path = _load_runtime_config(ctx, overrides=config_overrides)
    config = config.with_summary_overrides(
        provider=normalized_summary_provider,
        command=summary_command,
        model=summary_model,
        thinking_level=summary_thinking_level,
    )
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
    config, _ = _load_runtime_config(ctx, overrides=config_overrides)
    config = config.with_summary_overrides(
        provider=normalized_provider,
        command=summary_command,
        model=model,
        thinking_level=thinking_level,
    )
    result = summarize_video(video_id=video_id, config=config, model=model, thinking_level=thinking_level, force=force)
    console.print(f"Summarized {result.metadata.video_id} into {result.final_summary_path}.")


@app.command("submit-job")
def submit_job(
    url: str,
    server_url: str | None = typer.Option(
        None,
        envvar=SERVER_URL_ENV_VAR,
        help=f"Voxcraft server URL. Defaults to ${SERVER_URL_ENV_VAR} or {DEFAULT_SERVER_URL}.",
    ),
    token: str | None = typer.Option(
        None,
        envvar=SERVER_TOKEN_ENV_VAR,
        help=f"Voxcraft server API token. Defaults to ${SERVER_TOKEN_ENV_VAR}.",
    ),
    wait: float = typer.Option(0.0, min=0.0, help="Seconds to poll after submitting."),
    poll_interval: float = typer.Option(10.0, min=1.0, help="Seconds between status polls when --wait is used."),
    print_final: bool = typer.Option(False, help="Print final.md if the job is done before --wait expires."),
    output: Path | None = typer.Option(None, help="Write final.md to this local path if the job finishes."),
    language: str | None = typer.Option(None, help="Preferred subtitle/transcription language."),
    high_quality: bool = typer.Option(False, help="Use the highest-accuracy transcription profile."),
    force: bool = typer.Option(False, help="Ignore cached artifacts and recompute the pipeline."),
    asr_backend: str | None = typer.Option(None, help="Override ASR backend: qwen3-asr or whisper-cpp."),
    model: str | None = typer.Option(None, help="Override the ASR model alias or path."),
    diarize: bool = typer.Option(False, help="Enable pyannote speaker diarization for qwen3-asr."),
    num_speakers: int | None = typer.Option(None, min=1, help="Fixed speaker count for diarization."),
    min_speakers: int = typer.Option(1, min=1, help="Minimum speaker count for diarization auto mode."),
    max_speakers: int = typer.Option(8, min=1, help="Maximum speaker count for diarization auto mode."),
) -> None:
    """Submit a YouTube URL to a remote voxcraft server."""
    client = _server_client(server_url=server_url, token=token)
    try:
        try:
            payload = _job_payload(
                url=url,
                language=language,
                high_quality=high_quality,
                force=force,
                asr_backend=asr_backend,
                model=model,
                diarize=diarize,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers,
            )
        except ValueError as exc:
            raise typer.BadParameter(str(exc)) from exc
        response = client.create_job(payload)
        if wait > 0 and response.job.status in {"queued", "running"}:
            response = client.wait_for_job(response.job.id, timeout_sec=wait, poll_interval_sec=poll_interval)
        if response.job.status == "done" and (print_final or output is not None):
            _handle_final_markdown(client, response.job.id, print_final=print_final, output=output)
            return
        _print_job_response(response)
    except ServerClientError as exc:
        _handle_client_error(exc)


@app.command("check-job")
def check_job(
    job_id: str,
    server_url: str | None = typer.Option(
        None,
        envvar=SERVER_URL_ENV_VAR,
        help=f"Voxcraft server URL. Defaults to ${SERVER_URL_ENV_VAR} or {DEFAULT_SERVER_URL}.",
    ),
    token: str | None = typer.Option(
        None,
        envvar=SERVER_TOKEN_ENV_VAR,
        help=f"Voxcraft server API token. Defaults to ${SERVER_TOKEN_ENV_VAR}.",
    ),
    wait: float = typer.Option(0.0, min=0.0, help="Seconds to poll before returning."),
    poll_interval: float = typer.Option(10.0, min=1.0, help="Seconds between status polls when --wait is used."),
    print_final: bool = typer.Option(False, help="Print final.md instead of the status table when done."),
    output: Path | None = typer.Option(None, help="Write final.md to this local path when done."),
) -> None:
    """Check a remote voxcraft job."""
    client = _server_client(server_url=server_url, token=token)
    try:
        response = client.wait_for_job(job_id, timeout_sec=wait, poll_interval_sec=poll_interval) if wait > 0 else client.get_job(job_id)
        if response.job.status == "done" and (print_final or output is not None):
            _handle_final_markdown(client, job_id, print_final=print_final, output=output)
            return
        _print_job_response(response)
    except ServerClientError as exc:
        _handle_client_error(exc)


@app.command("latest-job")
def latest_job(
    server_url: str | None = typer.Option(
        None,
        envvar=SERVER_URL_ENV_VAR,
        help=f"Voxcraft server URL. Defaults to ${SERVER_URL_ENV_VAR} or {DEFAULT_SERVER_URL}.",
    ),
    token: str | None = typer.Option(
        None,
        envvar=SERVER_TOKEN_ENV_VAR,
        help=f"Voxcraft server API token. Defaults to ${SERVER_TOKEN_ENV_VAR}.",
    ),
) -> None:
    """Show the latest remote voxcraft job."""
    client = _server_client(server_url=server_url, token=token)
    try:
        _print_job_response(client.get_latest_job())
    except ServerClientError as exc:
        _handle_client_error(exc)


@app.command("fetch-final")
def fetch_final(
    job_id: str,
    server_url: str | None = typer.Option(
        None,
        envvar=SERVER_URL_ENV_VAR,
        help=f"Voxcraft server URL. Defaults to ${SERVER_URL_ENV_VAR} or {DEFAULT_SERVER_URL}.",
    ),
    token: str | None = typer.Option(
        None,
        envvar=SERVER_TOKEN_ENV_VAR,
        help=f"Voxcraft server API token. Defaults to ${SERVER_TOKEN_ENV_VAR}.",
    ),
    output: Path | None = typer.Option(None, help="Write final.md to this local path instead of stdout."),
) -> None:
    """Print or save a completed remote job's final.md."""
    client = _server_client(server_url=server_url, token=token)
    try:
        _handle_final_markdown(client, job_id, print_final=False, output=output)
    except ServerClientError as exc:
        _handle_client_error(exc)


@app.command("fetch-log")
def fetch_log(
    job_id: str,
    server_url: str | None = typer.Option(
        None,
        envvar=SERVER_URL_ENV_VAR,
        help=f"Voxcraft server URL. Defaults to ${SERVER_URL_ENV_VAR} or {DEFAULT_SERVER_URL}.",
    ),
    token: str | None = typer.Option(
        None,
        envvar=SERVER_TOKEN_ENV_VAR,
        help=f"Voxcraft server API token. Defaults to ${SERVER_TOKEN_ENV_VAR}.",
    ),
) -> None:
    """Print a remote job's pipeline log."""
    client = _server_client(server_url=server_url, token=token)
    try:
        sys.stdout.write(client.get_log(job_id))
    except ServerClientError as exc:
        _handle_client_error(exc)


@app.command()
def server(
    ctx: typer.Context,
    host: str = typer.Option("127.0.0.1", help="Host/interface to bind. Use a LAN or Tailscale IP for remote access."),
    port: int = typer.Option(8765, min=1, max=65535, help="Port to listen on."),
    token: str | None = typer.Option(
        None,
        envvar="VOXCRAFT_SERVER_TOKEN",
        help="API token. Defaults to $VOXCRAFT_SERVER_TOKEN.",
    ),
    jobs_db: Path | None = typer.Option(None, help="SQLite job database path."),
    data_dir: Path | None = typer.Option(None, help="Override the artifact root directory for server jobs."),
) -> None:
    """Run the authenticated async job server."""
    if not token:
        raise typer.BadParameter("Set --token or VOXCRAFT_SERVER_TOKEN before starting the server.")
    overrides = {"base_data_dir": data_dir} if data_dir is not None else None
    config, resolved_config_path = _load_runtime_config(ctx, overrides=overrides)

    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("The server dependencies are missing. Run `uv sync` first.") from exc

    from .server import create_app, default_jobs_db_path

    resolved_jobs_db = jobs_db or default_jobs_db_path(config)
    console.print(f"Starting voxcraft server on {host}:{port}")
    console.print(f"Config: {resolved_config_path or 'built-in defaults'}")
    console.print(f"Data root: {config.base_data_dir}")
    console.print(f"Jobs DB: {resolved_jobs_db}")
    app_instance = create_app(config=config, jobs_db_path=resolved_jobs_db, token=token)
    uvicorn.run(app_instance, host=host, port=port)


def main() -> None:
    app()
