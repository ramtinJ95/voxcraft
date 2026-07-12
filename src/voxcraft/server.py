from __future__ import annotations

import secrets
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Header, HTTPException, Response, status
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field, model_validator

from .config import PipelineConfig, normalize_asr_backend
from .jobs import JobOptions, JobRecord, JobStore
from .manifest import resolve_artifact_paths
from .pipeline import process_video
from .summarize import summarize_video
from .utils import extract_youtube_id, read_json


class CreateJobRequest(BaseModel):
    url: str
    language: str | None = None
    high_quality: bool = False
    force: bool = False
    asr_backend: str | None = None
    model: str | None = None
    diarize: bool = False
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int = Field(default=1, ge=1)
    max_speakers: int = Field(default=8, ge=1)

    @model_validator(mode="after")
    def validate_options(self) -> CreateJobRequest:
        if self.max_speakers < self.min_speakers:
            raise ValueError("max_speakers must be >= min_speakers")
        if self.asr_backend is not None:
            backend = normalize_asr_backend(self.asr_backend)
            if backend == "whisper-cpp" and self.diarize:
                raise ValueError("diarize is only supported with qwen3-asr")
            self.asr_backend = backend
        return self

    def to_job_options(self) -> JobOptions:
        return JobOptions(
            language=self.language,
            high_quality=self.high_quality,
            force=self.force,
            asr_backend=self.asr_backend,
            model=self.model,
            diarize=self.diarize,
            num_speakers=self.num_speakers,
            min_speakers=self.min_speakers,
            max_speakers=self.max_speakers,
        )


class JobResponse(BaseModel):
    job: JobRecord
    final_md_url: str | None = None
    log_url: str


def default_jobs_db_path(config: PipelineConfig) -> Path:
    return config.base_data_dir.parent / "server" / "jobs.sqlite3"


class JobWorker:
    def __init__(self, *, store: JobStore, config: PipelineConfig, poll_interval_sec: float = 2.0) -> None:
        self.store = store
        self.config = config
        self.poll_interval_sec = poll_interval_sec
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run_loop, name="voxcraft-job-worker", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _run_loop(self) -> None:
        while not self._stop_event.is_set():
            job = self.store.claim_next_queued()
            if job is None:
                self._stop_event.wait(self.poll_interval_sec)
                continue
            self._run_job(job)

    def _run_job(self, job: JobRecord) -> None:
        try:
            options = job.options
            self.store.update_running(job.id, message="Preparing transcript pipeline.")
            process_result = process_video(
                url=job.url,
                config=self.config,
                language=options.language,
                high_quality=options.high_quality,
                force=options.force,
                asr_backend=options.asr_backend,
                model=options.model,
                diarize=options.diarize,
                diarization_num_speakers=options.num_speakers,
                diarization_min_speakers=options.min_speakers,
                diarization_max_speakers=options.max_speakers,
            )
            workspace_path = str(process_result.artifact_root)
            log_path = str(process_result.artifact_root / "logs" / "pipeline.log")
            self.store.update_running(
                job.id,
                message="Summarizing transcript.",
                video_id=process_result.metadata.video_id,
                workspace_path=workspace_path,
                log_path=log_path,
            )
            summary_result = summarize_video(
                video_id=process_result.metadata.video_id,
                config=self.config,
                force=options.force,
            )
            final_md_path = _resolve_output_path(
                root=summary_result.artifact_root,
                path=summary_result.final_summary_path,
            )
            if final_md_path is None or not final_md_path.exists():
                raise RuntimeError("Summarization completed without a final.md path.")
            self.store.mark_done(
                job.id,
                video_id=summary_result.metadata.video_id,
                workspace_path=str(summary_result.artifact_root),
                final_md_path=str(final_md_path),
                log_path=str(summary_result.artifact_root / "logs" / "pipeline.log"),
            )
        except Exception as exc:
            recovered_paths = _recover_failure_paths(job=job, config=self.config)
            self.store.mark_failed(
                job.id,
                str(exc),
                video_id=recovered_paths.video_id,
                workspace_path=recovered_paths.workspace_path,
                log_path=recovered_paths.log_path,
            )


def create_app(*, config: PipelineConfig, jobs_db_path: Path, token: str, start_worker: bool = True) -> FastAPI:
    if not token:
        raise ValueError("A server API token is required.")
    store = JobStore(jobs_db_path)
    worker = JobWorker(store=store, config=config)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.initialize()
        reconcile_interrupted_jobs(store=store, config=config)
        if start_worker:
            worker.start()
        try:
            yield
        finally:
            if start_worker:
                worker.stop()

    app = FastAPI(title="voxcraft server", lifespan=lifespan)

    def require_token(
        authorization: Annotated[str | None, Header()] = None,
        x_voxcraft_token: Annotated[str | None, Header()] = None,
    ) -> None:
        supplied = _extract_bearer_token(authorization) or x_voxcraft_token
        if supplied is None or not secrets.compare_digest(supplied, token):
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid API token.")

    auth_dependency = Depends(require_token)

    @app.get("/healthz", dependencies=[auth_dependency])
    def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/jobs", dependencies=[auth_dependency], response_model=JobResponse, status_code=202)
    def create_job(request: CreateJobRequest, response: Response) -> JobResponse:
        _validate_request_against_config(request, config=config)
        job = store.create_job(url=request.url, options=request.to_job_options())
        response.headers["Location"] = f"/jobs/{job.id}"
        return _job_response(job)

    @app.get("/jobs/latest", dependencies=[auth_dependency], response_model=JobResponse)
    def latest_job() -> JobResponse:
        job = store.latest_job()
        if job is None:
            raise HTTPException(status_code=404, detail="No jobs found.")
        return _job_response(job)

    @app.get("/jobs/{job_id}", dependencies=[auth_dependency], response_model=JobResponse)
    def get_job(job_id: str) -> JobResponse:
        job = _require_job(store, job_id)
        return _job_response(job)

    @app.get("/jobs/{job_id}/final.md", dependencies=[auth_dependency], response_class=PlainTextResponse)
    def get_final_markdown(job_id: str) -> str:
        job = _require_job(store, job_id)
        if job.status != "done" or not job.final_md_path:
            raise HTTPException(status_code=409, detail="Job is not done yet.")
        path = Path(job.final_md_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="final.md was not found on disk.")
        return path.read_text(encoding="utf-8")

    @app.get("/jobs/{job_id}/log", dependencies=[auth_dependency], response_class=PlainTextResponse)
    def get_log(job_id: str) -> str:
        job = _require_job(store, job_id)
        if not job.log_path:
            raise HTTPException(status_code=404, detail="No log path recorded yet.")
        path = Path(job.log_path)
        if not path.exists():
            raise HTTPException(status_code=404, detail="Log file was not found on disk.")
        return path.read_text(encoding="utf-8")

    return app


class FailurePaths(BaseModel):
    video_id: str | None = None
    workspace_path: str | None = None
    log_path: str | None = None


def reconcile_interrupted_jobs(*, store: JobStore, config: PipelineConfig) -> int:
    reconciled_count = 0
    for job in store.running_jobs():
        paths = _recover_failure_paths(job=job, config=config)
        store.mark_failed(
            job.id,
            "Server restarted while this job was running.",
            message="Interrupted.",
            video_id=paths.video_id,
            workspace_path=paths.workspace_path,
            log_path=paths.log_path,
        )
        reconciled_count += 1
    return reconciled_count


def _validate_request_against_config(request: CreateJobRequest, *, config: PipelineConfig) -> None:
    effective_asr_backend = request.asr_backend or config.default_asr_backend
    if request.diarize and normalize_asr_backend(effective_asr_backend) == "whisper-cpp":
        raise HTTPException(
            status_code=422,
            detail="diarize is only supported with qwen3-asr",
        )


def _recover_failure_paths(*, job: JobRecord, config: PipelineConfig) -> FailurePaths:
    video_id = job.video_id or extract_youtube_id(job.url)
    root = Path(job.workspace_path) if job.workspace_path else None
    if root is not None and video_id is None:
        video_id = _read_workspace_video_id(root)
    if root is None and video_id is not None:
        root = resolve_artifact_paths(config.base_data_dir, video_id).root_dir
    if root is None:
        return FailurePaths(video_id=video_id)

    log_path = Path(job.log_path) if job.log_path else root / "logs" / "pipeline.log"
    workspace_path = str(root) if root.exists() else None
    log_path_string = str(log_path) if log_path.exists() else None

    return FailurePaths(
        video_id=video_id,
        workspace_path=workspace_path,
        log_path=log_path_string,
    )


def _read_workspace_video_id(root: Path) -> str | None:
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        return None
    try:
        metadata = read_json(metadata_path)
    except Exception:
        return None
    if not isinstance(metadata, dict):
        return None
    video_id = metadata.get("video_id")
    return video_id if isinstance(video_id, str) and video_id else None


def _job_response(job: JobRecord) -> JobResponse:
    return JobResponse(
        job=job,
        final_md_url=f"/jobs/{job.id}/final.md" if job.status == "done" and job.final_md_path else None,
        log_url=f"/jobs/{job.id}/log",
    )


def _require_job(store: JobStore, job_id: str) -> JobRecord:
    job = store.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Job not found.")
    return job


def _extract_bearer_token(authorization: str | None) -> str | None:
    if authorization is None:
        return None
    prefix = "Bearer "
    if not authorization.startswith(prefix):
        return None
    return authorization[len(prefix) :].strip() or None


def _resolve_output_path(*, root: Path, path: str | None) -> Path | None:
    if path is None:
        return None
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    return root / candidate
