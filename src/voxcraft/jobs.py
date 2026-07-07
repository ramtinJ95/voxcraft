from __future__ import annotations

import json
import secrets
import sqlite3
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

JobStatus = Literal["queued", "running", "done", "failed"]


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="microseconds").replace("+00:00", "Z")


def new_job_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    return f"{timestamp}-{secrets.token_hex(3)}"


class JobOptions(BaseModel):
    language: str | None = None
    high_quality: bool = False
    force: bool = False
    asr_backend: str | None = None
    model: str | None = None
    diarize: bool = False
    num_speakers: int | None = Field(default=None, ge=1)
    min_speakers: int = Field(default=1, ge=1)
    max_speakers: int = Field(default=8, ge=1)


class JobRecord(BaseModel):
    id: str
    url: str
    status: JobStatus
    created_at: str
    updated_at: str
    started_at: str | None = None
    finished_at: str | None = None
    video_id: str | None = None
    workspace_path: str | None = None
    final_md_path: str | None = None
    log_path: str | None = None
    message: str | None = None
    error: str | None = None
    options: JobOptions = Field(default_factory=JobOptions)


class JobStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.RLock()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    status TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    started_at TEXT,
                    finished_at TEXT,
                    video_id TEXT,
                    workspace_path TEXT,
                    final_md_path TEXT,
                    log_path TEXT,
                    message TEXT,
                    error TEXT,
                    options_json TEXT NOT NULL
                )
                """
            )

    def create_job(self, url: str, options: JobOptions | None = None) -> JobRecord:
        now = utc_now()
        job = JobRecord(
            id=new_job_id(),
            url=url,
            status="queued",
            created_at=now,
            updated_at=now,
            message="Queued.",
            options=options or JobOptions(),
        )
        with self._locked_connection() as connection:
            connection.execute(
                """
                INSERT INTO jobs (
                    id, url, status, created_at, updated_at, message, options_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    job.id,
                    job.url,
                    job.status,
                    job.created_at,
                    job.updated_at,
                    job.message,
                    job.options.model_dump_json(),
                ),
            )
        return job

    def get_job(self, job_id: str) -> JobRecord | None:
        with self._locked_connection() as connection:
            row = connection.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return self._row_to_job(row) if row is not None else None

    def latest_job(self) -> JobRecord | None:
        with self._locked_connection() as connection:
            row = connection.execute("SELECT * FROM jobs ORDER BY created_at DESC, id DESC LIMIT 1").fetchone()
        return self._row_to_job(row) if row is not None else None

    def running_jobs(self) -> list[JobRecord]:
        with self._locked_connection() as connection:
            rows = connection.execute(
                "SELECT * FROM jobs WHERE status = 'running' ORDER BY started_at ASC, id ASC"
            ).fetchall()
        return [self._row_to_job(row) for row in rows]

    def claim_next_queued(self) -> JobRecord | None:
        now = utc_now()
        with self._locked_connection() as connection:
            row = connection.execute(
                "SELECT * FROM jobs WHERE status = 'queued' ORDER BY created_at ASC, id ASC LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            connection.execute(
                """
                UPDATE jobs
                SET status = 'running', started_at = ?, updated_at = ?, message = ?, error = NULL
                WHERE id = ? AND status = 'queued'
                """,
                (now, now, "Starting pipeline.", row["id"]),
            )
            updated = connection.execute("SELECT * FROM jobs WHERE id = ?", (row["id"],)).fetchone()
        return self._row_to_job(updated) if updated is not None else None

    def update_running(
        self,
        job_id: str,
        *,
        message: str | None = None,
        video_id: str | None = None,
        workspace_path: str | None = None,
        final_md_path: str | None = None,
        log_path: str | None = None,
    ) -> None:
        updates: dict[str, Any] = {"updated_at": utc_now()}
        if message is not None:
            updates["message"] = message
        if video_id is not None:
            updates["video_id"] = video_id
        if workspace_path is not None:
            updates["workspace_path"] = workspace_path
        if final_md_path is not None:
            updates["final_md_path"] = final_md_path
        if log_path is not None:
            updates["log_path"] = log_path
        self._update(job_id, updates)

    def mark_done(
        self,
        job_id: str,
        *,
        video_id: str,
        workspace_path: str,
        final_md_path: str,
        log_path: str | None,
        message: str = "Done.",
    ) -> None:
        now = utc_now()
        self._update(
            job_id,
            {
                "status": "done",
                "updated_at": now,
                "finished_at": now,
                "video_id": video_id,
                "workspace_path": workspace_path,
                "final_md_path": final_md_path,
                "log_path": log_path,
                "message": message,
                "error": None,
            },
        )

    def mark_failed(
        self,
        job_id: str,
        error: str,
        *,
        message: str = "Failed.",
        video_id: str | None = None,
        workspace_path: str | None = None,
        log_path: str | None = None,
    ) -> None:
        now = utc_now()
        updates: dict[str, Any] = {
            "status": "failed",
            "updated_at": now,
            "finished_at": now,
            "message": message,
            "error": error,
        }
        if video_id is not None:
            updates["video_id"] = video_id
        if workspace_path is not None:
            updates["workspace_path"] = workspace_path
        if log_path is not None:
            updates["log_path"] = log_path
        self._update(job_id, updates)

    def _update(self, job_id: str, updates: dict[str, Any]) -> None:
        if not updates:
            return
        assignments = ", ".join(f"{column} = ?" for column in updates)
        values = [updates[column] for column in updates]
        with self._locked_connection() as connection:
            connection.execute(f"UPDATE jobs SET {assignments} WHERE id = ?", [*values, job_id])

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        connection.row_factory = sqlite3.Row
        return connection

    def _locked_connection(self):
        class LockedConnection:
            def __init__(self, store: JobStore) -> None:
                self.store = store
                self.connection: sqlite3.Connection | None = None

            def __enter__(self) -> sqlite3.Connection:
                self.store._lock.acquire()
                self.connection = self.store._connect()
                self.connection.execute("BEGIN IMMEDIATE")
                return self.connection

            def __exit__(self, exc_type, exc, traceback) -> None:
                assert self.connection is not None
                try:
                    if exc_type is None:
                        self.connection.commit()
                    else:
                        self.connection.rollback()
                finally:
                    self.connection.close()
                    self.store._lock.release()

        return LockedConnection(self)

    @staticmethod
    def _row_to_job(row: sqlite3.Row) -> JobRecord:
        return JobRecord(
            id=row["id"],
            url=row["url"],
            status=row["status"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            started_at=row["started_at"],
            finished_at=row["finished_at"],
            video_id=row["video_id"],
            workspace_path=row["workspace_path"],
            final_md_path=row["final_md_path"],
            log_path=row["log_path"],
            message=row["message"],
            error=row["error"],
            options=JobOptions.model_validate(json.loads(row["options_json"])),
        )
