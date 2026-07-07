from __future__ import annotations

from pathlib import Path

import pytest

from voxcraft.jobs import JobOptions, JobStore


def test_job_store_claims_jobs_fifo_and_persists_options(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()

    first = store.create_job("https://www.youtube.com/watch?v=first", JobOptions(language="en"))
    second = store.create_job("https://www.youtube.com/watch?v=second", JobOptions(force=True))

    claimed = store.claim_next_queued()

    assert claimed is not None
    assert claimed.id == first.id
    assert claimed.status == "running"
    assert claimed.options.language == "en"
    assert store.get_job(second.id).status == "queued"  # type: ignore[union-attr]


def test_job_store_marks_interrupted_running_jobs_failed(tmp_path: Path) -> None:
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    job = store.create_job("https://www.youtube.com/watch?v=abc123")
    assert store.claim_next_queued() is not None

    interrupted = store.mark_interrupted_running_jobs()
    updated = store.get_job(job.id)

    assert interrupted == 1
    assert updated is not None
    assert updated.status == "failed"
    assert updated.error == "Server restarted while this job was running."


def test_server_requires_token(tmp_path: Path) -> None:
    fastapi = pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from voxcraft.config import PipelineConfig
    from voxcraft.server import create_app

    app = create_app(
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        jobs_db_path=tmp_path / "jobs.sqlite3",
        token="secret",
        start_worker=False,
    )

    with TestClient(app) as client:
        assert client.get("/healthz").status_code == 401
        assert client.get("/healthz", headers={"Authorization": "Bearer secret"}).json() == {"status": "ok"}


def test_server_creates_queued_job(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from voxcraft.config import PipelineConfig
    from voxcraft.server import create_app

    app = create_app(
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        jobs_db_path=tmp_path / "jobs.sqlite3",
        token="secret",
        start_worker=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            headers={"X-Voxcraft-Token": "secret"},
            json={"url": "https://www.youtube.com/watch?v=abc123", "language": "en"},
        )

    assert response.status_code == 202
    payload = response.json()
    assert payload["job"]["url"] == "https://www.youtube.com/watch?v=abc123"
    assert payload["job"]["status"] == "queued"
    assert payload["job"]["options"]["language"] == "en"
    assert payload["log_url"].endswith("/log")


def test_server_rejects_diarization_when_default_backend_is_whisper(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from voxcraft.config import PipelineConfig
    from voxcraft.server import create_app

    app = create_app(
        config=PipelineConfig(base_data_dir=tmp_path / "videos", default_asr_backend="whisper-cpp"),
        jobs_db_path=tmp_path / "jobs.sqlite3",
        token="secret",
        start_worker=False,
    )

    with TestClient(app) as client:
        response = client.post(
            "/jobs",
            headers={"X-Voxcraft-Token": "secret"},
            json={"url": "https://www.youtube.com/watch?v=abc123", "diarize": True},
        )

    assert response.status_code == 422
    assert response.json()["detail"] == "diarize is only supported with qwen3-asr"


def test_worker_recovers_log_path_when_processing_fails(monkeypatch, tmp_path: Path) -> None:
    from voxcraft.config import PipelineConfig
    from voxcraft.server import JobWorker

    config = PipelineConfig(base_data_dir=tmp_path / "videos")
    store = JobStore(tmp_path / "jobs.sqlite3")
    store.initialize()
    job = store.create_job("https://www.youtube.com/watch?v=abc123")
    running_job = store.claim_next_queued()
    assert running_job is not None

    log_path = tmp_path / "videos" / "demo--abc123" / "logs" / "pipeline.log"

    def fake_process_video(**kwargs):
        log_path.parent.mkdir(parents=True)
        log_path.write_text("started\nfailed\n", encoding="utf-8")
        raise RuntimeError("transcription failed")

    monkeypatch.setattr("voxcraft.server.process_video", fake_process_video)

    JobWorker(store=store, config=config)._run_job(running_job)
    updated = store.get_job(job.id)

    assert updated is not None
    assert updated.status == "failed"
    assert updated.error == "transcription failed"
    assert updated.video_id == "abc123"
    assert updated.log_path == str(log_path)


def test_server_returns_final_markdown_for_done_job(tmp_path: Path) -> None:
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient

    from voxcraft.config import PipelineConfig
    from voxcraft.server import create_app

    jobs_db_path = tmp_path / "jobs.sqlite3"
    final_path = tmp_path / "videos" / "demo--abc123" / "final.md"
    final_path.parent.mkdir(parents=True)
    final_path.write_text("# Final Summary\n", encoding="utf-8")
    store = JobStore(jobs_db_path)
    store.initialize()
    job = store.create_job("https://www.youtube.com/watch?v=abc123")
    store.mark_done(
        job.id,
        video_id="abc123",
        workspace_path=str(final_path.parent),
        final_md_path=str(final_path),
        log_path=None,
    )

    app = create_app(
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        jobs_db_path=jobs_db_path,
        token="secret",
        start_worker=False,
    )

    with TestClient(app) as client:
        response = client.get(f"/jobs/{job.id}/final.md", headers={"Authorization": "Bearer secret"})

    assert response.status_code == 200
    assert response.text == "# Final Summary\n"
