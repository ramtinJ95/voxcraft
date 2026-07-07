from __future__ import annotations

import json
from email.message import Message
from io import BytesIO
from urllib.error import HTTPError

import pytest

from voxcraft.client import ServerClientError, VoxcraftServerClient


def _job_payload(status: str = "queued") -> dict[str, object]:
    return {
        "job": {
            "id": "job-123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "status": status,
            "created_at": "2026-07-07T00:00:00Z",
            "updated_at": "2026-07-07T00:00:00Z",
            "options": {},
        },
        "final_md_url": "/jobs/job-123/final.md" if status == "done" else None,
        "log_url": "/jobs/job-123/log",
    }


class FakeResponse:
    def __init__(self, body: str) -> None:
        self._body = body.encode("utf-8")
        self.headers = Message()

    def __enter__(self) -> FakeResponse:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        return None

    def read(self) -> bytes:
        return self._body


def test_server_client_posts_job_with_bearer_token(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        captured["authorization"] = request.get_header("Authorization")
        captured["content_type"] = request.get_header("Content-type")
        captured["timeout"] = timeout
        captured["body"] = json.loads(request.data.decode("utf-8"))
        return FakeResponse(json.dumps(_job_payload()))

    monkeypatch.setattr("voxcraft.client.urlopen", fake_urlopen)

    client = VoxcraftServerClient(base_url="http://mini.local:8765/", token="secret", timeout=12)
    response = client.create_job({"url": "https://www.youtube.com/watch?v=abc123", "diarize": True})

    assert response.job.id == "job-123"
    assert captured == {
        "url": "http://mini.local:8765/jobs",
        "method": "POST",
        "authorization": "Bearer secret",
        "content_type": "application/json",
        "timeout": 12,
        "body": {"url": "https://www.youtube.com/watch?v=abc123", "diarize": True},
    }


def test_server_client_fetches_final_markdown(monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_urlopen(request, timeout: float):
        captured["url"] = request.full_url
        captured["method"] = request.get_method()
        return FakeResponse("# Final Summary\n")

    monkeypatch.setattr("voxcraft.client.urlopen", fake_urlopen)

    client = VoxcraftServerClient(base_url="http://mini.local:8765", token="secret")

    assert client.get_final_markdown("job-123") == "# Final Summary\n"
    assert captured == {"url": "http://mini.local:8765/jobs/job-123/final.md", "method": "GET"}


def test_server_client_surfaces_http_error_detail(monkeypatch) -> None:
    def fake_urlopen(request, timeout: float):
        raise HTTPError(
            request.full_url,
            422,
            "Unprocessable Entity",
            hdrs=None,
            fp=BytesIO(b'{"detail":"diarize is only supported with qwen3-asr"}'),
        )

    monkeypatch.setattr("voxcraft.client.urlopen", fake_urlopen)
    client = VoxcraftServerClient(base_url="http://mini.local:8765", token="secret")

    with pytest.raises(ServerClientError) as exc_info:
        client.create_job({"url": "https://www.youtube.com/watch?v=abc123", "diarize": True})

    assert exc_info.value.status_code == 422
    assert str(exc_info.value) == "diarize is only supported with qwen3-asr"
