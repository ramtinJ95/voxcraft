from __future__ import annotations

import json
import time
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from pydantic import BaseModel

from .jobs import JobRecord

SERVER_URL_ENV_VAR = "VOXCRAFT_SERVER_URL"
SERVER_TOKEN_ENV_VAR = "VOXCRAFT_SERVER_TOKEN"
DEFAULT_SERVER_URL = "http://127.0.0.1:8765"


class ServerJobResponse(BaseModel):
    job: JobRecord
    final_md_url: str | None = None
    log_url: str


class ServerClientError(RuntimeError):
    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class VoxcraftServerClient:
    def __init__(self, *, base_url: str = DEFAULT_SERVER_URL, token: str, timeout: float = 30.0) -> None:
        if not token:
            raise ValueError("A server API token is required.")
        self.base_url = base_url.rstrip("/")
        self.token = token
        self.timeout = timeout

    def create_job(self, payload: dict[str, Any]) -> ServerJobResponse:
        return ServerJobResponse.model_validate(self._request_json("POST", "/jobs", payload=payload))

    def get_job(self, job_id: str) -> ServerJobResponse:
        return ServerJobResponse.model_validate(self._request_json("GET", f"/jobs/{job_id}"))

    def get_latest_job(self) -> ServerJobResponse:
        return ServerJobResponse.model_validate(self._request_json("GET", "/jobs/latest"))

    def get_final_markdown(self, job_id: str) -> str:
        return self._request_text("GET", f"/jobs/{job_id}/final.md")

    def get_log(self, job_id: str) -> str:
        return self._request_text("GET", f"/jobs/{job_id}/log")

    def wait_for_job(
        self,
        job_id: str,
        *,
        timeout_sec: float,
        poll_interval_sec: float = 10.0,
    ) -> ServerJobResponse:
        deadline = time.monotonic() + timeout_sec
        response = self.get_job(job_id)
        while response.job.status in {"queued", "running"} and time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            time.sleep(min(poll_interval_sec, remaining))
            response = self.get_job(job_id)
        return response

    def _request_json(
        self,
        method: str,
        path: str,
        *,
        payload: dict[str, Any] | None = None,
    ) -> Any:
        body = _json_body(payload) if payload is not None else None
        request = self._build_request(method, path, body=body)
        if body is not None:
            request.add_header("Content-Type", "application/json")
        text = self._open_text(request)
        try:
            return json.loads(text)
        except json.JSONDecodeError as exc:
            raise ServerClientError("Server returned invalid JSON.") from exc

    def _request_text(self, method: str, path: str) -> str:
        return self._open_text(self._build_request(method, path))

    def _build_request(self, method: str, path: str, *, body: bytes | None = None) -> Request:
        request = Request(
            _join_url(self.base_url, path),
            data=body,
            method=method,
        )
        request.add_header("Authorization", f"Bearer {self.token}")
        request.add_header("Accept", "application/json, text/plain")
        return request

    def _open_text(self, request: Request) -> str:
        try:
            with urlopen(request, timeout=self.timeout) as response:
                return response.read().decode(_response_charset(response.headers.get_content_charset()))
        except HTTPError as exc:
            raise ServerClientError(_http_error_message(exc), status_code=exc.code) from exc
        except URLError as exc:
            raise ServerClientError(f"Could not reach voxcraft server: {exc.reason}") from exc


def _json_body(payload: dict[str, Any]) -> bytes:
    return json.dumps(payload).encode("utf-8")


def _join_url(base_url: str, path: str) -> str:
    if not path.startswith("/"):
        path = f"/{path}"
    return f"{base_url}{path}"


def _response_charset(charset: str | None) -> str:
    return charset or "utf-8"


def _http_error_message(exc: HTTPError) -> str:
    raw_body = exc.read().decode("utf-8", errors="replace").strip()
    if raw_body:
        try:
            payload = json.loads(raw_body)
        except json.JSONDecodeError:
            return raw_body
        detail = payload.get("detail") if isinstance(payload, dict) else None
        if isinstance(detail, str):
            return detail
        if detail is not None:
            return json.dumps(detail, ensure_ascii=False)
    return f"Server returned HTTP {exc.code}."
