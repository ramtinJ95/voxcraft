from __future__ import annotations

from pathlib import Path
from typing import Any
from urllib.request import urlopen

from yt_dlp import YoutubeDL
from yt_dlp.utils import DownloadError

from .models import SubtitleCandidate
from .models import VideoMetadata
from .utils import write_json

EXTENSION_PRIORITY = {
    "vtt": 0,
    "srt": 1,
    "srv3": 2,
    "ttml": 3,
}


def build_probe_options() -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": False,
        "noplaylist": True,
    }


def build_subtitle_download_options(
    language: str,
    source_dir: Path,
) -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "writesubtitles": True,
        "writeautomaticsub": False,
        "subtitleslangs": [language],
        "subtitlesformat": "vtt/srt/best",
        "paths": {"home": str(source_dir)},
        "outtmpl": {"default": "%(id)s.%(ext)s"},
        "noplaylist": True,
    }


def build_audio_download_options(source_dir: Path) -> dict[str, Any]:
    return {
        "quiet": True,
        "no_warnings": True,
        "format": "bestaudio/best",
        "paths": {"home": str(source_dir)},
        "outtmpl": {"default": "audio.%(ext)s"},
        "noplaylist": True,
    }


def choose_subtitle_candidate(
    subtitles: dict[str, list[dict[str, Any]]] | None,
    preferred_language: str = "en",
) -> SubtitleCandidate | None:
    subtitles = subtitles or {}
    preferred_language = preferred_language.lower()

    def best_track(
        language: str,
        tracks: list[dict[str, Any]] | None,
        is_automatic: bool,
    ) -> SubtitleCandidate | None:
        if not tracks:
            return None

        ranked_tracks = sorted(
            tracks,
            key=lambda track: EXTENSION_PRIORITY.get(track.get("ext", ""), 99),
        )
        chosen = ranked_tracks[0]
        return SubtitleCandidate(
            language=language,
            ext=chosen.get("ext", "vtt"),
            url=chosen.get("url"),
            name=chosen.get("name"),
            is_automatic=is_automatic,
        )

    ordered_languages: list[str] = []
    for language in ("en", preferred_language):
        if language and language not in ordered_languages:
            ordered_languages.append(language)

    for language in ordered_languages:
        candidate = best_track(language, subtitles.get(language), is_automatic=False)
        if candidate is not None:
            return candidate

    return None


def probe_video(url: str) -> tuple[VideoMetadata, dict[str, Any]]:
    try:
        with YoutubeDL(build_probe_options()) as ydl:
            info = ydl.extract_info(url, download=False)
            sanitized = ydl.sanitize_info(info)
    except DownloadError as exc:
        raise RuntimeError(f"Failed to probe video metadata: {exc}") from exc

    metadata = VideoMetadata(
        video_id=sanitized["id"],
        url=sanitized.get("webpage_url") or sanitized.get("original_url") or url,
        title=sanitized.get("title"),
        channel=sanitized.get("channel") or sanitized.get("uploader"),
        duration_sec=float(sanitized["duration"]) if sanitized.get("duration") else None,
        subtitles=_subtitle_map(sanitized.get("subtitles")),
        automatic_captions=_subtitle_map(sanitized.get("automatic_captions")),
    )
    return metadata, sanitized


def write_metadata_artifacts(
    metadata: VideoMetadata,
    raw_info: dict[str, Any],
    metadata_path: Path,
    info_path: Path,
) -> None:
    write_json(metadata_path, metadata.model_dump(mode="json"))
    write_json(info_path, raw_info)


def download_subtitle_file(
    url: str,
    source_dir: Path,
    candidate: SubtitleCandidate,
    force: bool = False,
) -> Path:
    existing = _find_standardized_subtitle_file(source_dir, candidate.language)
    if existing is not None and not force:
        return existing

    try:
        with YoutubeDL(
            build_subtitle_download_options(
                language=candidate.language,
                source_dir=source_dir,
            )
        ) as ydl:
            ydl.extract_info(url, download=True)
    except DownloadError:
        if candidate.url is None:
            raise
        direct_path = source_dir / f"subtitles.{candidate.language}.{_preferred_subtitle_suffix(candidate)}"
        with urlopen(candidate.url) as response:
            direct_path.write_bytes(response.read())
        return direct_path

    downloaded = _find_downloaded_subtitle_file(source_dir, candidate.language)
    if downloaded is None:
        if candidate.url is None:
            raise RuntimeError(f"Subtitle download completed but no subtitle file was found for {candidate.language}.")
        direct_path = source_dir / f"subtitles.{candidate.language}.{_preferred_subtitle_suffix(candidate)}"
        with urlopen(candidate.url) as response:
            direct_path.write_bytes(response.read())
        return direct_path

    target = source_dir / f"subtitles.{candidate.language}{downloaded.suffix.lower()}"
    if downloaded != target:
        if target.exists():
            target.unlink()
        downloaded.replace(target)
    return target


def download_audio_file(
    url: str,
    source_dir: Path,
    force: bool = False,
) -> Path:
    existing = _find_audio_file(source_dir)
    if existing is not None and not force:
        return existing

    try:
        with YoutubeDL(build_audio_download_options(source_dir)) as ydl:
            info = ydl.extract_info(url, download=True)
            path = _extract_requested_filepath(info)
    except DownloadError as exc:
        raise RuntimeError(f"Failed to download audio: {exc}") from exc

    if path is not None and path.exists():
        return path

    downloaded = _find_audio_file(source_dir)
    if downloaded is None:
        raise RuntimeError("Audio download completed but no audio file was found.")
    return downloaded


def _subtitle_map(raw_map: dict[str, list[dict[str, Any]]] | None) -> dict[str, list[SubtitleCandidate]]:
    if not raw_map:
        return {}

    mapped: dict[str, list[SubtitleCandidate]] = {}
    for language, tracks in raw_map.items():
        mapped[language] = [
            SubtitleCandidate(
                language=language,
                ext=track.get("ext", "vtt"),
                url=track.get("url"),
                name=track.get("name"),
            )
            for track in tracks
        ]
    return mapped


def _find_standardized_subtitle_file(source_dir: Path, language: str) -> Path | None:
    for suffix in (".vtt", ".srt"):
        candidate = source_dir / f"subtitles.{language}{suffix}"
        if candidate.exists():
            return candidate
    return None


def _find_downloaded_subtitle_file(source_dir: Path, language: str) -> Path | None:
    patterns = (
        f"subtitles.{language}.vtt",
        f"subtitles.{language}.srt",
        f"*.{language}*.vtt",
        f"*.{language}*.srt",
    )
    for pattern in patterns:
        matches = sorted(path for path in source_dir.glob(pattern) if path.is_file())
        if matches:
            return matches[0]
    return None


def _find_audio_file(source_dir: Path) -> Path | None:
    candidates = sorted(
        path
        for path in source_dir.glob("audio.*")
        if path.is_file() and path.suffix.lower() not in {".part", ".ytdl", ".wav"}
    )
    return candidates[0] if candidates else None


def _extract_requested_filepath(info: dict[str, Any]) -> Path | None:
    requested = info.get("requested_downloads") or []
    for item in requested:
        filepath = item.get("filepath")
        if filepath:
            return Path(filepath)
    filepath = info.get("_filename")
    if filepath:
        return Path(filepath)
    return None


def _preferred_subtitle_suffix(candidate: SubtitleCandidate) -> str:
    if candidate.ext.lower() in {"vtt", "srt"}:
        return candidate.ext.lower()
    return "vtt"
