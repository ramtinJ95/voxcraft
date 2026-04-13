from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .models import (
    ArtifactPaths,
    ChunkManifestEntry,
    SourceKind,
    SummaryPayload,
    TranscriptionDetails,
    VideoMetadata,
)
from .utils import human_video_dirname, path_string, read_json, write_json


def build_artifact_paths(root_dir: Path, video_id: str) -> ArtifactPaths:
    source_dir = root_dir / "source"
    transcript_dir = root_dir / "transcript"
    chunks_dir = root_dir / "chunks"
    summary_input_dir = root_dir / "summary_input"
    summary_dir = root_dir / "summary"
    summary_prompts_dir = summary_dir / "prompts"
    logs_dir = root_dir / "logs"

    return ArtifactPaths(
        video_id=video_id,
        root_dir=root_dir,
        metadata_path=root_dir / "metadata.json",
        source_dir=source_dir,
        info_path=source_dir / "info.json",
        transcript_dir=transcript_dir,
        raw_transcript_path=transcript_dir / "raw.txt",
        clean_transcript_path=transcript_dir / "clean.txt",
        segments_path=transcript_dir / "segments.json",
        speaker_segments_path=transcript_dir / "speaker_segments.json",
        asr_output_json_path=transcript_dir / "asr_output.json",
        transcript_srt_path=transcript_dir / "transcript.srt",
        chunks_dir=chunks_dir,
        chunk_index_path=chunks_dir / "index.json",
        summary_input_dir=summary_input_dir,
        summary_payload_path=summary_input_dir / "payload.json",
        summary_dir=summary_dir,
        summary_manifest_path=summary_dir / "manifest.json",
        summary_prompts_dir=summary_prompts_dir,
        summary_final_path=summary_dir / "final.md",
        summary_final_prompt_path=summary_prompts_dir / "final.prompt.txt",
        logs_dir=logs_dir,
        pipeline_log_path=logs_dir / "pipeline.log",
    )


def resolve_artifact_paths(
    base_dir: Path,
    video_id: str,
    title: str | None = None,
) -> ArtifactPaths:
    return build_artifact_paths(
        root_dir=resolve_video_root(base_dir=base_dir, video_id=video_id, title=title),
        video_id=video_id,
    )


def resolve_video_root(
    base_dir: Path,
    video_id: str,
    title: str | None = None,
) -> Path:
    existing = find_existing_video_root(base_dir=base_dir, video_id=video_id)
    if existing is not None:
        return existing
    return base_dir / human_video_dirname(video_id=video_id, title=title)


def find_existing_video_root(base_dir: Path, video_id: str) -> Path | None:
    legacy = base_dir / video_id
    if legacy.is_dir():
        return legacy

    matches = sorted(path for path in base_dir.glob(f"*--{video_id}") if path.is_dir())
    if matches:
        return matches[0]

    for candidate in _metadata_dirs(base_dir):
        metadata_path = candidate / "metadata.json"
        if not metadata_path.exists():
            continue
        try:
            metadata = read_json(metadata_path)
        except Exception:
            continue
        if metadata.get("video_id") == video_id:
            return candidate

    return None


def _metadata_dirs(base_dir: Path) -> Iterable[Path]:
    if not base_dir.exists():
        return []
    return [path for path in base_dir.iterdir() if path.is_dir()]


def initialize_workspace(paths: ArtifactPaths) -> ArtifactPaths:
    for directory in (
        paths.root_dir,
        paths.source_dir,
        paths.transcript_dir,
        paths.chunks_dir,
        paths.summary_input_dir,
        paths.summary_dir,
        paths.summary_prompts_dir,
        paths.logs_dir,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    return paths


def build_summary_payload(
    metadata: VideoMetadata,
    source_kind: SourceKind,
    chunk_manifest: list[ChunkManifestEntry],
    paths: ArtifactPaths,
    segment_count: int = 0,
    artifacts: dict[str, str] | None = None,
    transcription: TranscriptionDetails | None = None,
    notes: list[str] | None = None,
) -> SummaryPayload:
    return SummaryPayload(
        video_id=metadata.video_id,
        url=metadata.url,
        title=metadata.title,
        channel=metadata.channel,
        duration_sec=metadata.duration_sec,
        source_kind=source_kind,
        transcript_path=path_string(paths.clean_transcript_path, paths.root_dir),
        segments_path=path_string(paths.segments_path, paths.root_dir),
        chunk_index_path=path_string(paths.chunk_index_path, paths.root_dir),
        chunk_count=len(chunk_manifest),
        segment_count=segment_count,
        artifacts=artifacts or {},
        transcription=transcription,
        notes=notes or [],
    )


def write_summary_payload(payload: SummaryPayload, destination: Path) -> None:
    write_json(destination, payload.model_dump(mode="json"))
