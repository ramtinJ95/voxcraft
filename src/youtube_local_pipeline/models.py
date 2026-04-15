from __future__ import annotations

from enum import Enum
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class SourceKind(str, Enum):
    MANUAL_SUBTITLES = "subtitles"
    LOCAL_ASR = "local-asr"


class SubtitleCandidate(BaseModel):
    language: str
    ext: str = "vtt"
    url: str | None = None
    name: str | None = None
    is_automatic: bool = False


class VideoMetadata(BaseModel):
    video_id: str
    url: str
    title: str | None = None
    channel: str | None = None
    duration_sec: float | None = None
    subtitles: dict[str, list[SubtitleCandidate]] = Field(default_factory=dict)
    automatic_captions: dict[str, list[SubtitleCandidate]] = Field(default_factory=dict)


class TranscriptSegment(BaseModel):
    start_sec: float
    end_sec: float
    text: str
    speaker: str | None = None


class TranscriptChunk(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    text: str


class ChunkManifestEntry(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    path: str
    char_count: int


class ArtifactPaths(BaseModel):
    video_id: str
    root_dir: Path
    metadata_path: Path
    source_dir: Path
    info_path: Path
    transcript_dir: Path
    raw_transcript_path: Path
    clean_transcript_path: Path
    segments_path: Path
    speaker_segments_path: Path
    asr_output_json_path: Path
    transcript_srt_path: Path
    chunks_dir: Path
    chunk_index_path: Path
    summary_input_dir: Path
    summary_payload_path: Path
    summary_dir: Path
    summary_manifest_path: Path
    summary_prompts_dir: Path
    summary_final_path: Path
    summary_final_prompt_path: Path
    logs_dir: Path
    pipeline_log_path: Path


class TranscriptionDetails(BaseModel):
    backend: str | None = None
    model: str | None = None
    model_path: str | None = None
    threads: int | None = None
    language: str | None = None
    diarized: bool | None = None
    speaker_count: int | None = None


class SummaryPayload(BaseModel):
    video_id: str
    url: str
    title: str | None = None
    channel: str | None = None
    duration_sec: float | None = None
    source_kind: SourceKind
    transcript_path: str
    segments_path: str
    chunk_index_path: str
    chunk_count: int
    segment_count: int = 0
    artifacts: dict[str, str] = Field(default_factory=dict)
    transcription: TranscriptionDetails | None = None
    notes: list[str] = Field(default_factory=list)


class ProcessResult(BaseModel):
    metadata: VideoMetadata
    source_kind: SourceKind
    artifact_root: Path
    chunk_count: int = 0
    notes: list[str] = Field(default_factory=list)
    used_cache: bool = False
    subtitle_path: str | None = None
    audio_source_path: str | None = None
    normalized_audio_path: str | None = None
    transcription: TranscriptionDetails | None = None
    summary_manifest_path: str | None = None
    final_summary_path: str | None = None


class ChunkSummaryEntry(BaseModel):
    index: int
    start_sec: float
    end_sec: float
    source_chunk_path: str
    prompt_path: str
    output_path: str


class SummaryManifest(BaseModel):
    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_codex_fields(cls, data):
        if not isinstance(data, dict):
            return data
        updated = dict(data)
        if "summary_provider" not in updated:
            updated["summary_provider"] = "codex"
        if "summary_command" not in updated and "codex_command" in updated:
            updated["summary_command"] = updated["codex_command"]
        if "summary_model" not in updated and "codex_model" in updated:
            updated["summary_model"] = updated["codex_model"]
        if "summary_thinking_level" not in updated:
            if "summary_reasoning_effort" in updated:
                updated["summary_thinking_level"] = updated["summary_reasoning_effort"]
            elif "codex_reasoning_effort" in updated:
                updated["summary_thinking_level"] = updated["codex_reasoning_effort"]
        return updated

    video_id: str
    summary_provider: str = "codex"
    summary_command: str
    summary_model: str | None = None
    summary_thinking_level: str | None = None
    chunk_summaries: list[ChunkSummaryEntry] = Field(default_factory=list)
    final_prompt_path: str
    final_summary_path: str
