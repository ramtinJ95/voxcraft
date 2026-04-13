from __future__ import annotations

from pathlib import Path
import json

from .audio import normalize_audio
from .chunk import chunk_segments, write_chunk_index, write_chunks
from .config import PipelineConfig
from .download import (
    choose_subtitle_candidate,
    download_audio_file,
    download_subtitle_file,
    probe_video,
    write_metadata_artifacts,
)
from .manifest import (
    build_summary_payload,
    initialize_workspace,
    resolve_artifact_paths,
    write_summary_payload,
)
from .models import (
    ProcessResult,
    SourceKind,
    SummaryPayload,
    TranscriptionDetails,
    VideoMetadata,
)
from .subtitles import load_segments, parse_subtitle_file, write_transcript_artifacts
from .transcribe import build_transcription_request, transcribe_audio_file
from .utils import append_log, path_string, read_json


def process_video(
    url: str,
    config: PipelineConfig,
    language: str | None = None,
    high_quality: bool = False,
    force: bool = False,
    asr_backend: str | None = None,
    model: str | None = None,
    diarize: bool = False,
    diarization_num_speakers: int | None = None,
    diarization_min_speakers: int = 1,
    diarization_max_speakers: int = 8,
    dry_run: bool = False,
) -> ProcessResult:
    metadata, raw_info = probe_video(url)
    candidate = choose_subtitle_candidate(
        subtitles={key: [item.model_dump(mode="json") for item in value] for key, value in metadata.subtitles.items()},
        preferred_language=language or config.language_preference,
    )
    source_kind = _planned_source_kind(candidate)
    transcription_details = _planned_transcription_details(
        config=config,
        language=language,
        high_quality=high_quality,
        asr_backend=asr_backend,
        model=model,
        diarize=diarize,
        diarization_num_speakers=diarization_num_speakers,
        source_kind=source_kind,
    )

    if dry_run:
        notes = [
            "Dry run only: metadata probed, no media downloaded.",
            _planned_note(candidate, metadata),
        ]
        return ProcessResult(
            metadata=metadata,
            source_kind=source_kind,
            artifact_root=config.video_root(metadata.video_id, title=metadata.title),
            notes=notes,
            transcription=transcription_details,
        )

    paths = initialize_workspace(
        resolve_artifact_paths(config.base_data_dir, metadata.video_id, title=metadata.title)
    )
    append_log(paths.pipeline_log_path, f"Starting pipeline for {metadata.url}")
    write_metadata_artifacts(
        metadata=metadata,
        raw_info=raw_info,
        metadata_path=paths.metadata_path,
        info_path=paths.info_path,
    )
    append_log(paths.pipeline_log_path, "Saved metadata artifacts")

    if _can_reuse_cached_artifacts(
        paths,
        config=config,
        force=force,
        requested_source_kind=source_kind,
        requested_transcription=transcription_details,
    ):
        append_log(paths.pipeline_log_path, "Reusing cached final artifacts")
        return _load_cached_result(metadata, paths)

    notes: list[str] = []
    subtitle_path: Path | None = None
    audio_source_path: Path | None = None
    normalized_audio_path: Path | None = None

    if config.subtitle_first and candidate is not None:
        append_log(paths.pipeline_log_path, f"Attempting subtitle-first processing with {candidate.language}")
        subtitle_path = download_subtitle_file(
            url=metadata.url,
            source_dir=paths.source_dir,
            candidate=candidate,
            force=force,
        )
        segments = parse_subtitle_file(subtitle_path)
        if segments:
            source_kind = SourceKind.MANUAL_SUBTITLES
            write_transcript_artifacts(paths, segments)
            notes.append(_planned_note(candidate, metadata))
            append_log(paths.pipeline_log_path, f"Subtitle branch succeeded with {subtitle_path.name}")
        else:
            notes.append("Subtitle file was empty after parsing; falling back to local ASR.")
            append_log(paths.pipeline_log_path, "Subtitle parse returned no segments; falling back to ASR")
            source_kind = SourceKind.LOCAL_ASR
    else:
        source_kind = SourceKind.LOCAL_ASR

    if source_kind == SourceKind.LOCAL_ASR:
        append_log(paths.pipeline_log_path, "Starting local ASR branch")
        audio_source_path = download_audio_file(
            url=metadata.url,
            source_dir=paths.source_dir,
            force=force,
        )
        normalized_audio_path = paths.source_dir / "audio.wav"
        if force or not normalized_audio_path.exists():
            normalize_audio(
                input_path=audio_source_path,
                output_path=normalized_audio_path,
                sample_rate=config.audio_sample_rate,
                channels=config.audio_channels,
            )
            append_log(paths.pipeline_log_path, f"Normalized audio to {normalized_audio_path.name}")
        else:
            append_log(paths.pipeline_log_path, f"Reused normalized audio {normalized_audio_path.name}")

        request = build_transcription_request(
            input_path=normalized_audio_path,
            config=config,
            language=language,
            high_quality=high_quality,
            asr_backend=asr_backend,
            model=model,
        )
        transcription_result = transcribe_audio_file(
            request=request,
            qwen_command=config.qwen_command,
            qwen_context=config.qwen_context,
            qwen_diarize=diarize,
            qwen_num_speakers=diarization_num_speakers,
            qwen_min_speakers=diarization_min_speakers,
            qwen_max_speakers=diarization_max_speakers,
            qwen_forced_aligner=config.qwen_forced_aligner,
            qwen_pyannote_model=config.qwen_pyannote_model,
            qwen_dtype=config.qwen_dtype,
            qwen_draft_model=config.qwen_draft_model,
            qwen_num_draft_tokens=config.qwen_num_draft_tokens,
            whisper_cpp_model_path=config.whisper_cpp_model_path,
            whisper_cpp_model_dir=config.whisper_cpp_model_dir,
            whisper_cpp_command=config.whisper_cpp_command,
            whisper_cpp_threads=config.whisper_cpp_threads,
            output_base=paths.asr_output_json_path.with_suffix(""),
        )
        if transcription_result.speaker_segments:
            paths.speaker_segments_path.write_text(
                json_dumps(transcription_result.speaker_segments),
                encoding="utf-8",
            )
        elif paths.speaker_segments_path.exists():
            paths.speaker_segments_path.unlink()
        write_transcript_artifacts(
            paths=paths,
            segments=transcription_result.segments,
            raw_text=transcription_result.text,
        )
        transcription_details = transcription_result.details
        notes.append(
            f"Local ASR completed with {transcription_details.backend} "
            f"using {transcription_details.model}."
        )
        append_log(
            paths.pipeline_log_path,
            f"ASR branch succeeded with {transcription_details.backend} {transcription_details.model}",
        )

    cleaned_segments = load_segments(paths.segments_path)
    chunk_manifest = _write_summary_artifacts(
        metadata=metadata,
        source_kind=source_kind,
        paths=paths,
        segments=cleaned_segments,
        notes=notes,
        transcription=transcription_details,
        subtitle_path=subtitle_path,
        audio_source_path=audio_source_path,
        normalized_audio_path=normalized_audio_path,
        chunk_target_chars=config.chunk_target_chars,
    )
    append_log(paths.pipeline_log_path, f"Wrote {len(chunk_manifest)} chunk files and summary payload")

    return ProcessResult(
        metadata=metadata,
        source_kind=source_kind,
        artifact_root=paths.root_dir,
        chunk_count=len(chunk_manifest),
        notes=notes,
        subtitle_path=path_string(subtitle_path, paths.root_dir) if subtitle_path else None,
        audio_source_path=path_string(audio_source_path, paths.root_dir) if audio_source_path else None,
        normalized_audio_path=path_string(normalized_audio_path, paths.root_dir) if normalized_audio_path else None,
        transcription=transcription_details,
    )


def rechunk_video(
    video_id: str,
    config: PipelineConfig,
) -> ProcessResult:
    paths = initialize_workspace(resolve_artifact_paths(config.base_data_dir, video_id))
    metadata = VideoMetadata.model_validate(read_json(paths.metadata_path))
    summary = _load_summary_payload(paths.summary_payload_path)
    segments = load_segments(paths.segments_path)
    chunk_manifest = _write_summary_artifacts(
        metadata=metadata,
        source_kind=summary.source_kind if summary else SourceKind.LOCAL_ASR,
        paths=paths,
        segments=segments,
        notes=summary.notes if summary else ["Chunk files regenerated from existing segments."],
        transcription=summary.transcription if summary else None,
        subtitle_path=_resolve_artifact_path(paths, summary, "subtitle_source"),
        audio_source_path=_resolve_artifact_path(paths, summary, "audio_source"),
        normalized_audio_path=_resolve_artifact_path(paths, summary, "audio_normalized"),
        chunk_target_chars=config.chunk_target_chars,
    )
    append_log(paths.pipeline_log_path, f"Rechunked transcript into {len(chunk_manifest)} chunks")
    return ProcessResult(
        metadata=metadata,
        source_kind=summary.source_kind if summary else SourceKind.LOCAL_ASR,
        artifact_root=paths.root_dir,
        chunk_count=len(chunk_manifest),
        notes=summary.notes if summary else [],
        subtitle_path=summary.artifacts.get("subtitle_source") if summary else None,
        audio_source_path=summary.artifacts.get("audio_source") if summary else None,
        normalized_audio_path=summary.artifacts.get("audio_normalized") if summary else None,
        transcription=summary.transcription if summary else None,
    )


def prepare_summary_input(
    video_id: str,
    config: PipelineConfig,
) -> ProcessResult:
    paths = initialize_workspace(resolve_artifact_paths(config.base_data_dir, video_id))
    metadata = VideoMetadata.model_validate(read_json(paths.metadata_path))
    summary = _load_summary_payload(paths.summary_payload_path)
    segments = load_segments(paths.segments_path)
    chunk_manifest = _write_summary_artifacts(
        metadata=metadata,
        source_kind=summary.source_kind if summary else SourceKind.LOCAL_ASR,
        paths=paths,
        segments=segments,
        notes=summary.notes if summary else ["Summary payload regenerated from existing segments."],
        transcription=summary.transcription if summary else None,
        subtitle_path=_resolve_artifact_path(paths, summary, "subtitle_source"),
        audio_source_path=_resolve_artifact_path(paths, summary, "audio_source"),
        normalized_audio_path=_resolve_artifact_path(paths, summary, "audio_normalized"),
        chunk_target_chars=config.chunk_target_chars,
    )

    append_log(paths.pipeline_log_path, f"Prepared summary payload with {len(chunk_manifest)} chunks")
    refreshed_summary = _load_summary_payload(paths.summary_payload_path)
    return ProcessResult(
        metadata=metadata,
        source_kind=refreshed_summary.source_kind if refreshed_summary else SourceKind.LOCAL_ASR,
        artifact_root=paths.root_dir,
        chunk_count=refreshed_summary.chunk_count if refreshed_summary else len(chunk_manifest),
        notes=refreshed_summary.notes if refreshed_summary else [],
        subtitle_path=refreshed_summary.artifacts.get("subtitle_source") if refreshed_summary else None,
        audio_source_path=refreshed_summary.artifacts.get("audio_source") if refreshed_summary else None,
        normalized_audio_path=refreshed_summary.artifacts.get("audio_normalized") if refreshed_summary else None,
        transcription=refreshed_summary.transcription if refreshed_summary else None,
    )


def _can_reuse_cached_artifacts(
    paths,
    config: PipelineConfig,
    force: bool,
    requested_source_kind: SourceKind,
    requested_transcription: TranscriptionDetails | None,
) -> bool:
    if not (
        config.reuse_cached_artifacts
        and not force
        and paths.metadata_path.exists()
        and paths.info_path.exists()
        and paths.clean_transcript_path.exists()
        and paths.segments_path.exists()
        and paths.chunk_index_path.exists()
        and paths.summary_payload_path.exists()
    ):
        return False

    summary = _load_summary_payload(paths.summary_payload_path)
    if summary is None or summary.source_kind != requested_source_kind:
        return False

    if requested_source_kind != SourceKind.LOCAL_ASR:
        return True

    return _transcription_details_match(summary.transcription, requested_transcription)


def _load_cached_result(metadata: VideoMetadata, paths) -> ProcessResult:
    summary = _load_summary_payload(paths.summary_payload_path)
    if summary is None:
        return ProcessResult(
            metadata=metadata,
            source_kind=SourceKind.LOCAL_ASR,
            artifact_root=paths.root_dir,
            used_cache=True,
            notes=["Reused cached artifacts."],
        )
    return ProcessResult(
        metadata=metadata,
        source_kind=summary.source_kind,
        artifact_root=paths.root_dir,
        chunk_count=summary.chunk_count,
        notes=["Reused cached artifacts.", *summary.notes],
        used_cache=True,
        subtitle_path=summary.artifacts.get("subtitle_source"),
        audio_source_path=summary.artifacts.get("audio_source"),
        normalized_audio_path=summary.artifacts.get("audio_normalized"),
        transcription=summary.transcription,
    )


def _transcription_details_match(
    cached: TranscriptionDetails | None,
    requested: TranscriptionDetails | None,
) -> bool:
    if requested is None:
        return cached is None
    if cached is None:
        return False

    if cached.backend != requested.backend:
        return False
    if cached.model != requested.model:
        return False
    if (cached.language or None) != (requested.language or None):
        return False
    if bool(cached.diarized) != bool(requested.diarized):
        return False
    if requested.diarized and requested.speaker_count is not None:
        return cached.speaker_count == requested.speaker_count
    return True


def _write_summary_artifacts(
    metadata: VideoMetadata,
    source_kind: SourceKind,
    paths,
    segments,
    notes: list[str],
    transcription: TranscriptionDetails | None,
    subtitle_path: Path | None,
    audio_source_path: Path | None,
    normalized_audio_path: Path | None,
    chunk_target_chars: int,
):
    if not segments:
        raise RuntimeError("No transcript segments are available to build summary artifacts.")

    for existing in paths.chunks_dir.glob("chunk-*.txt"):
        existing.unlink()

    chunk_list = chunk_segments(segments, target_chars=chunk_target_chars)
    chunk_manifest = write_chunks(chunk_list, paths.chunks_dir, root_dir=paths.root_dir)
    write_chunk_index(chunk_manifest, paths.chunk_index_path)

    artifact_map = {
        "metadata": path_string(paths.metadata_path, paths.root_dir),
        "source_info": path_string(paths.info_path, paths.root_dir),
        "transcript_raw": path_string(paths.raw_transcript_path, paths.root_dir),
        "transcript_clean": path_string(paths.clean_transcript_path, paths.root_dir),
        "segments": path_string(paths.segments_path, paths.root_dir),
        "transcript_srt": path_string(paths.transcript_srt_path, paths.root_dir),
        "chunk_index": path_string(paths.chunk_index_path, paths.root_dir),
    }
    if subtitle_path is not None:
        artifact_map["subtitle_source"] = path_string(subtitle_path, paths.root_dir)
    if audio_source_path is not None:
        artifact_map["audio_source"] = path_string(audio_source_path, paths.root_dir)
    if normalized_audio_path is not None:
        artifact_map["audio_normalized"] = path_string(normalized_audio_path, paths.root_dir)
    if paths.asr_output_json_path.exists():
        artifact_map["asr_output_json"] = path_string(paths.asr_output_json_path, paths.root_dir)
    if paths.speaker_segments_path.exists():
        artifact_map["speaker_segments"] = path_string(paths.speaker_segments_path, paths.root_dir)

    payload = build_summary_payload(
        metadata=metadata,
        source_kind=source_kind,
        chunk_manifest=chunk_manifest,
        paths=paths,
        segment_count=len(segments),
        artifacts=artifact_map,
        transcription=transcription,
        notes=notes,
    )
    write_summary_payload(payload, paths.summary_payload_path)
    return chunk_manifest


def _planned_source_kind(candidate) -> SourceKind:
    if candidate is None:
        return SourceKind.LOCAL_ASR
    return SourceKind.MANUAL_SUBTITLES


def _planned_note(candidate, metadata: VideoMetadata) -> str:
    if candidate is None:
        if metadata.automatic_captions:
            return "Only auto-captions were found and they are ignored; local ASR will be used."
        return "No creator-provided subtitles were found; local ASR will be used."
    return f"Creator-provided subtitles are available in {candidate.language}; ASR can be skipped."


def _planned_transcription_details(
    config: PipelineConfig,
    language: str | None,
    high_quality: bool,
    asr_backend: str | None,
    model: str | None,
    diarize: bool,
    diarization_num_speakers: int | None,
    source_kind: SourceKind,
) -> TranscriptionDetails | None:
    if source_kind != SourceKind.LOCAL_ASR:
        return None
    request = build_transcription_request(
        input_path=Path("audio.wav"),
        config=config,
        language=language,
        high_quality=high_quality,
        asr_backend=asr_backend,
        model=model,
    )
    return TranscriptionDetails(
        backend=request.backend,
        model=request.model,
        language=request.language,
        threads=config.whisper_cpp_threads if request.backend == "whisper-cpp" else None,
        diarized=diarize if request.backend == "qwen3-asr" else False,
        speaker_count=diarization_num_speakers if diarize else None,
    )


def _load_summary_payload(path: Path) -> SummaryPayload | None:
    if not path.exists():
        return None
    return SummaryPayload.model_validate(read_json(path))


def _resolve_artifact_path(paths, summary: SummaryPayload | None, key: str) -> Path | None:
    if summary is None:
        return None
    relative = summary.artifacts.get(key)
    if not relative:
        return None
    return paths.root_dir / relative


def json_dumps(payload: object) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
