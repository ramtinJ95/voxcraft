from __future__ import annotations

from pathlib import Path

from .clean import (
    clean_segments,
    normalize_transcript_text,
    render_segment_text,
    segments_to_paragraphs,
    segments_to_text,
)
from .models import ArtifactPaths, TranscriptSegment
from .utils import read_json, seconds_to_srt_timestamp, write_json, write_text


def parse_subtitle_file(path: Path) -> list[TranscriptSegment]:
    try:
        import webvtt  # type: ignore
    except ModuleNotFoundError as exc:
        raise RuntimeError("webvtt-py is not installed in the current environment.") from exc

    suffix = path.suffix.lower()
    if suffix == ".vtt":
        captions = webvtt.read(str(path))
    elif suffix == ".srt":
        captions = webvtt.from_srt(str(path))
    else:
        raise RuntimeError(f"Unsupported subtitle format: {path.suffix}")

    segments: list[TranscriptSegment] = []

    for caption in captions:
        text = normalize_transcript_text(caption.text)
        if not text:
            continue
        segments.append(
            TranscriptSegment(
                start_sec=float(caption.start_in_seconds),
                end_sec=float(caption.end_in_seconds),
                text=text,
            )
        )

    return segments


def build_transcript_artifacts(
    segments: list[TranscriptSegment],
    raw_text: str | None = None,
    paragraph_target_chars: int = 400,
) -> tuple[str, str, list[TranscriptSegment], str]:
    raw_text_value = normalize_transcript_text(raw_text) if raw_text else segments_to_text(segments)
    cleaned_segments = clean_segments(segments)
    clean_text = segments_to_paragraphs(
        cleaned_segments,
        target_chars=paragraph_target_chars,
    )
    return raw_text_value, clean_text, cleaned_segments, segments_to_srt_text(cleaned_segments)


def write_transcript_artifacts(
    paths: ArtifactPaths,
    segments: list[TranscriptSegment],
    raw_text: str | None = None,
    paragraph_target_chars: int = 400,
) -> tuple[str, str, list[TranscriptSegment]]:
    raw_text_value, clean_text, cleaned_segments, srt_text = build_transcript_artifacts(
        segments=segments,
        raw_text=raw_text,
        paragraph_target_chars=paragraph_target_chars,
    )
    write_text(paths.raw_transcript_path, raw_text_value + "\n")
    write_text(paths.clean_transcript_path, clean_text + "\n")
    write_json(
        paths.segments_path,
        [segment.model_dump(mode="json") for segment in cleaned_segments],
    )
    write_text(paths.transcript_srt_path, srt_text)
    return raw_text_value, clean_text, cleaned_segments


def segments_to_srt_text(segments: list[TranscriptSegment]) -> str:
    lines: list[str] = []
    for index, segment in enumerate(segments, start=1):
        lines.append(str(index))
        lines.append(
            f"{seconds_to_srt_timestamp(segment.start_sec)} --> "
            f"{seconds_to_srt_timestamp(segment.end_sec)}"
        )
        lines.append(render_segment_text(segment))
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def load_segments(path: Path) -> list[TranscriptSegment]:
    return [TranscriptSegment.model_validate(item) for item in read_json(path)]
