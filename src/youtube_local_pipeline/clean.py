from __future__ import annotations

import re

from .models import TranscriptSegment

WHITESPACE_RE = re.compile(r"\s+")
PUNCTUATION_SPACING_RE = re.compile(r"\s+([,.;:!?])")
OPENING_BRACKET_SPACING_RE = re.compile(r"([(\[{])\s+")


def normalize_transcript_text(text: str) -> str:
    normalized = text.replace("\u00a0", " ").replace("\n", " ")
    return WHITESPACE_RE.sub(" ", normalized).strip()


def remove_adjacent_duplicate_lines(lines: list[str]) -> list[str]:
    deduped: list[str] = []
    previous = ""

    for line in lines:
        normalized = normalize_transcript_text(line)
        if not normalized or normalized == previous:
            continue
        deduped.append(normalized)
        previous = normalized

    return deduped


def clean_segments(segments: list[TranscriptSegment]) -> list[TranscriptSegment]:
    cleaned: list[TranscriptSegment] = []
    previous: tuple[str | None, str] | None = None

    for segment in segments:
        text = normalize_transcript_text(segment.text)
        if not text:
            if previous == (segment.speaker, text):
                continue
            continue
        if previous == (segment.speaker, text):
            continue
        cleaned.append(
            TranscriptSegment(
                start_sec=segment.start_sec,
                end_sec=segment.end_sec,
                text=text,
                speaker=segment.speaker,
            )
        )
        previous = (segment.speaker, text)

    return cleaned


def segments_to_text(segments: list[TranscriptSegment]) -> str:
    return "\n".join(
        rendered
        for segment in segments
        if (rendered := render_segment_text(segment))
    )


def segments_to_paragraphs(
    segments: list[TranscriptSegment],
    target_chars: int = 400,
) -> str:
    paragraphs: list[str] = []
    current_lines: list[str] = []
    current_chars = 0

    for segment in segments:
        text = render_segment_text(segment)
        if not text:
            continue

        projected_size = current_chars + len(text) + (1 if current_lines else 0)
        if current_lines and projected_size > target_chars:
            paragraphs.append(" ".join(current_lines))
            current_lines = [text]
            current_chars = len(text)
            continue

        current_lines.append(text)
        current_chars = projected_size

    if current_lines:
        paragraphs.append(" ".join(current_lines))

    return "\n\n".join(paragraphs)


def render_segment_text(segment: TranscriptSegment) -> str:
    text = normalize_transcript_text(segment.text)
    if not text:
        return ""

    speaker = normalize_transcript_text(segment.speaker or "")
    if not speaker:
        return text

    return f"{speaker}: {text}"


def join_transcript_tokens(tokens: list[str], language: str | None = None) -> str:
    cleaned_tokens = [normalize_transcript_text(token) for token in tokens if normalize_transcript_text(token)]
    if not cleaned_tokens:
        return ""

    normalized_language = (language or "").strip().lower()
    if normalized_language in {"chinese", "zh", "zh-cn", "zh-tw", "cantonese", "yue", "japanese", "ja", "jp", "korean", "ko", "kr"}:
        return "".join(cleaned_tokens)

    joined = " ".join(cleaned_tokens)
    joined = PUNCTUATION_SPACING_RE.sub(r"\1", joined)
    joined = OPENING_BRACKET_SPACING_RE.sub(r"\1", joined)
    return joined.strip()
