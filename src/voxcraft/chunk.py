from __future__ import annotations

from pathlib import Path

from .clean import render_segment_text
from .models import ChunkManifestEntry, TranscriptChunk, TranscriptSegment
from .utils import path_string, write_json, write_text


def chunk_segments(
    segments: list[TranscriptSegment],
    target_chars: int = 10000,
) -> list[TranscriptChunk]:
    cleaned_segments = [segment for segment in segments if segment.text.strip()]
    if not cleaned_segments:
        return []

    chunks: list[TranscriptChunk] = []
    current_segments: list[TranscriptSegment] = []
    current_chars = 0

    for segment in cleaned_segments:
        segment_length = len(segment.text) + 1
        if current_segments and current_chars + segment_length > target_chars:
            chunks.append(_build_chunk(len(chunks) + 1, current_segments))
            current_segments = [segment]
            current_chars = segment_length
            continue

        current_segments.append(segment)
        current_chars += segment_length

    if current_segments:
        chunks.append(_build_chunk(len(chunks) + 1, current_segments))

    return chunks


def write_chunks(
    chunks: list[TranscriptChunk],
    chunks_dir: Path,
    root_dir: Path | None = None,
) -> list[ChunkManifestEntry]:
    chunks_dir.mkdir(parents=True, exist_ok=True)
    manifest_entries: list[ChunkManifestEntry] = []
    relative_root = root_dir or chunks_dir.parent

    for chunk in chunks:
        chunk_path = chunks_dir / f"chunk-{chunk.index:03}.txt"
        write_text(chunk_path, chunk.text + "\n")
        manifest_entries.append(
            ChunkManifestEntry(
                index=chunk.index,
                start_sec=chunk.start_sec,
                end_sec=chunk.end_sec,
                path=path_string(chunk_path, relative_root),
                char_count=len(chunk.text),
            )
        )

    return manifest_entries


def write_chunk_index(
    manifest_entries: list[ChunkManifestEntry],
    chunk_index_path: Path,
) -> None:
    write_json(
        chunk_index_path,
        [entry.model_dump(mode="json") for entry in manifest_entries],
    )


def _build_chunk(index: int, segments: list[TranscriptSegment]) -> TranscriptChunk:
    text = "\n".join(render_segment_text(segment) for segment in segments)
    return TranscriptChunk(
        index=index,
        start_sec=segments[0].start_sec,
        end_sec=segments[-1].end_sec,
        text=text,
    )
