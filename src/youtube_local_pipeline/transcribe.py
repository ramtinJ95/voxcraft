from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from .clean import join_transcript_tokens, normalize_transcript_text
from .config import PipelineConfig
from .models import TranscriptSegment, TranscriptionDetails

MODEL_FILE_EXTENSIONS = (".bin", ".gguf")
DEFAULT_SPEAKER_LABEL = "SPEAKER_00"
QWEN_WRAPPER_COMMAND = "yt-transcriber-qwen"
QWEN_WRAPPER_MODULE = "youtube_local_pipeline.qwen_cli"


class TranscriptionRequest(BaseModel):
    input_path: Path
    backend: str
    model: str
    language: str | None = None


class TranscriptionResult(BaseModel):
    text: str
    language: str | None = None
    segments: list[TranscriptSegment]
    speaker_segments: list[dict[str, object]] | None = None
    details: TranscriptionDetails


def build_transcription_request(
    input_path: Path,
    config: PipelineConfig,
    language: str | None = None,
    high_quality: bool = False,
    asr_backend: str | None = None,
    model: str | None = None,
) -> TranscriptionRequest:
    profile = config.transcription_profile(
        language=language,
        high_quality=high_quality,
        asr_backend=asr_backend,
        model=model,
    )
    return TranscriptionRequest(
        input_path=input_path,
        backend=profile.backend,
        model=profile.model,
        language=profile.language,
    )


def transcribe_audio_file(
    request: TranscriptionRequest,
    qwen_command: str = "mlx-qwen3-asr",
    qwen_context: str = "",
    qwen_diarize: bool = False,
    qwen_num_speakers: int | None = None,
    qwen_min_speakers: int = 1,
    qwen_max_speakers: int = 8,
    qwen_forced_aligner: str = "Qwen/Qwen3-ForcedAligner-0.6B",
    qwen_pyannote_model: str = "pyannote/speaker-diarization-community-1",
    qwen_dtype: str = "float16",
    qwen_draft_model: str | None = None,
    qwen_num_draft_tokens: int = 4,
    whisper_cpp_model_path: Path | None = None,
    whisper_cpp_model_dir: Path | None = None,
    whisper_cpp_command: str = "whisper-cli",
    whisper_cpp_threads: int = 4,
    output_base: Path | None = None,
) -> TranscriptionResult:
    if request.backend == "qwen3-asr":
        return _transcribe_with_qwen3_asr(
            request=request,
            command=qwen_command,
            context=qwen_context,
            diarize=qwen_diarize,
            num_speakers=qwen_num_speakers,
            min_speakers=qwen_min_speakers,
            max_speakers=qwen_max_speakers,
            forced_aligner=qwen_forced_aligner,
            pyannote_model=qwen_pyannote_model,
            dtype=qwen_dtype,
            draft_model=qwen_draft_model,
            num_draft_tokens=qwen_num_draft_tokens,
            output_base=output_base,
        )

    if request.backend == "whisper-cpp":
        return _transcribe_with_whisper_cpp(
            request=request,
            model_path=whisper_cpp_model_path,
            model_dir=whisper_cpp_model_dir,
            command=whisper_cpp_command,
            threads=whisper_cpp_threads,
            output_base=output_base,
        )

    raise RuntimeError(f"Unsupported transcription backend: {request.backend}")


def _transcribe_with_qwen3_asr(
    request: TranscriptionRequest,
    command: str,
    context: str,
    diarize: bool,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
    forced_aligner: str,
    pyannote_model: str,
    dtype: str,
    draft_model: str | None,
    num_draft_tokens: int,
    output_base: Path | None = None,
) -> TranscriptionResult:
    command_prefix = resolve_qwen_command_args(command)

    output_base = output_base or request.input_path.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="qwen-asr-", dir=output_base.parent) as temp_dir_name:
        temp_dir = Path(temp_dir_name)
        command_args = [
            *command_prefix,
            str(request.input_path),
            "--model",
            request.model,
            "--output-dir",
            str(temp_dir),
            "--output-format",
            "json",
            "--timestamps",
            "--forced-aligner",
            forced_aligner,
            "--dtype",
            dtype,
            "--quiet",
            "--no-progress",
        ]
        if request.language:
            command_args.extend(["--language", request.language])
        if context:
            command_args.extend(["--context", context])
        if draft_model:
            command_args.extend(["--draft-model", draft_model, "--num-draft-tokens", str(num_draft_tokens)])

        env = dict(os.environ)

        completed = subprocess.run(
            command_args,
            capture_output=True,
            text=True,
            check=False,
            env=env,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.strip() or completed.stdout.strip() or "mlx-qwen3-asr exited with a non-zero status."
            raise RuntimeError(stderr)

        temp_output_json_path = temp_dir / f"{request.input_path.stem}.json"
        if not temp_output_json_path.exists():
            raise RuntimeError(f"mlx-qwen3-asr did not produce the expected JSON output: {temp_output_json_path}")

        payload = json.loads(temp_output_json_path.read_text(encoding="utf-8"))

    if diarize:
        speaker_segments, labeled_word_segments = _diarize_qwen_payload_with_pyannote(
            audio_path=request.input_path,
            payload=payload,
            model_id=pyannote_model,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        if labeled_word_segments:
            payload["segments"] = labeled_word_segments
        if speaker_segments:
            payload["speaker_segments"] = speaker_segments

    output_json_path = output_base.with_suffix(".json")
    output_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    language = str(payload.get("language") or request.language or "")
    speaker_segments = payload.get("speaker_segments")
    transcript_segments = _segments_from_qwen_payload(payload, language=language)
    speaker_count = _speaker_count(speaker_segments if isinstance(speaker_segments, list) else None)

    return TranscriptionResult(
        text=normalize_transcript_text(str(payload.get("text", ""))),
        language=language or request.language,
        segments=transcript_segments,
        speaker_segments=speaker_segments if isinstance(speaker_segments, list) else None,
        details=TranscriptionDetails(
            backend="qwen3-asr",
            model=request.model,
            language=language or request.language,
            diarized=bool(speaker_segments),
            speaker_count=speaker_count,
        ),
    )


def _transcribe_with_whisper_cpp(
    request: TranscriptionRequest,
    model_path: Path | None,
    model_dir: Path | None,
    command: str,
    threads: int,
    output_base: Path | None = None,
) -> TranscriptionResult:
    if shutil.which(command) is None:
        raise RuntimeError(f"{command} is not available on PATH.")
    resolved_model_path = resolve_whisper_cpp_model_path(
        requested_model=request.model,
        explicit_model_path=model_path,
        explicit_model_dir=model_dir,
    )

    output_base = output_base or request.input_path.with_suffix("")
    output_base.parent.mkdir(parents=True, exist_ok=True)
    command_args = [
        command,
        "-m",
        str(resolved_model_path),
        "-f",
        str(request.input_path),
        "-t",
        str(threads),
        "-oj",
        "-of",
        str(output_base),
        "-np",
    ]
    command_args.extend(["-l", request.language or "auto"])

    completed = subprocess.run(
        command_args,
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or completed.stdout.strip() or "whisper.cpp exited with a non-zero status."
        raise RuntimeError(stderr)

    output_json_path = output_base.with_suffix(".json")
    if not output_json_path.exists():
        raise RuntimeError(f"whisper.cpp did not produce the expected JSON output: {output_json_path}")

    payload = json.loads(output_json_path.read_text(encoding="utf-8"))
    transcription = payload.get("transcription", [])
    segments = [
        TranscriptSegment(
            start_sec=float(item["offsets"]["from"]) / 1000.0,
            end_sec=float(item["offsets"]["to"]) / 1000.0,
            text=normalize_transcript_text(item["text"]),
        )
        for item in transcription
        if normalize_transcript_text(item.get("text", ""))
    ]
    text = normalize_transcript_text(" ".join(segment.text for segment in segments))
    language = payload.get("result", {}).get("language") or request.language

    return TranscriptionResult(
        text=text,
        language=language,
        segments=segments,
        speaker_segments=None,
        details=TranscriptionDetails(
            backend="whisper-cpp",
            model=request.model,
            model_path=str(resolved_model_path),
            threads=threads,
            language=language,
            diarized=False,
        ),
    )


def resolve_whisper_cpp_model_path(
    requested_model: str,
    explicit_model_path: Path | None = None,
    explicit_model_dir: Path | None = None,
) -> Path:
    if explicit_model_path is not None:
        resolved_explicit_path = explicit_model_path.expanduser()
        if not resolved_explicit_path.exists():
            raise RuntimeError(f"Configured whisper.cpp model file does not exist: {resolved_explicit_path}")
        return resolved_explicit_path

    direct_path = Path(requested_model).expanduser()
    if _looks_like_model_path(requested_model):
        if direct_path.exists():
            return direct_path
        raise RuntimeError(f"Requested whisper.cpp model file does not exist: {direct_path}")

    candidate_names = _candidate_model_names(requested_model)
    search_dirs = _candidate_model_dirs(explicit_model_dir)
    searched_paths: list[Path] = []
    for search_dir in search_dirs:
        for candidate_name in candidate_names:
            candidate_path = search_dir / candidate_name
            searched_paths.append(candidate_path)
            if candidate_path.exists():
                return candidate_path

    searched = "\n".join(f"- {path}" for path in searched_paths)
    raise RuntimeError(
        "Unable to locate a whisper.cpp model file for "
        f"'{requested_model}'. Set WHISPER_CPP_MODEL, set WHISPER_CPP_MODEL_DIR, pass "
        "--whisper-cpp-model, or place the model under ./models.\n"
        f"Searched:\n{searched}"
    )


def resolve_qwen_command_args(command: str) -> list[str]:
    direct_path = Path(command).expanduser()
    if direct_path.exists():
        return [str(direct_path.resolve())]

    resolved = shutil.which(command)
    if resolved is not None:
        return [resolved]

    if direct_path.name == QWEN_WRAPPER_COMMAND:
        return [sys.executable, "-m", QWEN_WRAPPER_MODULE]

    raise RuntimeError(f"{command} is not available on PATH.")


def describe_qwen_command(command: str) -> str | None:
    try:
        return " ".join(resolve_qwen_command_args(command))
    except RuntimeError:
        return None


def _candidate_model_dirs(explicit_model_dir: Path | None) -> list[Path]:
    candidates = [
        explicit_model_dir.expanduser() if explicit_model_dir is not None else None,
        Path("models"),
        Path("whisper.cpp/models"),
        Path("/opt/homebrew/opt/whisper-cpp/share/whisper-cpp"),
        Path("/opt/homebrew/share/whisper-cpp"),
    ]
    unique_candidates: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        if candidate is None:
            continue
        resolved = candidate.expanduser()
        if resolved in seen:
            continue
        seen.add(resolved)
        unique_candidates.append(resolved)
    return unique_candidates


def _candidate_model_names(requested_model: str) -> list[str]:
    if any(requested_model.endswith(extension) for extension in MODEL_FILE_EXTENSIONS):
        return [requested_model]

    candidate_names = [
        f"ggml-{requested_model}.bin",
        f"ggml-{requested_model}.gguf",
        f"{requested_model}.bin",
        f"{requested_model}.gguf",
        requested_model,
    ]
    unique_names: list[str] = []
    seen: set[str] = set()
    for name in candidate_names:
        if name in seen:
            continue
        seen.add(name)
        unique_names.append(name)
    return unique_names


def _looks_like_model_path(value: str) -> bool:
    return "/" in value or value.startswith(".") or any(value.endswith(extension) for extension in MODEL_FILE_EXTENSIONS)


def _segments_from_qwen_payload(
    payload: dict[str, object],
    *,
    language: str,
) -> list[TranscriptSegment]:
    raw_speaker_segments = payload.get("speaker_segments")
    if isinstance(raw_speaker_segments, list) and raw_speaker_segments:
        segments = [
            TranscriptSegment(
                start_sec=float(item.get("start", 0.0)),
                end_sec=float(item.get("end", 0.0)),
                text=normalize_transcript_text(str(item.get("text", ""))),
                speaker=_normalize_speaker(str(item.get("speaker", ""))),
            )
            for item in raw_speaker_segments
            if normalize_transcript_text(str(item.get("text", "")))
        ]
        return _merge_adjacent_segments(segments)

    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list):
        return []
    return _group_qwen_word_segments(raw_segments, language=language)


def _group_qwen_word_segments(
    raw_segments: list[object],
    *,
    language: str,
    max_gap_sec: float = 0.8,
    max_duration_sec: float = 6.0,
    max_chars: int = 160,
) -> list[TranscriptSegment]:
    normalized_items: list[dict[str, object]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = normalize_transcript_text(str(item.get("text", "")))
        if not text:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end < start:
            end = start
        normalized_items.append(
            {
                "start": start,
                "end": end,
                "text": text,
                "speaker": _normalize_speaker(str(item.get("speaker", ""))),
            }
        )

    if not normalized_items:
        return []

    groups: list[list[dict[str, object]]] = []
    current_group: list[dict[str, object]] = []

    for item in normalized_items:
        if not current_group:
            current_group = [item]
            continue

        previous_item = current_group[-1]
        previous_speaker = previous_item.get("speaker")
        current_speaker = item.get("speaker")
        gap = float(item["start"]) - float(previous_item["end"])
        duration = float(item["end"]) - float(current_group[0]["start"])
        projected_text = join_transcript_tokens(
            [str(segment["text"]) for segment in [*current_group, item]],
            language=language,
        )
        should_break = (
            previous_speaker != current_speaker
            or gap > max_gap_sec
            or duration > max_duration_sec
            or len(projected_text) > max_chars
            or _ends_sentence(str(previous_item["text"]))
        )
        if should_break:
            groups.append(current_group)
            current_group = [item]
            continue

        current_group.append(item)

    if current_group:
        groups.append(current_group)

    segments = [
        TranscriptSegment(
            start_sec=float(group[0]["start"]),
            end_sec=float(group[-1]["end"]),
            text=join_transcript_tokens([str(item["text"]) for item in group], language=language),
            speaker=group[0].get("speaker") or None,
        )
        for group in groups
    ]
    return _merge_adjacent_segments(segments)


def _merge_adjacent_segments(
    segments: list[TranscriptSegment],
    max_gap_sec: float = 0.35,
) -> list[TranscriptSegment]:
    if not segments:
        return []

    merged: list[TranscriptSegment] = [segments[0]]
    for segment in segments[1:]:
        previous = merged[-1]
        gap = segment.start_sec - previous.end_sec
        if (
            previous.speaker == segment.speaker
            and gap <= max_gap_sec
            and not _ends_sentence(previous.text)
        ):
            merged[-1] = TranscriptSegment(
                start_sec=previous.start_sec,
                end_sec=segment.end_sec,
                text=join_transcript_tokens([previous.text, segment.text]),
                speaker=previous.speaker,
            )
            continue
        merged.append(segment)
    return merged


def _ends_sentence(text: str) -> bool:
    return bool(re.search(r"[.!?。！？…]$", normalize_transcript_text(text)))


def _normalize_speaker(value: str) -> str | None:
    normalized = normalize_transcript_text(value)
    return normalized or None


def _speaker_count(speaker_segments: list[object] | None) -> int | None:
    if not speaker_segments:
        return None
    speakers = {
        _normalize_speaker(str(item.get("speaker", "")))
        for item in speaker_segments
        if isinstance(item, dict)
    }
    speakers.discard(None)
    return len(speakers) or None


def _diarize_qwen_payload_with_pyannote(
    *,
    audio_path: Path,
    payload: dict[str, object],
    model_id: str,
    num_speakers: int | None,
    min_speakers: int,
    max_speakers: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    raw_segments = payload.get("segments")
    if not isinstance(raw_segments, list) or not raw_segments:
        return [], []

    token = _resolve_pyannote_token()
    pipeline = _load_pyannote_pipeline(model_id=model_id, token=token)
    kwargs: dict[str, int] = {}
    if num_speakers is not None:
        kwargs["num_speakers"] = num_speakers
    else:
        kwargs["min_speakers"] = min_speakers
        kwargs["max_speakers"] = max_speakers

    try:
        diarization_output = pipeline(str(audio_path), **kwargs)
    except TypeError:
        diarization_output = pipeline(str(audio_path))
    except Exception as exc:
        raise RuntimeError(
            f"pyannote diarization inference failed for '{model_id}'."
        ) from exc

    annotation = _select_pyannote_annotation(diarization_output)
    speaker_turns = _annotation_to_turns(annotation)
    if not speaker_turns:
        return [], []

    labeled_word_segments = _assign_speakers_to_word_segments(raw_segments, speaker_turns)
    speaker_segments = _build_speaker_segments_from_turns(
        speaker_turns=speaker_turns,
        word_segments=labeled_word_segments,
    )
    return speaker_segments, labeled_word_segments


def _resolve_pyannote_token() -> str:
    token = (
        os.getenv("PYANNOTE_AUTH_TOKEN")
        or os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or ""
    ).strip()
    if not token:
        raise RuntimeError(
            "Pyannote diarization requires PYANNOTE_AUTH_TOKEN, HF_TOKEN, or HUGGINGFACE_TOKEN "
            "after accepting the Hugging Face model terms."
        )
    return token


def _load_pyannote_pipeline(*, model_id: str, token: str) -> object:
    try:
        from pyannote.audio import Pipeline
    except ImportError as exc:
        raise RuntimeError(
            "Pyannote diarization requires pyannote.audio. Install the diarization extras first."
        ) from exc

    try:
        return Pipeline.from_pretrained(model_id, token=token)
    except Exception as exc:
        raise RuntimeError(
            f"Failed to initialize pyannote pipeline '{model_id}'. "
            "Confirm that the model terms are accepted and the token has access."
        ) from exc


def _select_pyannote_annotation(diarization_output: object) -> object:
    annotation = getattr(diarization_output, "exclusive_speaker_diarization", None)
    if annotation is None:
        annotation = getattr(diarization_output, "speaker_diarization", None)
    if annotation is None:
        annotation = diarization_output
    if not hasattr(annotation, "itertracks"):
        raise RuntimeError("Pyannote diarization output did not expose an annotation timeline.")
    return annotation


def _annotation_to_turns(annotation: object) -> list[dict[str, object]]:
    turns: list[dict[str, object]] = []
    for turn, _, speaker in annotation.itertracks(yield_label=True):
        start = float(getattr(turn, "start", 0.0))
        end = float(getattr(turn, "end", start))
        if end <= start:
            continue
        turns.append(
            {
                "speaker": _normalize_speaker(str(speaker)) or DEFAULT_SPEAKER_LABEL,
                "start": start,
                "end": end,
            }
        )
    turns.sort(key=lambda item: (float(item["start"]), float(item["end"])))
    return turns


def _assign_speakers_to_word_segments(
    raw_segments: list[object],
    speaker_turns: list[dict[str, object]],
) -> list[dict[str, object]]:
    labeled: list[dict[str, object]] = []
    for item in raw_segments:
        if not isinstance(item, dict):
            continue
        text = normalize_transcript_text(str(item.get("text", "")))
        if not text:
            continue
        start = float(item.get("start", 0.0))
        end = float(item.get("end", start))
        if end < start:
            end = start
        labeled.append(
            {
                **item,
                "start": start,
                "end": end,
                "text": text,
                "speaker": _speaker_for_interval(start, end, speaker_turns),
            }
        )
    return labeled


def _speaker_for_interval(
    start: float,
    end: float,
    speaker_turns: list[dict[str, object]],
) -> str:
    if not speaker_turns:
        return DEFAULT_SPEAKER_LABEL

    midpoint = (start + end) / 2.0
    best_speaker = _normalize_speaker(str(speaker_turns[0].get("speaker", DEFAULT_SPEAKER_LABEL))) or DEFAULT_SPEAKER_LABEL
    best_overlap = -1.0
    for turn in speaker_turns:
        turn_start = float(turn.get("start", 0.0))
        turn_end = float(turn.get("end", turn_start))
        overlap = max(0.0, min(end, turn_end) - max(start, turn_start))
        if overlap > best_overlap:
            best_overlap = overlap
            best_speaker = _normalize_speaker(str(turn.get("speaker", DEFAULT_SPEAKER_LABEL))) or DEFAULT_SPEAKER_LABEL
            continue
        if best_overlap <= 0.0 and turn_start <= midpoint <= turn_end:
            best_speaker = _normalize_speaker(str(turn.get("speaker", DEFAULT_SPEAKER_LABEL))) or DEFAULT_SPEAKER_LABEL
    return best_speaker


def _build_speaker_segments_from_turns(
    *,
    speaker_turns: list[dict[str, object]],
    word_segments: list[dict[str, object]],
    max_gap_sec: float = 0.2,
) -> list[dict[str, object]]:
    if not speaker_turns:
        return []

    words = sorted(
        (dict(item) for item in word_segments),
        key=lambda item: (float(item.get("start", 0.0)), float(item.get("end", 0.0))),
    )
    out: list[dict[str, object]] = []
    word_index = 0
    for turn in speaker_turns:
        start = max(0.0, float(turn.get("start", 0.0)))
        end = max(start, float(turn.get("end", start)))
        speaker = _normalize_speaker(str(turn.get("speaker", DEFAULT_SPEAKER_LABEL))) or DEFAULT_SPEAKER_LABEL

        while word_index < len(words) and float(words[word_index].get("end", 0.0)) <= start:
            word_index += 1

        text_parts: list[str] = []
        scan_index = word_index
        while scan_index < len(words) and float(words[scan_index].get("start", 0.0)) < end:
            word = words[scan_index]
            word_start = float(word.get("start", 0.0))
            word_end = float(word.get("end", word_start))
            overlap = max(0.0, min(end, word_end) - max(start, word_start))
            if overlap > 0.0:
                token = normalize_transcript_text(str(word.get("text", "")))
                if token:
                    text_parts.append(token)
            scan_index += 1

        out.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": join_transcript_tokens(text_parts),
            }
        )

    merged: list[dict[str, object]] = []
    for item in out:
        if not merged:
            merged.append(dict(item))
            continue
        previous = merged[-1]
        gap = float(item["start"]) - float(previous["end"])
        if previous["speaker"] == item["speaker"] and gap <= max_gap_sec:
            previous["end"] = max(float(previous["end"]), float(item["end"]))
            previous["text"] = join_transcript_tokens(
                [str(previous.get("text", "")), str(item.get("text", ""))]
            )
            continue
        merged.append(dict(item))
    return merged
