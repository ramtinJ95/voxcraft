from __future__ import annotations

import os
import sys
from pathlib import Path

from pydantic import BaseModel, Field

from .manifest import resolve_video_root

QWEN_1_7B_8BIT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN_FORCED_ALIGNER_FP16_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
PYANNOTE_COMMUNITY_1_MODEL = "pyannote/speaker-diarization-community-1"


def _default_whisper_cpp_model_path() -> Path | None:
    value = os.getenv("WHISPER_CPP_MODEL")
    if not value:
        return None
    return Path(value).expanduser()


def _default_whisper_cpp_model_dir() -> Path | None:
    value = os.getenv("WHISPER_CPP_MODEL_DIR")
    if not value:
        return None
    return Path(value).expanduser()


def _default_whisper_cpp_threads() -> int:
    cpu_count = os.cpu_count() or 4
    return max(4, min(cpu_count, 8))


def _default_console_script(name: str) -> str:
    candidate_dirs = [
        Path(sys.prefix).resolve() / "bin",
        Path(sys.executable).resolve().parent,
    ]
    for candidate_dir in candidate_dirs:
        local_script = candidate_dir / name
        if local_script.exists():
            return str(local_script)
    return name


class TranscriptionProfile(BaseModel):
    backend: str = "qwen3-asr"
    model: str
    language: str | None = None


class PipelineConfig(BaseModel):
    base_data_dir: Path = Path("data/videos")
    language_preference: str = "en"
    subtitle_first: bool = True
    audio_sample_rate: int = 16000
    audio_channels: int = 1
    default_asr_backend: str = "qwen3-asr"
    qwen_default_model_english: str = QWEN_1_7B_8BIT_MODEL
    qwen_default_model_multilingual: str = QWEN_1_7B_8BIT_MODEL
    qwen_high_quality_model: str = QWEN_1_7B_8BIT_MODEL
    qwen_command: str = Field(default_factory=lambda: _default_console_script("yt-transcriber-qwen"))
    qwen_context: str = ""
    qwen_forced_aligner: str = QWEN_FORCED_ALIGNER_FP16_MODEL
    qwen_pyannote_model: str = PYANNOTE_COMMUNITY_1_MODEL
    qwen_dtype: str = "float16"
    qwen_draft_model: str | None = None
    qwen_num_draft_tokens: int = 4
    whisper_default_model_english: str = "large-v3"
    whisper_default_model_multilingual: str = "large-v3"
    whisper_high_quality_model: str = "large-v3"
    chunk_target_chars: int = 10000
    reuse_cached_artifacts: bool = True
    whisper_cpp_command: str = "whisper-cli"
    whisper_cpp_model_path: Path | None = Field(default_factory=_default_whisper_cpp_model_path)
    whisper_cpp_model_dir: Path | None = Field(default_factory=_default_whisper_cpp_model_dir)
    whisper_cpp_threads: int = Field(default_factory=_default_whisper_cpp_threads)
    codex_command: str = "codex"
    codex_summary_model: str | None = "gpt-5.4"

    def video_root(self, video_id: str, title: str | None = None) -> Path:
        return resolve_video_root(self.base_data_dir, video_id=video_id, title=title)

    def transcription_profile(
        self,
        language: str | None = None,
        high_quality: bool = False,
        asr_backend: str | None = None,
        model: str | None = None,
    ) -> TranscriptionProfile:
        normalized_language = (language or self.language_preference).lower()
        backend = normalize_asr_backend(asr_backend or self.default_asr_backend)
        if normalized_language in {"", "auto"}:
            normalized_language = ""

        if backend == "qwen3-asr":
            return _qwen_transcription_profile(
                normalized_language=normalized_language,
                high_quality=high_quality,
                model=model,
                config=self,
            )

        return _whisper_transcription_profile(
            normalized_language=normalized_language,
            high_quality=high_quality,
            model=model,
            config=self,
        )


def normalize_asr_backend(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "qwen": "qwen3-asr",
        "qwen3": "qwen3-asr",
        "qwen3-asr": "qwen3-asr",
        "whisper": "whisper-cpp",
        "whisper-cpp": "whisper-cpp",
        "whispercpp": "whisper-cpp",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported ASR backend: {value}") from exc


def _qwen_transcription_profile(
    *,
    normalized_language: str,
    high_quality: bool,
    model: str | None,
    config: PipelineConfig,
) -> TranscriptionProfile:
    if high_quality:
        profile = TranscriptionProfile(
            backend="qwen3-asr",
            model=config.qwen_high_quality_model,
            language=normalized_language or None,
        )
        return _apply_profile_overrides(profile, model=model)

    if not normalized_language:
        profile = TranscriptionProfile(
            backend="qwen3-asr",
            model=config.qwen_default_model_multilingual,
            language=None,
        )
        return _apply_profile_overrides(profile, model=model)

    if normalized_language.startswith("en"):
        profile = TranscriptionProfile(
            backend="qwen3-asr",
            model=config.qwen_default_model_english,
            language="en",
        )
        return _apply_profile_overrides(profile, model=model)

    profile = TranscriptionProfile(
        backend="qwen3-asr",
        model=config.qwen_default_model_multilingual,
        language=normalized_language,
    )
    return _apply_profile_overrides(profile, model=model)


def _whisper_transcription_profile(
    *,
    normalized_language: str,
    high_quality: bool,
    model: str | None,
    config: PipelineConfig,
) -> TranscriptionProfile:
    if high_quality:
        profile = TranscriptionProfile(
            backend="whisper-cpp",
            model=config.whisper_high_quality_model,
            language=normalized_language or None,
        )
        return _apply_profile_overrides(profile, model=model)

    if not normalized_language:
        profile = TranscriptionProfile(
            backend="whisper-cpp",
            model=config.whisper_default_model_multilingual,
            language=None,
        )
        return _apply_profile_overrides(profile, model=model)

    if normalized_language.startswith("en"):
        profile = TranscriptionProfile(
            backend="whisper-cpp",
            model=config.whisper_default_model_english,
            language="en",
        )
        return _apply_profile_overrides(profile, model=model)

    profile = TranscriptionProfile(
        backend="whisper-cpp",
        model=config.whisper_default_model_multilingual,
        language=normalized_language,
    )
    return _apply_profile_overrides(profile, model=model)


def _apply_profile_overrides(
    profile: TranscriptionProfile,
    model: str | None,
) -> TranscriptionProfile:
    updates: dict[str, str | None] = {}
    if model is not None:
        updates["model"] = model
    if not updates:
        return profile
    return profile.model_copy(update=updates)
