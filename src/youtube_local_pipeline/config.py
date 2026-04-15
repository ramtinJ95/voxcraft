from __future__ import annotations

import os
import sys
from typing import Any
from pathlib import Path

from pydantic import BaseModel, Field, model_validator

from .manifest import resolve_video_root
from .utils import read_json

QWEN_1_7B_8BIT_MODEL = "mlx-community/Qwen3-ASR-1.7B-8bit"
QWEN_FORCED_ALIGNER_FP16_MODEL = "Qwen/Qwen3-ForcedAligner-0.6B"
PYANNOTE_COMMUNITY_1_MODEL = "pyannote/speaker-diarization-community-1"
DEFAULT_SUMMARY_PROVIDER = "codex"
DEFAULT_CODEX_SUMMARY_MODEL = "gpt-5.4"
DEFAULT_CODEX_THINKING_LEVEL = "high"
SUPPORTED_SUMMARY_PROVIDERS = ("codex", "claude", "gemini", "pi")
CONFIG_ENV_VAR = "YT_TRANSCRIBER_CONFIG"
CONFIG_DIR_NAME = "yt-transcriber"
CONFIG_FILE_NAME = "config.json"


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


def default_config_path() -> Path:
    config_home = os.getenv("XDG_CONFIG_HOME")
    if config_home:
        return Path(config_home).expanduser() / CONFIG_DIR_NAME / CONFIG_FILE_NAME
    return Path.home() / ".config" / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def resolve_config_path(config_path: Path | None = None) -> Path | None:
    if config_path is not None:
        return config_path.expanduser()

    env_value = os.getenv(CONFIG_ENV_VAR)
    if env_value:
        return Path(env_value).expanduser()

    candidate = default_config_path()
    if candidate.exists():
        return candidate
    return None


def load_pipeline_config(
    *,
    config_path: Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> tuple["PipelineConfig", Path | None]:
    resolved_config_path = resolve_config_path(config_path)
    payload: dict[str, Any] = {}

    if resolved_config_path is not None:
        if not resolved_config_path.exists():
            raise FileNotFoundError(f"Config file not found: {resolved_config_path}")
        raw_payload = read_json(resolved_config_path)
        if not isinstance(raw_payload, dict):
            raise ValueError(f"Config file must contain a top-level JSON object: {resolved_config_path}")
        payload.update(raw_payload)

    if overrides:
        payload.update(overrides)

    return PipelineConfig.model_validate(payload), resolved_config_path


class SummaryHarnessConfig(BaseModel):
    command: str | None = None
    model: str | None = None
    thinking_level: str | None = None


def _default_summary_profile(provider: str) -> SummaryHarnessConfig:
    if provider == "codex":
        return SummaryHarnessConfig(
            command=_default_console_script("codex"),
            model=DEFAULT_CODEX_SUMMARY_MODEL,
            thinking_level=DEFAULT_CODEX_THINKING_LEVEL,
        )
    return SummaryHarnessConfig(command=_default_console_script(provider))


def _default_summary_profiles() -> dict[str, SummaryHarnessConfig]:
    return {provider: _default_summary_profile(provider) for provider in SUPPORTED_SUMMARY_PROVIDERS}


def _normalize_summary_profiles(value: dict[str, SummaryHarnessConfig | dict[str, Any]] | None) -> dict[str, SummaryHarnessConfig]:
    profiles = _default_summary_profiles()
    if not value:
        return profiles
    for raw_provider, raw_profile in value.items():
        provider = normalize_summary_provider(raw_provider)
        profile = SummaryHarnessConfig.model_validate(raw_profile)
        profiles[provider] = profiles[provider].model_copy(update=profile.model_dump(exclude_none=True))
    return profiles


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
    summary_provider: str = DEFAULT_SUMMARY_PROVIDER
    summary_profiles: dict[str, SummaryHarnessConfig] = Field(default_factory=_default_summary_profiles)
    summary_command: str | None = None
    summary_model: str | None = None
    summary_thinking_level: str | None = None
    codex_command: str | None = None
    codex_summary_model: str | None = None
    codex_summary_thinking_level: str | None = None

    @model_validator(mode="after")
    def _hydrate_summary_settings(self) -> PipelineConfig:
        self.default_asr_backend = normalize_asr_backend(self.default_asr_backend)
        provider = normalize_summary_provider(self.summary_provider)
        profiles = _normalize_summary_profiles(self.summary_profiles)
        codex_updates: dict[str, str] = {}
        if self.codex_command:
            codex_updates["command"] = self.codex_command
        if self.codex_summary_model is not None:
            codex_updates["model"] = self.codex_summary_model
        if self.codex_summary_thinking_level is not None:
            codex_updates["thinking_level"] = self.codex_summary_thinking_level
        if codex_updates:
            profiles["codex"] = profiles["codex"].model_copy(update=codex_updates)

        selected_profile = profiles[provider]
        selected_updates: dict[str, str] = {}
        if self.summary_command is not None:
            selected_updates["command"] = self.summary_command
        if self.summary_model is not None:
            selected_updates["model"] = self.summary_model
        if self.summary_thinking_level is not None:
            selected_updates["thinking_level"] = self.summary_thinking_level
        if selected_updates:
            selected_profile = selected_profile.model_copy(update=selected_updates)
            profiles[provider] = selected_profile

        self.summary_provider = provider
        self.summary_profiles = profiles
        self.summary_command = selected_profile.command
        self.summary_model = selected_profile.model
        self.summary_thinking_level = selected_profile.thinking_level
        self.codex_command = profiles["codex"].command
        self.codex_summary_model = profiles["codex"].model
        self.codex_summary_thinking_level = profiles["codex"].thinking_level
        return self

    def summary_harness(self, provider: str | None = None) -> SummaryHarnessConfig:
        resolved_provider = normalize_summary_provider(provider or self.summary_provider)
        return self.summary_profiles[resolved_provider]

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


def normalize_summary_provider(value: str) -> str:
    normalized = value.strip().lower()
    aliases = {
        "anthropic": "claude",
        "claude": "claude",
        "claude-code": "claude",
        "codex": "codex",
        "gemini": "gemini",
        "gemini-cli": "gemini",
        "google": "gemini",
        "pi": "pi",
        "pi-agent": "pi",
        "openai": "codex",
    }
    try:
        return aliases[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported summary provider: {value}") from exc


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
