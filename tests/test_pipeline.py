from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from youtube_local_pipeline.config import (
    CONFIG_ENV_VAR,
    PipelineConfig,
    SummaryHarnessConfig,
    default_config_path,
    load_pipeline_config,
    normalize_summary_provider,
    resolve_config_path,
)
from youtube_local_pipeline.download import choose_subtitle_candidate
from youtube_local_pipeline.manifest import build_artifact_paths, initialize_workspace, resolve_video_root
from youtube_local_pipeline.models import SourceKind, TranscriptSegment, TranscriptionDetails, VideoMetadata
from youtube_local_pipeline.pipeline import _transcription_details_match, process_video
from youtube_local_pipeline.qwen_cli import apply_mlx_qwen3_asr_patch
from youtube_local_pipeline.subtitles import load_segments, write_transcript_artifacts
from youtube_local_pipeline.summarize import _build_summary_command, summarize_video, wrap_markdown_text
from youtube_local_pipeline.transcribe import (
    TranscriptionRequest,
    resolve_whisper_cpp_model_path,
    resolve_qwen_command_args,
    transcribe_audio_file,
)
from youtube_local_pipeline.utils import read_json, write_json, write_text


def test_choose_subtitle_candidate_prefers_manual_english_first() -> None:
    candidate = choose_subtitle_candidate(
        subtitles={
            "en": [{"ext": "vtt", "url": "https://example.com/en.vtt"}],
            "es": [{"ext": "vtt", "url": "https://example.com/es.vtt"}],
        },
        preferred_language="es",
    )

    assert candidate is not None
    assert candidate.language == "en"
    assert candidate.is_automatic is False


def test_choose_subtitle_candidate_ignores_auto_captions_by_default() -> None:
    candidate = choose_subtitle_candidate(
        subtitles={},
        preferred_language="en",
    )

    assert candidate is None


def test_process_video_dry_run_plans_asr_without_subtitles(monkeypatch, tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        title="Test Video",
        subtitles={},
        automatic_captions={},
    )

    def fake_probe_video(url: str) -> tuple[VideoMetadata, dict[str, object]]:
        return metadata, {"id": metadata.video_id, "webpage_url": metadata.url}

    monkeypatch.setattr("youtube_local_pipeline.pipeline.probe_video", fake_probe_video)

    result = process_video(
        url=metadata.url,
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        language="en",
        dry_run=True,
    )

    assert result.source_kind == SourceKind.LOCAL_ASR
    assert result.transcription is not None
    assert result.transcription.backend == "qwen3-asr"
    assert result.transcription.model == "mlx-community/Qwen3-ASR-1.7B-8bit"
    assert result.transcription.language == "en"
    assert result.artifact_root == (tmp_path / "videos" / "test-video--abc123")


def test_resolve_qwen_command_args_falls_back_to_module_wrapper(monkeypatch) -> None:
    monkeypatch.setattr("youtube_local_pipeline.transcribe.shutil.which", lambda command: None)
    monkeypatch.setattr("youtube_local_pipeline.transcribe.sys.executable", "/tmp/python")

    resolved = resolve_qwen_command_args("yt-transcriber-qwen")

    assert resolved == ["/tmp/python", "-m", "youtube_local_pipeline.qwen_cli"]


def test_process_video_dry_run_applies_explicit_model_override(monkeypatch, tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="xyz789",
        url="https://www.youtube.com/watch?v=xyz789",
        title="Override Video",
        subtitles={},
        automatic_captions={},
    )

    def fake_probe_video(url: str) -> tuple[VideoMetadata, dict[str, object]]:
        return metadata, {"id": metadata.video_id, "webpage_url": metadata.url}

    monkeypatch.setattr("youtube_local_pipeline.pipeline.probe_video", fake_probe_video)

    result = process_video(
        url=metadata.url,
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        language="en",
        model="Qwen/Qwen3-ASR-0.6B",
        dry_run=True,
    )

    assert result.source_kind == SourceKind.LOCAL_ASR
    assert result.transcription is not None
    assert result.transcription.model == "Qwen/Qwen3-ASR-0.6B"


def test_apply_mlx_qwen3_asr_patch_supports_hybrid_quantized_checkpoint(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    write_text(model_dir / "config.json", "{}\n")

    quantize_calls: list[dict[str, object]] = []

    class FakeLogger:
        def info(self, message: str) -> None:
            pass

    class FakeConfig:
        @staticmethod
        def from_dict(raw_config):
            return {"raw": raw_config}

    class FakeModel:
        def __init__(self, config):
            self.config = config
            self.loaded_weights: list[tuple[str, object]] = []
            self.eval_called = False

        def load_weights(self, items):
            self.loaded_weights.extend(items)

        def parameters(self):
            return {"model.embed_tokens.weight": SimpleNamespace(shape=(2, 64), ndim=2)}

        def eval(self):
            self.eval_called = True

    class FakeNN:
        @staticmethod
        def quantize(model, bits, group_size, class_predicate):
            quantize_calls.append(
                {
                    "bits": bits,
                    "group_size": group_size,
                    "embed_tokens": class_predicate("model.embed_tokens", object()),
                    "audio_tower": class_predicate("audio_tower.conv2d1", object()),
                }
            )

    class FakeMX:
        float32 = "float32"

        @staticmethod
        def eval(params):
            return None

    fake_load_models = SimpleNamespace(
        _resolve_path=lambda _: model_dir,
        Qwen3ASRConfig=FakeConfig,
        _load_safetensors=lambda _: {
            "model.embed_tokens.weight": "embed-weight",
            "model.embed_tokens.scales": "embed-scales",
            "model.embed_tokens.biases": "embed-biases",
            "audio_tower.conv2d1.weight": "audio-weight",
        },
        remap_weights=lambda weights: weights,
        Qwen3ASRModel=FakeModel,
        _read_quantization_config=lambda _: None,
        _is_quantized_weights=lambda weights: True,
        _infer_quantization_params=lambda weights, model: (8, 64),
        nn=FakeNN,
        mx=FakeMX,
        mlx_utils=SimpleNamespace(tree_flatten=lambda tree: list(tree.items())),
        _cast_tree_dtype=lambda params, dtype: params,
        logger=FakeLogger(),
    )

    apply_mlx_qwen3_asr_patch(fake_load_models)

    model, _, _ = fake_load_models._load_model_with_resolved_path("repo", "float16")
    loaded_weight_keys = [key for key, _ in model.loaded_weights]

    assert "lm_head.weight" in loaded_weight_keys
    assert "lm_head.scales" in loaded_weight_keys
    assert "lm_head.biases" in loaded_weight_keys
    assert model.eval_called is True
    assert quantize_calls == [
        {
            "bits": 8,
            "group_size": 64,
            "embed_tokens": True,
            "audio_tower": False,
        }
    ]


def test_write_transcript_artifacts_round_trip(tmp_path: Path) -> None:
    paths = initialize_workspace(build_artifact_paths(tmp_path / "video123", "video123"))
    segments = [
        TranscriptSegment(start_sec=0.0, end_sec=1.0, text="Hello"),
        TranscriptSegment(start_sec=1.0, end_sec=2.0, text="Hello"),
        TranscriptSegment(start_sec=2.0, end_sec=4.0, text="World"),
    ]

    _, clean_text, cleaned_segments = write_transcript_artifacts(paths, segments)

    assert "Hello" in clean_text
    assert len(cleaned_segments) == 2
    assert paths.transcript_srt_path.exists()

    loaded_segments = load_segments(paths.segments_path)
    assert loaded_segments == cleaned_segments


def test_whisper_cpp_transcription_parses_json(monkeypatch, tmp_path: Path) -> None:
    model_path = tmp_path / "ggml-base.en.bin"
    model_path.write_text("placeholder", encoding="utf-8")
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_base = tmp_path / "whisper-output"

    def fake_run(command, capture_output, text, check):
        output_json = output_base.with_suffix(".json")
        output_json.write_text(
            """
            {
              "result": {"language": "en"},
              "transcription": [
                {
                  "offsets": {"from": 0, "to": 1530},
                  "text": " Test line "
                }
              ]
            }
            """,
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    monkeypatch.setattr("youtube_local_pipeline.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr("youtube_local_pipeline.transcribe.subprocess.run", fake_run)

    result = transcribe_audio_file(
        request=TranscriptionRequest(
            input_path=audio_path,
            backend="whisper-cpp",
            model="base.en",
            language="en",
        ),
        whisper_cpp_model_path=model_path,
        output_base=output_base,
    )

    assert result.language == "en"
    assert result.segments[0].start_sec == 0.0
    assert result.segments[0].end_sec == 1.53
    assert result.segments[0].text == "Test line"
    assert result.details.model == "base.en"
    assert result.details.model_path == str(model_path)


def test_qwen_transcription_parses_json_and_speaker_segments(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_base = tmp_path / "asr-output"
    captured_diarize_args: dict[str, object] = {}

    def fake_run(command, capture_output, text, check, env):
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        output_json = output_dir / "audio.json"
        output_json.write_text(
            """
            {
              "text": "Hello there general Kenobi",
              "language": "en",
              "segments": [
                {"start": 0.0, "end": 0.5, "text": "Hello"},
                {"start": 0.5, "end": 0.9, "text": "there"},
                {"start": 1.4, "end": 1.9, "text": "general"},
                {"start": 1.9, "end": 2.4, "text": "Kenobi"}
              ]
            }
            """,
            encoding="utf-8",
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_diarize(*, audio_path, payload, model_id, num_speakers, min_speakers, max_speakers):
        captured_diarize_args.update(
            {
                "audio_path": audio_path,
                "model_id": model_id,
                "num_speakers": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
        )
        return (
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.9, "text": "Hello there"},
                {"speaker": "SPEAKER_01", "start": 1.4, "end": 2.4, "text": "general Kenobi"},
            ],
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5, "text": "Hello"},
                {"speaker": "SPEAKER_00", "start": 0.5, "end": 0.9, "text": "there"},
                {"speaker": "SPEAKER_01", "start": 1.4, "end": 1.9, "text": "general"},
                {"speaker": "SPEAKER_01", "start": 1.9, "end": 2.4, "text": "Kenobi"},
            ],
        )

    monkeypatch.setattr("youtube_local_pipeline.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr("youtube_local_pipeline.transcribe.subprocess.run", fake_run)
    monkeypatch.setattr("youtube_local_pipeline.transcribe._diarize_qwen_payload_with_pyannote", fake_diarize)

    result = transcribe_audio_file(
        request=TranscriptionRequest(
            input_path=audio_path,
            backend="qwen3-asr",
            model="mlx-community/Qwen3-ASR-1.7B-8bit",
            language="en",
        ),
        qwen_diarize=True,
        output_base=output_base,
    )

    assert result.language == "en"
    assert result.details.backend == "qwen3-asr"
    assert result.details.diarized is True
    assert result.details.speaker_count == 2
    assert result.segments[0].speaker == "SPEAKER_00"
    assert result.segments[1].speaker == "SPEAKER_01"
    assert output_base.with_suffix(".json").exists()
    assert captured_diarize_args["audio_path"] == audio_path
    assert captured_diarize_args["model_id"] == "pyannote/speaker-diarization-community-1"
    assert captured_diarize_args["num_speakers"] is None


def test_resolve_whisper_cpp_model_path_finds_named_model_in_local_models_dir(tmp_path: Path) -> None:
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    model_path = model_dir / "ggml-large-v3.bin"
    model_path.write_text("placeholder", encoding="utf-8")

    resolved = resolve_whisper_cpp_model_path(
        requested_model="large-v3",
        explicit_model_dir=model_dir,
    )

    assert resolved == model_path


def test_process_video_dry_run_allows_whisper_cpp_override(monkeypatch, tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="whisper123",
        url="https://www.youtube.com/watch?v=whisper123",
        title="Whisper Override",
        subtitles={},
        automatic_captions={},
    )

    def fake_probe_video(url: str) -> tuple[VideoMetadata, dict[str, object]]:
        return metadata, {"id": metadata.video_id, "webpage_url": metadata.url}

    monkeypatch.setattr("youtube_local_pipeline.pipeline.probe_video", fake_probe_video)

    result = process_video(
        url=metadata.url,
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        language="en",
        asr_backend="whisper-cpp",
        dry_run=True,
    )

    assert result.transcription is not None
    assert result.transcription.backend == "whisper-cpp"
    assert result.transcription.model == "large-v3"


def test_transcription_details_match_rejects_cached_backend_mismatch() -> None:
    cached = TranscriptionDetails(
        backend="whisper-cpp",
        model="large-v3",
        language="en",
        diarized=False,
    )
    requested = TranscriptionDetails(
        backend="qwen3-asr",
        model="mlx-community/Qwen3-ASR-1.7B-8bit",
        language="en",
        diarized=False,
    )

    assert _transcription_details_match(cached, requested) is False


def test_normalize_summary_provider_accepts_common_aliases() -> None:
    assert normalize_summary_provider("codex") == "codex"
    assert normalize_summary_provider("openai") == "codex"
    assert normalize_summary_provider("claude-code") == "claude"
    assert normalize_summary_provider("anthropic") == "claude"
    assert normalize_summary_provider("gemini-cli") == "gemini"
    assert normalize_summary_provider("google") == "gemini"
    assert normalize_summary_provider("pi-agent") == "pi"


def test_pipeline_config_resolves_provider_specific_summary_profile() -> None:
    config = PipelineConfig(
        summary_provider="pi",
        summary_profiles={
            "pi": SummaryHarnessConfig(
                command="pi-custom",
                model="openai/gpt-5.4",
                thinking_level="high",
            )
        },
    )

    assert config.summary_command == "pi-custom"
    assert config.summary_model == "openai/gpt-5.4"
    assert config.summary_thinking_level == "high"
    assert config.summary_harness("pi").command == "pi-custom"
    assert config.summary_harness("pi").model == "openai/gpt-5.4"
    assert config.summary_harness("pi").thinking_level == "high"
    assert config.summary_harness("codex").model == "gpt-5.4"
    assert config.summary_harness("codex").thinking_level == "high"


def test_resolve_config_path_prefers_env_and_default_locations(monkeypatch, tmp_path: Path) -> None:
    env_config = tmp_path / "env-config.json"
    monkeypatch.setenv(CONFIG_ENV_VAR, str(env_config))

    assert resolve_config_path() == env_config

    monkeypatch.delenv(CONFIG_ENV_VAR)
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path / "xdg"))
    expected_default_path = default_config_path()
    expected_default_path.parent.mkdir(parents=True, exist_ok=True)
    write_text(expected_default_path, "{}\n")

    assert resolve_config_path() == expected_default_path


def test_load_pipeline_config_reads_json_and_applies_overrides(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_json(
        config_path,
        {
            "default_asr_backend": "whisper",
            "summary_provider": "codex",
            "summary_profiles": {
                "codex": {
                    "command": "codex-custom",
                    "model": "gpt-5.4",
                    "thinking_level": "medium",
                },
                "pi": {
                    "command": "pi-custom",
                    "model": "openai/gpt-5.4",
                    "thinking_level": "high",
                },
            },
        },
    )

    config, resolved_path = load_pipeline_config(
        config_path=config_path,
        overrides={
            "summary_provider": "pi",
            "summary_model": "openai/gpt-5.5",
        },
    )

    assert resolved_path == config_path
    assert config.default_asr_backend == "whisper-cpp"
    assert config.summary_provider == "pi"
    assert config.summary_command == "pi-custom"
    assert config.summary_model == "openai/gpt-5.5"
    assert config.summary_thinking_level == "high"


def test_build_summary_command_for_claude_disables_tools() -> None:
    command, writes_to_stdout = _build_summary_command(
        provider="claude",
        command="claude",
        model="claude-sonnet-4-5",
        thinking_level=None,
        workdir=Path("/tmp/workdir"),
        output_path=Path("/tmp/workdir/out.md"),
    )

    assert writes_to_stdout is True
    assert command == [
        "claude",
        "--print",
        "Follow the piped prompt exactly and output only the requested markdown. Do not use tools.",
        "--output-format",
        "text",
        "--no-session-persistence",
        "--permission-mode",
        "default",
        "--bare",
        "--max-turns",
        "1",
        "--tools",
        "",
        "--model",
        "claude-sonnet-4-5",
    ]


def test_build_summary_command_for_gemini_uses_noninteractive_prompt_mode() -> None:
    command, writes_to_stdout = _build_summary_command(
        provider="gemini",
        command="gemini",
        model="gemini-2.5-pro",
        thinking_level=None,
        workdir=Path("/tmp/workdir"),
        output_path=Path("/tmp/workdir/out.md"),
    )

    assert writes_to_stdout is True
    assert command == [
        "gemini",
        "--prompt",
        "Follow the piped prompt exactly and output only the requested markdown. Do not use tools.",
        "--output-format",
        "text",
        "--approval-mode",
        "default",
        "--model",
        "gemini-2.5-pro",
    ]


def test_build_summary_command_for_pi_uses_print_mode_and_thinking_level() -> None:
    command, writes_to_stdout = _build_summary_command(
        provider="pi",
        command="pi",
        model="openai/gpt-5.4",
        thinking_level="high",
        workdir=Path("/tmp/workdir"),
        output_path=Path("/tmp/workdir/out.md"),
    )

    assert writes_to_stdout is True
    assert command == [
        "pi",
        "-p",
        "Follow the piped prompt exactly and output only the requested markdown. Do not use tools.",
        "--no-session",
        "--no-tools",
        "--model",
        "openai/gpt-5.4",
        "--thinking",
        "high",
    ]


def test_summarize_video_writes_chunk_and_final_outputs(monkeypatch, tmp_path: Path) -> None:
    paths = initialize_workspace(build_artifact_paths(tmp_path / "video123", "video123"))
    metadata = VideoMetadata(
        video_id="video123",
        url="https://www.youtube.com/watch?v=video123",
        title="Summary Test",
        channel="Example Channel",
        duration_sec=42.0,
        subtitles={"en": []},
    )
    write_json(paths.metadata_path, metadata.model_dump(mode="json"))
    write_json(
        paths.chunk_index_path,
        [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 12.0,
                "path": "chunks/chunk-001.txt",
                "char_count": 20,
            }
        ],
    )
    write_text(paths.chunks_dir / "chunk-001.txt", "alpha beta gamma\n")
    write_json(
        paths.summary_payload_path,
        {
            "video_id": "video123",
            "url": metadata.url,
            "title": metadata.title,
            "channel": metadata.channel,
            "duration_sec": metadata.duration_sec,
            "source_kind": "subtitles",
            "transcript_path": "transcript/clean.txt",
            "segments_path": "transcript/segments.json",
            "chunk_index_path": "chunks/index.json",
            "chunk_count": 1,
            "segment_count": 1,
            "artifacts": {},
            "transcription": None,
            "notes": [],
        },
    )

    captured_summary_calls: list[dict[str, str | None]] = []

    def fake_run_summary_cli(
        prompt: str,
        output_path: Path,
        workdir: Path,
        provider: str,
        command: str,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> str:
        captured_summary_calls.append(
            {
                "provider": provider,
                "command": command,
                "model": model,
                "thinking_level": thinking_level,
            }
        )
        if "Combine these chunk summaries" in prompt:
            content = (
                "# Final Summary\n\n"
                "This is a deliberately long final summary paragraph that should be wrapped by "
                "the pipeline after the summary CLI finishes writing the final markdown file so "
                "that no prose line exceeds the configured maximum width.\n\n"
                "## Main Takeaways\n"
                "- This bullet point is also deliberately long so the wrapper has to reflow "
                "it instead of leaving a single oversized markdown list line in place.\n\n"
                "## Timeline\n"
                "- One long timeline bullet that should also be wrapped correctly.\n\n"
                "## Open Questions Or Uncertainties\n"
                "- None."
            )
        else:
            content = "## Chunk Summary\n\nChunk summary.\n\n## Key Points\n- One\n\n## Notable Details\n- Detail"
        write_text(output_path, content + "\n")
        return content

    monkeypatch.setattr("youtube_local_pipeline.summarize.run_summary_cli", fake_run_summary_cli)

    result = summarize_video(
        video_id="video123",
        config=PipelineConfig(base_data_dir=tmp_path),
    )

    assert result.final_summary_path == "final.md"
    assert result.summary_manifest_path == "summary/manifest.json"
    assert paths.summary_final_path.exists()
    assert paths.summary_manifest_path.exists()
    assert captured_summary_calls == [
        {
            "provider": "codex",
            "command": "codex",
            "model": "gpt-5.4",
            "thinking_level": "high",
        },
        {
            "provider": "codex",
            "command": "codex",
            "model": "gpt-5.4",
            "thinking_level": "high",
        },
    ]
    manifest = read_json(paths.summary_manifest_path)
    assert manifest["summary_provider"] == "codex"
    assert manifest["summary_command"] == "codex"
    assert manifest["summary_model"] == "gpt-5.4"
    assert manifest["summary_thinking_level"] == "high"
    assert all(len(line) <= 80 for line in paths.summary_final_path.read_text(encoding="utf-8").splitlines())


def test_summarize_video_migrates_legacy_final_summary_without_rerunning_codex(tmp_path: Path) -> None:
    paths = initialize_workspace(build_artifact_paths(tmp_path / "video123", "video123"))
    metadata = VideoMetadata(
        video_id="video123",
        url="https://www.youtube.com/watch?v=video123",
        title="Summary Test",
        channel="Example Channel",
        duration_sec=42.0,
        subtitles={"en": []},
    )
    write_json(paths.metadata_path, metadata.model_dump(mode="json"))
    write_json(
        paths.chunk_index_path,
        [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 12.0,
                "path": "chunks/chunk-001.txt",
                "char_count": 20,
            }
        ],
    )
    write_text(paths.chunks_dir / "chunk-001.txt", "alpha beta gamma\n")
    write_text(paths.summary_dir / "chunk-001.md", "## Chunk Summary\n\nChunk summary.\n")
    write_json(
        paths.summary_payload_path,
        {
            "video_id": "video123",
            "url": metadata.url,
            "title": metadata.title,
            "channel": metadata.channel,
            "duration_sec": metadata.duration_sec,
            "source_kind": "subtitles",
            "transcript_path": "transcript/clean.txt",
            "segments_path": "transcript/segments.json",
            "chunk_index_path": "chunks/index.json",
            "chunk_count": 1,
            "segment_count": 1,
            "artifacts": {},
            "transcription": None,
            "notes": [],
        },
    )
    write_text(paths.summary_dir / "final.md", "# Final Summary\n\nLegacy summary.\n")
    write_json(
        paths.summary_manifest_path,
        {
            "video_id": "video123",
            "codex_command": "codex",
            "codex_model": "gpt-5.4",
            "codex_reasoning_effort": "high",
            "chunk_summaries": [
                {
                    "index": 1,
                    "start_sec": 0.0,
                    "end_sec": 12.0,
                    "source_chunk_path": "chunks/chunk-001.txt",
                    "prompt_path": "summary/prompts/chunk-001.prompt.txt",
                    "output_path": "summary/chunk-001.md",
                }
            ],
            "final_prompt_path": "summary/prompts/final.prompt.txt",
            "final_summary_path": "summary/final.md",
        },
    )

    result = summarize_video(
        video_id="video123",
        config=PipelineConfig(base_data_dir=tmp_path),
    )

    assert result.final_summary_path == "final.md"
    assert paths.summary_final_path.read_text(encoding="utf-8") == "# Final Summary\n\nLegacy summary.\n"
    assert not (paths.summary_dir / "final.md").exists()
    manifest = read_json(paths.summary_manifest_path)
    assert manifest["final_summary_path"] == "final.md"
    assert manifest["summary_provider"] == "codex"
    assert manifest["summary_command"] == "codex"


def test_wrap_markdown_text_wraps_paragraphs_and_bullets_to_80_columns() -> None:
    content = (
        "# Final Summary\n\n"
        "This paragraph should be wrapped by the markdown formatter because it is much longer "
        "than eighty characters and should therefore not remain as a single line in the final "
        "summary document.\n\n"
        "## Main Takeaways\n"
        "- This bullet should also be wrapped cleanly so that continuation lines align under "
        "the bullet body instead of exceeding the line width.\n"
    )

    wrapped = wrap_markdown_text(content, width=80)
    wrapped_lines = wrapped.splitlines()

    assert wrapped_lines[0] == "# Final Summary"
    assert "## Main Takeaways" in wrapped_lines
    takeaway_index = wrapped_lines.index("## Main Takeaways")
    assert wrapped_lines[takeaway_index + 1].startswith("- ")
    assert all(len(line) <= 80 for line in wrapped_lines)


def test_resolve_video_root_prefers_human_readable_name_and_finds_legacy_dirs(tmp_path: Path) -> None:
    base_dir = tmp_path / "videos"
    base_dir.mkdir()

    assert resolve_video_root(base_dir, "abc123", title="Test Video") == base_dir / "test-video--abc123"

    legacy = base_dir / "abc123"
    legacy.mkdir()
    assert resolve_video_root(base_dir, "abc123", title="Changed Title") == legacy
