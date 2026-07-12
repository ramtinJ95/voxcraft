from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from voxcraft.cli import _command_status
from voxcraft.config import (
    CONFIG_ENV_VAR,
    PipelineConfig,
    SummaryHarnessConfig,
    default_config_path,
    load_pipeline_config,
    normalize_summary_provider,
    resolve_config_path,
)
from voxcraft.download import _download_direct_subtitle, choose_subtitle_candidate, write_metadata_artifacts
from voxcraft.manifest import build_artifact_paths, initialize_workspace, resolve_video_root
from voxcraft.models import SourceKind, SubtitleCandidate, TranscriptSegment, TranscriptionDetails, VideoMetadata
from voxcraft.pipeline import _cached_subtitle_languages, _transcription_details_match, process_video, rechunk_video
from voxcraft.qwen_cli import apply_mlx_qwen3_asr_patch
from voxcraft.subtitles import load_segments, write_transcript_artifacts
from voxcraft.summarize import _build_summary_command, summarize_video, wrap_markdown_text
from voxcraft.transcribe import (
    TranscriptionRequest,
    _load_reusable_qwen_payload,
    _qwen_payload_marker,
    resolve_whisper_cpp_model_path,
    resolve_qwen_command_args,
    transcribe_audio_file,
)
from voxcraft.utils import read_json, write_json, write_text


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


def test_choose_subtitle_candidate_can_prefer_explicit_language() -> None:
    candidate = choose_subtitle_candidate(
        subtitles={
            "en": [{"ext": "vtt", "url": "https://example.com/en.vtt"}],
            "es": [{"ext": "vtt", "url": "https://example.com/es.vtt"}],
        },
        preferred_language="es",
        prefer_english=False,
    )

    assert candidate is not None
    assert candidate.language == "es"


def test_choose_subtitle_candidate_falls_back_to_any_creator_language() -> None:
    candidate = choose_subtitle_candidate(
        subtitles={
            "sv": [{"ext": "vtt", "url": "https://example.com/sv.vtt"}],
            "fr": [{"ext": "vtt", "url": "https://example.com/fr.vtt"}],
        },
        preferred_language="en",
    )

    assert candidate is not None
    assert candidate.language == "fr"


def test_choose_subtitle_candidate_skips_languages_without_tracks() -> None:
    candidate = choose_subtitle_candidate(
        subtitles={
            "en": [],
            "sv": [{"ext": "vtt", "url": "https://example.com/sv.vtt"}],
        },
        preferred_language="en",
    )

    assert candidate is not None
    assert candidate.language == "sv"


def test_cached_subtitle_languages_skips_legacy_empty_tracks() -> None:
    assert _cached_subtitle_languages(
        {
            "subtitles": {
                "en": [],
                "sv": [{"ext": "vtt"}],
            }
        }
    ) == {"sv"}


def test_choose_subtitle_candidate_ignores_auto_captions_by_default() -> None:
    candidate = choose_subtitle_candidate(
        subtitles={},
        preferred_language="en",
    )

    assert candidate is None


def test_write_metadata_artifacts_keeps_top_level_metadata_compact(tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="compact123",
        url="https://www.youtube.com/watch?v=compact123",
        title="Compact Metadata",
        channel="Example Channel",
        duration_sec=120.0,
        upload_date="2026-06-11",
        subtitles={
            "de": [],
            "en": [
                {
                    "language": "en",
                    "ext": "vtt",
                    "url": "https://example.com/transient-subtitle-url",
                    "name": "English",
                }
            ]
        },
        automatic_captions={
            "fi": [],
            "sv": [
                {
                    "language": "sv",
                    "ext": "vtt",
                    "url": "https://example.com/transient-auto-caption-url",
                    "name": "Swedish",
                }
            ]
        },
    )
    raw_info = {
        "id": "compact123",
        "title": "Compact Metadata",
        "subtitles": {"en": [{"url": "https://example.com/transient-subtitle-url"}]},
    }
    metadata_path = tmp_path / "metadata.json"
    info_path = tmp_path / "source" / "info.json"

    write_metadata_artifacts(
        metadata=metadata,
        raw_info=raw_info,
        metadata_path=metadata_path,
        info_path=info_path,
    )

    compact_metadata = read_json(metadata_path)
    assert compact_metadata == {
        "video_id": "compact123",
        "url": "https://www.youtube.com/watch?v=compact123",
        "title": "Compact Metadata",
        "channel": "Example Channel",
        "duration_sec": 120.0,
        "upload_date": "2026-06-11",
        "subtitle_languages": ["en"],
        "automatic_caption_languages": ["sv"],
    }
    assert read_json(info_path) == raw_info


def test_download_direct_subtitle_uses_timeout(monkeypatch, tmp_path: Path) -> None:
    captured_call: dict[str, object] = {}

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback):
            return None

        @staticmethod
        def read() -> bytes:
            return b"WEBVTT\n\n"

    def fake_urlopen(url: str, timeout: int):
        captured_call["url"] = url
        captured_call["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr("voxcraft.download.urlopen", fake_urlopen)

    path = _download_direct_subtitle(
        tmp_path,
        SubtitleCandidate(language="en", ext="vtt", url="https://example.com/subtitles.vtt"),
    )

    assert captured_call == {"url": "https://example.com/subtitles.vtt", "timeout": 30}
    assert path.read_text(encoding="utf-8") == "WEBVTT\n\n"


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

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fake_probe_video)

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


def test_process_video_dry_run_uses_upload_date_in_new_workspace_name(monkeypatch, tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="dated123",
        url="https://www.youtube.com/watch?v=dated123",
        title="Dated Test Video",
        upload_date="2026-06-11",
        subtitles={},
        automatic_captions={},
    )

    def fake_probe_video(url: str) -> tuple[VideoMetadata, dict[str, object]]:
        return metadata, {"id": metadata.video_id, "webpage_url": metadata.url}

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fake_probe_video)

    result = process_video(
        url=metadata.url,
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        language="en",
        dry_run=True,
    )

    assert result.artifact_root == (
        tmp_path / "videos" / "2026-06-11--dated-test-video--dated123"
    )


def test_process_video_reuses_cached_subtitle_artifacts_without_probe(monkeypatch, tmp_path: Path) -> None:
    base_dir = tmp_path / "videos"
    paths = initialize_workspace(build_artifact_paths(base_dir / "test-video--abc123", "abc123"))
    write_json(
        paths.metadata_path,
        {
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Test Video",
            "subtitle_languages": ["en"],
            "automatic_caption_languages": [],
        },
    )
    write_json(paths.info_path, {"id": "abc123"})
    write_text(paths.clean_transcript_path, "Hello world\n")
    write_json(
        paths.segments_path,
        [TranscriptSegment(start_sec=0.0, end_sec=1.0, text="Hello world").model_dump(mode="json")],
    )
    write_json(
        paths.chunk_index_path,
        [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "path": "chunks/chunk-001.txt",
                "char_count": 11,
            }
        ],
    )
    write_json(
        paths.summary_payload_path,
        {
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Test Video",
            "source_kind": "subtitles",
            "transcript_path": "transcript/clean.txt",
            "segments_path": "transcript/segments.json",
            "chunk_index_path": "chunks/index.json",
            "chunk_count": 1,
            "segment_count": 1,
            "artifacts": {"subtitle_language": "en"},
            "transcription": None,
            "notes": ["Creator-provided subtitles are available in en; ASR can be skipped."],
        },
    )

    def fail_probe_video(url: str):
        raise AssertionError("metadata probe should be skipped for compatible cached artifacts")

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fail_probe_video)

    result = process_video(
        url="https://www.youtube.com/watch?v=abc123",
        config=PipelineConfig(base_data_dir=base_dir),
        language="en",
    )

    assert result.used_cache is True
    assert result.source_kind == SourceKind.MANUAL_SUBTITLES
    assert result.chunk_count == 1


def test_process_video_reruns_cached_subtitles_for_explicit_language(monkeypatch, tmp_path: Path) -> None:
    base_dir = tmp_path / "videos"
    paths = initialize_workspace(build_artifact_paths(base_dir / "test-video--abc123", "abc123"))
    write_json(
        paths.metadata_path,
        {
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Test Video",
            "subtitle_languages": ["en", "es"],
            "automatic_caption_languages": [],
        },
    )
    write_json(paths.info_path, {"id": "abc123"})
    write_text(paths.clean_transcript_path, "Hello world\n")
    write_json(
        paths.segments_path,
        [TranscriptSegment(start_sec=0.0, end_sec=1.0, text="Hello world").model_dump(mode="json")],
    )
    write_json(
        paths.chunk_index_path,
        [
            {
                "index": 1,
                "start_sec": 0.0,
                "end_sec": 1.0,
                "path": "chunks/chunk-001.txt",
                "char_count": 11,
            }
        ],
    )
    write_json(
        paths.summary_payload_path,
        {
            "video_id": "abc123",
            "url": "https://www.youtube.com/watch?v=abc123",
            "title": "Test Video",
            "source_kind": "subtitles",
            "transcript_path": "transcript/clean.txt",
            "segments_path": "transcript/segments.json",
            "chunk_index_path": "chunks/index.json",
            "chunk_count": 1,
            "segment_count": 1,
            "artifacts": {"subtitle_language": "en"},
            "transcription": None,
            "notes": ["Creator-provided subtitles are available in en; ASR can be skipped."],
        },
    )
    metadata = VideoMetadata(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
        title="Test Video",
        subtitles={
            "en": [SubtitleCandidate(language="en", ext="vtt", url="https://example.com/en.vtt")],
            "es": [SubtitleCandidate(language="es", ext="vtt", url="https://example.com/es.vtt")],
        },
        automatic_captions={},
    )
    downloaded_languages: list[str] = []

    def fake_probe_video(url: str) -> tuple[VideoMetadata, dict[str, object]]:
        return metadata, {"id": metadata.video_id, "webpage_url": metadata.url}

    def fake_download_subtitle_file(
        url: str,
        source_dir: Path,
        candidate: SubtitleCandidate,
        force: bool = False,
    ) -> Path:
        downloaded_languages.append(candidate.language)
        subtitle_path = source_dir / f"subtitles.{candidate.language}.vtt"
        write_text(
            subtitle_path,
            "WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHola mundo\n",
        )
        return subtitle_path

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fake_probe_video)
    monkeypatch.setattr("voxcraft.pipeline.download_subtitle_file", fake_download_subtitle_file)

    result = process_video(
        url="https://www.youtube.com/watch?v=abc123",
        config=PipelineConfig(base_data_dir=base_dir),
        language="es",
    )

    assert result.used_cache is False
    assert result.source_kind == SourceKind.MANUAL_SUBTITLES
    assert downloaded_languages == ["es"]
    assert paths.clean_transcript_path.read_text(encoding="utf-8") == "Hola mundo\n"
    assert read_json(paths.summary_payload_path)["artifacts"]["subtitle_language"] == "es"


def test_rechunk_video_reports_refreshed_chunk_count(tmp_path: Path) -> None:
    paths = initialize_workspace(build_artifact_paths(tmp_path / "video123", "video123"))
    metadata = VideoMetadata(
        video_id="video123",
        url="https://www.youtube.com/watch?v=video123",
        title="Chunk Count Test",
    )
    segments = [
        TranscriptSegment(start_sec=0.0, end_sec=1.0, text="alpha"),
        TranscriptSegment(start_sec=1.0, end_sec=2.0, text="beta"),
        TranscriptSegment(start_sec=2.0, end_sec=3.0, text="gamma"),
    ]
    write_json(paths.metadata_path, metadata.model_dump(mode="json"))
    write_json(paths.info_path, {"id": "video123"})
    write_json(paths.segments_path, [segment.model_dump(mode="json") for segment in segments])
    write_text(paths.clean_transcript_path, "alpha beta gamma\n")
    write_text(paths.raw_transcript_path, "alpha beta gamma\n")
    write_text(paths.transcript_srt_path, "1\n00:00:00,000 --> 00:00:03,000\nalpha beta gamma\n")
    write_json(
        paths.summary_payload_path,
        {
            "video_id": "video123",
            "url": metadata.url,
            "title": metadata.title,
            "source_kind": "local-asr",
            "transcript_path": "transcript/clean.txt",
            "segments_path": "transcript/segments.json",
            "chunk_index_path": "chunks/index.json",
            "chunk_count": 1,
            "segment_count": 3,
            "artifacts": {},
            "transcription": None,
            "notes": [],
        },
    )

    result = rechunk_video(
        video_id="video123",
        config=PipelineConfig(base_data_dir=tmp_path, chunk_target_chars=6),
    )

    assert result.chunk_count == 3
    assert read_json(paths.summary_payload_path)["chunk_count"] == 3


def test_rechunk_video_rejects_missing_summary_payload(tmp_path: Path) -> None:
    config = PipelineConfig(base_data_dir=tmp_path / "videos")
    paths = initialize_workspace(
        build_artifact_paths(config.base_data_dir / "abc123", "abc123")
    )
    metadata = VideoMetadata(
        video_id="abc123",
        url="https://www.youtube.com/watch?v=abc123",
    )
    write_json(paths.metadata_path, metadata.model_dump(mode="json"))
    write_json(
        paths.segments_path,
        [TranscriptSegment(start_sec=0, end_sec=1, text="hello").model_dump(mode="json")],
    )

    with pytest.raises(RuntimeError, match="rerun process"):
        rechunk_video("abc123", config)


def test_resolve_qwen_command_args_falls_back_to_module_wrapper(monkeypatch) -> None:
    monkeypatch.setattr("voxcraft.transcribe.shutil.which", lambda command: None)
    monkeypatch.setattr("voxcraft.transcribe.sys.executable", "/tmp/python")

    resolved = resolve_qwen_command_args("voxcraft-qwen")

    assert resolved == ["/tmp/python", "-m", "voxcraft.qwen_cli"]


def test_command_status_marks_missing_required_tools() -> None:
    assert _command_status(None, required=True) == "missing"
    assert _command_status(None, required=False) == "optional"
    assert _command_status("/usr/local/bin/tool", required=True) == "ok"


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

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fake_probe_video)

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


def test_process_video_dry_run_uses_high_quality_qwen_model(monkeypatch, tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="hq123",
        url="https://www.youtube.com/watch?v=hq123",
        title="High Quality Video",
        subtitles={},
        automatic_captions={},
    )

    def fake_probe_video(url: str) -> tuple[VideoMetadata, dict[str, object]]:
        return metadata, {"id": metadata.video_id, "webpage_url": metadata.url}

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fake_probe_video)

    result = process_video(
        url=metadata.url,
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        language="en",
        high_quality=True,
        dry_run=True,
    )

    assert result.transcription is not None
    assert result.transcription.model == "Qwen/Qwen3-ASR-1.7B"


def test_process_video_dry_run_prefers_explicit_subtitle_language(monkeypatch, tmp_path: Path) -> None:
    metadata = VideoMetadata(
        video_id="lang123",
        url="https://www.youtube.com/watch?v=lang123",
        title="Language Override Video",
        subtitles={
            "en": [SubtitleCandidate(language="en", ext="vtt", url="https://example.com/en.vtt")],
            "es": [SubtitleCandidate(language="es", ext="vtt", url="https://example.com/es.vtt")],
        },
        automatic_captions={},
    )

    def fake_probe_video(url: str) -> tuple[VideoMetadata, dict[str, object]]:
        return metadata, {"id": metadata.video_id, "webpage_url": metadata.url}

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fake_probe_video)

    result = process_video(
        url=metadata.url,
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        language="es",
        dry_run=True,
    )

    assert result.source_kind == SourceKind.MANUAL_SUBTITLES
    assert any("Creator-provided subtitles are available in es" in note for note in result.notes)


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

    monkeypatch.setattr("voxcraft.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr("voxcraft.transcribe.subprocess.run", fake_run)

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

    monkeypatch.setattr("voxcraft.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr("voxcraft.transcribe.subprocess.run", fake_run)
    monkeypatch.setattr("voxcraft.transcribe._diarize_qwen_payload_with_pyannote", fake_diarize)

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


def test_qwen_transcription_persists_raw_output_before_diarization(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_base = tmp_path / "asr-output"

    def fake_run(command, capture_output, text, check, env):
        output_dir = Path(command[command.index("--output-dir") + 1])
        write_json(
            output_dir / "audio.json",
            {
                "text": "Hello",
                "language": "en",
                "segments": [{"start": 0.0, "end": 0.5, "text": "Hello"}],
            },
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fail_diarize(**kwargs):
        raise RuntimeError("diarization failed")

    monkeypatch.setattr("voxcraft.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr("voxcraft.transcribe.subprocess.run", fake_run)
    monkeypatch.setattr("voxcraft.transcribe._diarize_qwen_payload_with_pyannote", fail_diarize)

    with pytest.raises(RuntimeError, match="diarization failed"):
        transcribe_audio_file(
            request=TranscriptionRequest(
                input_path=audio_path,
                backend="qwen3-asr",
                model="mlx-community/Qwen3-ASR-1.7B-8bit",
                language="en",
            ),
            qwen_diarize=True,
            output_base=output_base,
        )

    saved = read_json(output_base.with_suffix(".json"))
    assert saved["text"] == "Hello"
    assert "_voxcraft" in saved
    assert "speaker_segments" not in saved


def test_qwen_diarization_retry_clears_stale_speakers(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_path = tmp_path / "asr-output.json"
    request = TranscriptionRequest(
        input_path=audio_path,
        backend="qwen3-asr",
        model="mlx-community/Qwen3-ASR-1.7B-8bit",
        language="en",
    )
    write_json(
        output_path,
        {
            "text": "Hello",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "Hello", "speaker": "STALE"}
            ],
            "speaker_segments": [
                {"start": 0.0, "end": 0.5, "text": "Hello", "speaker": "STALE"}
            ],
            "_voxcraft": _qwen_payload_marker(
                request=request,
                context="",
                forced_aligner="Qwen/Qwen3-ForcedAligner-0.6B",
                dtype="float16",
                draft_model=None,
                num_draft_tokens=4,
            ),
        },
    )

    monkeypatch.setattr("voxcraft.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr(
        "voxcraft.transcribe.subprocess.run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("Qwen should not rerun")),
    )
    monkeypatch.setattr(
        "voxcraft.transcribe._diarize_qwen_payload_with_pyannote",
        lambda **kwargs: ([], []),
    )

    result = transcribe_audio_file(request=request, qwen_diarize=True, output_base=output_path.with_suffix(""))
    saved = read_json(output_path)

    assert result.details.diarized is False
    assert result.segments[0].speaker is None
    assert "speaker_segments" not in saved
    assert "speaker" not in saved["segments"][0]


def test_reusable_qwen_payload_treats_invalid_or_changed_input_as_cache_miss(tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_path = tmp_path / "asr-output.json"
    request = TranscriptionRequest(
        input_path=audio_path,
        backend="qwen3-asr",
        model="model",
        language="en",
    )
    options = {
        "output_json_path": output_path,
        "request": request,
        "context": "",
        "forced_aligner": "aligner",
        "dtype": "float16",
        "draft_model": None,
        "num_draft_tokens": 4,
    }
    output_path.write_text("{truncated", encoding="utf-8")
    assert _load_reusable_qwen_payload(**options) is None

    write_json(
        output_path,
        {
            "segments": [],
            "_voxcraft": _qwen_payload_marker(
                request=request,
                context="",
                forced_aligner="aligner",
                dtype="float16",
                draft_model=None,
                num_draft_tokens=4,
            ),
        },
    )
    audio_path.write_bytes(b"RIFF-changed")
    assert _load_reusable_qwen_payload(**options) is None


def test_qwen_transcription_reuses_saved_payload_for_diarization_retry(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_base = tmp_path / "asr-output"
    write_json(
        output_base.with_suffix(".json"),
        {
            "text": "Hello there",
            "language": "en",
            "segments": [
                {"start": 0.0, "end": 0.5, "text": "Hello"},
                {"start": 0.5, "end": 0.9, "text": "there"},
            ],
            "_voxcraft": {
                "backend": "qwen3-asr",
                "context": "",
                "dtype": "float16",
                "draft_model": None,
                "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
                "input_path": str(audio_path.resolve()),
                "input_size": audio_path.stat().st_size,
                "input_mtime_ns": audio_path.stat().st_mtime_ns,
                "input_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                "language": "en",
                "model": "mlx-community/Qwen3-ASR-1.7B-8bit",
                "num_draft_tokens": 4,
            },
        },
    )

    def fail_run(*args, **kwargs):
        raise AssertionError("saved ASR output should avoid rerunning Qwen")

    def fake_diarize(*, audio_path, payload, model_id, num_speakers, min_speakers, max_speakers):
        return (
            [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.9, "text": "Hello there"}],
            [
                {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5, "text": "Hello"},
                {"speaker": "SPEAKER_00", "start": 0.5, "end": 0.9, "text": "there"},
            ],
        )

    monkeypatch.setattr("voxcraft.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr("voxcraft.transcribe.subprocess.run", fail_run)
    monkeypatch.setattr("voxcraft.transcribe._diarize_qwen_payload_with_pyannote", fake_diarize)

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

    assert result.details.diarized is True
    assert result.details.speaker_count == 1
    assert result.segments[0].speaker == "SPEAKER_00"
    assert read_json(output_base.with_suffix(".json"))["speaker_segments"] == [
        {"speaker": "SPEAKER_00", "start": 0.0, "end": 0.9, "text": "Hello there"}
    ]


def test_qwen_transcription_can_force_rerun_with_saved_payload(monkeypatch, tmp_path: Path) -> None:
    audio_path = tmp_path / "audio.wav"
    audio_path.write_bytes(b"RIFF")
    output_base = tmp_path / "asr-output"
    write_json(
        output_base.with_suffix(".json"),
        {
            "text": "Stale text",
            "language": "en",
            "segments": [{"start": 0.0, "end": 0.5, "text": "Stale"}],
            "_voxcraft": {
                "backend": "qwen3-asr",
                "context": "",
                "dtype": "float16",
                "draft_model": None,
                "forced_aligner": "Qwen/Qwen3-ForcedAligner-0.6B",
                "input_path": str(audio_path.resolve()),
                "input_size": audio_path.stat().st_size,
                "input_mtime_ns": audio_path.stat().st_mtime_ns,
                "input_sha256": hashlib.sha256(audio_path.read_bytes()).hexdigest(),
                "language": "en",
                "model": "mlx-community/Qwen3-ASR-1.7B-8bit",
                "num_draft_tokens": 4,
            },
        },
    )
    subprocess_calls = 0

    def fake_run(command, capture_output, text, check, env):
        nonlocal subprocess_calls
        subprocess_calls += 1
        output_dir = Path(command[command.index("--output-dir") + 1])
        output_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            output_dir / "audio.json",
            {
                "text": "Fresh text",
                "language": "en",
                "segments": [{"start": 0.0, "end": 0.5, "text": "Fresh"}],
            },
        )
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    def fake_diarize(*, audio_path, payload, model_id, num_speakers, min_speakers, max_speakers):
        return (
            [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5, "text": "Fresh"}],
            [{"speaker": "SPEAKER_00", "start": 0.0, "end": 0.5, "text": "Fresh"}],
        )

    monkeypatch.setattr("voxcraft.transcribe.shutil.which", lambda command: f"/usr/local/bin/{command}")
    monkeypatch.setattr("voxcraft.transcribe.subprocess.run", fake_run)
    monkeypatch.setattr("voxcraft.transcribe._diarize_qwen_payload_with_pyannote", fake_diarize)

    result = transcribe_audio_file(
        request=TranscriptionRequest(
            input_path=audio_path,
            backend="qwen3-asr",
            model="mlx-community/Qwen3-ASR-1.7B-8bit",
            language="en",
        ),
        qwen_diarize=True,
        output_base=output_base,
        reuse_qwen_output=False,
    )

    assert subprocess_calls == 1
    assert result.text == "Fresh text"
    assert read_json(output_base.with_suffix(".json"))["text"] == "Fresh text"


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

    monkeypatch.setattr("voxcraft.pipeline.probe_video", fake_probe_video)

    result = process_video(
        url=metadata.url,
        config=PipelineConfig(base_data_dir=tmp_path / "videos"),
        language="en",
        asr_backend="whisper-cpp",
        dry_run=True,
    )

    assert result.transcription is not None
    assert result.transcription.backend == "whisper-cpp"
    assert result.transcription.model == "base"


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


def test_transcription_details_match_rejects_fingerprint_mismatch() -> None:
    cached = TranscriptionDetails(
        backend="qwen3-asr",
        model="mlx-community/Qwen3-ASR-1.7B-8bit",
        language="en",
        diarized=False,
        fingerprint="old-fingerprint",
    )
    requested = TranscriptionDetails(
        backend="qwen3-asr",
        model="mlx-community/Qwen3-ASR-1.7B-8bit",
        language="en",
        diarized=False,
        fingerprint="new-fingerprint",
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
                model="openai/gpt-5.5",
                thinking_level="high",
            )
        },
    )

    assert config.summary_command == "pi-custom"
    assert config.summary_model == "openai/gpt-5.5"
    assert config.summary_thinking_level == "high"
    assert config.summary_harness("pi").command == "pi-custom"
    assert config.summary_harness("pi").model == "openai/gpt-5.5"
    assert config.summary_harness("pi").thinking_level == "high"
    assert config.summary_harness("codex").model == "gpt-5.5"
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
                    "model": "gpt-5.5",
                    "thinking_level": "medium",
                },
                "pi": {
                    "command": "pi-custom",
                    "model": "openai/gpt-5.5",
                    "thinking_level": "high",
                },
            },
        },
    )

    config, resolved_path = load_pipeline_config(config_path=config_path)
    config = config.with_summary_overrides(
        provider="pi",
        model="openai/gpt-5.5",
    )

    assert resolved_path == config_path
    assert config.default_asr_backend == "whisper-cpp"
    assert config.summary_provider == "pi"
    assert config.summary_command == "pi-custom"
    assert config.summary_model == "openai/gpt-5.5"
    assert config.summary_thinking_level == "high"


def test_load_pipeline_config_rejects_unknown_keys(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_json(config_path, {"summary_provder": "codex"})

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_pipeline_config(config_path=config_path)


def test_load_pipeline_config_rejects_removed_subtitle_first_setting(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_json(config_path, {"subtitle_first": False})

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_pipeline_config(config_path=config_path)


def test_load_pipeline_config_rejects_legacy_summary_settings(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    write_json(config_path, {"summary_model": "gpt-5.5"})

    with pytest.raises(ValueError, match="Extra inputs are not permitted"):
        load_pipeline_config(config_path=config_path)


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
        model="openai/gpt-5.5",
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
        "openai/gpt-5.5",
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

    monkeypatch.setattr("voxcraft.summarize.run_summary_cli", fake_run_summary_cli)

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
            "model": "gpt-5.5",
            "thinking_level": "high",
        },
        {
            "provider": "codex",
            "command": "codex",
            "model": "gpt-5.5",
            "thinking_level": "high",
        },
    ]
    manifest = read_json(paths.summary_manifest_path)
    assert manifest["summary_provider"] == "codex"
    assert manifest["summary_command"] == "codex"
    assert manifest["summary_model"] == "gpt-5.5"
    assert manifest["summary_thinking_level"] == "high"
    assert manifest["prompt_version"] == 1
    assert manifest["chunk_summaries"][0]["prompt_sha256"]
    assert manifest["chunk_summaries"][0]["output_sha256"]
    assert manifest["final_prompt_sha256"]
    assert manifest["final_summary_sha256"]
    assert all(len(line) <= 80 for line in paths.summary_final_path.read_text(encoding="utf-8").splitlines())

    captured_summary_calls.clear()
    summarize_video(video_id="video123", config=PipelineConfig(base_data_dir=tmp_path))
    assert captured_summary_calls == []

    write_text(paths.summary_dir / "chunk-001.md", "corrupted summary\n")
    summarize_video(video_id="video123", config=PipelineConfig(base_data_dir=tmp_path))
    assert len(captured_summary_calls) == 2

    captured_summary_calls.clear()
    write_text(paths.summary_final_path, "truncated\n")
    summarize_video(video_id="video123", config=PipelineConfig(base_data_dir=tmp_path))
    assert len(captured_summary_calls) == 1

    captured_summary_calls.clear()
    changed_metadata = metadata.model_copy(update={"title": "Changed title"})
    write_json(paths.metadata_path, changed_metadata.model_dump(mode="json"))
    summarize_video(video_id="video123", config=PipelineConfig(base_data_dir=tmp_path))
    assert len(captured_summary_calls) == 2


def test_summarize_video_reruns_when_summary_settings_change(monkeypatch, tmp_path: Path) -> None:
    paths = initialize_workspace(build_artifact_paths(tmp_path / "video123", "video123"))
    metadata = VideoMetadata(
        video_id="video123",
        url="https://www.youtube.com/watch?v=video123",
        title="Summary Test",
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
    write_text(paths.summary_dir / "chunk-001.md", "stale chunk\n")
    write_text(paths.summary_final_path, "stale final\n")
    write_json(
        paths.summary_manifest_path,
        {
            "video_id": "video123",
            "summary_provider": "codex",
            "summary_command": "codex",
            "summary_model": "gpt-5.5",
            "summary_thinking_level": "high",
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
            "final_summary_path": "final.md",
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
        content = "# Final Summary\n\nfresh final\n" if "Combine these chunk summaries" in prompt else "fresh chunk\n"
        write_text(output_path, content)
        return content

    monkeypatch.setattr("voxcraft.summarize.run_summary_cli", fake_run_summary_cli)

    summarize_video(
        video_id="video123",
        config=PipelineConfig(
            base_data_dir=tmp_path,
            summary_provider="pi",
            summary_profiles={
                "pi": SummaryHarnessConfig(
                    command="pi",
                    model="openai/gpt-5.5",
                    thinking_level="high",
                )
            },
        ),
    )

    assert captured_summary_calls == [
        {
            "provider": "pi",
            "command": "pi",
            "model": "openai/gpt-5.5",
            "thinking_level": "high",
        },
        {
            "provider": "pi",
            "command": "pi",
            "model": "openai/gpt-5.5",
            "thinking_level": "high",
        },
    ]
    assert (paths.summary_dir / "chunk-001.md").read_text(encoding="utf-8") == "fresh chunk\n"
    assert paths.summary_final_path.read_text(encoding="utf-8") == "# Final Summary\n\nfresh final\n"
    manifest = read_json(paths.summary_manifest_path)
    assert manifest["summary_provider"] == "pi"
    assert manifest["summary_command"] == "pi"


def test_summarize_video_reruns_when_chunk_content_changes(monkeypatch, tmp_path: Path) -> None:
    paths = initialize_workspace(build_artifact_paths(tmp_path / "video123", "video123"))
    metadata = VideoMetadata(
        video_id="video123",
        url="https://www.youtube.com/watch?v=video123",
        title="Summary Test",
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
                "char_count": 19,
            }
        ],
    )
    write_text(paths.chunks_dir / "chunk-001.txt", "changed chunk text\n")
    write_json(
        paths.summary_payload_path,
        {
            "video_id": "video123",
            "url": metadata.url,
            "title": metadata.title,
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
    write_text(paths.summary_dir / "chunk-001.md", "stale chunk summary\n")
    write_text(paths.summary_final_path, "stale final\n")
    write_json(
        paths.summary_manifest_path,
        {
            "video_id": "video123",
            "summary_provider": "codex",
            "summary_command": "codex",
            "summary_model": "gpt-5.5",
            "summary_thinking_level": "high",
            "chunk_summaries": [
                {
                    "index": 1,
                    "start_sec": 0.0,
                    "end_sec": 12.0,
                    "source_chunk_path": "chunks/chunk-001.txt",
                    "source_chunk_sha256": hashlib.sha256(b"old chunk text").hexdigest(),
                    "prompt_path": "summary/prompts/chunk-001.prompt.txt",
                    "output_path": "summary/chunk-001.md",
                }
            ],
            "final_prompt_path": "summary/prompts/final.prompt.txt",
            "final_summary_path": "final.md",
        },
    )
    captured_prompts: list[str] = []

    def fake_run_summary_cli(
        prompt: str,
        output_path: Path,
        workdir: Path,
        provider: str,
        command: str,
        model: str | None = None,
        thinking_level: str | None = None,
    ) -> str:
        captured_prompts.append(prompt)
        content = "# Final Summary\n\nfresh final\n" if "Combine these chunk summaries" in prompt else "fresh chunk\n"
        write_text(output_path, content)
        return content

    monkeypatch.setattr("voxcraft.summarize.run_summary_cli", fake_run_summary_cli)

    summarize_video(video_id="video123", config=PipelineConfig(base_data_dir=tmp_path))

    assert len(captured_prompts) == 2
    assert (paths.summary_dir / "chunk-001.md").read_text(encoding="utf-8") == "fresh chunk\n"
    assert paths.summary_final_path.read_text(encoding="utf-8") == "# Final Summary\n\nfresh final\n"
    manifest = read_json(paths.summary_manifest_path)
    assert manifest["chunk_summaries"][0]["source_chunk_sha256"] == hashlib.sha256(
        b"changed chunk text"
    ).hexdigest()


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

    assert resolve_video_root(
        base_dir,
        "abc123",
        title="Test Video",
        upload_date="2026-06-11",
    ) == base_dir / "2026-06-11--test-video--abc123"

    legacy = base_dir / "abc123"
    legacy.mkdir()
    assert resolve_video_root(
        base_dir,
        "abc123",
        title="Changed Title",
        upload_date="2026-06-12",
    ) == legacy


def test_resolve_video_root_finds_existing_undated_human_readable_dir(tmp_path: Path) -> None:
    base_dir = tmp_path / "videos"
    existing = base_dir / "test-video--abc123"
    existing.mkdir(parents=True)

    assert resolve_video_root(
        base_dir,
        "abc123",
        title="Changed Title",
        upload_date="2026-06-12",
    ) == existing
