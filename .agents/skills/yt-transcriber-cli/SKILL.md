---
name: yt-transcriber-cli
description: Operate the local `yt-transcriber` CLI for YouTube transcript preparation, diarization, chunking, and Codex summarization. Use when Codex needs to run or diagnose this repo's pipeline, process a YouTube URL end to end, inspect transcript artifacts, rerun summaries, choose between `qwen3-asr` and `whisper.cpp`, or explain where outputs are stored and how the workflow behaves.
---

# YT Transcriber CLI

Use this skill when working in the `yt-transcriber-cli` repository.

## Quick Start

Run the CLI from the repo venv:

```bash
.venv/bin/yt-transcriber doctor
.venv/bin/yt-transcriber process "<youtube-url>"
.venv/bin/yt-transcriber process "<youtube-url>" --summarize
.venv/bin/yt-transcriber summarize <youtube_id>
```

Prefer `.venv/bin/yt-transcriber` over assuming a global install.

## Default Behavior

Assume these defaults unless the code or README has changed:

- Subtitle policy: creator-provided subtitles only. Never use auto-captions.
- Default ASR backend: `qwen3-asr`
- Default Qwen model: `mlx-community/Qwen3-ASR-1.7B-8bit`
- Default forced aligner: `Qwen/Qwen3-ForcedAligner-0.6B`
- Default diarization model: `pyannote/speaker-diarization-community-1`
- Default language: `en`
- Summary model: `gpt-5.4`
- Summary reasoning effort: `high`

The pipeline downloads source audio, normalizes it to `source/audio.wav`, then feeds that WAV to ASR.

## Command Selection

Use `doctor` first when the task is environment verification or dependency diagnosis.

Use `process "<url>" --dry-run` first when you need to inspect the planned branch without downloading media.

Use `process "<url>"` when the user wants a full local transcript pipeline run.

Use `process "<url>" --summarize` when the user wants the full pipeline plus chunk summaries and a final summary in one step.

Use `summarize <video_id>` when transcript artifacts already exist and only the summary step needs to run or rerun.

Use `rechunk <video_id>` when chunk boundaries need regeneration from existing transcript segments.

Use `prepare-summary <video_id>` when summary payload artifacts need regeneration from an existing transcript.

## High-Value Flags

Prefer these flags when they are actually needed:

- `--summarize` to run the full end-to-end summary path
- `--force` to ignore cached artifacts and recompute
- `--dry-run` to inspect the planned path without downloading media
- `--diarize` to enable pyannote speaker labeling on the Qwen path
- `--num-speakers <n>` when speaker count is known; prefer this over loose estimation
- `--min-speakers <n>` and `--max-speakers <n>` when speaker count is uncertain but bounded
- `--asr-backend whisper-cpp` when Qwen is unavailable, unstable, or the user explicitly wants the safer fallback
- `--model ...` only when overriding the backend default intentionally
- `--whisper-cpp-model <path>` when using the whisper fallback and the model file must be pinned explicitly
- `--data-dir <path>` when outputs must be written somewhere other than `data/videos`

Do not combine `--diarize` with `--asr-backend whisper-cpp`; the CLI rejects that combination.

## Recommended Run Patterns

Use these patterns directly:

```bash
.venv/bin/yt-transcriber process "<youtube-url>"
.venv/bin/yt-transcriber process "<youtube-url>" --summarize
.venv/bin/yt-transcriber process "<youtube-url>" --diarize --summarize
.venv/bin/yt-transcriber process "<youtube-url>" --asr-backend whisper-cpp --whisper-cpp-model ./models/ggml-large-v3.bin
.venv/bin/yt-transcriber summarize <youtube_id> --force
```

When testing a new URL, a good sequence is:

1. `doctor`
2. `process "<url>" --dry-run`
3. `process "<url>" --summarize`

## Output Layout

Artifacts are stored under:

```text
data/videos/<title-slug>--<youtube_id>/
```

Common files to inspect:

- `metadata.json`
- `source/info.json`
- `source/audio.wav`
- `transcript/raw.txt`
- `transcript/clean.txt`
- `transcript/segments.json`
- `transcript/speaker_segments.json` when diarization is enabled
- `transcript/asr_output.json`
- `chunks/index.json`
- `summary_input/payload.json`
- `summary/manifest.json`
- `summary/final.md`
- `logs/pipeline.log`

## Diagnosis Workflow

When a run looks wrong, inspect in this order:

1. `logs/pipeline.log`
2. `summary_input/payload.json`
3. `transcript/segments.json`
4. `transcript/speaker_segments.json` when diarization is involved
5. `summary/manifest.json` and `summary/final.md` for summary-only issues

Use this logic:

- If creator subtitles exist but transcript quality is poor, inspect the downloaded subtitle artifact and parsed segments.
- If no creator subtitles exist, the pipeline should go through local ASR.
- If transcript coverage is short, check segment count and final segment end time in `segments.json`.
- If speaker attribution is poor, check whether diarization was enabled and whether `num-speakers` should have been set explicitly.
- If a rerun unexpectedly reused old work, inspect the cached workspace and use `--force`.

## Practical Notes

- Prefer `qwen3-asr` for the main path.
- Prefer `whisper.cpp` when the user asks for the fallback or when the Qwen runtime is failing.
- Treat `community-1` as the main diarization quality lever; changing Qwen precision is a secondary lever for speaker attribution.
- A future move to `mlx-community/Qwen3-ASR-1.7B-bf16` is reasonable only if real transcript quality, not diarization, becomes the limiting factor.
- Avoid deleting workspaces unless the user explicitly asks for a clean run.

## Verify Before Explaining

Before answering questions about behavior, prefer checking:

- `README.md`
- `src/youtube_local_pipeline/cli.py`
- `src/youtube_local_pipeline/config.py`
- `src/youtube_local_pipeline/pipeline.py`
- `src/youtube_local_pipeline/transcribe.py`

The repo has changed quickly; verify the live defaults before stating them as facts.
