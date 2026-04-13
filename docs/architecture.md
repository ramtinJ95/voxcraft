# Architecture

This document explains how `yt-transcriber-cli` works internally and how the runtime pieces fit together.

## Design Goals

The project is opinionated in a few specific ways:

- prefer creator-provided subtitles over ASR
- never use YouTube auto-captions
- keep transcript preparation local
- keep artifacts durable and inspectable on disk
- allow a safer fallback backend when the default ASR stack is unavailable
- make summaries downstream of a reusable transcript workspace instead of mixing everything into one opaque command

## Pipeline Overview

The main entrypoint is `yt-transcriber process "<youtube-url>"`.

The end-to-end control flow is:

```text
CLI
  -> probe video metadata
  -> choose source path
     -> creator subtitles
     -> or local ASR
  -> build transcript artifacts
  -> chunk transcript
  -> write summary handoff payload
  -> optionally summarize via codex
```

Expanded:

```text
YouTube URL
  -> yt_dlp metadata probe
  -> creator subtitles available?
     -> yes:
        -> download subtitle file
        -> parse VTT/SRT
        -> clean transcript segments
     -> no:
        -> download best audio source
        -> ffmpeg normalize to source/audio.wav
        -> build transcription request
        -> qwen3-asr default path
           -> yt-transcriber-qwen wrapper
           -> mlx-qwen3-asr
           -> forced aligner
           -> optional pyannote diarization
        -> or whisper.cpp fallback
  -> write raw.txt / clean.txt / segments.json / transcript.srt
  -> chunk transcript into chunk-*.txt
  -> write summary_input/payload.json
  -> optional codex exec chunk summaries
  -> optional codex exec final summary
```

## Module Responsibilities

### CLI

[src/youtube_local_pipeline/cli.py](../src/youtube_local_pipeline/cli.py)

Defines the public commands:
- `doctor`
- `process`
- `summarize`
- `rechunk`
- `prepare-summary`

It also validates flag combinations such as:
- `--summarize` cannot be used with `--dry-run`
- `--diarize` is only valid on the Qwen backend

### Config

[src/youtube_local_pipeline/config.py](../src/youtube_local_pipeline/config.py)

Holds the runtime defaults:
- default ASR backend
- default models
- default language
- chunk size
- tool command names

The current effective defaults are:
- backend: `qwen3-asr`
- Qwen model: `mlx-community/Qwen3-ASR-1.7B-8bit`
- aligner: `Qwen/Qwen3-ForcedAligner-0.6B`
- diarization model: `pyannote/speaker-diarization-community-1`
- fallback backend: `whisper-cpp`
- summary model: `gpt-5.4`

### Download And Probe

[src/youtube_local_pipeline/download.py](../src/youtube_local_pipeline/download.py)

Uses the Python `yt_dlp` library directly.

Responsibilities:
- probe metadata
- map subtitle tracks into typed models
- choose a creator subtitle candidate
- download creator subtitles
- download source audio

Important behavior:
- creator subtitles only
- auto-captions are always ignored
- English is preferred first when available

### Audio Normalization

[src/youtube_local_pipeline/audio.py](../src/youtube_local_pipeline/audio.py)

Normalizes downloaded audio with `ffmpeg` to:
- mono
- 16 kHz
- WAV

This normalized file becomes `source/audio.wav`, which is the ASR input for the Qwen path.

### Transcription

[src/youtube_local_pipeline/transcribe.py](../src/youtube_local_pipeline/transcribe.py)

This module does three jobs:
- plan the transcription request
- execute the selected backend
- convert backend output into repo-level transcript segments

There are two backend code paths.

#### Qwen Path

Default backend: `qwen3-asr`

Flow:
- resolve the Qwen command
- call `yt-transcriber-qwen`
- request JSON output with timestamps
- optionally run pyannote diarization in repo code
- convert backend segments into `TranscriptSegment` objects

The Qwen command is not `mlx-qwen3-asr` directly. It is the repo wrapper in:

[src/youtube_local_pipeline/qwen_cli.py](../src/youtube_local_pipeline/qwen_cli.py)

That wrapper patches the upstream model loader at runtime so the hybrid `mlx-community/Qwen3-ASR-1.7B-8bit` checkpoint loads correctly.

The patch exists because the upstream loader quantized the entire model whenever it saw any `.scales` tensors, but this checkpoint is only partially quantized. The wrapper fixes that by:
- injecting tied `lm_head` weights from `model.embed_tokens`
- quantizing only submodules that actually have quantized tensors

#### Whisper Fallback

Fallback backend: `whisper-cpp`

Flow:
- resolve a local Whisper model path
- invoke `whisper-cli`
- parse the resulting JSON
- convert it into repo-level transcript segments

This path does not support diarization in the current CLI.

### Diarization

Still owned by [src/youtube_local_pipeline/transcribe.py](../src/youtube_local_pipeline/transcribe.py).

The repo intentionally does not rely on the upstream `mlx-qwen3-asr --diarize` behavior. Instead:
- Qwen ASR runs first
- pyannote runs afterward in repo code
- speaker turns are mapped back onto timestamped word segments
- merged speaker segments are written separately

The current diarization model is:
- `pyannote/speaker-diarization-community-1`

Access notes:
- inference is local
- model downloads are gated by Hugging Face terms/token access

Required environment variables for gated access:
- `PYANNOTE_AUTH_TOKEN`
- or `HF_TOKEN`
- or `HUGGINGFACE_TOKEN`

### Transcript Shaping

[src/youtube_local_pipeline/subtitles.py](../src/youtube_local_pipeline/subtitles.py) and [src/youtube_local_pipeline/clean.py](../src/youtube_local_pipeline/clean.py)

These modules:
- parse VTT or SRT
- normalize whitespace
- remove adjacent duplicates
- render speaker-prefixed transcript text where applicable
- generate:
  - `raw.txt`
  - `clean.txt`
  - `segments.json`
  - `transcript.srt`

### Chunking

[src/youtube_local_pipeline/chunk.py](../src/youtube_local_pipeline/chunk.py)

Chunks are built from already-cleaned transcript segments.

Current policy:
- target roughly `10000` characters
- never split a segment mid-segment
- preserve transcript order

Outputs:
- `chunks/chunk-001.txt`, etc.
- `chunks/index.json`

### Artifact Management

[src/youtube_local_pipeline/manifest.py](../src/youtube_local_pipeline/manifest.py)

Creates the workspace layout and summary handoff payload.

Workspace root:

```text
data/videos/<title-slug>--<youtube_id>/
```

This module also preserves compatibility with older `<youtube_id>`-only directories by resolving existing workspaces before creating new ones.

### Summarization

[src/youtube_local_pipeline/summarize.py](../src/youtube_local_pipeline/summarize.py)

This is the only non-local part of the pipeline.

Flow:
- read chunk index
- build a prompt per chunk
- call `codex exec`
- write `summary/chunk-*.md`
- build a final synthesis prompt
- call `codex exec` again
- write `summary/final.md`

Summary settings:
- default model: `gpt-5.4`
- reasoning effort: `high` when using `gpt-5.4`

The prompts are currently detail-preserving and technical by default.

## Cache And Reuse

The project reuses existing artifacts only when the requested run is compatible with the cached one.

For local ASR runs, cache matching checks:
- backend
- model
- language
- diarization enabled or disabled
- requested speaker count, if explicitly fixed

This prevents unsafe reuse such as:
- returning a whisper transcript for a requested Qwen run
- returning a non-diarized transcript for a diarized request

Use `--force` to recompute.

## Output Files

A typical local-ASR workspace contains:

```text
metadata.json
source/info.json
source/audio.webm
source/audio.wav
transcript/raw.txt
transcript/clean.txt
transcript/segments.json
transcript/speaker_segments.json
transcript/asr_output.json
transcript/transcript.srt
chunks/chunk-001.txt
chunks/index.json
summary_input/payload.json
summary/manifest.json
summary/final.md
logs/pipeline.log
```

Not every file appears on every run:
- no `speaker_segments.json` unless diarization is enabled
- no `summary/*` outputs unless summarization is run
- no audio source artifacts when creator subtitles are used successfully

## External Dependencies And Boundaries

### Fully Local

- metadata probe
- subtitle download and parsing
- audio download
- audio normalization
- local ASR
- local diarization
- transcript cleanup
- chunk generation
- summary payload generation

### Not Local

- `codex exec` summarization

### Gated Downloads

- Hugging Face model weights for pyannote
- Hugging Face model weights for Qwen and aligner on first use

## Known Operational Caveats

- The primary ASR path is Apple Silicon specific in practice because it depends on MLX.
- Long Qwen runs use `--quiet --no-progress`, so they can look stuck even when they are still computing.
- First runs can be slow because the models may need to be downloaded and cached.
- `whisper.cpp` is the operational fallback when the Qwen stack is unavailable or unstable.
- The current summary step assumes Codex CLI auth is already configured outside this repo.

## Public-Repo Replication Checklist

For a new user to reproduce the workflow successfully, they need:
- Apple Silicon macOS
- Python `3.11`
- `ffmpeg`
- `uv sync --group dev --python 3.11`
- optional `whisper-cli` if they want the fallback backend
- accepted Hugging Face terms plus token if they want diarization
- authenticated `codex` CLI if they want summarization

The most important user-facing command to verify after setup is:

```bash
yt-transcriber doctor
```
