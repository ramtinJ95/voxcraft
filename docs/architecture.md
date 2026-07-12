# Architecture

This document explains how `voxcraft` works internally and how the runtime pieces fit together.

## Design Goals

The project is opinionated in a few specific ways:

- prefer creator-provided subtitles over ASR
- never use YouTube auto-captions
- keep transcript preparation local
- keep artifacts durable and inspectable on disk
- allow a safer fallback backend when the default ASR stack is unavailable
- make summaries downstream of a reusable transcript workspace instead of mixing everything into one opaque command

## Pipeline Overview

The main entrypoint is `voxcraft process "<youtube-url>"`.

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
  -> optionally summarize via a supported summary CLI
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
           -> voxcraft-qwen wrapper
           -> mlx-qwen3-asr
           -> forced aligner
           -> optional pyannote diarization
        -> or whisper.cpp fallback
  -> write raw.txt / clean.txt / segments.json / transcript.srt
  -> chunk transcript into chunk-*.txt
  -> write summary_input/payload.json
  -> optional summary CLI chunk summaries
  -> optional summary CLI final summary
```

## Module Responsibilities

### CLI

[src/voxcraft/cli.py](../src/voxcraft/cli.py)

Defines the public commands:
- `doctor`
- `process`
- `summarize`
- `rechunk`

It also exposes a global `--config` option so commands can load a runtime `config.json`
before applying any per-run overrides.

It also validates flag combinations such as:
- `--summarize` cannot be used with `--dry-run`
- `--diarize` is only valid on the Qwen backend

### Config

[src/voxcraft/config.py](../src/voxcraft/config.py)

Holds the runtime defaults:
- default ASR backend
- default models
- default language
- chunk size
- tool command names

Runtime config is loaded in this order:
- `--config /path/to/config.json`
- `VOXCRAFT_CONFIG`
- `~/.config/voxcraft/config.json`
- built-in defaults only when no file is present

The current effective defaults are:
- backend: `qwen3-asr`
- Qwen model: `mlx-community/Qwen3-ASR-1.7B-8bit`
- aligner: `Qwen/Qwen3-ForcedAligner-0.6B`
- diarization model: `pyannote/speaker-diarization-community-1`
- fallback backend: `whisper-cpp`
- default summary provider: `codex`
- default Codex summary model: `gpt-5.5`
- default Codex thinking level: `high`

Summary CLI settings are stored per provider in `PipelineConfig.summary_profiles`.

Each provider profile can define:
- command
- model
- thinking level

The selected provider is exposed through:
- `summary_provider`
- `summary_command`
- `summary_model`
- `summary_thinking_level`

CLI flags like `--summary-provider`, `--summary-model`, and `--thinking-level` now
act only as per-run overrides on top of the loaded config.

### Download And Probe

[src/voxcraft/download.py](../src/voxcraft/download.py)

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

[src/voxcraft/audio.py](../src/voxcraft/audio.py)

Normalizes downloaded audio with `ffmpeg` to:
- mono
- 16 kHz
- WAV

This normalized file becomes `source/audio.wav`, which is the ASR input for the Qwen path.

### Transcription

[src/voxcraft/transcribe.py](../src/voxcraft/transcribe.py)

This module does three jobs:
- plan the transcription request
- execute the selected backend
- convert backend output into repo-level transcript segments

There are two backend code paths.

#### Qwen Path

Default backend: `qwen3-asr`

Flow:
- resolve the Qwen command
- call `voxcraft-qwen`
- request JSON output with timestamps
- optionally run pyannote diarization in repo code
- convert backend segments into `TranscriptSegment` objects

The Qwen command is not `mlx-qwen3-asr` directly. It is the repo wrapper in:

[src/voxcraft/qwen_cli.py](../src/voxcraft/qwen_cli.py)

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

Still owned by [src/voxcraft/transcribe.py](../src/voxcraft/transcribe.py).

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

[src/voxcraft/subtitles.py](../src/voxcraft/subtitles.py) and [src/voxcraft/clean.py](../src/voxcraft/clean.py)

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

[src/voxcraft/chunk.py](../src/voxcraft/chunk.py)

Chunks are built from already-cleaned transcript segments.

Current policy:
- target roughly `10000` characters
- never split a segment mid-segment
- preserve transcript order

Outputs:
- `chunks/chunk-001.txt`, etc.
- `chunks/index.json`

### Artifact Management

[src/voxcraft/manifest.py](../src/voxcraft/manifest.py)

Creates the workspace layout and summary handoff payload.

Workspace root:

```text
data/videos/<upload-date>--<title-slug>--<youtube_id>/
```

The upload date is formatted as `YYYY-MM-DD` when YouTube exposes it, which keeps the
folder list chronologically sortable. If no upload date is available, the folder falls
back to `<title-slug>--<youtube_id>/`.

This module also preserves compatibility with older `<youtube_id>`-only and
`<title-slug>--<youtube_id>` directories by resolving existing workspaces before
creating new ones.

### Summarization

[src/voxcraft/summarize.py](../src/voxcraft/summarize.py)

This is the only non-local part of the pipeline.

Flow:
- read chunk index
- build a prompt per chunk
- call the selected summary CLI
- write `summary/chunk-*.md`
- build a final synthesis prompt
- call the selected summary CLI again
- write `final.md`

Summary settings:
- supported providers: `codex`, `claude`, `gemini`, `pi`
- default provider: `codex`
- default Codex model: `gpt-5.5`
- default Codex thinking level: `high`
- thinking-level passthrough is currently implemented for `codex` and `pi`

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
final.md
logs/pipeline.log
```

`metadata.json` is intentionally compact: it contains stable video fields and subtitle
language lists. The full `yt-dlp` probe payload, including transient subtitle URLs and
format details, is kept in `source/info.json`.

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

- summary CLI summarization

### Gated Downloads

- Hugging Face model weights for pyannote
- Hugging Face model weights for Qwen and aligner on first use

## Known Operational Caveats

- The primary ASR path is Apple Silicon specific in practice because it depends on MLX.
- Long Qwen runs use `--quiet --no-progress`, so they can look stuck even when they are still computing.
- First runs can be slow because the models may need to be downloaded and cached.
- `whisper.cpp` is the operational fallback when the Qwen stack is unavailable or unstable.
- The current summary step assumes one supported summary CLI is installed and authenticated outside this repo.
- `claude` and `gemini` do not currently receive a headless thinking-level CLI flag from this repo because their official CLI docs do not expose the same level-based control as `codex` and `pi`.

## Public-Repo Replication Checklist

For a new user to reproduce the workflow successfully, they need:
- Apple Silicon macOS
- Python `3.11`
- `ffmpeg`
- `uv sync --group dev --python 3.11`
- optional `whisper-cli` if they want the fallback backend
- accepted Hugging Face terms plus token if they want diarization
- authenticated `codex`, `claude`, `gemini`, or `pi` CLI if they want summarization

The most important user-facing command to verify after setup is:

```bash
voxcraft doctor
```
