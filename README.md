# yt-transcriber-cli

Local YouTube transcript-prep pipeline for Apple Silicon Macs.

The workflow is:
- prefer creator-provided subtitles
- never use auto-captions
- otherwise download audio and transcribe locally
- optionally diarize speakers
- chunk the transcript
- optionally summarize chunks and the full video through a supported agent CLI

Outputs are written per video under `data/videos/<title-slug>--<youtube_id>/`.

For the implementation-level overview, see [docs/architecture.md](docs/architecture.md).

## Status

The current default stack is:
- ASR backend: `qwen3-asr`
- Qwen model: `mlx-community/Qwen3-ASR-1.7B-8bit`
- Forced aligner: `Qwen/Qwen3-ForcedAligner-0.6B`
- Diarization model: `pyannote/speaker-diarization-community-1`
- Fallback backend: `whisper.cpp`
- Default summary provider: `codex`
- Default Codex summary model: `gpt-5.4`
- Default Codex thinking level: `high`

The default Qwen path is invoked through the repo-owned `yt-transcriber-qwen` wrapper. That wrapper patches the upstream loader so the hybrid `mlx-community/Qwen3-ASR-1.7B-8bit` checkpoint works in a fresh environment without hand-editing the venv.

## Known Limitations

- The default ASR path is Apple Silicon oriented because it depends on MLX.
- Summarization is not local. It requires an authenticated supported summary CLI.
- Diarization requires Hugging Face access to the gated pyannote model.
- Long Qwen runs currently use `--quiet --no-progress`, so they can appear idle while they are still computing.

## Requirements

Supported target:
- Apple Silicon macOS
- Python `3.11`

Required external tool:
- `ffmpeg`

Optional external tools:
- `codex`
  Optional summary CLI
- `claude`
  Optional summary CLI
- `gemini`
  Optional summary CLI
- `pi`
  Optional summary CLI
- `whisper-cli`
  Required only for `--asr-backend whisper-cpp`

Python dependencies are installed with `uv sync` from [pyproject.toml](pyproject.toml).

## Install

Install the system prerequisites:

```bash
brew install python@3.11 uv ffmpeg
```

If you want the fallback backend too:

```bash
brew install whisper-cpp
```

Create the environment and install the project:

```bash
uv sync --group dev --python 3.11
source .venv/bin/activate
yt-transcriber doctor
```

## Auth And Gating

There are two important gated integrations:

1. Pyannote diarization
   The diarization model runs locally, but its weights are hosted on Hugging Face and are access-gated.

   Before using `--diarize`:
   - accept the model terms for `pyannote/speaker-diarization-community-1`
   - export a token

   Example:

   ```bash
   export PYANNOTE_AUTH_TOKEN=hf_...
   ```

   The code also accepts:
   - `HF_TOKEN`
   - `HUGGINGFACE_TOKEN`

2. Summary CLI authentication
   The transcript pipeline is local. Summarization is not.

   `yt-transcriber summarize ...` and `yt-transcriber process ... --summarize` require:
   - one supported summary CLI on `PATH`: `codex`, `claude`, `gemini`, or `pi`
   - an authenticated session for that CLI

## Environment Variables

Supported runtime environment variables:

- `YT_TRANSCRIBER_CONFIG`
  Override the runtime `config.json` path
- `PYANNOTE_AUTH_TOKEN`
  Hugging Face token for gated pyannote diarization models
- `HF_TOKEN`
  Alternate token name accepted by the diarization path
- `HUGGINGFACE_TOKEN`
  Alternate token name accepted by the diarization path
- `WHISPER_CPP_MODEL`
  Exact whisper.cpp model file path
- `WHISPER_CPP_MODEL_DIR`
  Directory containing whisper.cpp model files

The CLI does not load a `.env` file by itself. These variables must already be present in the shell environment.

## Runtime Config

The CLI now treats a JSON config as the primary source of runtime defaults.

Lookup order:
- `yt-transcriber --config /path/to/config.json ...`
- `$YT_TRANSCRIBER_CONFIG`
- `~/.config/yt-transcriber/config.json`

CLI flags still work, but they are now per-run overrides on top of the loaded config.

Example config:

```json
{
  "base_data_dir": "data/videos",
  "language_preference": "en",
  "default_asr_backend": "qwen3-asr",
  "summary_provider": "pi",
  "summary_profiles": {
    "codex": {
      "command": "codex",
      "model": "gpt-5.4",
      "thinking_level": "high"
    },
    "claude": {
      "command": "claude",
      "model": "claude-sonnet-4-5"
    },
    "gemini": {
      "command": "gemini",
      "model": "gemini-2.5-pro"
    },
    "pi": {
      "command": "pi",
      "model": "openai/gpt-5.4",
      "thinking_level": "high"
    }
  }
}
```

An example file is included at [config.example.json](config.example.json).

## Quick Start

Probe a video without downloading media:

```bash
yt-transcriber process "https://www.youtube.com/watch?v=..." --dry-run
```

Run the normal local pipeline:

```bash
yt-transcriber process "https://www.youtube.com/watch?v=..."
```

Run the pipeline and summarize:

```bash
yt-transcriber process "https://www.youtube.com/watch?v=..." --summarize
```

Use a specific config file for the whole run:

```bash
yt-transcriber --config ./config.json process "https://www.youtube.com/watch?v=..." --summarize
```

Temporarily override the configured summary CLI:

```bash
yt-transcriber process "https://www.youtube.com/watch?v=..." \
  --summarize \
  --summary-provider claude
```

Enable diarization for multi-speaker audio:

```bash
yt-transcriber process "https://www.youtube.com/watch?v=..." --diarize --num-speakers 2
```

Use the safe fallback backend:

```bash
yt-transcriber process "https://www.youtube.com/watch?v=..." \
  --asr-backend whisper-cpp \
  --whisper-cpp-model ./models/ggml-large-v3.bin
```

Summarize an already-processed video:

```bash
yt-transcriber summarize <youtube_id>
```

Summarize with a specific config file:

```bash
yt-transcriber --config ./config.json summarize <youtube_id>
```

Override the configured Pi model for one run:

```bash
yt-transcriber summarize <youtube_id> \
  --provider pi \
  --model openai/gpt-5.4 \
  --thinking-level high
```

## Commands

Available commands:

```bash
yt-transcriber doctor
yt-transcriber --config ./config.json doctor
yt-transcriber process "<youtube-url>" --dry-run
yt-transcriber process "<youtube-url>"
yt-transcriber process "<youtube-url>" --summarize
yt-transcriber --config ./config.json process "<youtube-url>" --summarize
yt-transcriber process "<youtube-url>" --summarize --summary-provider claude
yt-transcriber process "<youtube-url>" --summarize --summary-provider pi --summary-model openai/gpt-5.4 --summary-thinking-level high
yt-transcriber process "<youtube-url>" --diarize
yt-transcriber summarize <youtube_id>
yt-transcriber --config ./config.json summarize <youtube_id>
yt-transcriber summarize <youtube_id> --provider gemini
yt-transcriber summarize <youtube_id> --provider pi --model openai/gpt-5.4 --thinking-level high
yt-transcriber rechunk <youtube_id>
yt-transcriber prepare-summary <youtube_id>
```

Command behavior:
- `doctor`
  Shows environment status, installed Python packages, auth-related readiness, and the loaded config path
- `process --dry-run`
  Probes metadata and prints the planned source path without downloading media
- `process`
  Runs the real subtitle or ASR pipeline
- `summarize`
  Reuses existing chunk artifacts and generates chunk summaries plus `final.md`
  `--summary-provider` / `--provider`, `--summary-model` / `--model`, and `--summary-thinking-level` / `--thinking-level` are per-run overrides on top of the loaded config
- `rechunk`
  Regenerates chunk files from existing transcript segments
- `prepare-summary`
  Rebuilds the summary payload from existing transcript artifacts

## Workflow

At a high level:

```text
YouTube URL
  -> yt-dlp metadata probe
  -> creator subtitles?
     -> yes: download + parse subtitles
     -> no: download source audio -> ffmpeg -> audio.wav -> local ASR
  -> optional pyannote diarization
  -> transcript cleanup
  -> transcript chunks
  -> summary_input/payload.json
  -> optional summary CLI chunk summaries
  -> optional summary CLI final summary
```

Important policy decisions:
- creator-provided subtitles are used when available
- auto-captions are always ignored
- local ASR consumes the normalized `source/audio.wav`
- diarization is supported only on the Qwen path

## Output Layout

Each processed video gets a folder like:

```text
data/videos/<title-slug>--<youtube_id>/
```

Common files:
- `metadata.json`
- `source/info.json`
- `source/audio.<ext>`
- `source/audio.wav`
- `transcript/raw.txt`
- `transcript/clean.txt`
- `transcript/segments.json`
- `transcript/speaker_segments.json`
- `transcript/asr_output.json`
- `transcript/transcript.srt`
- `chunks/chunk-*.txt`
- `chunks/index.json`
- `summary_input/payload.json`
- `summary/manifest.json`
- `final.md`
- `logs/pipeline.log`

## Sample Output

Example workspace after a successful run:

```text
data/videos/how-hardware-makes-threads-less-of-a-nightmare--IMceN4_rieo/
├── metadata.json
├── final.md
├── source/
│   ├── info.json
│   ├── audio.webm
│   └── audio.wav
├── transcript/
│   ├── raw.txt
│   ├── clean.txt
│   ├── segments.json
│   ├── asr_output.json
│   └── transcript.srt
├── chunks/
│   ├── chunk-001.txt
│   ├── chunk-002.txt
│   └── index.json
├── summary_input/
│   └── payload.json
├── summary/
│   ├── chunk-001.md
│   ├── chunk-002.md
│   ├── manifest.json
│   └── prompts/
└── logs/
    └── pipeline.log
```

Typical `process` result fields shown by the CLI:

```text
video_id: IMceN4_rieo
source_kind: local-asr
subtitle_policy: creator-only
cached: False
chunk_count: 2
transcription_backend: qwen3-asr
transcription_model: mlx-community/Qwen3-ASR-1.7B-8bit
language: en
diarized: False
```

Typical final outputs to inspect:
- `final.md`
- `summary/manifest.json`
- `summary_input/payload.json`
- `transcript/segments.json`
- `logs/pipeline.log`

## Reproducibility Notes

Caching is per video workspace and is not purely keyed by URL. The pipeline checks the requested processing mode before reusing artifacts.

For local ASR runs, cache reuse depends on:
- backend
- model
- language
- diarization enabled or disabled
- fixed speaker count, if explicitly requested

Use `--force` to recompute.

## Operational Caveats

- The primary ASR path is Apple Silicon oriented because it depends on MLX.
- First runs can be slow because model weights may need to be downloaded from Hugging Face.
- Long Qwen runs currently use `--quiet --no-progress`, so they can look idle even when they are still computing.
- `whisper.cpp` is the safer fallback path if the Qwen stack is unavailable or unstable on a given machine.
- Summary provider, command, model, and thinking level can all be set in `config.json`; CLI flags override only the selected provider for the current run.
- The loaded runtime config comes from `--config`, then `$YT_TRANSCRIBER_CONFIG`, then `~/.config/yt-transcriber/config.json`.
- `--summary-thinking-level` and `--thinking-level` are currently applied to providers that expose a comparable headless flag: `codex` and `pi`.
- The `python3.11` and `yt-dlp` shell commands are convenient but not strictly required after the environment is built, because the runtime uses the active interpreter and the installed `yt_dlp` Python package.

## Code Map

Main modules:
- [src/youtube_local_pipeline/cli.py](src/youtube_local_pipeline/cli.py)
- [src/youtube_local_pipeline/config.py](src/youtube_local_pipeline/config.py)
- [src/youtube_local_pipeline/pipeline.py](src/youtube_local_pipeline/pipeline.py)
- [src/youtube_local_pipeline/download.py](src/youtube_local_pipeline/download.py)
- [src/youtube_local_pipeline/audio.py](src/youtube_local_pipeline/audio.py)
- [src/youtube_local_pipeline/transcribe.py](src/youtube_local_pipeline/transcribe.py)
- [src/youtube_local_pipeline/qwen_cli.py](src/youtube_local_pipeline/qwen_cli.py)
- [src/youtube_local_pipeline/subtitles.py](src/youtube_local_pipeline/subtitles.py)
- [src/youtube_local_pipeline/clean.py](src/youtube_local_pipeline/clean.py)
- [src/youtube_local_pipeline/chunk.py](src/youtube_local_pipeline/chunk.py)
- [src/youtube_local_pipeline/summarize.py](src/youtube_local_pipeline/summarize.py)
- [src/youtube_local_pipeline/manifest.py](src/youtube_local_pipeline/manifest.py)
- [src/youtube_local_pipeline/models.py](src/youtube_local_pipeline/models.py)

## Tests

Run:

```bash
.venv/bin/pytest -q
```

The current repo test suite covers the main planning and artifact-generation paths, including the repo-owned Qwen wrapper patch logic.

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
