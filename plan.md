# YouTube Transcriber CLI Plan

## Goal

Set up a local YouTube transcript-prep pipeline on a **Mac mini with M2 Pro and 16GB unified memory**.

### Constraints
- All steps **before summarization** must run **locally**.
- No cloud transcription or cloud preprocessing.
- Final summarization will be done later with a SOTA model like OpenAI GPT, but that is outside the scope of this setup.
- Prefer a **Python-native implementation**.

---

## Short Recommendation

For a Python-native local setup on this machine, use:

- **Homebrew** — system packages
- **Python 3.11 + `uv`** — env/package management
- **`yt-dlp`** — fetch YouTube metadata, subtitles, or audio
- **`ffmpeg`** — audio extraction + normalization
- **`lightning-whisper-mlx`** — local transcription on Apple Silicon
- **`webvtt-py`** — parse subtitle files when captions exist
- **`typer`** — local CLI wrapper
- **stdlib JSON/filesystem** — artifact storage

Keep **`whisper-cpp` as a fallback**, not the default.

---

# 1. Framework Plan

## A. System/setup layer

### 1) Homebrew
Use Homebrew for machine-level dependencies.

### Install via Brew
- `python@3.11`
- `uv`
- `ffmpeg`

Optional:
- `yt-dlp` CLI for manual debugging
- `whisper-cpp` as fallback

### Why
- simplest on macOS
- clean install story for agents
- easy upgrades and diagnostics

---

## B. Python runtime layer

### 2) Python 3.11
Use a dedicated project env.

### 3) `uv`
Use `uv` for:
- virtualenv creation
- dependency install
- lock/freeze if wanted

### Why
- fast
- clean reproducibility
- good default for agent setup

If less tooling is preferred, fallback to:
- `python3 -m venv .venv`
- `pip install ...`

---

## C. Media retrieval layer

### 4) `yt-dlp`
Use the **Python package**, not just the CLI.

Use it for:
- video metadata
- subtitle availability check
- subtitle download
- audio-only download fallback

### Why
- robust YouTube support
- subtitle-first flow is easy
- Python API keeps everything in one pipeline

Current docs support subtitle download and audio extraction via Python config, including:
- `writesubtitles`
- `writeautomaticsub`
- `subtitleslangs`
- FFmpeg postprocessors

---

## D. Audio processing layer

### 5) `ffmpeg`
Use system `ffmpeg` via subprocess from Python.

Use it for:
- converting source audio to WAV
- mono / 16k normalization
- trimming or re-encoding if needed

### Why
- stable
- universal
- best tool for audio prep

### Recommended ASR audio output
- WAV
- mono
- 16 kHz

---

## E. Local transcription layer

### 6) `lightning-whisper-mlx`
This is the core of Option 2.

Use it for:
- local transcription
- Apple Silicon acceleration
- configurable model size / batch size / quantization

### Why this over `whisper-cpp` for Option 2
- more Python-native
- easier integration into a single Python app
- good fit for Apple Silicon
- model downloads automatically on first use
- supports tuning like `batch_size`, `language`, `quant`

### Model guidance for this machine
For **M2 Pro / 16GB**:

#### Default
- `distil-medium.en`
- `batch_size=12`
- English-only content

#### Multilingual
- `medium`
- `batch_size=8` or `10`

#### Higher quality, slower
- `large-v3`
- `quant="4bit"`
- `batch_size=6`

Do **not** make `large-v3` the default on 16GB.

---

## F. Subtitle parsing / transcript cleanup layer

### 7) `webvtt-py`
Use it if subtitles exist.

Use it for:
- parsing `.vtt`
- extracting timestamped caption text
- converting captions into clean transcript paragraphs

### Why
- simpler than writing a VTT parser
- keeps subtitle-first path lightweight

If zero extra dependency is desired here, VTT/SRT can be parsed manually, but `webvtt-py` is the preferred choice.

---

## G. CLI/orchestration layer

### 8) `typer`
Use it to expose commands like:

- `process <url>`
- `process <url> --model distil-medium.en --language en`
- `rechunk <video_id>`
- `prepare-summary <video_id>`

### Why
- nice CLI ergonomics
- good for agents and humans
- simpler than building a full UI

---

# 2. Recommended Architecture

## Main rule
**Always try subtitles first. Only transcribe if subtitles are missing or unusable.**

That gives the best combination of:
- speed
- lower compute
- better timestamp alignment
- less local processing

---

## Pipeline

```text
YouTube URL
  -> yt-dlp metadata probe
  -> subtitles available?
       yes -> download VTT/SRT -> parse -> clean transcript
       no  -> download audio only -> ffmpeg normalize -> local MLX Whisper transcription
  -> segment-aware cleanup
  -> chunk transcript
  -> write artifacts for OpenAI summary step
```

---

# 3. What the Project Should Produce

For each video, store a self-contained artifact folder.

## Suggested output structure

```text
data/
  videos/
    <youtube_id>/
      metadata.json
      source/
        info.json
        subtitles.en.vtt
        audio.m4a
        audio.wav
      transcript/
        raw.txt
        clean.txt
        segments.json
        transcript.srt
      chunks/
        chunk-001.txt
        chunk-002.txt
        index.json
      summary_input/
        payload.json
      logs/
        pipeline.log
```

### Important outputs
- `metadata.json` — title, channel, duration, url, ids
- `clean.txt` — best transcript text for summarization
- `segments.json` — timestamped segments
- `chunks/*.txt` — chunked text for OpenAI
- `summary_input/payload.json` — ready-to-send structured handoff

---

# 4. Implementation Plan for the Agent

## Phase 1 — Base setup

### Install machine dependencies
```bash
brew install python@3.11 uv ffmpeg
```

Optional:
```bash
brew install yt-dlp whisper-cpp
```

### Create project
```bash
mkdir -p ~/youtube-local-pipeline
cd ~/youtube-local-pipeline
uv venv --python 3.11
source .venv/bin/activate
uv pip install yt-dlp lightning-whisper-mlx webvtt-py typer pydantic rich
```

If avoiding `uv`:
```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install yt-dlp lightning-whisper-mlx webvtt-py typer pydantic rich
```

---

## Phase 2 — Project structure

### Suggested code layout
```text
youtube_local_pipeline/
  pyproject.toml
  README.md
  src/
    youtube_local_pipeline/
      __init__.py
      cli.py
      config.py
      download.py
      subtitles.py
      audio.py
      transcribe.py
      clean.py
      chunk.py
      manifest.py
      models.py
      utils.py
  data/
```

---

## Phase 3 — Core modules

### `download.py`
Responsibilities:
- accept YouTube URL
- fetch metadata with `yt-dlp`
- detect available subtitles / auto-captions
- download subtitles if present
- otherwise download audio only

### `subtitles.py`
Responsibilities:
- parse `.vtt`
- flatten caption fragments
- dedupe repeated caption lines
- preserve timestamps
- emit:
  - `raw.txt`
  - `clean.txt`
  - `segments.json`

### `audio.py`
Responsibilities:
- convert downloaded source audio to normalized WAV
- use `ffmpeg`
- standardize:
  - mono
  - 16kHz
  - WAV

### `transcribe.py`
Responsibilities:
- run `lightning-whisper-mlx`
- choose model from config
- output:
  - transcript text
  - segments
  - detected language
- save JSON + TXT artifacts

### `clean.py`
Responsibilities:
- normalize whitespace
- merge tiny fragments into paragraphs
- remove caption duplication
- keep timestamped segments untouched separately

### `chunk.py`
Responsibilities:
- chunk transcript for later OpenAI calls
- avoid splitting mid-segment
- produce:
  - text chunks
  - chunk index JSON
  - optional time ranges per chunk

### `manifest.py`
Responsibilities:
- write a single summary-ready manifest
- list all artifact paths
- capture whether source was:
  - subtitles
  - auto-captions
  - local ASR

---

# 5. Processing Logic

## Step 1 — Probe metadata
Use `yt-dlp` to get:
- video id
- title
- duration
- channel
- subtitle availability
- automatic caption availability

Save raw metadata immediately.

---

## Step 2 — Subtitle-first branch

If subtitles or auto-generated captions exist:

1. download them
2. prefer English if available
3. prefer creator subtitles over auto-captions
4. parse `.vtt`
5. build cleaned transcript
6. skip ASR entirely

### Priority order
1. manual English subtitles
2. auto English subtitles
3. requested language subtitles
4. fallback to local transcription

---

## Step 3 — Audio fallback branch

If subtitles are missing or poor:

1. download bestaudio only
2. save original source audio
3. normalize with `ffmpeg`:
   ```bash
   ffmpeg -y -i input.m4a -ar 16000 -ac 1 output.wav
   ```
4. pass WAV into `lightning-whisper-mlx`

---

## Step 4 — Local transcription

Use `lightning-whisper-mlx`.

### Recommended defaults
For this machine:

#### English
- model: `distil-medium.en`
- batch size: `12`
- quant: `None`

#### Multilingual
- model: `medium`
- batch size: `8`

#### Higher accuracy
- model: `large-v3`
- quant: `4bit`
- batch size: `6`

### Save
- transcript text
- segment list
- language
- model config used

---

## Step 5 — Clean and normalize transcript

The agent should:
- remove repeated caption artifacts
- merge broken line fragments
- preserve sentence boundaries where possible
- keep a plain-text summarization version
- keep timestamped segments separate

### Outputs
- `raw.txt`
- `clean.txt`
- `segments.json`

---

## Step 6 — Chunk for OpenAI

Chunking should happen **locally** before any cloud call.

### Recommended chunk strategy
- chunk by segment boundaries
- target roughly **8k–12k characters**
- include:
  - chunk index
  - start time
  - end time
  - text

### Output
`chunks/index.json`
```json
[
  {
    "index": 1,
    "start_sec": 0,
    "end_sec": 540,
    "path": "chunks/chunk-001.txt"
  }
]
```

This makes the later OpenAI summarization step much cleaner.

---

# 6. Runtime Defaults

## Default config
```text
language_preference = en
subtitle_first = true
allow_auto_subtitles = true
audio_sample_rate = 16000
audio_channels = 1
default_model_english = distil-medium.en
default_model_multilingual = medium
default_batch_size_english = 12
default_batch_size_multilingual = 8
chunk_target_chars = 10000
reuse_cached_artifacts = true
```

---

# 7. Important Operational Notes

## First-run model download
`lightning-whisper-mlx` will download model weights on first use.

So this setup is:
- **local at runtime**
- but **not air-gapped during initial setup**

If fully offline use is desired after setup, tell the agent to:
1. pre-run the model once
2. verify it is cached locally
3. document cache location

---

## Fallback plan
If `lightning-whisper-mlx` gives trouble on the machine, fallback to:
- `whisper-cpp` installed via Homebrew
- same Python pipeline, but transcription done by subprocess

That fallback is worth keeping in the README.

---

## Reprocessing/caching
The agent should make reruns idempotent:
- key output folder by YouTube video ID
- skip steps if artifacts already exist
- allow `--force` to recompute

---

## No cloud before summary
The code should not call:
- OpenAI
- external transcription APIs
- cloud diarization services

Only local processing until the handoff files are ready.

---

# 8. Success Criteria

The agent’s implementation is successful if:

1. Given a YouTube URL with captions:
   - it downloads captions
   - produces a clean transcript
   - skips local ASR

2. Given a YouTube URL without captions:
   - it downloads audio only
   - normalizes audio
   - transcribes locally with MLX Whisper
   - produces transcript + segments

3. In both cases:
   - it writes chunk files
   - it writes a summary-ready payload
   - no cloud API is called

---

# 9. Exact Setup Recommendation

## Recommended primary path
### Python-native MLX stack
- `python@3.11`
- `uv`
- `ffmpeg`
- `yt-dlp`
- `lightning-whisper-mlx`
- `webvtt-py`
- `typer`

## Recommended fallback
### CLI ASR fallback
- `whisper-cpp`

---

# 10. Copy/Paste Brief for the Agent on the Mac mini

```text
Set up a local YouTube transcript-prep pipeline on this Mac mini (M2 Pro, 16GB unified memory).

Constraints:
- All steps before summarization must run locally.
- No cloud transcription or cloud preprocessing.
- Final summarization will be done later with OpenAI, but do not implement that yet unless needed for handoff formatting.
- Prefer a Python-native implementation.

Recommended stack:
- Homebrew: python@3.11, uv, ffmpeg
- Python packages: yt-dlp, lightning-whisper-mlx, webvtt-py, typer, pydantic, rich
- Optional fallback: whisper-cpp via Homebrew

What to build:
1. A CLI tool that accepts a YouTube URL.
2. It should probe metadata with yt-dlp.
3. It should try subtitles first:
   - prefer manual English subtitles
   - then auto English subtitles
   - parse VTT and create a clean transcript
4. If subtitles are unavailable:
   - download audio only
   - normalize to mono 16k WAV with ffmpeg
   - transcribe locally with lightning-whisper-mlx
5. Save artifacts under a folder keyed by YouTube video ID:
   - metadata.json
   - source subtitles/audio
   - transcript raw.txt
   - transcript clean.txt
   - transcript segments.json
   - chunks/chunk-*.txt
   - chunks/index.json
   - summary_input/payload.json
6. Make reruns idempotent and cached.
7. Provide a README with install/run instructions.
8. Add a fallback mode using whisper-cpp if MLX transcription has issues.

Suggested defaults:
- English default model: distil-medium.en, batch_size=12
- Multilingual fallback: medium, batch_size=8
- Optional higher-quality mode: large-v3, quant=4bit, batch_size=6

Success criteria:
- Works for one URL with captions and one URL without captions
- No cloud calls before summary preparation
- Outputs clean transcript and chunked summary-ready files
```

---

# 11. Recommendation Between the Two Local Transcription Engines

## Best for Option 2
**`lightning-whisper-mlx`**

## Best fallback / safety net
**`whisper-cpp`**

If instructing the agent directly:
- build the app around **MLX**
- keep **`whisper-cpp`** as a backup path

---

# 12. Optional Next Steps

If desired later, the next useful docs to create would be:
1. a cleaner agent prompt
2. a file-by-file implementation spec
3. a first draft `README.md` and setup checklist for the machine

