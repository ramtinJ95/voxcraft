from __future__ import annotations

import shutil
import subprocess
from pathlib import Path


def ffmpeg_normalize_command(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-i",
        str(input_path),
        "-ar",
        str(sample_rate),
        "-ac",
        str(channels),
        str(output_path),
    ]


def normalize_audio(
    input_path: Path,
    output_path: Path,
    sample_rate: int = 16000,
    channels: int = 1,
    dry_run: bool = False,
) -> list[str]:
    command = ffmpeg_normalize_command(
        input_path=input_path,
        output_path=output_path,
        sample_rate=sample_rate,
        channels=channels,
    )

    if dry_run:
        return command

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is not available on PATH.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    completed = subprocess.run(
        command,
        capture_output=True,
        check=False,
        text=True,
    )

    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "ffmpeg exited with a non-zero status."
        raise RuntimeError(stderr)

    return command
