from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from voxcraft.utils import write_bytes, write_json, write_text


def test_write_text_atomically_replaces_existing_content(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"
    destination.write_text("old content\n", encoding="utf-8")

    write_text(destination, "new content\n")

    assert destination.read_text(encoding="utf-8") == "new content\n"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_write_bytes_preserves_existing_content_when_replace_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    destination = tmp_path / "artifact.bin"
    destination.write_bytes(b"old content")

    def fail_replace(source: Path, target: Path) -> None:
        assert Path(source).parent == destination.parent
        assert Path(target) == destination
        raise OSError("replace failed")

    monkeypatch.setattr("voxcraft.utils.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        write_bytes(destination, b"new content")

    assert destination.read_bytes() == b"old content"
    assert list(tmp_path.glob(f".{destination.name}.*.tmp")) == []


def test_write_json_publishes_complete_json(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.json"

    write_json(destination, {"title": "Résumé", "items": [1, 2, 3]})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "title": "Résumé",
        "items": [1, 2, 3],
    }
    assert destination.read_text(encoding="utf-8").endswith("\n")


@pytest.mark.skipif(os.name == "nt", reason="POSIX file modes are not stable on Windows")
def test_write_text_preserves_existing_file_mode(tmp_path: Path) -> None:
    destination = tmp_path / "artifact.txt"
    destination.write_text("old content\n", encoding="utf-8")
    destination.chmod(0o640)

    write_text(destination, "new content\n")

    assert stat.S_IMODE(destination.stat().st_mode) == 0o640
