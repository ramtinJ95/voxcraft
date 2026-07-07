from __future__ import annotations

from pathlib import Path

from typer.testing import CliRunner

from voxcraft.cli import app, _save_text_output


runner = CliRunner()


class FakeServerClient:
    def __init__(self, *, base_url: str, token: str, timeout: float = 30.0) -> None:
        self.base_url = base_url
        self.token = token
        self.timeout = timeout

    @staticmethod
    def get_final_markdown(job_id: str) -> str:
        assert job_id == "job-123"
        return "# Final Summary\n"


def test_save_text_output_writes_exact_path(tmp_path: Path) -> None:
    output_path = tmp_path / "summary.md"

    saved_path = _save_text_output("# Final Summary\n", output_path, default_name="final.md")

    assert saved_path == output_path
    assert output_path.read_text(encoding="utf-8") == "# Final Summary\n"


def test_save_text_output_writes_default_name_for_existing_directory(tmp_path: Path) -> None:
    saved_path = _save_text_output("# Final Summary\n", tmp_path, default_name="final.md")

    assert saved_path == tmp_path / "final.md"
    assert saved_path.read_text(encoding="utf-8") == "# Final Summary\n"


def test_fetch_final_can_save_to_local_file(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("voxcraft.cli.VoxcraftServerClient", FakeServerClient)
    output_path = tmp_path / "downloaded-final.md"

    result = runner.invoke(app, ["fetch-final", "job-123", "--token", "secret", "--output", str(output_path)])

    assert result.exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "# Final Summary\n"
    assert "Saved final.md" in result.output
    assert "# Final Summary" not in result.output
