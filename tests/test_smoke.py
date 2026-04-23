"""Smoke tests — the bootstrap build must produce an importable package."""

from __future__ import annotations

import pytest


def test_import_version() -> None:
    import imas_ink

    assert isinstance(imas_ink.__version__, str)
    assert imas_ink.__version__


def test_cli_version_flags(capsys: pytest.CaptureFixture[str]) -> None:
    """`imas-ink version`, `--version`, and `-V` all print the version."""
    import imas_ink
    from imas_ink.cli import main

    for flag in ("version", "--version", "-V"):
        rc = main([flag])
        captured = capsys.readouterr()
        assert rc == 0, f"{flag!r} returned {rc}"
        assert captured.out.strip() == imas_ink.__version__
