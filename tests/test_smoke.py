"""Smoke test — the bootstrap build must produce an importable package."""

from __future__ import annotations


def test_import_version() -> None:
    import imas_ink

    assert isinstance(imas_ink.__version__, str)
    assert imas_ink.__version__
