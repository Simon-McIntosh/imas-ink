"""Tests for the ``imas-ink demo iter-coilset`` CLI command."""

from __future__ import annotations

import subprocess
import sys


class TestDemoUsage:
    def test_demo_no_args_exits_nonzero(self):
        """``imas-ink demo`` without a sub-command prints usage and exits 1."""
        result = subprocess.run(
            [sys.executable, "-m", "imas_ink.cli", "demo"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "iter-coilset" in result.stderr

    def test_demo_iter_coilset_no_uri_exits_nonzero(self):
        """``imas-ink demo iter-coilset`` without --uri exits 1."""
        result = subprocess.run(
            [sys.executable, "-m", "imas_ink.cli", "demo", "iter-coilset"],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0
        assert "--uri" in result.stderr


class TestLazyImport:
    def test_version_does_not_import_vtk(self):
        """``imas-ink version`` must not import pyvista or vtk."""
        result = subprocess.run(
            [
                sys.executable,
                "-c",
                "import sys; "
                "from imas_ink.cli import main; "
                "main(['version']); "
                "print('pyvista' in sys.modules, 'vtk' in sys.modules)",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        # The last line of stdout should be "False False"
        lines = result.stdout.strip().splitlines()
        assert lines[-1] == "False False"
