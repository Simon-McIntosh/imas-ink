"""Annular vessel skins in the 2D geometry figure.

A ``wall`` IDS whose vessel unit carries filled ``annular.outline_inner`` and
``annular.outline_outer`` must surface as two closed shells from
``extract_vessel_shells``, flow through ``extract_geometry`` into
``MachineGeometry.vessel_shells``, and be drawn by both renderers.
"""

from __future__ import annotations

import types

import numpy as np
import pytest


def _ns(**kwargs):
    """Shorthand for ``types.SimpleNamespace``."""
    return types.SimpleNamespace(**kwargs)


def _closed_ellipse(r0: float, a: float, z0: float = 0.0, b: float = 1.0):
    """Return a closed RZ ellipse (first point == last point)."""
    theta = np.linspace(0.0, 2.0 * np.pi, 51)
    return r0 + a * np.cos(theta), z0 + b * np.sin(theta)


# ---------------------------------------------------------------------------
# Fixtures: a wall IDS with one limiter unit and one annular vessel unit
# ---------------------------------------------------------------------------

def _make_limiter_unit():
    r, z = _closed_ellipse(5.0, 1.0)
    return _ns(outline=_ns(r=r, z=z))


def _make_vessel_unit():
    r_in, z_in = _closed_ellipse(4.0, 1.0)
    r_out, z_out = _closed_ellipse(6.0, 1.0)
    return _ns(
        annular=_ns(
            outline_inner=_ns(r=r_in, z=z_in),
            outline_outer=_ns(r=r_out, z=z_out),
            centreline=_ns(r=np.array([]), z=np.array([])),
        )
    )


def _make_wall_ids():
    """One description_2d: one limiter unit and one annular vessel unit."""
    desc = _ns(
        limiter=_ns(unit=[_make_limiter_unit()], type=_ns(index=1)),
        vessel=_ns(unit=[_make_vessel_unit()]),
    )
    return _ns(description_2d=[desc])


def _make_pf_ids():
    return _ns(coil=[])


def _make_two_desc_wall_ids():
    """desc[0] untyped (index 0 sentinel) missing vessel; desc[1] typed with vessel.

    ``_select_description_2d`` picks the typed entry (index 1) even though it is
    not at index 0, so the vessel must come from that entry.
    """
    limiter0 = _ns(unit=[_make_limiter_unit()], type=_ns(index=-999999999))
    desc0 = _ns(limiter=limiter0, vessel=_ns(unit=[]))
    limiter1 = _ns(unit=[_make_limiter_unit()], type=_ns(index=1))
    desc1 = _ns(limiter=limiter1, vessel=_ns(unit=[_make_vessel_unit()]))
    return _ns(description_2d=[desc0, desc1])


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------

class TestExtractVesselShells:
    def test_two_closed_shells_from_annular_outlines(self):
        from imas_ink.extract import extract_vessel_shells

        shells = extract_vessel_shells(_make_wall_ids())

        assert len(shells) == 2, f"expected inner+outer, got {len(shells)}"
        inner, outer = shells
        assert inner.name == "vessel_0_inner"
        assert outer.name == "vessel_0_outer"
        assert inner.is_closed is True
        assert outer.is_closed is True
        # Inner is the smaller-radius skin, outer the larger — order is preserved.
        assert inner.r.max() < outer.r.max()

    def test_vessel_read_from_selected_description_entry(self):
        """The vessel comes from the entry ``_select_description_2d`` chose."""
        from imas_ink.extract import extract_geometry

        geom = extract_geometry(_make_two_desc_wall_ids(), _make_pf_ids())

        assert len(geom.vessel_shells) == 2, (
            "vessel must be read from the selected (index 1) entry, not index 0"
        )


# ---------------------------------------------------------------------------
# geometry_figure_mpl draws the shells
# ---------------------------------------------------------------------------

class TestGeometryFigureDrawsShells:
    def _geom(self):
        from imas_ink.extract import extract_geometry

        return extract_geometry(_make_two_desc_wall_ids(), _make_pf_ids())

    def test_both_shells_drawn_on_axes(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from imas_ink.figures import geometry_figure_mpl

        geom = self._geom()
        fig, ax = geometry_figure_mpl(geom)
        try:
            # one limiter unit line + two vessel-shell lines
            assert len(ax.lines) == 3, (
                f"expected 3 wall lines (1 limiter + 2 shells), got {len(ax.lines)}"
            )
            max_r = max(float(np.max(line.get_xdata())) for line in ax.lines)
            assert max_r > 6.9, f"outer shell (r≈7) not drawn; max r = {max_r}"
        finally:
            plt.close(fig)

    def test_vessel_skin_inside_axes_limits(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from imas_ink.figures import geometry_figure_mpl

        geom = self._geom()
        # The vessel outer skin reaches r≈7, beyond the limiter (r≈6) and the
        # (empty) coil extent; the viewport must still contain it.
        assert max(float(s.r.max()) for s in geom.vessel_shells) > 6.9
        fig, ax = geometry_figure_mpl(geom)
        try:
            xlim = ax.get_xlim()
            ylim = ax.get_ylim()
            assert xlim[1] >= 6.9, f"viewport rmax {xlim[1]} excludes vessel skin"
            assert ylim[0] <= -0.99 and ylim[1] >= 0.99
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Altair renderer emits a mark per limiter unit and per shell
# ---------------------------------------------------------------------------

class TestWallOutlineAlt:
    def test_alt_marks_every_unit_and_shell(self):
        alt = pytest.importorskip("altair")  # noqa: F841
        from imas_ink.alt import render_alt
        from imas_ink.components import WallOutline

        r_lim, z_lim = _closed_ellipse(5.0, 1.0)
        r_in, z_in = _closed_ellipse(4.0, 1.0)
        r_out, z_out = _closed_ellipse(6.0, 1.0)
        from imas_ink._types import VesselShell

        wall = WallOutline(
            wall_r=r_lim,
            wall_z=z_lim,
            wall_units=[(r_lim, z_lim)],
            vessel_shells=[
                VesselShell(r=r_in, z=z_in, name="inner", is_closed=True),
                VesselShell(r=r_out, z=z_out, name="outer", is_closed=True),
            ],
        )

        chart = render_alt(wall)
        seg_ids = set(chart.data["seg_id"].tolist())
        assert len(seg_ids) == 3, f"expected 3 segments, got {seg_ids}"


# ---------------------------------------------------------------------------
# The 2D import graph must not load the 3D subpackage
# ---------------------------------------------------------------------------

class TestImportGraph:
    def test_2d_modules_do_not_load_three_d(self):
        import os
        import subprocess
        import sys
        from pathlib import Path

        # The repository root holds the source under test; put it on the
        # child's path so the subprocess imports this tree, not an install.
        root = Path(__file__).resolve().parents[2]
        code = (
            "import sys\n"
            "import imas_ink.extract, imas_ink.figures\n"
            "print('THREE_D' if 'imas_ink.three_d' in sys.modules else 'CLEAN')\n"
        )
        env = dict(os.environ, PYTHONPATH=str(root))
        out = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            cwd=str(root),
            env=env,
            check=True,
        )
        assert "CLEAN" in out.stdout, (
            f"importing imas_ink.extract/figures loaded imas_ink.three_d: {out.stdout!r}"
        )
