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


def _make_single_limiter_wall_ids():
    """One description_2d with a single limiter unit and no vessel."""
    desc = _ns(
        limiter=_ns(unit=[_make_limiter_unit()], type=_ns(index=1)),
        vessel=_ns(unit=[]),
    )
    return _ns(description_2d=[desc])


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


def _make_one_skin_vessel_unit(*, skin: str):
    """An annular vessel unit carrying only one of its two outlines.

    The wall IDS permits a unit with just one filled annular outline; the
    unfilled outline carries the EMPTY sentinel (a zero-length array).
    """
    empty = np.array([])
    r_in, z_in = _closed_ellipse(4.0, 1.0)
    r_out, z_out = _closed_ellipse(6.0, 1.0)
    if skin == "outer":
        inner, outer = _ns(r=empty, z=empty), _ns(r=r_out, z=z_out)
    else:
        inner, outer = _ns(r=r_in, z=z_in), _ns(r=empty, z=empty)
    return _ns(
        annular=_ns(
            outline_inner=inner,
            outline_outer=outer,
            centreline=_ns(r=empty, z=empty),
        )
    )


def _make_one_skin_wall_ids(skin: str):
    """One description_2d with a single annular vessel unit carrying one skin."""
    desc = _ns(
        limiter=_ns(unit=[_make_limiter_unit()], type=_ns(index=1)),
        vessel=_ns(unit=[_make_one_skin_vessel_unit(skin=skin)]),
    )
    return _ns(description_2d=[desc])


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


class TestAnnularOneSkin:
    def test_outer_only_returns_that_one_closed_shell(self):
        from imas_ink.extract import extract_vessel_shells

        shells = extract_vessel_shells(_make_one_skin_wall_ids("outer"))

        assert len(shells) == 1, f"expected the one filled outer skin, got {len(shells)}"
        shell = shells[0]
        assert shell.name == "vessel_0_outer"
        assert shell.is_closed is True
        assert shell.r.max() > 6.9, f"outer skin (r≈7) not returned: {shell.r.max()}"

    def test_inner_only_returns_that_one_closed_shell(self):
        from imas_ink.extract import extract_vessel_shells

        shells = extract_vessel_shells(_make_one_skin_wall_ids("inner"))

        assert len(shells) == 1, f"expected the one filled inner skin, got {len(shells)}"
        shell = shells[0]
        assert shell.name == "vessel_0_inner"
        assert shell.is_closed is True
        assert shell.r.max() < 5.1, f"inner skin (r≈5) not returned: {shell.r.max()}"


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
# Composing two machine states on caller-owned axes
# ---------------------------------------------------------------------------

class TestGeometryFigureComposesIntoAxes:
    def _geom_with_vessel(self):
        from imas_ink.extract import extract_geometry

        return extract_geometry(_make_two_desc_wall_ids(), _make_pf_ids())

    def _geom_limiter_only(self):
        from imas_ink.extract import extract_geometry

        return extract_geometry(_make_single_limiter_wall_ids(), _make_pf_ids())

    def test_two_geometries_share_one_figure_via_ax(self):
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from imas_ink.figures import geometry_figure_mpl

        fig, (ax0, ax1) = plt.subplots(1, 2)
        n_before = len(plt.get_fignums())
        try:
            fig0, ret0 = geometry_figure_mpl(self._geom_limiter_only(), ax=ax0)
            fig1, ret1 = geometry_figure_mpl(self._geom_with_vessel(), ax=ax1)

            # No figure was created: the figure count is unchanged and each
            # call returned the caller's own figure and axes.
            assert len(plt.get_fignums()) == n_before, "geometry_figure_mpl created a figure"
            assert fig0 is fig and fig1 is fig
            assert ret0 is ax0 and ret1 is ax1

            # Each axes holds its own geometry's wall lines: one limiter unit
            # for ax0, one limiter unit plus two vessel shells for ax1.
            assert len(ax0.lines) == 1, f"ax0: expected 1 wall line, got {len(ax0.lines)}"
            assert len(ax1.lines) == 3, f"ax1: expected 3 wall lines, got {len(ax1.lines)}"
        finally:
            plt.close(fig)


# ---------------------------------------------------------------------------
# Wall / vessel line style follows InkStyle.wall_linestyle
# ---------------------------------------------------------------------------

class TestWallLinestyle:
    def _geom(self):
        from imas_ink.extract import extract_geometry

        return extract_geometry(_make_two_desc_wall_ids(), _make_pf_ids())

    def test_dashed_wall_and_vessel_lines(self):
        import matplotlib
        matplotlib.use("Agg")
        from dataclasses import replace

        import matplotlib.pyplot as plt

        from imas_ink.figures import geometry_figure_mpl
        from imas_ink.style import DEFAULT_STYLE

        style = replace(DEFAULT_STYLE, wall_linestyle="dashed")
        fig, ax = geometry_figure_mpl(self._geom(), style=style)
        try:
            assert len(ax.lines) == 3
            for line in ax.lines:
                assert line.get_linestyle() in ("--", "dashed"), (
                    f"wall line not dashed: {line.get_linestyle()!r}"
                )
        finally:
            plt.close(fig)

    def test_default_wall_lines_are_solid(self):
        # Control: the default style draws solid, so the dashed assertion in
        # the sibling test measures the style field, not a fixed pattern.
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from imas_ink.figures import geometry_figure_mpl

        fig, ax = geometry_figure_mpl(self._geom())
        try:
            assert len(ax.lines) == 3
            for line in ax.lines:
                assert line.get_linestyle() in ("-", "solid"), (
                    f"wall line not solid: {line.get_linestyle()!r}"
                )
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

    def test_alt_wall_dash_follows_style(self):
        alt = pytest.importorskip("altair")  # noqa: F841
        from dataclasses import replace

        from imas_ink.alt import STROKE_DASH, render_alt
        from imas_ink.components import WallOutline
        from imas_ink.style import DEFAULT_STYLE

        r_lim, z_lim = _closed_ellipse(5.0, 1.0)
        style = replace(DEFAULT_STYLE, wall_linestyle="dashed")
        wall = WallOutline(wall_r=r_lim, wall_z=z_lim, wall_units=[(r_lim, z_lim)], style=style)
        spec = render_alt(wall).to_dict()
        assert spec["mark"]["strokeDash"] == STROKE_DASH["dashed"]


# ---------------------------------------------------------------------------
# A panel may hold several TimeSeries drawn on shared axes
# ---------------------------------------------------------------------------


class TestTracePanels:
    def _ts(self, values, *, label="", ylabel="", units="", style=None):
        from imas_ink.components import TimeSeries
        from imas_ink.style import DEFAULT_STYLE

        t = np.linspace(0.0, 1.0, 11)
        return TimeSeries(
            t,
            np.asarray(values, dtype=float),
            label=label,
            ylabel=ylabel,
            units=units,
            style=style if style is not None else DEFAULT_STYLE)

    def test_panel_list_draws_second_in_first_colour_and_style(self):
        import matplotlib
        matplotlib.use("Agg")
        from dataclasses import replace

        import matplotlib.pyplot as plt

        from imas_ink.figures import time_trace_figure_mpl
        from imas_ink.style import DEFAULT_STYLE

        first_style = replace(DEFAULT_STYLE, trace_linestyle="solid")
        second_style = replace(DEFAULT_STYLE, trace_linestyle="dashed")
        first = self._ts([1.0] * 11, ylabel="Ip", units="MA", style=first_style)
        second = self._ts([2.0] * 11, label="raw", ylabel="other", style=second_style)

        fig, axes = time_trace_figure_mpl([[first, second]])
        try:
            ax = axes[0]
            assert len(ax.lines) == 2, f"expected 2 lines on the panel, got {len(ax.lines)}"
            first_line, second_line = ax.lines
            assert second_line.get_linestyle() in ("--", "dashed"), (
                f"second series not dashed: {second_line.get_linestyle()!r}"
            )
            assert second_line.get_color() == first_line.get_color(), (
                f"second line colour {second_line.get_color()!r} != "
                f"first {first_line.get_color()!r}"
            )
            assert ax.get_ylabel() == "Ip [MA]", (
                f"panel y-label is the second series', not the first: {ax.get_ylabel()!r}"
            )
            assert ax.get_legend() is None, "later series added a legend"
        finally:
            plt.close(fig)

    def test_single_series_panel_draws_one_line(self):
        import matplotlib
        matplotlib.use("Agg")

        import matplotlib.pyplot as plt

        from imas_ink.figures import time_trace_figure_mpl

        fig, axes = time_trace_figure_mpl([self._ts([1.0] * 11, ylabel="Ip")])
        try:
            assert len(axes[0].lines) == 1, (
                f"single-series panel drew {len(axes[0].lines)} lines"
            )
        finally:
            plt.close(fig)


class TestStrokeDashShorthand:
    def test_wall_and_trace_dash_map_shorthand(self):
        alt = pytest.importorskip("altair")  # noqa: F841
        from dataclasses import replace

        from imas_ink.alt import STROKE_DASH, render_alt
        from imas_ink.components import TimeSeries, WallOutline
        from imas_ink.style import DEFAULT_STYLE

        # Matplotlib shorthand "for each" resolves to the same pattern the
        # named form does, in both backends.
        assert STROKE_DASH["--"] == STROKE_DASH["dashed"]
        assert STROKE_DASH["-"] == STROKE_DASH["solid"]
        assert STROKE_DASH[":"] == STROKE_DASH["dotted"]
        assert STROKE_DASH["-."] == STROKE_DASH["dashdot"]

        style = replace(DEFAULT_STYLE, wall_linestyle="--", trace_linestyle="--")
        r_lim, z_lim = _closed_ellipse(5.0, 1.0)
        wall = WallOutline(wall_r=r_lim, wall_z=z_lim, wall_units=[(r_lim, z_lim)], style=style)
        wall_spec = render_alt(wall).to_dict()
        assert wall_spec["mark"]["strokeDash"] == STROKE_DASH["dashed"]

        t = np.linspace(0.0, 1.0, 11)
        ts = TimeSeries(t, t, style=style)
        trace_spec = render_alt(ts).to_dict()
        line_mark = trace_spec["layer"][0]["mark"]
        assert line_mark["strokeDash"] == STROKE_DASH["dashed"], (
            f"trace mark strokeDash {line_mark.get('strokeDash')!r}"
        )


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
