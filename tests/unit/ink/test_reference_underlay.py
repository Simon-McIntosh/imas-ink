"""Unit tests for the reference-equilibrium underlay feature.

Tests cover:
1. ReferenceContours and ReferenceLcfs component construction.
2. mpl renderers add artists to axes without error.
3. Level matching: reference contours at primary absolute psi levels.
4. Normalised-level fallback when psi frames are incompatible.
5. equilibrium_figure_mpl with reference_slice produces additional artists.
6. Empty / sentinel-axis reference slice is handled gracefully.
7. X-point extraction from boundary.x_point / constraints / contour_tree.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from imas_ink._cocos import make_levels
from imas_ink.components import ReferenceContours, ReferenceLcfs, ReferenceXPoints
from imas_ink.contours import ContourExtractor
from imas_ink.extract import _extract_x_points
from imas_ink.style import DEFAULT_STYLE, InkStyle


def _ns(**kw):
    return types.SimpleNamespace(**kw)


# ---------------------------------------------------------------------------
# 7. X-point extraction (DD-version-robust, IDS-verbatim)
# ---------------------------------------------------------------------------


class TestExtractXPoints:
    """_extract_x_points reads from 3 IDS sources in priority order."""

    def test_boundary_x_point_ddv3(self):
        """boundary.x_point (DDv3 / aliased) is the first-priority source."""
        ts = _ns(
            boundary=_ns(x_point=[_ns(r=5.1, z=-3.4)]),
            constraints=_ns(x_point=[]),
            contour_tree=_ns(node=[]),
        )
        pts = _extract_x_points(ts)
        assert pts == [(pytest.approx(5.1), pytest.approx(-3.4))]

    def test_constraints_x_point_ddv4(self):
        """constraints.x_point.position_reconstructed used when boundary empty."""
        ts = _ns(
            boundary=_ns(x_point=[]),
            constraints=_ns(x_point=[_ns(position_reconstructed=_ns(r=4.8, z=-3.2))]),
            contour_tree=_ns(node=[]),
        )
        pts = _extract_x_points(ts)
        assert pts == [(pytest.approx(4.8), pytest.approx(-3.2))]

    def test_contour_tree_xpoint_ddv4(self):
        """contour_tree node with critical_type==1 is read (forward models)."""
        ts = _ns(
            boundary=_ns(x_point=[]),
            constraints=_ns(x_point=[]),
            contour_tree=_ns(
                node=[
                    _ns(critical_type=0, r=6.38, z=0.47),  # O-point — ignored
                    _ns(critical_type=1, r=5.12, z=-3.41),  # primary X-point
                    _ns(critical_type=1, r=4.73, z=4.50),  # secondary X-point
                ]
            ),
        )
        pts = _extract_x_points(ts)
        assert (pytest.approx(5.12), pytest.approx(-3.41)) in pts
        assert (pytest.approx(4.73), pytest.approx(4.50)) in pts
        # O-point must NOT be included
        assert (pytest.approx(6.38), pytest.approx(0.47)) not in pts
        assert len(pts) == 2

    def test_boundary_takes_priority_over_contour_tree(self):
        """When boundary.x_point present, contour_tree is not consulted."""
        ts = _ns(
            boundary=_ns(x_point=[_ns(r=5.0, z=-3.0)]),
            constraints=_ns(x_point=[]),
            contour_tree=_ns(node=[_ns(critical_type=1, r=9.9, z=9.9)]),
        )
        pts = _extract_x_points(ts)
        assert pts == [(pytest.approx(5.0), pytest.approx(-3.0))]

    def test_honest_absence_limiter(self):
        """No X-points anywhere → empty list (limiter slice)."""
        ts = _ns(
            boundary=_ns(x_point=[]),
            constraints=_ns(x_point=[]),
            contour_tree=_ns(node=[_ns(critical_type=0, r=6.0, z=0.0)]),
        )
        assert _extract_x_points(ts) == []

    def test_sentinel_values_skipped(self):
        """Sentinel / NaN X-point coordinates are filtered out."""
        ts = _ns(
            boundary=_ns(
                x_point=[
                    _ns(r=-9.0e40, z=-9.0e40),  # sentinel
                    _ns(r=5.1, z=-3.4),  # valid
                ]
            ),
            constraints=_ns(x_point=[]),
            contour_tree=_ns(node=[]),
        )
        pts = _extract_x_points(ts)
        assert pts == [(pytest.approx(5.1), pytest.approx(-3.4))]

    def test_missing_attributes_no_crash(self):
        """A time-slice lacking all three structures returns []."""
        ts = _ns()  # no boundary, constraints, or contour_tree
        assert _extract_x_points(ts) == []


# ---------------------------------------------------------------------------
# Synthetic equilibrium helpers
# ---------------------------------------------------------------------------


def _synthetic_eq_slice(
    psi_axis: float = -50.0,
    psi_bnd: float = 2.0,
    n_r: int = 32,
    n_z: int = 48,
    r_min: float = 4.0,
    r_max: float = 8.0,
    z_min: float = -4.0,
    z_max: float = 4.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float, float, float]:
    """Return (R_2d, Z_2d, psi_2d, psi_axis, psi_bnd, r_axis, z_axis).

    The psi field is a simple elliptic paraboloid so the contours are
    closed ellipses around (r_axis, z_axis = centroid).  ``psi_axis`` is
    the value at the centroid; ``psi_bnd`` is a level outside the axis.
    """
    r_ax = (r_min + r_max) / 2.0
    z_ax = (z_min + z_max) / 2.0
    r_1d = np.linspace(r_min, r_max, n_r)
    z_1d = np.linspace(z_min, z_max, n_z)
    R, Z = np.meshgrid(r_1d, z_1d, indexing="ij")
    # Radially squashed paraboloid: axis at centre
    dr = (r_max - r_min) / 2.0
    dz = (z_max - z_min) / 2.0
    dist_sq = ((R - r_ax) / dr) ** 2 + ((Z - z_ax) / dz) ** 2
    # psi_axis most-negative, increases outward (COCOS-17 compatible)
    psi_2d = psi_axis + (psi_bnd - psi_axis) * dist_sq
    return R, Z, psi_2d, psi_axis, psi_bnd, r_ax, z_ax


def _make_eq_slice_ns(
    psi_axis: float = -50.0,
    psi_bnd: float = 2.0,
    include_boundary: bool = True,
    boundary_n: int = 60,
) -> types.SimpleNamespace:
    """Return a SimpleNamespace mimicking a minimal EquilibriumSlice."""
    R, Z, psi_2d, pa, pb, ra, za = _synthetic_eq_slice(psi_axis=psi_axis, psi_bnd=psi_bnd)
    # Build a synthetic closed boundary around the magnetic axis
    if include_boundary:
        theta = np.linspace(0, 2 * np.pi, boundary_n + 1)
        br = ra + 1.5 * np.cos(theta)
        bz = za + 1.8 * np.sin(theta)
    else:
        br = np.array([])
        bz = np.array([])
    return types.SimpleNamespace(
        psi_2d=psi_2d,
        R_2d=R,
        Z_2d=Z,
        psi_axis=pa,
        psi_boundary=pb,
        r_axis=ra,
        z_axis=za,
        boundary_r=br if include_boundary else None,
        boundary_z=bz if include_boundary else None,
        x_points=[],
        time=0.5,
        converged=True,
        ip=1e6,
        beta_pol=None,
        li_3=None,
        q95=None,
    )


# ---------------------------------------------------------------------------
# 1. Component construction tests
# ---------------------------------------------------------------------------


class TestReferenceContoursComponent:
    def test_construct_empty(self):
        rc = ReferenceContours(segments=[], ref_name="DINA")
        assert rc.ref_name == "DINA"
        assert rc.psi_matched is True
        assert rc.style is DEFAULT_STYLE

    def test_construct_with_segments(self):
        segs = [[np.array([[4.0, -1.0], [5.0, 0.0], [4.0, 1.0]])]]
        rc = ReferenceContours(segments=segs, ref_name="NICE")
        assert len(rc.segments) == 1

    def test_custom_style(self):
        custom = InkStyle(ref_color="#ff0000")
        rc = ReferenceContours(segments=[], style=custom)
        assert rc.style.ref_color == "#ff0000"

    def test_psi_matched_false(self):
        rc = ReferenceContours(segments=[], psi_matched=False)
        assert rc.psi_matched is False


class TestReferenceLcfsComponent:
    def test_construct(self):
        r = np.array([4.0, 5.0, 6.0, 5.0, 4.0])
        z = np.array([0.0, 1.0, 0.0, -1.0, 0.0])
        rl = ReferenceLcfs(r=r, z=z, ref_name="EPM")
        assert rl.ref_name == "EPM"
        assert rl.style is DEFAULT_STYLE

    def test_empty_arrays(self):
        rl = ReferenceLcfs(r=np.array([]), z=np.array([]), ref_name="X")
        assert len(rl.r) == 0


# ---------------------------------------------------------------------------
# 2. Renderer tests (matplotlib)
# ---------------------------------------------------------------------------


class TestReferenceContoursRenderer:
    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close(self.fig)

    def test_render_empty_no_artists(self):
        """Empty ReferenceContours renders without error, no line artists added."""
        from imas_ink.mpl import render_mpl

        rc = ReferenceContours(segments=[], ref_name="DINA")
        n_before = len(self.ax.collections)
        render_mpl(self.ax, rc)
        assert len(self.ax.collections) == n_before  # no LineCollection added

    def test_render_adds_collection(self):
        """Non-empty ReferenceContours adds a LineCollection per level."""
        from imas_ink.mpl import render_mpl

        segs = [[np.array([[4.0, -1.0], [5.0, 0.0], [4.0, 1.0]])]]
        rc = ReferenceContours(segments=segs, ref_name="DINA")
        n_before = len(self.ax.collections)
        render_mpl(self.ax, rc)
        assert len(self.ax.collections) > n_before

    def test_render_adds_legend_proxy(self):
        """Non-empty ReferenceContours adds a legend proxy line."""
        from imas_ink.mpl import render_mpl

        segs = [[np.array([[4.0, -1.0], [5.0, 0.0], [4.0, 1.0]])]]
        rc = ReferenceContours(segments=segs, ref_name="DINA")
        render_mpl(self.ax, rc)
        labels = [h.get_label() for h in self.ax.get_lines()]
        assert any("DINA" in lbl for lbl in labels)

    def test_render_psi_norm_fallback_note(self):
        """psi_matched=False adds '[ψ_norm]' to the legend label."""
        from imas_ink.mpl import render_mpl

        segs = [[np.array([[4.0, -1.0], [5.0, 0.0], [4.0, 1.0]])]]
        rc = ReferenceContours(segments=segs, ref_name="EPM", psi_matched=False)
        render_mpl(self.ax, rc)
        labels = [h.get_label() for h in self.ax.get_lines()]
        assert any("ψ_norm" in lbl for lbl in labels)


class TestReferenceLcfsRenderer:
    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close(self.fig)

    def test_render_empty_no_lines(self):
        """Empty ReferenceLcfs renders without error, no lines added."""
        from imas_ink.mpl import render_mpl

        rl = ReferenceLcfs(r=np.array([]), z=np.array([]), ref_name="X")
        n_before = len(self.ax.lines)
        render_mpl(self.ax, rl)
        assert len(self.ax.lines) == n_before

    def test_render_closed_adds_line(self):
        """Closed ReferenceLcfs adds a line and a legend label."""
        from imas_ink.mpl import render_mpl

        theta = np.linspace(0, 2 * np.pi, 50)
        r = 5.5 + 1.0 * np.cos(theta)
        z = 0.0 + 1.2 * np.sin(theta)
        rl = ReferenceLcfs(r=r, z=z, ref_name="NICE")
        n_before = len(self.ax.lines)
        render_mpl(self.ax, rl)
        assert len(self.ax.lines) > n_before
        labels = [h.get_label() for h in self.ax.lines]
        assert any("NICE" in lbl for lbl in labels)

    def test_render_style_applied(self):
        """Custom ref_lcfs_linewidth is applied to the rendered line."""
        from dataclasses import replace

        from imas_ink.mpl import render_mpl

        custom = replace(DEFAULT_STYLE, ref_lcfs_linewidth=3.5)
        theta = np.linspace(0, 2 * np.pi, 40)
        r = 5.0 + 0.8 * np.cos(theta)
        z = 0.0 + 0.8 * np.sin(theta)
        rl = ReferenceLcfs(r=r, z=z, ref_name="X", style=custom)
        render_mpl(self.ax, rl)
        line = [ln for ln in self.ax.lines if "X" in ln.get_label()]
        assert line
        assert pytest.approx(line[0].get_linewidth(), abs=0.1) == 3.5


class TestReferenceXPointsComponentAndRenderer:
    """ReferenceXPoints component + renderer."""

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        self.fig, self.ax = plt.subplots()

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close(self.fig)

    def test_construct(self):
        rxp = ReferenceXPoints([(5.12, -3.41)], ref_name="DINA")
        assert rxp.ref_name == "DINA"
        assert rxp.points == [(5.12, -3.41)]
        assert rxp.style is DEFAULT_STYLE

    def test_render_empty_no_line(self):
        from imas_ink.mpl import render_mpl

        rxp = ReferenceXPoints([], ref_name="DINA")
        n_before = len(self.ax.lines)
        render_mpl(self.ax, rxp)
        assert len(self.ax.lines) == n_before

    def test_render_adds_marker_and_label(self):
        from imas_ink.mpl import render_mpl

        rxp = ReferenceXPoints([(5.12, -3.41), (4.73, 4.50)], ref_name="DINA")
        render_mpl(self.ax, rxp)
        labels = [h.get_label() for h in self.ax.lines]
        assert any("DINA" in lbl and "X-pt" in lbl for lbl in labels)
        # Marker is the configured ref_xpt_marker, not the primary 'x'
        marker_lines = [ln for ln in self.ax.lines if "X-pt" in ln.get_label()]
        assert marker_lines
        assert marker_lines[0].get_marker() == DEFAULT_STYLE.ref_xpt_marker

    def test_marker_distinct_from_primary(self):
        """Reference X marker shape differs from the primary X marker ('x')."""
        assert DEFAULT_STYLE.ref_xpt_marker != DEFAULT_STYLE.xpt_marker


# ---------------------------------------------------------------------------
# 3. Level-matching tests
# ---------------------------------------------------------------------------


class TestLevelMatching:
    """Test that reference contours are drawn at primary absolute psi levels."""

    def test_levels_computed_from_primary(self):
        """Levels computed from primary psi_axis/psi_bnd are applied to reference."""
        # Primary slice
        _R_p, _Z_p, _psi_p, psi_ax_p, psi_bnd_p, _r_ax, _z_ax = _synthetic_eq_slice(
            psi_axis=-50.0, psi_bnd=2.0
        )
        # Reference slice: slightly different field, same psi convention
        R_r, Z_r, psi_r, _psi_ax_r, _psi_bnd_r, _, _ = _synthetic_eq_slice(
            psi_axis=-55.0, psi_bnd=2.5
        )
        n_levels = 4
        primary_levels = make_levels(psi_ax_p, psi_bnd_p, n=n_levels)
        cx_ref = ContourExtractor(R_r, Z_r, psi_r)
        # Extract contours at primary levels (same as the underlay does)
        segs_by_level = [cx_ref.lines_at(lev) for lev in primary_levels]
        # Each level can produce 0 or more segments — verify no crash
        assert len(segs_by_level) == n_levels

    def test_level_values_identical(self):
        """make_levels with same args always returns identical arrays."""
        levels_a = make_levels(-50.0, 2.0, n=6)
        levels_b = make_levels(-50.0, 2.0, n=6)
        np.testing.assert_array_equal(levels_a, levels_b)

    def test_primary_levels_in_reference_psi_range(self):
        """Primary levels lie within reference psi range when fields are similar."""
        _, _, psi_r, _psi_ax_r, _, _, _ = _synthetic_eq_slice(psi_axis=-55.0, psi_bnd=2.5)
        primary_levels = make_levels(-50.0, 2.0, n=4)
        # Levels must be within the reference field's psi range for contours to exist
        in_range = (primary_levels >= psi_r.min()) & (primary_levels <= psi_r.max())
        assert in_range.any(), "At least some levels should be in the reference field range"


# ---------------------------------------------------------------------------
# 4. Normalised fallback tests
# ---------------------------------------------------------------------------


def _ref_contours_label(ax) -> str | None:
    """Return the reference-contours legend label from an Axes, or None.

    The reference-contours proxy line carries a label beginning with
    'reference ('.  The normalised-fallback path appends ' [ψ_norm]'.
    """
    for h in ax.get_lines():
        lbl = h.get_label()
        if lbl.startswith("reference ("):
            return lbl
    return None


class TestNormalisedFallback:
    """ψ_norm fallback vs absolute matching, exercised through the real
    ``equilibrium_figure_mpl`` (behavioural, not a re-implemented condition)."""

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def _geom(self):
        from imas_ink._types import MachineGeometry

        r = np.array([3.5, 9.5, 9.5, 3.5, 3.5])
        z = np.array([-5.0, -5.0, 5.0, 5.0, -5.0])
        return MachineGeometry(
            wall_r=r,
            wall_z=z,
            coil_rects=[],
            wall_clip_vertices=np.column_stack([r, z]),
            wall_units=[(r, z)],
            probe_r=np.array([]),
            probe_z=np.array([]),
            probe_angle=np.array([]),
            flux_loop_r=np.array([]),
            flux_loop_z=np.array([]),
        )

    def _slice(self, psi_axis, psi_bnd, sentinel_axis=False):
        from imas_ink._types import EquilibriumSlice

        R, Z, psi_2d, pa, pb, ra, za = _synthetic_eq_slice(psi_axis=psi_axis, psi_bnd=psi_bnd)
        theta = np.linspace(0, 2 * np.pi, 40)
        br = ra + 1.5 * np.cos(theta)
        bz = za + 1.8 * np.sin(theta)
        return EquilibriumSlice(
            psi_2d=psi_2d,
            r_grid=R[:, 0],
            z_grid=Z[0, :],
            psi_axis=(float("nan") if sentinel_axis else pa),
            psi_boundary=pb,
            r_axis=ra,
            z_axis=za,
            ip=1e6,
            time=0.5,
            converged=True,
            x_points=[],
            boundary_r=br,
            boundary_z=bz,
        )

    def test_compatible_sign_uses_absolute(self):
        """Same-sign psi_bnd + valid axes → absolute matching (no ψ_norm note)."""
        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._slice(-50.0, 2.0)
        ref = self._slice(-55.0, 2.5)
        _fig, ax = equilibrium_figure_mpl(
            sl, self._geom(), reference_slice=ref, reference_name="DINA"
        )
        label = _ref_contours_label(ax)
        assert label is not None
        assert "ψ_norm" not in label, "Compatible frames must use absolute levels"

    def test_sentinel_psi_axis_infers_and_uses_absolute(self):
        """Reference with sentinel (NaN) psi_axis but valid field + same-sign
        psi_bnd: axis is inferred from psi_2d and absolute matching is used.
        This is the DINA case — the intended behaviour."""
        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._slice(-50.0, 2.0)
        ref = self._slice(-55.0, 2.5, sentinel_axis=True)
        _fig, ax = equilibrium_figure_mpl(
            sl, self._geom(), reference_slice=ref, reference_name="DINA"
        )
        label = _ref_contours_label(ax)
        assert label is not None
        assert "ψ_norm" not in label, (
            "Sentinel psi_axis with valid field should infer axis and use absolute matching"
        )

    def test_opposite_sign_triggers_normalised(self):
        """Opposite-sign psi_bnd → normalised fallback, ψ_norm note present."""
        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._slice(-50.0, 2.0)
        # Reference with opposite-sign psi field (psi_axis > 0, psi_bnd < 0)
        ref = self._slice(50.0, -2.0)
        _fig, ax = equilibrium_figure_mpl(sl, self._geom(), reference_slice=ref, reference_name="X")
        label = _ref_contours_label(ax)
        assert label is not None
        assert "ψ_norm" in label, "Opposite-sign frames must fall back to ψ_norm"


# ---------------------------------------------------------------------------
# 5. equilibrium_figure_mpl integration tests
# ---------------------------------------------------------------------------


class TestEquilibriumFigureWithReference:
    """Integration tests for equilibrium_figure_mpl with reference_slice."""

    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def _make_geom(self):
        """Minimal MachineGeometry for test figures."""
        from imas_ink._types import MachineGeometry

        r = np.array([3.5, 9.5, 9.5, 3.5, 3.5])
        z = np.array([-5.0, -5.0, 5.0, 5.0, -5.0])
        return MachineGeometry(
            wall_r=r,
            wall_z=z,
            coil_rects=[],
            wall_clip_vertices=np.column_stack([r, z]),
            wall_units=[(r, z)],
            probe_r=np.array([]),
            probe_z=np.array([]),
            probe_angle=np.array([]),
            flux_loop_r=np.array([]),
            flux_loop_z=np.array([]),
        )

    def _make_slice(self, psi_axis=-50.0, psi_bnd=2.0, x_points=None):
        """Convert synthetic ns to EquilibriumSlice."""
        from imas_ink._types import EquilibriumSlice

        R, Z, psi_2d, pa, pb, ra, za = _synthetic_eq_slice(psi_axis=psi_axis, psi_bnd=psi_bnd)
        theta = np.linspace(0, 2 * np.pi, 40)
        br = ra + 1.5 * np.cos(theta)
        bz = za + 1.8 * np.sin(theta)
        return EquilibriumSlice(
            psi_2d=psi_2d,
            r_grid=R[:, 0],
            z_grid=Z[0, :],
            psi_axis=pa,
            psi_boundary=pb,
            r_axis=ra,
            z_axis=za,
            ip=1e6,
            time=0.5,
            converged=True,
            x_points=list(x_points) if x_points else [],
            boundary_r=br,
            boundary_z=bz,
        )

    def test_no_reference_produces_figure(self):
        """Without reference_slice, figure is produced as before."""
        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_slice()
        geom = self._make_geom()
        fig, ax = equilibrium_figure_mpl(sl, geom)
        assert fig is not None
        assert ax is not None

    def test_with_reference_adds_underlay_artists(self):
        """With reference_slice, more artists are added to the axes."""
        import matplotlib.pyplot as plt

        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_slice()
        ref = self._make_slice(psi_axis=-55.0, psi_bnd=2.5)
        geom = self._make_geom()

        fig_no_ref, ax_no_ref = equilibrium_figure_mpl(sl, geom)
        n_cols_no_ref = len(ax_no_ref.collections)

        fig_with_ref, ax_with_ref = equilibrium_figure_mpl(
            sl, geom, reference_slice=ref, reference_name="DINA"
        )
        n_cols_with_ref = len(ax_with_ref.collections)
        plt.close(fig_no_ref)
        plt.close(fig_with_ref)

        # With reference, at least one additional LineCollection should appear
        assert n_cols_with_ref > n_cols_no_ref

    def test_with_reference_adds_legend_entries(self):
        """With reference_slice, legend entries for reference name appear."""
        import matplotlib.pyplot as plt

        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_slice()
        ref = self._make_slice(psi_axis=-55.0, psi_bnd=2.5)
        geom = self._make_geom()
        fig, ax = equilibrium_figure_mpl(sl, geom, reference_slice=ref, reference_name="DINA")
        labels = [h.get_label() for h in ax.lines + list(ax.collections)]
        plt.close(fig)
        # At least one label must contain "DINA"
        assert any("DINA" in lbl for lbl in labels)

    def test_reference_xpoints_rendered(self):
        """Reference slice with x_points -> reference X-pt marker + legend."""
        import matplotlib.pyplot as plt

        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_slice()  # limited primary (no x_points)
        # Reference is lower-diverted: one X-point at the bottom
        ref = self._make_slice(psi_axis=-55.0, psi_bnd=2.5, x_points=[(5.1, -3.4)])
        geom = self._make_geom()
        fig, ax = equilibrium_figure_mpl(sl, geom, reference_slice=ref, reference_name="DINA")
        labels = [h.get_label() for h in ax.lines]
        plt.close(fig)
        # A "ref X-pt (DINA)" legend entry must be present
        assert any("X-pt" in lbl and "DINA" in lbl for lbl in labels)

    def test_no_reference_xpoints_when_absent(self):
        """Reference slice without x_points -> no reference X-pt marker."""
        import matplotlib.pyplot as plt

        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_slice()
        ref = self._make_slice(psi_axis=-55.0, psi_bnd=2.5, x_points=[])
        geom = self._make_geom()
        fig, ax = equilibrium_figure_mpl(sl, geom, reference_slice=ref, reference_name="DINA")
        labels = [h.get_label() for h in ax.lines]
        plt.close(fig)
        assert not any("X-pt" in lbl and "DINA" in lbl for lbl in labels)

    def test_with_none_reference_unchanged(self):
        """Passing reference_slice=None is backward compatible."""
        import matplotlib.pyplot as plt

        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_slice()
        geom = self._make_geom()
        fig, _ax = equilibrium_figure_mpl(sl, geom, reference_slice=None)
        plt.close(fig)
        assert fig is not None

    def _make_fieldless_ref(self, x_points):
        """Reference slice with NO 2D ψ field (e.g. NICE) — boundary + X only."""
        from imas_ink._types import EquilibriumSlice

        theta = np.linspace(0, 2 * np.pi, 40)
        br = 6.0 + 1.4 * np.cos(theta)
        bz = 0.0 + 1.6 * np.sin(theta)
        return EquilibriumSlice(
            psi_2d=np.empty((0, 0)),  # no field
            r_grid=np.array([]),
            z_grid=np.array([]),
            psi_axis=float("nan"),
            psi_boundary=float("nan"),
            r_axis=6.0,
            z_axis=0.0,
            ip=1e6,
            time=0.5,
            converged=True,
            x_points=list(x_points),
            boundary_r=br,
            boundary_z=bz,
        )

    def test_fieldless_reference_lcfs_and_xpoints_only(self):
        """Reference with no psi_2d: no contour underlay, but LCFS + X-pts shown."""
        import matplotlib.pyplot as plt

        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_slice()
        ref = self._make_fieldless_ref(x_points=[(5.4, -2.8)])
        geom = self._make_geom()
        fig, ax = equilibrium_figure_mpl(sl, geom, reference_slice=ref, reference_name="NICE")
        line_labels = [h.get_label() for h in ax.lines]
        plt.close(fig)
        # LCFS + X-pt legend entries present (boundary mismatch still visible)
        assert any("ref LCFS (NICE)" in lbl for lbl in line_labels)
        assert any("X-pt" in lbl and "NICE" in lbl for lbl in line_labels)
        # No "reference (NICE)" CONTOUR proxy (there is no field to contour)
        assert not any(lbl.startswith("reference (NICE)") for lbl in line_labels)


# ---------------------------------------------------------------------------
# 6. Graceful degradation
# ---------------------------------------------------------------------------


class TestGracefulDegradation:
    """Sentinel and missing-data paths handled without crash."""

    def test_ref_lcfs_empty_no_render(self):
        """ReferenceLcfs with empty arrays renders nothing."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from imas_ink.mpl import render_mpl

        fig, ax = plt.subplots()
        rl = ReferenceLcfs(r=np.array([]), z=np.array([]), ref_name="empty")
        render_mpl(ax, rl)  # must not raise
        plt.close(fig)

    def test_ref_contours_empty_levels_no_artists(self):
        """ReferenceContours with all-empty levels renders nothing."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        from imas_ink.mpl import render_mpl

        fig, ax = plt.subplots()
        # All level groups empty
        rc = ReferenceContours(segments=[[], [], []], ref_name="empty")
        n_before = len(ax.collections)
        render_mpl(ax, rc)
        assert len(ax.collections) == n_before
        plt.close(fig)
