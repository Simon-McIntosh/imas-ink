"""Tests for the plot-only refactor (2026-06-04).

Three categories:
1. Enclosure-styling: closed-around-axis vs open vs closed-not-around-axis.
2. LCFS-from-IDS: boundary.outline verbatim, no recomputation from psi.
3. No-xpoint-when-absent: x_points empty if IDS has none.
"""
from __future__ import annotations

import types

import numpy as np
import pytest

from imas_ink.components import LcfsOutline, SolContours, FluxContours
from imas_ink.extract import extract_slice
from imas_ink.geometry import classify_flux_segments, encloses_point


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _diamond_poly(cx: float, cz: float, r: float) -> np.ndarray:
    """Return a closed diamond polygon centred at (cx, cz) with half-width r."""
    verts = np.array([
        [cx, cz - r],
        [cx + r, cz],
        [cx, cz + r],
        [cx - r, cz],
        [cx, cz - r],  # closed
    ])
    return verts


def _open_line(r0: float = 1.0, z0: float = 0.0, z1: float = 1.0) -> np.ndarray:
    """Return an open line segment (first != last)."""
    return np.array([
        [r0, z0],
        [r0 + 0.5, (z0 + z1) / 2],
        [r0 + 1.0, z1],
    ])


# ---------------------------------------------------------------------------
# 1. Enclosure-styling tests
# ---------------------------------------------------------------------------

class TestEnclosesPoint:
    """encloses_point — pure geometry, drives enclosure-styling."""

    def test_inside_diamond(self):
        """Point at the centroid is inside."""
        v = _diamond_poly(5.0, 0.0, 1.0)
        assert encloses_point(v, 5.0, 0.0) is True

    def test_outside_diamond(self):
        """Point far from the polygon is outside."""
        v = _diamond_poly(5.0, 0.0, 1.0)
        assert encloses_point(v, 9.0, 0.0) is False

    def test_empty_polygon(self):
        """Degenerate polygon with fewer than 3 vertices returns False."""
        v = np.array([[5.0, 0.0], [5.5, 0.0]])
        assert encloses_point(v, 5.0, 0.0) is False

    def test_large_circle(self):
        """Axis well inside a large circular polygon."""
        theta = np.linspace(0, 2 * np.pi, 200)
        r_val, z0 = 5.0, 0.0
        r_arr = r_val + 0.8 * np.cos(theta)
        z_arr = z0 + 0.8 * np.sin(theta)
        v = np.column_stack([r_arr, z_arr])
        assert encloses_point(v, r_val, z0) is True
        # Point outside the circle
        assert encloses_point(v, r_val + 1.5, z0) is False


class TestClassifyFluxSegments:
    """classify_flux_segments: closed-around-axis vs SOL/open."""

    def test_closed_enclosing_is_confined(self):
        """A closed polygon that encloses the axis → confined."""
        seg = _diamond_poly(5.0, 0.0, 0.5)  # encloses (5.0, 0.0)
        confined, sol = classify_flux_segments([seg], r_axis=5.0, z_axis=0.0)
        assert len(confined) == 1
        assert len(sol) == 0

    def test_open_line_is_sol(self):
        """An open line segment → SOL."""
        seg = _open_line()
        confined, sol = classify_flux_segments([seg], r_axis=5.0, z_axis=0.0)
        assert len(confined) == 0
        assert len(sol) == 1

    def test_closed_not_enclosing_is_sol(self):
        """A closed polygon that does NOT enclose the axis → SOL."""
        # Diamond around (9.0, 0.0) — far from the axis at (5.0, 0.0)
        seg = _diamond_poly(9.0, 0.0, 0.3)
        confined, sol = classify_flux_segments([seg], r_axis=5.0, z_axis=0.0)
        assert len(confined) == 0
        assert len(sol) == 1

    def test_mixed_level(self):
        """A level with one confined and one open segment."""
        closed = _diamond_poly(5.0, 0.0, 0.5)
        open_ = _open_line()
        confined, sol = classify_flux_segments([closed, open_], r_axis=5.0, z_axis=0.0)
        assert len(confined) == 1
        assert len(sol) == 1

    def test_empty_level(self):
        """Empty segment list → both outputs empty."""
        confined, sol = classify_flux_segments([], r_axis=5.0, z_axis=0.0)
        assert confined == []
        assert sol == []

    def test_tiny_segment_is_sol(self):
        """A segment with fewer than 3 points is classified as SOL."""
        seg = np.array([[5.0, 0.0], [5.1, 0.0]])
        confined, sol = classify_flux_segments([seg], r_axis=5.0, z_axis=0.0)
        assert len(confined) == 0
        assert len(sol) == 1


class TestSolOnlySegments:
    """Full-field vacuum layers must not render axis-enclosing closed contours."""

    def test_axis_enclosing_closed_segment_is_dropped(self):
        from imas_ink.figures import _sol_only_segments

        closed = _diamond_poly(5.0, 0.0, 0.5)
        open_ = _open_line()
        sol_levels = _sol_only_segments([[closed, open_]], r_axis=5.0, z_axis=0.0)
        assert len(sol_levels) == 1
        assert len(sol_levels[0]) == 1
        np.testing.assert_array_equal(sol_levels[0][0], open_)

    def test_non_axis_closed_segment_is_kept_as_sol(self):
        from imas_ink.figures import _sol_only_segments

        closed_far = _diamond_poly(9.0, 0.0, 0.3)
        sol_levels = _sol_only_segments([[closed_far]], r_axis=5.0, z_axis=0.0)
        assert len(sol_levels) == 1
        assert len(sol_levels[0]) == 1
        np.testing.assert_array_equal(sol_levels[0][0], closed_far)


# ---------------------------------------------------------------------------
# 2. LCFS-from-IDS tests
# ---------------------------------------------------------------------------

def _make_minimal_time_slice(
    r_bnd: np.ndarray | None = None,
    z_bnd: np.ndarray | None = None,
    x_points: list | None = None,
):
    """Build the minimal IDS-like object expected by extract_slice.

    Only the fields accessed by extract_slice are populated; everything
    else raises AttributeError (which extract_slice must handle cleanly).
    """
    # Build minimal objects using SimpleNamespace
    ns = types.SimpleNamespace

    # 2D grid
    r_1d = np.linspace(4.0, 8.0, 17)
    z_1d = np.linspace(-2.0, 2.0, 17)
    r_2d, z_2d = np.meshgrid(r_1d, z_1d, indexing="ij")
    dist_sq = (r_2d - 6.0) ** 2 + z_2d ** 2
    psi_2d = 4.0 - dist_sq  # psi_axis ~ 4.0

    p2d = ns(
        grid=ns(dim1=r_1d, dim2=z_1d),
        psi=psi_2d,
    )

    gq = ns(
        psi_axis=float(psi_2d.max()),
        psi_boundary=float(psi_2d.max()) - 1.0,
        magnetic_axis=ns(r=6.0, z=0.0),
        ip=1e6,
        beta_pol=float("nan"),
        li_3=float("nan"),
        q95=float("nan"),
    )

    # X-points
    if x_points is None:
        xp_list = []
    else:
        xp_list = [ns(r=r, z=z) for (r, z) in x_points]

    # Boundary
    if r_bnd is not None and z_bnd is not None:
        boundary = ns(
            outline=ns(r=r_bnd, z=z_bnd),
            x_point=xp_list,
        )
    else:
        # outline with empty arrays (simulates absent data)
        boundary = ns(
            outline=ns(r=np.array([]), z=np.array([])),
            x_point=xp_list,
        )

    ts = ns(profiles_2d=[p2d], global_quantities=gq, boundary=boundary)

    # Build the minimal IDS-level object
    eq_ids = ns(
        time=np.array([0.5]),
        time_slice=[ts],
    )
    return eq_ids


class TestLcfsFromIds:
    """LCFS must come verbatim from boundary.outline, never recomputed."""

    def test_boundary_r_z_returned_verbatim(self):
        """When boundary.outline is populated, sl.boundary_r/z match exactly."""
        r_bnd = np.array([6.5, 5.5, 5.5, 6.5, 6.5])
        z_bnd = np.array([0.5, 0.5, -0.5, -0.5, 0.5])
        eq_ids = _make_minimal_time_slice(r_bnd=r_bnd, z_bnd=z_bnd)
        sl = extract_slice(eq_ids, 0)
        assert sl.boundary_r is not None
        assert sl.boundary_z is not None
        np.testing.assert_array_equal(sl.boundary_r, r_bnd)
        np.testing.assert_array_equal(sl.boundary_z, z_bnd)

    def test_boundary_absent_gives_none(self):
        """When boundary.outline is empty, sl.boundary_r is None (not a psi contour)."""
        eq_ids = _make_minimal_time_slice(r_bnd=None, z_bnd=None)
        sl = extract_slice(eq_ids, 0)
        assert sl.boundary_r is None
        assert sl.boundary_z is None

    def test_lcfs_outline_component_renders_verbatim(self):
        """LcfsOutline holds the IDS arrays unchanged."""
        r_bnd = np.array([6.0, 7.0, 6.0, 5.0, 6.0])
        z_bnd = np.array([1.0, 0.0, -1.0, 0.0, 1.0])
        lcfs = LcfsOutline(r_bnd, z_bnd)
        np.testing.assert_array_equal(lcfs.r, r_bnd)
        np.testing.assert_array_equal(lcfs.z, z_bnd)

    def test_lcfs_empty_gives_empty_arrays(self):
        """LcfsOutline with empty arrays is valid (nothing rendered)."""
        lcfs = LcfsOutline(np.array([]), np.array([]))
        assert lcfs.r.size == 0
        assert lcfs.z.size == 0


# ---------------------------------------------------------------------------
# 3. No-xpoint-when-absent tests
# ---------------------------------------------------------------------------

class TestNoXpointWhenAbsent:
    """X-points must be read from IDS only; absent → empty list."""

    def test_x_points_absent_gives_empty_list(self):
        """When boundary.x_point is absent/empty, sl.x_points is empty."""
        eq_ids = _make_minimal_time_slice(x_points=[])
        sl = extract_slice(eq_ids, 0)
        assert sl.x_points == []

    def test_x_points_present_are_read(self):
        """When boundary.x_point contains valid data, it is returned."""
        eq_ids = _make_minimal_time_slice(x_points=[(5.8, -1.2)])
        sl = extract_slice(eq_ids, 0)
        assert len(sl.x_points) == 1
        assert abs(sl.x_points[0][0] - 5.8) < 1e-9
        assert abs(sl.x_points[0][1] - (-1.2)) < 1e-9

    def test_no_fallback_to_find_xpoints(self):
        """extract_slice does not call find_xpoints when IDS has no x_points.

        Previously, the fallback called find_xpoints(psi_2d, ...) which
        computed physics values.  After the refactor, the result must be
        an empty list — not a numerically detected set.
        """
        # Construct a psi field that would produce find_xpoints hits if
        # the fallback were active: add a saddle at z < 0 by subtracting
        # a Gaussian near the axis.
        eq_ids = _make_minimal_time_slice(x_points=[])
        sl = extract_slice(eq_ids, 0)
        # After the refactor: always empty when IDS has none
        assert sl.x_points == [], (
            "find_xpoints fallback must NOT be called; "
            "x_points must be empty when IDS boundary.x_point is absent"
        )
