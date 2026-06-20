"""Tests for efit.ink.geometry — backend-neutral geometry processing."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from imas_ink.geometry import (
    close_polygon,
    is_closed_contour,
    mask_pfr,
    split_path_segs,
    wall_clip_vertices,
)


# ---------------------------------------------------------------------------
# Helper: synthetic psi fields
# ---------------------------------------------------------------------------
def _circular_psi(n: int = 65, r0: float = 6.0, z0: float = 0.0):
    """Create a circular psi field: psi = (R-R0)^2 + (Z-Z0)^2.

    psi_axis = 0 at the centre, psi increases outward.
    To make it COCOS-17 compatible (psi_axis > psi_bnd), we invert:
    psi = psi_max - ((R-R0)^2 + (Z-Z0)^2).
    """
    r_1d = np.linspace(4.0, 8.0, n)
    z_1d = np.linspace(-2.0, 2.0, n)
    r_2d, z_2d = np.meshgrid(r_1d, z_1d, indexing="ij")
    dist_sq = (r_2d - r0) ** 2 + (z_2d - z0) ** 2
    psi_max = dist_sq.max()
    psi_2d = psi_max - dist_sq  # max at centre
    psi_axis = float(psi_2d.max())  # at (R0, Z0)
    # boundary at some radius
    psi_bnd = psi_max - 1.0  # circle of radius 1
    return psi_2d, r_2d, z_2d, r_1d, z_1d, psi_axis, psi_bnd, r0, z0


# ---------------------------------------------------------------------------
# mask_pfr
# ---------------------------------------------------------------------------
class TestMaskPfr:
    """mask_pfr() — private flux region masking."""

    def test_core_pixels_survive(self):
        """Core pixels near the axis should remain finite."""
        psi_2d, r_2d, z_2d, _, _, psi_axis, psi_bnd, r0, z0 = _circular_psi()
        masked = mask_pfr(psi_2d, r_2d, z_2d, psi_axis, psi_bnd, r0, z0)
        # The pixel closest to the axis should be finite
        dist_sq = (r_2d - r0) ** 2 + (z_2d - z0) ** 2
        axis_idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
        assert np.isfinite(masked[axis_idx])

    def test_outer_pixels_masked(self):
        """Pixels far outside the boundary should be NaN."""
        psi_2d, r_2d, z_2d, _, _, psi_axis, psi_bnd, r0, z0 = _circular_psi()
        masked = mask_pfr(psi_2d, r_2d, z_2d, psi_axis, psi_bnd, r0, z0)
        assert np.isnan(masked).any(), "Some pixels should be masked"

    def test_does_not_modify_input(self):
        """mask_pfr returns a copy, not modifying the input."""
        psi_2d, r_2d, z_2d, _, _, psi_axis, psi_bnd, r0, z0 = _circular_psi()
        original = psi_2d.copy()
        _ = mask_pfr(psi_2d, r_2d, z_2d, psi_axis, psi_bnd, r0, z0)
        assert_allclose(psi_2d, original)

    def test_output_shape(self):
        """Output has same shape as input."""
        psi_2d, r_2d, z_2d, _, _, psi_axis, psi_bnd, r0, z0 = _circular_psi()
        masked = mask_pfr(psi_2d, r_2d, z_2d, psi_axis, psi_bnd, r0, z0)
        assert masked.shape == psi_2d.shape


# ---------------------------------------------------------------------------
# wall_clip_vertices
# ---------------------------------------------------------------------------
class TestWallClipVertices:
    """wall_clip_vertices() — polygon closure."""

    def test_open_polygon_gets_closed(self):
        """An open polygon is closed by appending the first vertex."""
        r = np.array([1.0, 2.0, 2.0, 1.0])
        z = np.array([0.0, 0.0, 1.0, 1.0])
        verts = wall_clip_vertices(r, z)
        assert verts.shape == (5, 2)
        assert_allclose(verts[0], verts[-1])

    def test_already_closed_stays_same(self):
        """A closed polygon is not modified."""
        r = np.array([1.0, 2.0, 2.0, 1.0, 1.0])
        z = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        verts = wall_clip_vertices(r, z)
        assert verts.shape == (5, 2)
        assert_allclose(verts[0], verts[-1])

    def test_output_columns(self):
        """Output has two columns: R and Z."""
        r = np.array([1.0, 2.0, 3.0])
        z = np.array([0.0, 1.0, 0.0])
        verts = wall_clip_vertices(r, z)
        assert verts.shape[1] == 2


# ---------------------------------------------------------------------------
# split_path_segs
# ---------------------------------------------------------------------------
class TestSplitPathSegs:
    """split_path_segs() — splitting multi-segment contour paths."""

    MOVETO = 1
    LINETO = 2
    CLOSEPOLY = 79

    def test_none_codes_returns_single(self):
        """None codes → the entire vertex array as one segment."""
        verts = np.array([[0, 0], [1, 0], [1, 1]])
        segs = split_path_segs(verts, None)
        assert len(segs) == 1
        assert_allclose(segs[0], verts)

    def test_single_open_segment(self):
        """A single MOVETO + LINETOs without CLOSEPOLY."""
        verts = np.array([[0, 0], [1, 0], [2, 0], [3, 0]])
        codes = np.array([1, 2, 2, 2])
        segs = split_path_segs(verts, codes)
        assert len(segs) == 1
        assert len(segs[0]) == 4

    def test_single_closed_segment(self):
        """A single closed segment: MOVETO + LINETOs + CLOSEPOLY."""
        verts = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        codes = np.array([1, 2, 2, 79])
        segs = split_path_segs(verts, codes)
        assert len(segs) == 1
        assert len(segs[0]) == 4

    def test_two_segments(self):
        """Two separate sub-paths split at MOVETO boundaries."""
        verts = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],  # segment 1
                [5, 5],
                [6, 5],
                [6, 6],  # segment 2
            ]
        )
        codes = np.array([1, 2, 2, 1, 2, 2])
        segs = split_path_segs(verts, codes)
        assert len(segs) == 2
        assert len(segs[0]) == 3
        assert len(segs[1]) == 3

    def test_mixed_closed_and_open(self):
        """First segment closed, second open."""
        verts = np.array(
            [
                [0, 0],
                [1, 0],
                [1, 1],
                [0, 0],  # closed
                [5, 5],
                [6, 5],
                [6, 6],  # open
            ]
        )
        codes = np.array([1, 2, 2, 79, 1, 2, 2])
        segs = split_path_segs(verts, codes)
        assert len(segs) == 2

    def test_all_segments_have_two_columns(self):
        """Every segment should have shape (N, 2)."""
        verts = np.array([[0, 0], [1, 0], [2, 0], [3, 3], [4, 4]])
        codes = np.array([1, 2, 2, 1, 2])
        segs = split_path_segs(verts, codes)
        for seg in segs:
            assert seg.shape[1] == 2


# ---------------------------------------------------------------------------
# close_polygon
# ---------------------------------------------------------------------------
class TestClosePolygon:
    """close_polygon() — ensure first vertex == last vertex."""

    def test_open_polygon(self):
        v = np.array([[0, 0], [1, 0], [1, 1]])
        result = close_polygon(v)
        assert result.shape == (4, 2)
        assert_allclose(result[0], result[-1])

    def test_already_closed(self):
        v = np.array([[0, 0], [1, 0], [1, 1], [0, 0]])
        result = close_polygon(v)
        assert result.shape == (4, 2)  # no extra vertex

    def test_single_vertex(self):
        """A single vertex cannot be closed."""
        v = np.array([[0, 0]])
        result = close_polygon(v)
        assert result.shape == (1, 2)

    def test_two_vertices_open(self):
        v = np.array([[0, 0], [1, 1]])
        result = close_polygon(v)
        assert result.shape == (3, 2)
        assert_allclose(result[0], result[-1])


# ---------------------------------------------------------------------------
# is_closed_contour
# ---------------------------------------------------------------------------
class TestIsClosedContour:
    """is_closed_contour() — CLOSEPOLY detection."""

    def test_closed(self):
        codes = np.array([1, 2, 2, 79])
        assert is_closed_contour(codes) is True

    def test_open(self):
        codes = np.array([1, 2, 2])
        assert is_closed_contour(codes) is False

    def test_none(self):
        assert is_closed_contour(None) is False

    def test_empty_array(self):
        assert is_closed_contour(np.array([])) is False

    def test_only_closepoly(self):
        """Single CLOSEPOLY code."""
        codes = np.array([79])
        assert is_closed_contour(codes) is True

    def test_moveto_only(self):
        codes = np.array([1])
        assert is_closed_contour(codes) is False
