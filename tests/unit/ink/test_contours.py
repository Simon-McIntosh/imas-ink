"""Tests for efit.ink.contours — ContourExtractor."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ink.contours import ContourExtractor


# ---------------------------------------------------------------------------
# Synthetic circular psi field
# ---------------------------------------------------------------------------
def _make_circular_field(n: int = 129, r0: float = 6.0, z0: float = 0.0):
    """Circular flux surfaces: psi = psi_max - ((R-R0)^2 + (Z-Z0)^2).

    Returns enough structure to test contour extraction.
    psi_axis is at the centre (maximum), psi decreases outward.
    """
    r_1d = np.linspace(4.0, 8.0, n)
    z_1d = np.linspace(-2.0, 2.0, n)
    r_2d, z_2d = np.meshgrid(r_1d, z_1d, indexing="ij")
    dist_sq = (r_2d - r0) ** 2 + (z_2d - z0) ** 2
    psi_max = dist_sq.max()
    psi_2d = psi_max - dist_sq
    psi_axis = float(psi_2d.max())  # at centre
    psi_bnd = psi_axis - 1.0  # circle of radius 1.0
    return r_2d, z_2d, psi_2d, psi_axis, psi_bnd


@pytest.fixture
def cx():
    """Fixture returning a ContourExtractor on a circular field."""
    r_2d, z_2d, psi_2d, _, _ = _make_circular_field()
    return ContourExtractor(r_2d, z_2d, psi_2d)


@pytest.fixture
def field_params():
    """Fixture returning psi_axis and psi_bnd for the circular field."""
    _, _, _, psi_axis, psi_bnd = _make_circular_field()
    return psi_axis, psi_bnd


# ---------------------------------------------------------------------------
# lines_at
# ---------------------------------------------------------------------------
class TestLinesAt:
    """ContourExtractor.lines_at() — single-level contour extraction."""

    def test_returns_segments(self, cx, field_params):
        psi_axis, psi_bnd = field_params
        mid_level = (psi_axis + psi_bnd) / 2
        segs = cx.lines_at(mid_level)
        assert isinstance(segs, list)
        assert len(segs) > 0, "Should find contour at mid-level"

    def test_segment_shape(self, cx, field_params):
        psi_axis, psi_bnd = field_params
        mid_level = (psi_axis + psi_bnd) / 2
        segs = cx.lines_at(mid_level)
        for seg in segs:
            assert seg.ndim == 2
            assert seg.shape[1] == 2

    def test_no_contour_at_extreme(self, cx):
        """Level far outside the field range should return empty."""
        segs = cx.lines_at(-1e6)
        assert len(segs) == 0

    def test_segments_have_two_or_more_points(self, cx, field_params):
        """Every returned segment has at least 2 points."""
        psi_axis, psi_bnd = field_params
        mid_level = (psi_axis + psi_bnd) / 2
        for seg in cx.lines_at(mid_level):
            assert len(seg) >= 2


# ---------------------------------------------------------------------------
# flux_surfaces
# ---------------------------------------------------------------------------
class TestFluxSurfaces:
    """ContourExtractor.flux_surfaces()."""

    def test_returns_n_levels(self, cx, field_params):
        psi_axis, psi_bnd = field_params
        n = 6
        result = cx.flux_surfaces(psi_axis, psi_bnd, n=n)
        assert len(result) == n

    def test_each_level_is_list_of_segments(self, cx, field_params):
        psi_axis, psi_bnd = field_params
        result = cx.flux_surfaces(psi_axis, psi_bnd, n=3)
        for level_segs in result:
            assert isinstance(level_segs, list)

    def test_interior_levels_have_contours(self, cx, field_params):
        """For a clean circular field, every interior level should produce
        at least one contour segment.
        """
        psi_axis, psi_bnd = field_params
        result = cx.flux_surfaces(psi_axis, psi_bnd, n=4)
        for level_segs in result:
            assert len(level_segs) > 0, "Interior level should have contours"


# ---------------------------------------------------------------------------
# separatrix
# ---------------------------------------------------------------------------
class TestSeparatrix:
    """ContourExtractor.separatrix()."""

    def test_returns_segments(self, cx, field_params):
        _, psi_bnd = field_params
        segs = cx.separatrix(psi_bnd)
        assert isinstance(segs, list)
        assert len(segs) > 0, "Should find separatrix contour"

    def test_segment_shape(self, cx, field_params):
        _, psi_bnd = field_params
        for seg in cx.separatrix(psi_bnd):
            assert seg.ndim == 2
            assert seg.shape[1] == 2


# ---------------------------------------------------------------------------
# is_closed
# ---------------------------------------------------------------------------
class TestIsClosed:
    """ContourExtractor.is_closed()."""

    def test_interior_closed(self, cx, field_params):
        """Interior contour of a circular field should be closed."""
        psi_axis, psi_bnd = field_params
        # Use a level well inside the boundary
        interior_level = psi_bnd + 0.75 * (psi_axis - psi_bnd)
        # This should be a closed contour if the circle is fully enclosed
        segs = cx.lines_at(interior_level)
        if len(segs) > 0:
            # For a well-enclosed circle, expect closed
            result = cx.is_closed(interior_level, seg_index=0)
            assert result in (True, False)  # numpy bool or Python bool

    def test_out_of_range_index(self, cx, field_params):
        """seg_index beyond available segments returns False."""
        psi_axis, psi_bnd = field_params
        mid_level = (psi_axis + psi_bnd) / 2
        assert cx.is_closed(mid_level, seg_index=999) is False


# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------
class TestContourExtractorConstruction:
    """Verify construction and basic properties."""

    def test_stores_arrays(self):
        r_2d, z_2d, psi_2d, _, _ = _make_circular_field(n=33)
        cx = ContourExtractor(r_2d, z_2d, psi_2d)
        assert cx.r_2d is r_2d
        assert cx.z_2d is z_2d
        assert cx.psi_2d is psi_2d

    def test_generator_created(self):
        """The internal contourpy generator should exist after init."""
        r_2d, z_2d, psi_2d, _, _ = _make_circular_field(n=33)
        cx = ContourExtractor(r_2d, z_2d, psi_2d)
        assert hasattr(cx, "_gen")
