"""Tests for imas_ink/three_d/flux_projection.py — ψ projection onto 3D cap surfaces."""

from __future__ import annotations

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista not installed")

from imas_ink._cocos import make_levels
from imas_ink.three_d.equilibrium import EquilibriumSlice2D
from imas_ink.three_d.flux_projection import (
    FluxOverlay,
    build_flux_overlay,
    contours_on_cap,
    offset_along_normal,
    sample_psi_on_cap,
)


# ---------------------------------------------------------------------------
# Helpers — synthetic data
# ---------------------------------------------------------------------------


def _make_slice_rz(
    r_range: tuple[float, float] = (1.0, 3.0),
    z_range: tuple[float, float] = (-1.5, 1.5),
    n: int = 50,
    psi_func=None,
) -> EquilibriumSlice2D:
    """Build a synthetic EquilibriumSlice2D with a prescribed ψ(R, Z)."""
    R_1d = np.linspace(r_range[0], r_range[1], n)
    Z_1d = np.linspace(z_range[0], z_range[1], n)

    if psi_func is None:
        # Default: ψ = R + Z (simple linear field)
        R_2d, Z_2d = np.meshgrid(R_1d, Z_1d, indexing="ij")
        psi_2d = R_2d + Z_2d
    else:
        R_2d, Z_2d = np.meshgrid(R_1d, Z_1d, indexing="ij")
        psi_2d = psi_func(R_2d, Z_2d)

    psi_axis = float(np.min(psi_2d))
    psi_boundary = float(np.max(psi_2d))

    return EquilibriumSlice2D(
        time=1.0,
        psi_axis=psi_axis,
        psi_boundary=psi_boundary,
        R_1d=R_1d,
        Z_1d=Z_1d,
        psi_2d=psi_2d,
        boundary_r=np.empty(0),
        boundary_z=np.empty(0),
        o_point=None,
        x_points=(),
    )


def _make_cap_quad(
    r_range: tuple[float, float] = (1.0, 2.0),
    z_range: tuple[float, float] = (-1.0, 1.0),
    y: float = 0.0,
    n_r: int = 5,
    n_z: int = 5,
) -> pv.PolyData:
    """Build a rectangular cap mesh in the y = *y* plane.

    Returns a triangulated PolyData with vertices spanning
    ``R ∈ r_range``, ``Z ∈ z_range``, all at ``y = y``.
    """
    r_vals = np.linspace(r_range[0], r_range[1], n_r)
    z_vals = np.linspace(z_range[0], z_range[1], n_z)

    grid = pv.RectilinearGrid(r_vals, [y], z_vals)
    return grid.extract_surface().triangulate()


# ---------------------------------------------------------------------------
# test_sample_psi_on_cap_synthetic
# ---------------------------------------------------------------------------


class TestSamplePsiOnCap:
    def test_synthetic_linear(self):
        """ψ = R + Z; sampled values on cap should match R + Z at each vertex."""
        sl = _make_slice_rz(psi_func=lambda R, Z: R + Z)
        cap = _make_cap_quad(r_range=(1.0, 2.0), z_range=(-1.0, 1.0))

        psi = sample_psi_on_cap(cap, sl)

        pts = np.asarray(cap.points, dtype=float)
        R_exp = np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2)
        Z_exp = pts[:, 2]
        expected = R_exp + Z_exp

        np.testing.assert_allclose(psi, expected, atol=0.05)

    def test_outside_returns_nan(self):
        """Points outside the (R, Z) grid sample to NaN."""
        # Slice covers R ∈ [1, 3], Z ∈ [-1.5, 1.5]
        sl = _make_slice_rz(r_range=(1.0, 3.0), z_range=(-1.5, 1.5))
        # Cap far outside the grid
        cap = _make_cap_quad(r_range=(10.0, 12.0), z_range=(5.0, 7.0))

        psi = sample_psi_on_cap(cap, sl)
        assert np.all(np.isnan(psi)), "All values should be NaN for out-of-bounds cap"

    def test_axis_not_y_raises(self):
        """axis != 'y' should raise NotImplementedError."""
        sl = _make_slice_rz()
        cap = _make_cap_quad()
        with pytest.raises(NotImplementedError, match="axis"):
            sample_psi_on_cap(cap, sl, axis="x")


# ---------------------------------------------------------------------------
# test_contours_on_cap_synthetic
# ---------------------------------------------------------------------------


class TestContoursOnCap:
    def test_radial_contours(self):
        """ψ = R² + Z²: contour lines should be circular arcs on y = 0."""
        sl = _make_slice_rz(
            r_range=(0.5, 3.0),
            z_range=(-2.0, 2.0),
            n=80,
            psi_func=lambda R, Z: R**2 + Z**2,
        )
        cap = _make_cap_quad(r_range=(0.5, 3.0), z_range=(-2.0, 2.0), n_r=20, n_z=20)

        contours, levels = contours_on_cap(cap, sl, n_levels=4)

        assert len(levels) == 4
        # At least some contours should be generated
        assert len(contours) > 0

        # All contour points lie on y ≈ 0
        for poly in contours:
            pts = np.asarray(poly.points, dtype=float)
            np.testing.assert_allclose(pts[:, 1], 0.0, atol=1e-9)

    def test_explicit_levels_used(self):
        """When explicit levels are passed, returned levels match."""
        sl = _make_slice_rz(psi_func=lambda R, Z: R + Z)
        cap = _make_cap_quad()
        explicit = np.array([1.2, 1.5, 1.8])

        _, returned_levels = contours_on_cap(cap, sl, levels=explicit)
        np.testing.assert_array_equal(returned_levels, explicit)

    def test_clipped_to_polygon(self):
        """Contour points should be clipped to lie inside the cap polygon."""
        # Small cap, large ψ field
        sl = _make_slice_rz(
            r_range=(0.5, 10.0),
            z_range=(-5.0, 5.0),
            n=100,
            psi_func=lambda R, Z: R + Z,
        )
        cap = _make_cap_quad(r_range=(5.0, 6.0), z_range=(-0.5, 0.5), n_r=10, n_z=10)

        contours, _ = contours_on_cap(cap, sl, n_levels=4)

        for poly in contours:
            pts = np.asarray(poly.points, dtype=float)
            R_pts = np.abs(pts[:, 0])
            Z_pts = pts[:, 2]
            # All points should be within the cap bounds (with tolerance)
            assert np.all(R_pts >= 5.0 - 0.2), f"R min: {R_pts.min()}"
            assert np.all(R_pts <= 6.0 + 0.2), f"R max: {R_pts.max()}"
            assert np.all(Z_pts >= -0.5 - 0.2), f"Z min: {Z_pts.min()}"
            assert np.all(Z_pts <= 0.5 + 0.2), f"Z max: {Z_pts.max()}"


# ---------------------------------------------------------------------------
# test_build_flux_overlay_modes
# ---------------------------------------------------------------------------


class TestBuildFluxOverlay:
    @pytest.fixture
    def overlay_inputs(self):
        sl = _make_slice_rz(psi_func=lambda R, Z: R + Z)
        cap = _make_cap_quad()
        return cap, sl

    def test_contours_and_field(self, overlay_inputs):
        cap, sl = overlay_inputs
        ov = build_flux_overlay(cap, sl, mode="contours_and_field", n_levels=4)
        assert isinstance(ov, FluxOverlay)
        assert ov.field is not None
        assert ov.field.shape == (cap.n_points,)
        assert len(ov.levels) == 4
        # contours may or may not be present depending on geometry
        assert isinstance(ov.contours, list)

    def test_field_only(self, overlay_inputs):
        cap, sl = overlay_inputs
        ov = build_flux_overlay(cap, sl, mode="field_only", n_levels=4)
        assert ov.field is not None
        assert ov.field.shape == (cap.n_points,)
        assert ov.contours == []

    def test_contours_only(self, overlay_inputs):
        cap, sl = overlay_inputs
        ov = build_flux_overlay(cap, sl, mode="contours_only", n_levels=4)
        assert ov.field is None
        assert isinstance(ov.contours, list)

    def test_cap_normal_unit_vector(self, overlay_inputs):
        cap, sl = overlay_inputs
        ov = build_flux_overlay(cap, sl, mode="field_only")
        n = np.asarray(ov.cap_normal)
        assert abs(np.linalg.norm(n) - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# test_offset_along_normal_translation
# ---------------------------------------------------------------------------


class TestOffsetAlongNormal:
    def test_translation(self):
        """Output vertices are translated by exactly epsilon * normal."""
        pts = np.array([[1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [3.0, 0.0, 2.0]])
        lines = np.array([3, 0, 1, 2])
        poly = pv.PolyData(pts, lines=lines)

        normal = (0.0, 1.0, 0.0)
        eps = 0.005
        result = offset_along_normal(poly, normal, epsilon=eps)

        expected = pts + np.array([[0.0, eps, 0.0]])
        np.testing.assert_allclose(result.points, expected, atol=1e-12)

    def test_does_not_mutate_input(self):
        """The original PolyData must not be modified."""
        pts = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        lines = np.array([2, 0, 1])
        poly = pv.PolyData(pts, lines=lines)
        original_pts = poly.points.copy()

        offset_along_normal(poly, (0, 0, 1), epsilon=1.0)

        np.testing.assert_array_equal(poly.points, original_pts)

    def test_unnormalized_normal_still_works(self):
        """A non-unit normal should be normalised internally."""
        pts = np.array([[0.0, 0.0, 0.0]])
        poly = pv.PolyData(pts)

        result = offset_along_normal(poly, (0, 0, 3.0), epsilon=1.0)
        np.testing.assert_allclose(result.points[0], [0, 0, 1.0], atol=1e-12)


# ---------------------------------------------------------------------------
# test_levels_use_make_levels_default
# ---------------------------------------------------------------------------


class TestLevelsDefault:
    def test_default_levels_match_make_levels(self):
        """When levels=None, returned levels match make_levels()."""
        sl = _make_slice_rz(psi_func=lambda R, Z: R + Z)
        cap = _make_cap_quad()
        n = 6

        _, returned_levels = contours_on_cap(cap, sl, n_levels=n)
        expected_levels = make_levels(sl.psi_axis, sl.psi_boundary, n=n)

        np.testing.assert_array_equal(returned_levels, expected_levels)
