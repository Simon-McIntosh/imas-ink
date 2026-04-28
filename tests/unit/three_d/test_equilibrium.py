"""Tests for imas_ink/three_d/equilibrium.py — DD resolution, IDS extraction, interpolation."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# DD version resolution tests (no imas dependency)
# ---------------------------------------------------------------------------

from imas_ink._dd import DEFAULT_DD_VERSION, resolve_dd_version


class TestResolveDD:
    def test_resolve_dd_version_explicit_wins(self):
        """Explicit kwarg takes highest precedence."""
        assert resolve_dd_version("3.42.0") == "3.42.0"

    def test_resolve_dd_version_explicit_wins_over_env(self, monkeypatch):
        """Explicit kwarg wins even when IMAS_VERSION is set."""
        monkeypatch.setenv("IMAS_VERSION", "3.99.0")
        assert resolve_dd_version("3.42.0") == "3.42.0"

    def test_resolve_dd_version_env_var(self, monkeypatch):
        """IMAS_VERSION env var is used when no explicit kwarg is given."""
        monkeypatch.setenv("IMAS_VERSION", "4.0.0")
        monkeypatch.delenv("IMAS_VERSION", raising=False)
        monkeypatch.setenv("IMAS_VERSION", "4.0.0")
        assert resolve_dd_version() == "4.0.0"

    def test_resolve_dd_version_default_fallback(self, monkeypatch):
        """Falls back to DEFAULT_DD_VERSION when nothing else is set."""
        monkeypatch.delenv("IMAS_VERSION", raising=False)
        result = resolve_dd_version()
        assert result == DEFAULT_DD_VERSION
        assert result == "4.1.0"

    def test_resolve_dd_version_empty_env_is_ignored(self, monkeypatch):
        """Empty IMAS_VERSION is treated as unset; falls back to default."""
        monkeypatch.setenv("IMAS_VERSION", "   ")
        assert resolve_dd_version() == DEFAULT_DD_VERSION


# ---------------------------------------------------------------------------
# Synthetic IDS tests — require imas-python
# ---------------------------------------------------------------------------

imas = pytest.importorskip("imas", reason="imas-python not installed")

from imas_ink.three_d.equilibrium import (  # noqa: E402
    EquilibriumSlice2D,
    extract_slice_2d,
    psi_grid_interpolator,
)


def _make_synthetic_eq(dd_version: str = DEFAULT_DD_VERSION):
    """Build a minimal 5×5 equilibrium IDS for testing."""
    factory = imas.IDSFactory(version=dd_version)
    eq = factory.new("equilibrium")

    # One time slice
    eq.time = np.array([1.5])
    eq.time_slice.resize(1)
    ts = eq.time_slice[0]

    # 5×5 ψ grid
    r_vals = np.linspace(5.0, 7.0, 5)
    z_vals = np.linspace(-1.0, 1.0, 5)
    psi_vals = np.outer(r_vals - 6.0, z_vals - 0.0) + 0.5  # simple saddle

    ts.profiles_2d.resize(1)
    p2d = ts.profiles_2d[0]
    p2d.grid.dim1 = r_vals
    p2d.grid.dim2 = z_vals
    p2d.psi = psi_vals

    # Global quantities
    gq = ts.global_quantities
    gq.psi_axis = 0.5
    gq.psi_boundary = 1.2
    gq.magnetic_axis.r = 6.0
    gq.magnetic_axis.z = 0.1

    # Boundary outline
    ts.boundary.outline.r = np.array([5.5, 6.0, 6.5, 6.0, 5.5])
    ts.boundary.outline.z = np.array([0.0, 0.8, 0.0, -0.8, 0.0])

    return eq


class TestExtractSlice2DSynthetic:
    """Round-trip tests with a synthetic IDS."""

    @pytest.fixture(scope="class")
    def slice_2d(self):
        eq = _make_synthetic_eq()
        return extract_slice_2d(eq, time_index=0)

    def test_time_round_trips(self, slice_2d):
        assert slice_2d.time == pytest.approx(1.5)

    def test_psi_axis_round_trips(self, slice_2d):
        assert slice_2d.psi_axis == pytest.approx(0.5)

    def test_psi_boundary_round_trips(self, slice_2d):
        assert slice_2d.psi_boundary == pytest.approx(1.2)

    def test_r_1d_shape(self, slice_2d):
        assert slice_2d.R_1d.shape == (5,)
        assert slice_2d.R_1d[0] == pytest.approx(5.0)
        assert slice_2d.R_1d[-1] == pytest.approx(7.0)

    def test_z_1d_shape(self, slice_2d):
        assert slice_2d.Z_1d.shape == (5,)

    def test_psi_2d_shape_ij(self, slice_2d):
        assert slice_2d.psi_2d.shape == (5, 5)

    def test_o_point_round_trips(self, slice_2d):
        assert slice_2d.o_point is not None
        r_ax, z_ax = slice_2d.o_point
        assert r_ax == pytest.approx(6.0)
        assert z_ax == pytest.approx(0.1)

    def test_boundary_outline_present(self, slice_2d):
        assert slice_2d.boundary_r.size == 5
        assert slice_2d.boundary_z.size == 5

    def test_dataclass_is_frozen(self, slice_2d):
        with pytest.raises(Exception):
            slice_2d.time = 99.0  # type: ignore[misc]

    def test_returns_equilbrium_slice_2d_type(self, slice_2d):
        assert isinstance(slice_2d, EquilibriumSlice2D)


class TestPsiGridInterpolator:
    """Tests for psi_grid_interpolator."""

    @pytest.fixture(scope="class")
    def interp(self):
        eq = _make_synthetic_eq()
        sl = extract_slice_2d(eq, time_index=0)
        return sl, psi_grid_interpolator(sl)

    def test_interior_matches_grid_value(self, interp):
        sl, itp = interp
        # Sample at a grid node — should return the exact psi value
        r_idx, z_idx = 2, 2
        r_val = sl.R_1d[r_idx]
        z_val = sl.Z_1d[z_idx]
        expected = sl.psi_2d[r_idx, z_idx]
        result = itp([[r_val, z_val]])[0]
        assert result == pytest.approx(expected, abs=1e-10)

    def test_exterior_returns_nan(self, interp):
        _sl, itp = interp
        # Far outside the grid
        result = itp([[100.0, 100.0]])[0]
        assert np.isnan(result)

    def test_batch_mix_inside_outside(self, interp):
        sl, itp = interp
        points = [
            [sl.R_1d[1], sl.Z_1d[1]],  # inside
            [0.0, 0.0],  # outside (R too small)
        ]
        vals = itp(points)
        assert not np.isnan(vals[0])
        assert np.isnan(vals[1])


class TestEmptyDoubleSentinelFiltering:
    """Ensure EMPTY_DOUBLE sentinel (-9.0e40) is handled correctly."""

    def test_psi_axis_sentinel_becomes_nan(self):
        """psi_axis = EMPTY_DOUBLE → extracted value is NaN."""
        eq = _make_synthetic_eq()
        ts = eq.time_slice[0]
        ts.global_quantities.psi_axis = -9.0e40
        sl = extract_slice_2d(eq, time_index=0)
        assert np.isnan(sl.psi_axis), "EMPTY_DOUBLE psi_axis should map to NaN"

    def test_psi_boundary_sentinel_becomes_nan(self):
        """psi_boundary = EMPTY_DOUBLE → extracted value is NaN."""
        eq = _make_synthetic_eq()
        ts = eq.time_slice[0]
        ts.global_quantities.psi_boundary = -9.0e40
        sl = extract_slice_2d(eq, time_index=0)
        assert np.isnan(sl.psi_boundary)

    def test_magnetic_axis_sentinel_makes_o_point_none(self):
        """Sentinel in magnetic_axis.r → o_point is None."""
        eq = _make_synthetic_eq()
        ts = eq.time_slice[0]
        ts.global_quantities.magnetic_axis.r = -9.0e40
        sl = extract_slice_2d(eq, time_index=0)
        assert sl.o_point is None, "Sentinel in magnetic_axis.r should give o_point=None"


# ---------------------------------------------------------------------------
# Optional integration test against ITER 135013 reference data
# ---------------------------------------------------------------------------

_REF_PATH = Path(
    "/home/ITER/mcintos/Code/efitpp/tests/data/imas/ITER/135013/reference/run_set_1"
)

_REF_H5 = _REF_PATH / "equilibrium.h5"


@pytest.mark.skipif(not _REF_H5.exists(), reason="Reference data absent")
def test_read_iter_135013_reference_equilibrium():
    """Open the ITER 135013 run_set_1 equilibrium and check the magnetic axis."""
    from imas_ink.three_d.equilibrium import read_equilibrium

    uri = f"imas:hdf5?path={_REF_PATH}/"
    eq_ids = read_equilibrium(uri)
    sl = extract_slice_2d(eq_ids, time_index=0)

    assert sl.o_point is not None, "Magnetic axis should be present"
    r_ax, z_ax = sl.o_point
    # Expected: R≈6.50 m, Z≈0.45 m (tolerance 0.5 m per spec)
    assert abs(r_ax - 6.50) < 0.5, f"R_axis={r_ax:.3f} m not near 6.50 m"
    assert abs(z_ax - 0.45) < 0.5, f"Z_axis={z_ax:.3f} m not near 0.45 m"
    assert sl.psi_2d.shape[0] == 65
    assert sl.psi_2d.shape[1] == 65
