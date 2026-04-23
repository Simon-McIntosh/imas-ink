"""Tests for efit.ink.extract — IDS extractor functions.

All tests use lightweight mock objects (SimpleNamespace) that mimic the
IDS structure. No IMAS installation required.
"""

from __future__ import annotations

import types

import numpy as np
import pytest
from numpy.testing import assert_allclose

from imas_ink.extract import (
    extract_profiles_1d,
    extract_slice,
    extract_time_traces,
)


# ---------------------------------------------------------------------------
# Mock IDS factory helpers
# ---------------------------------------------------------------------------
def _ns(**kw):
    """Shorthand for types.SimpleNamespace."""
    return types.SimpleNamespace(**kw)


def _make_eq_ids(
    n_r: int = 33,
    n_z: int = 33,
    psi_axis_val: float = 10.0,
    psi_bnd_val: float = 2.0,
    ip_val: float = 1e6,
    time_val: float = 0.5,
    n_slices: int = 1,
    include_boundary: bool = True,
    include_xpoints: bool = False,
    include_profiles_1d: bool = False,
    inject_sentinel: bool = False,
    beta_pol_val: float = 0.8,
    li_3_val: float = 1.2,
    q95_val: float = 3.5,
):
    """Build a mock equilibrium IDS object."""
    r_grid = np.linspace(4.0, 8.0, n_r)
    z_grid = np.linspace(-2.0, 2.0, n_z)
    r_2d, z_2d = np.meshgrid(r_grid, z_grid, indexing="ij")
    # Circular psi
    r0, z0 = 6.0, 0.0
    dist_sq = (r_2d - r0) ** 2 + (z_2d - z0) ** 2
    psi_2d = psi_axis_val - dist_sq

    if inject_sentinel:
        # Place some sentinel values in the psi field
        psi_2d[0, 0] = 1e30
        psi_2d[-1, -1] = -9e40

    time_arr = np.linspace(time_val, time_val + 0.1 * (n_slices - 1), n_slices)

    slices = []
    for _i in range(n_slices):
        grid = _ns(dim1=r_grid, dim2=z_grid)
        p2d = _ns(grid=grid, psi=psi_2d)
        mag_axis = _ns(r=r0, z=z0)
        gq = _ns(
            psi_axis=psi_axis_val,
            psi_boundary=psi_bnd_val,
            magnetic_axis=mag_axis,
            ip=ip_val,
            beta_pol=beta_pol_val,
            li_3=li_3_val,
            q95=q95_val,
        )

        # Boundary
        if include_boundary:
            theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
            br = r0 + 1.0 * np.cos(theta)
            bz = z0 + 1.0 * np.sin(theta)
            boundary = _ns(outline=_ns(r=br, z=bz), x_point=[])
        else:
            boundary = _ns(outline=_ns(r=np.array([]), z=np.array([])), x_point=[])

        if include_xpoints:
            xp = _ns(r=5.5, z=-1.0)
            boundary.x_point = [xp]

        ts = _ns(
            profiles_2d=[p2d],
            global_quantities=gq,
            boundary=boundary,
        )

        # Optional profiles_1d
        if include_profiles_1d:
            psi_norm = np.linspace(0, 1, 65)
            pressure = np.linspace(1e4, 0, 65)
            j_tor = np.linspace(1e5, 0, 65)
            q_prof = np.linspace(1.0, 5.0, 65)
            dpressure_dpsi = np.gradient(pressure, psi_norm)
            f_df_dpsi = np.zeros(65)
            ts.profiles_1d = _ns(
                psi=psi_norm,
                pressure=pressure,
                j_tor=j_tor,
                q=q_prof,
                dpressure_dpsi=dpressure_dpsi,
                f_df_dpsi=f_df_dpsi,
            )

        slices.append(ts)

    return _ns(time_slice=slices, time=time_arr)


# ---------------------------------------------------------------------------
# extract_slice
# ---------------------------------------------------------------------------
class TestExtractSlice:
    """extract_slice() — single time-slice extraction."""

    def test_basic_extraction(self):
        eq = _make_eq_ids()
        sl = extract_slice(eq, 0)
        assert sl.time == pytest.approx(0.5)
        assert sl.psi_axis == pytest.approx(10.0)
        assert sl.psi_boundary == pytest.approx(2.0)
        assert sl.ip == pytest.approx(1e6)
        assert sl.converged is True

    def test_grid_shapes(self):
        eq = _make_eq_ids(n_r=33, n_z=33)
        sl = extract_slice(eq, 0)
        assert sl.psi_2d.shape == (33, 33)
        assert sl.r_grid.shape == (33,)
        assert sl.z_grid.shape == (33,)

    def test_axis_position(self):
        eq = _make_eq_ids()
        sl = extract_slice(eq, 0)
        assert sl.r_axis == pytest.approx(6.0)
        assert sl.z_axis == pytest.approx(0.0)

    def test_boundary_extracted(self):
        eq = _make_eq_ids(include_boundary=True)
        sl = extract_slice(eq, 0)
        assert sl.boundary_r is not None
        assert sl.boundary_z is not None
        assert len(sl.boundary_r) == 64

    def test_no_boundary(self):
        eq = _make_eq_ids(include_boundary=False)
        sl = extract_slice(eq, 0)
        assert sl.boundary_r is None
        assert sl.boundary_z is None

    def test_xpoints_from_ids(self):
        eq = _make_eq_ids(include_xpoints=True)
        sl = extract_slice(eq, 0)
        assert len(sl.x_points) == 1
        assert sl.x_points[0] == pytest.approx((5.5, -1.0))

    def test_optional_scalars(self):
        eq = _make_eq_ids(beta_pol_val=0.8, li_3_val=1.2, q95_val=3.5)
        sl = extract_slice(eq, 0)
        assert sl.beta_pol == pytest.approx(0.8)
        assert sl.li_3 == pytest.approx(1.2)
        assert sl.q95 == pytest.approx(3.5)

    def test_sentinel_scalars_become_none(self):
        """Sentinel values in global quantities should become None."""
        eq = _make_eq_ids(beta_pol_val=1e30, li_3_val=-9e40, q95_val=1e30)
        sl = extract_slice(eq, 0)
        assert sl.beta_pol is None
        assert sl.li_3 is None
        assert sl.q95 is None

    def test_meshgrid_properties(self):
        """R_2d and Z_2d properties should produce correct meshgrids."""
        eq = _make_eq_ids(n_r=17, n_z=21)
        sl = extract_slice(eq, 0)
        assert sl.R_2d.shape == (17, 21)
        assert sl.Z_2d.shape == (17, 21)

    def test_second_time_index(self):
        """Extract the second slice from a multi-slice IDS."""
        eq = _make_eq_ids(n_slices=3)
        sl = extract_slice(eq, 1)
        assert sl.time == pytest.approx(eq.time[1])


# ---------------------------------------------------------------------------
# extract_time_traces
# ---------------------------------------------------------------------------
class TestExtractTimeTraces:
    """extract_time_traces() — global scalar time traces."""

    def test_shapes_match(self):
        eq = _make_eq_ids(n_slices=5)
        tt = extract_time_traces(eq)
        assert tt.time.shape == (5,)
        assert tt.ip.shape == (5,)
        assert tt.beta_pol.shape == (5,)
        assert tt.li_3.shape == (5,)
        assert tt.q95.shape == (5,)

    def test_values(self):
        eq = _make_eq_ids(n_slices=3, ip_val=1e6)
        tt = extract_time_traces(eq)
        assert_allclose(tt.ip, 1e6)

    def test_sentinel_becomes_nan(self):
        """Sentinel ip values should become NaN."""
        eq = _make_eq_ids(n_slices=2, ip_val=1e30)
        tt = extract_time_traces(eq)
        assert np.all(np.isnan(tt.ip))

    def test_converged_defaults_false(self):
        eq = _make_eq_ids(n_slices=3)
        tt = extract_time_traces(eq)
        assert tt.converged.dtype == bool
        assert tt.converged.shape == (3,)


# ---------------------------------------------------------------------------
# extract_profiles_1d
# ---------------------------------------------------------------------------
class TestExtractProfiles1d:
    """extract_profiles_1d() — 1D radial profiles."""

    def test_basic_extraction(self):
        eq = _make_eq_ids(include_profiles_1d=True)
        rp = extract_profiles_1d(eq, 0)
        assert rp.psi_norm.shape == (65,)
        assert rp.pressure.shape == (65,)
        assert rp.j_tor.shape == (65,)
        assert rp.q.shape == (65,)

    def test_normalised_psi(self):
        """If psi is already normalised (max <= 1.5), it should be kept."""
        eq = _make_eq_ids(include_profiles_1d=True)
        rp = extract_profiles_1d(eq, 0)
        assert rp.psi_norm[0] == pytest.approx(0.0)
        assert rp.psi_norm[-1] == pytest.approx(1.0)

    def test_unnormalised_psi_gets_normalised(self):
        """If psi > 1.5, it should be normalised using psi_axis/psi_bnd."""
        eq = _make_eq_ids(
            include_profiles_1d=True,
            psi_axis_val=10.0,
            psi_bnd_val=2.0,
        )
        # Override psi in profiles_1d to be un-normalised
        ts = eq.time_slice[0]
        ts.profiles_1d.psi = np.linspace(2.0, 10.0, 65)
        rp = extract_profiles_1d(eq, 0)
        assert rp.psi_norm[0] == pytest.approx(0.0)
        assert rp.psi_norm[-1] == pytest.approx(1.0)

    def test_sentinel_in_profile_becomes_nan(self):
        """Sentinel values in pressure should become NaN."""
        eq = _make_eq_ids(include_profiles_1d=True)
        ts = eq.time_slice[0]
        ts.profiles_1d.pressure[10] = 1e30
        ts.profiles_1d.pressure[20] = -9e40
        rp = extract_profiles_1d(eq, 0)
        assert np.isnan(rp.pressure[10])
        assert np.isnan(rp.pressure[20])

    def test_missing_attribute_returns_nan_array(self):
        """If an attribute is missing, a NaN-filled array should be returned."""
        eq = _make_eq_ids(include_profiles_1d=True)
        ts = eq.time_slice[0]
        del ts.profiles_1d.j_tor  # remove the attribute
        rp = extract_profiles_1d(eq, 0)
        assert np.all(np.isnan(rp.j_tor))

    def test_time_extracted(self):
        eq = _make_eq_ids(include_profiles_1d=True, time_val=1.5)
        rp = extract_profiles_1d(eq, 0)
        assert rp.time == pytest.approx(1.5)
