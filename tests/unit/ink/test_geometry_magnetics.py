"""Tests for MachineGeometry magnetics extraction from IMAS data."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ink._types import MachineGeometry


class TestMachineGeometryMagneticsDefaults:
    """MachineGeometry defaults — magnetics arrays are empty when not provided."""

    def test_default_probe_arrays_empty(self):
        """Probe arrays default to empty when not explicitly passed."""
        geom = MachineGeometry(
            wall_r=np.array([1.0, 2.0]),
            wall_z=np.array([0.0, 1.0]),
            coil_rects=[],
            wall_clip_vertices=np.array([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]),
        )
        assert len(geom.probe_r) == 0
        assert len(geom.probe_z) == 0
        assert len(geom.probe_angle) == 0

    def test_default_flux_loop_arrays_empty(self):
        """Flux loop arrays default to empty when not explicitly passed."""
        geom = MachineGeometry(
            wall_r=np.array([1.0, 2.0]),
            wall_z=np.array([0.0, 1.0]),
            coil_rects=[],
            wall_clip_vertices=np.array([[1, 0], [2, 0], [2, 1], [1, 1], [1, 0]]),
        )
        assert len(geom.flux_loop_r) == 0
        assert len(geom.flux_loop_z) == 0

    def test_backwards_compatible(self):
        """Existing callers that don't pass magnetics fields still work."""
        # This is the legacy call pattern — must not raise
        geom = MachineGeometry(
            wall_r=np.array([1.0]),
            wall_z=np.array([0.0]),
            coil_rects=[],
            wall_clip_vertices=np.array([[1, 0], [1, 0]]),
        )
        assert geom.wall_r[0] == 1.0
        assert len(geom.probe_r) == 0

    def test_with_magnetics(self):
        """MachineGeometry stores magnetics arrays when passed."""
        pr = np.array([3.0, 4.0, 5.0])
        pz = np.array([-1.0, 0.0, 1.0])
        pa = np.array([0.0, np.pi / 2, np.nan])
        fr = np.array([3.5, 4.5])
        fz = np.array([-2.0, 2.0])
        geom = MachineGeometry(
            wall_r=np.array([1.0]),
            wall_z=np.array([0.0]),
            coil_rects=[],
            wall_clip_vertices=np.array([[1, 0], [1, 0]]),
            probe_r=pr,
            probe_z=pz,
            probe_angle=pa,
            flux_loop_r=fr,
            flux_loop_z=fz,
        )
        assert len(geom.probe_r) == 3
        assert len(geom.flux_loop_r) == 2
        assert np.isnan(geom.probe_angle[2])


# ---------------------------------------------------------------------------
# Integration test with real IMAS data (requires test fixtures)
# ---------------------------------------------------------------------------
@pytest.fixture
def iter_magnetics_geom(iter_imas_data):
    """Load MachineGeometry from ITER 135013 test data (DDv4)."""
    import imas

    from imas_ink.extract import extract_geometry

    uri = f"imas:hdf5?path={iter_imas_data}"
    entry = imas.DBEntry(uri, "r", dd_version=None)
    try:
        wall = entry.get("wall")
        pf = entry.get("pf_active")
        mag = entry.get("magnetics")
    finally:
        entry.close()
    return extract_geometry(wall, pf, mag)


class TestGeometryFromIMAS:
    """Integration tests loading magnetics from real HDF5 data."""

    def test_probes_populated(self, iter_magnetics_geom):
        """Probes are loaded from ITER 135013 magnetics IDS."""
        geom = iter_magnetics_geom
        assert len(geom.probe_r) > 0, "Should find B-pol probes"
        assert len(geom.probe_z) == len(geom.probe_r)
        assert len(geom.probe_angle) == len(geom.probe_r)

    def test_flux_loops_populated(self, iter_magnetics_geom):
        """Flux loops are loaded from ITER 135013 magnetics IDS."""
        geom = iter_magnetics_geom
        assert len(geom.flux_loop_r) > 0, "Should find flux loops"
        assert len(geom.flux_loop_z) == len(geom.flux_loop_r)

    def test_probe_positions_physical(self, iter_magnetics_geom):
        """Probe R coordinates should be physically reasonable for ITER."""
        geom = iter_magnetics_geom
        assert np.all(geom.probe_r > 1.0), "All probe R > 1m"
        assert np.all(geom.probe_r < 12.0), "All probe R < 12m"

    def test_probe_angles_finite_or_nan(self, iter_magnetics_geom):
        """Probe angles are either finite or NaN (no sentinels)."""
        geom = iter_magnetics_geom
        finite_mask = np.isfinite(geom.probe_angle)
        # All finite angles should be in [0, 2π] or [-π, π]
        finite_angles = geom.probe_angle[finite_mask]
        assert np.all(np.abs(finite_angles) < 100), "Angles should not be sentinels"

    def test_no_magnetics_ids_graceful(self, iter_imas_data):
        """extract_geometry without magnetics_ids works (empty arrays)."""
        import imas

        from imas_ink.extract import extract_geometry

        uri = f"imas:hdf5?path={iter_imas_data}"
        entry = imas.DBEntry(uri, "r", dd_version=None)
        try:
            wall = entry.get("wall")
            pf = entry.get("pf_active")
        finally:
            entry.close()
        geom = extract_geometry(wall, pf)  # no magnetics
        assert len(geom.probe_r) == 0
        assert len(geom.flux_loop_r) == 0
