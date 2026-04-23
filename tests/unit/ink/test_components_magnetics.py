"""Tests for MagneticProbes and FluxLoops component dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ink.components import FluxLoops, MagneticProbes
from imas_ink.style import DEFAULT_STYLE, InkStyle


class TestMagneticProbes:
    """MagneticProbes — B-pol probe positions and angles."""

    def test_construct(self):
        r = np.array([3.0, 4.0, 5.0])
        z = np.array([-1.0, 0.0, 1.0])
        angles = np.array([0.0, np.pi / 2, np.nan])
        mp = MagneticProbes(positions_r=r, positions_z=z, angles=angles)
        assert len(mp.positions_r) == 3
        assert len(mp.positions_z) == 3
        assert len(mp.angles) == 3

    def test_default_style(self):
        mp = MagneticProbes(
            positions_r=np.array([]),
            positions_z=np.array([]),
            angles=np.array([]),
        )
        assert mp.style is DEFAULT_STYLE

    def test_custom_style(self):
        custom = InkStyle(probe_color="#ff0000")
        mp = MagneticProbes(
            positions_r=np.array([1.0]),
            positions_z=np.array([0.0]),
            angles=np.array([0.0]),
            style=custom,
        )
        assert mp.style.probe_color == "#ff0000"

    def test_nan_angles_preserved(self):
        """NaN angles must round-trip through the dataclass."""
        angles = np.array([np.nan, 1.5, np.nan, 0.0])
        mp = MagneticProbes(
            positions_r=np.zeros(4),
            positions_z=np.zeros(4),
            angles=angles,
        )
        assert np.isnan(mp.angles[0])
        assert np.isnan(mp.angles[2])
        assert mp.angles[1] == pytest.approx(1.5)
        assert mp.angles[3] == pytest.approx(0.0)

    def test_mutable(self):
        """Components are not frozen — style can be reassigned."""
        mp = MagneticProbes(
            positions_r=np.array([]),
            positions_z=np.array([]),
            angles=np.array([]),
        )
        custom = InkStyle(probe_color="#abcdef")
        mp.style = custom
        assert mp.style.probe_color == "#abcdef"

    def test_empty_arrays(self):
        mp = MagneticProbes(
            positions_r=np.array([]),
            positions_z=np.array([]),
            angles=np.array([]),
        )
        assert len(mp.positions_r) == 0
        assert len(mp.positions_z) == 0
        assert len(mp.angles) == 0


class TestFluxLoops:
    """FluxLoops — flux loop marker positions."""

    def test_construct(self):
        r = np.array([3.5, 4.5])
        z = np.array([-2.0, 2.0])
        fl = FluxLoops(positions_r=r, positions_z=z)
        assert len(fl.positions_r) == 2
        assert len(fl.positions_z) == 2

    def test_default_style(self):
        fl = FluxLoops(positions_r=np.array([]), positions_z=np.array([]))
        assert fl.style is DEFAULT_STYLE

    def test_custom_style(self):
        custom = InkStyle(flux_loop_color="#00ff00")
        fl = FluxLoops(
            positions_r=np.array([1.0]),
            positions_z=np.array([0.0]),
            style=custom,
        )
        assert fl.style.flux_loop_color == "#00ff00"

    def test_mutable(self):
        fl = FluxLoops(positions_r=np.array([]), positions_z=np.array([]))
        custom = InkStyle(flux_loop_color="#123456")
        fl.style = custom
        assert fl.style.flux_loop_color == "#123456"

    def test_empty_arrays(self):
        fl = FluxLoops(positions_r=np.array([]), positions_z=np.array([]))
        assert len(fl.positions_r) == 0
        assert len(fl.positions_z) == 0
