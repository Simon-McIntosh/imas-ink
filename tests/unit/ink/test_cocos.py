"""Tests for efit.ink._cocos — COCOS 17 flux level utilities."""

from __future__ import annotations

import numpy as np
from numpy.testing import assert_allclose

from imas_ink._cocos import make_levels


class TestMakeLevelsBasic:
    """Standard level generation."""

    def test_three_levels(self):
        """make_levels(1.0, 0.0, 3) → [0.25, 0.5, 0.75]."""
        levels = make_levels(1.0, 0.0, n=3)
        assert_allclose(levels, [0.25, 0.5, 0.75])

    def test_six_levels(self):
        """make_levels(1.0, 0.0, 6) → 6 values strictly between 0 and 1."""
        levels = make_levels(1.0, 0.0, n=6)
        assert len(levels) == 6
        assert np.all(levels > 0.0)
        assert np.all(levels < 1.0)
        expected = np.array([1 / 7, 2 / 7, 3 / 7, 4 / 7, 5 / 7, 6 / 7])
        assert_allclose(levels, expected)

    def test_single_level(self):
        """n=1 → single midpoint level."""
        levels = make_levels(1.0, 0.0, n=1)
        assert len(levels) == 1
        assert_allclose(levels, [0.5])

    def test_zero_levels(self):
        """n=0 → empty array."""
        levels = make_levels(1.0, 0.0, n=0)
        assert len(levels) == 0
        assert isinstance(levels, np.ndarray)


class TestMakeLevelsCOCOS17:
    """COCOS 17 sign convention: psi_axis > psi_boundary."""

    def test_reversed_sign(self):
        """psi_axis=10, psi_bnd=2 → levels between 2 and 10."""
        levels = make_levels(psi_axis=10.0, psi_bnd=2.0, n=4)
        assert len(levels) == 4
        assert np.all(levels > 2.0)
        assert np.all(levels < 10.0)
        # dpsi = (10 - 2) / 5 = 1.6
        expected = np.array([3.6, 5.2, 6.8, 8.4])
        assert_allclose(levels, expected)

    def test_negative_psi_values(self):
        """Negative psi range: psi_axis=-1, psi_bnd=-5."""
        levels = make_levels(psi_axis=-1.0, psi_bnd=-5.0, n=3)
        assert len(levels) == 3
        assert np.all(levels > -5.0)
        assert np.all(levels < -1.0)


class TestMakeLevelsSorting:
    """Levels must always be sorted."""

    def test_sorted_ascending(self):
        """Output is always sorted regardless of psi_axis vs psi_bnd sign."""
        levels = make_levels(1.0, 0.0, n=5)
        assert np.all(np.diff(levels) > 0)

    def test_sorted_when_axis_less_than_bnd(self):
        """When psi_axis < psi_bnd (non-COCOS-17), still sorted."""
        levels = make_levels(psi_axis=0.0, psi_bnd=1.0, n=5)
        assert np.all(np.diff(levels) > 0)


class TestMakeLevelsEdgeCases:
    """Edge cases and degenerate inputs."""

    def test_axis_equals_boundary(self):
        """When psi_axis == psi_bnd, dpsi=0 → all levels identical."""
        levels = make_levels(5.0, 5.0, n=3)
        assert len(levels) == 3
        assert_allclose(levels, [5.0, 5.0, 5.0])

    def test_large_n(self):
        """Large number of levels: verify count and bounds."""
        levels = make_levels(1.0, 0.0, n=100)
        assert len(levels) == 100
        assert np.all(levels > 0.0)
        assert np.all(levels < 1.0)

    def test_return_type(self):
        """Result is always a numpy array."""
        levels = make_levels(1.0, 0.0, n=3)
        assert isinstance(levels, np.ndarray)
