"""Tests for efit.ink._sentinel — EMPTY value guards."""

from __future__ import annotations

import math

import numpy as np
import pytest
from numpy.testing import assert_array_equal

from imas_ink._sentinel import EMPTY_THRESHOLD, is_empty, safe_float


# ---------------------------------------------------------------------------
# is_empty — scalar cases
# ---------------------------------------------------------------------------
class TestIsEmptyScalar:
    """Scalar inputs to is_empty()."""

    @pytest.mark.parametrize(
        "value, expected",
        [
            (0, False),
            (1.0, False),
            (-42.5, False),
            (1e8, False),  # below default threshold (1e9)
            (-1e10, True),  # above threshold
            (1e30, True),
            (-9e40, True),  # classic IMAS EMPTY_DOUBLE
        ],
    )
    def test_scalar_cases(self, value, expected):
        assert is_empty(value) == expected

    def test_nan_is_not_empty(self):
        """NaN has |NaN| → NaN, and NaN > tol is False."""
        assert is_empty(float("nan")) is np.False_

    def test_inf_is_empty(self):
        """±Inf is above any finite threshold."""
        assert is_empty(float("inf"))
        assert is_empty(float("-inf"))


# ---------------------------------------------------------------------------
# is_empty — array cases
# ---------------------------------------------------------------------------
class TestIsEmptyArray:
    """Array inputs to is_empty()."""

    def test_mixed_array(self):
        arr = np.array([0.0, 1e8, 1e30, -9e40, 3.14])
        result = is_empty(arr)
        expected = np.array([False, False, True, True, False])
        assert_array_equal(result, expected)

    def test_all_normal(self):
        arr = np.array([1.0, 2.0, 3.0])
        assert not np.any(is_empty(arr))

    def test_all_empty(self):
        arr = np.array([1e30, -9e40, 1e11])
        assert np.all(is_empty(arr))

    def test_empty_array(self):
        arr = np.array([])
        result = is_empty(arr)
        assert result.size == 0


# ---------------------------------------------------------------------------
# is_empty — custom tolerance
# ---------------------------------------------------------------------------
class TestIsEmptyCustomTol:
    """Custom tolerance parameter."""

    def test_lower_threshold(self):
        """With tol=100, values above 100 are flagged."""
        assert is_empty(200.0, tol=100)
        assert not is_empty(50.0, tol=100)

    def test_higher_threshold(self):
        """With tol=1e20, the default sentinels at 1e10 are not flagged."""
        assert not is_empty(1e10, tol=1e20)
        assert is_empty(1e30, tol=1e20)


# ---------------------------------------------------------------------------
# safe_float — normal operation
# ---------------------------------------------------------------------------
class TestSafeFloat:
    """safe_float() scalar extraction."""

    def test_valid_value(self):
        assert safe_float(1.23) == 1.23

    def test_zero(self):
        assert safe_float(0.0) == 0.0

    def test_negative(self):
        assert safe_float(-42.5) == -42.5

    def test_sentinel_returns_nan(self):
        result = safe_float(-9e40)
        assert math.isnan(result)

    def test_large_positive_returns_nan(self):
        result = safe_float(1e30)
        assert math.isnan(result)

    def test_just_below_threshold(self):
        """Value just below threshold should be returned."""
        val = EMPTY_THRESHOLD * 0.99
        assert safe_float(val) == val

    def test_just_above_threshold(self):
        """Value just above threshold should return default."""
        val = EMPTY_THRESHOLD * 1.01
        assert math.isnan(safe_float(val))


# ---------------------------------------------------------------------------
# safe_float — default and error handling
# ---------------------------------------------------------------------------
class TestSafeFloatEdgeCases:
    """Edge cases: TypeError, custom default, custom tol."""

    def test_none_returns_default(self):
        result = safe_float(None)
        assert math.isnan(result)

    def test_none_custom_default(self):
        assert safe_float(None, default=-1.0) == -1.0

    def test_string_returns_default(self):
        result = safe_float("not_a_number")
        assert math.isnan(result)

    def test_custom_default_on_sentinel(self):
        assert safe_float(-9e40, default=0.0) == 0.0

    def test_custom_tol(self):
        """With tol=100, 200 is treated as EMPTY."""
        assert safe_float(200.0, tol=100) != 200.0
        assert safe_float(50.0, tol=100) == 50.0

    def test_int_input(self):
        """Integer input should be converted to float."""
        assert safe_float(42) == 42.0
        assert isinstance(safe_float(42), float)

    def test_numpy_scalar(self):
        """np.float64 should work like a regular float."""
        assert safe_float(np.float64(3.14)) == pytest.approx(3.14)

    def test_list_raises_type_error(self):
        """A list cannot be converted to float — returns default."""
        result = safe_float([1, 2, 3])
        assert math.isnan(result)
