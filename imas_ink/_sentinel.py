"""IMAS EMPTY value guard.

The IMAS Access Layer fills missing data with sentinel values:
  - EMPTY_DOUBLE = -9.0E40  (Fortran AL)
  - Convention: |v| > 1e30 is treated as EMPTY

All sentinel filtering happens in extract.py. Downstream code (geometry,
contours, renderers) never encounters sentinels.
"""

from __future__ import annotations

import numpy as np

EMPTY_THRESHOLD: float = 1e9


def is_empty(value, tol: float = EMPTY_THRESHOLD):
    """Return True where |value| > tol.

    Works on scalars and numpy arrays alike.

    Parameters
    ----------
    value : scalar or array_like
        Value(s) to test.
    tol : float
        Threshold above which a value is considered EMPTY.

    Returns
    -------
    bool or np.ndarray of bool

    Examples
    --------
    >>> is_empty(-9e40)
    True
    >>> is_empty(3.14)
    False
    >>> is_empty(np.array([1.0, 1e30, -9e40]))
    array([False,  True,  True])
    """
    return np.abs(value) > tol


def safe_float(value, default: float = float("nan"), tol: float = EMPTY_THRESHOLD) -> float:
    """Extract a scalar, returning *default* if it is an EMPTY sentinel.

    Parameters
    ----------
    value : scalar
        Raw value from an IDS field.
    default : float
        Fallback when *value* is EMPTY or cannot be converted.
    tol : float
        Threshold for the EMPTY test.

    Returns
    -------
    float

    Examples
    --------
    >>> safe_float(-9e40)
    nan
    >>> safe_float(1.23)
    1.23
    """
    try:
        v = float(value)
    except (TypeError, ValueError):
        return default
    return default if abs(v) > tol else v
