"""COCOS 17 flux level utilities.

COCOS 17: psi_axis > psi_boundary (flux increases inward).
Interior flux surfaces: psi_boundary + k * dpsi,
  k = 1..n, dpsi = (psi_axis - psi_boundary) / (n+1).
"""

from __future__ import annotations

import numpy as np


def make_levels(psi_axis: float, psi_bnd: float, n: int = 6) -> np.ndarray:
    """Generate *n* interior flux levels between psi_boundary and psi_axis.

    Uses COCOS 17 convention where ``psi_axis > psi_boundary``.
    Levels are evenly spaced between the LCFS and the magnetic axis,
    excluding the boundary and axis values themselves.

    Parameters
    ----------
    psi_axis : float
        Poloidal flux at the magnetic axis.
    psi_bnd : float
        Poloidal flux at the last closed flux surface (boundary).
    n : int
        Number of interior levels to generate.

    Returns
    -------
    np.ndarray
        Sorted array of *n* flux levels strictly inside the LCFS.

    Examples
    --------
    >>> make_levels(1.0, 0.0, n=3)
    array([0.25, 0.5 , 0.75])
    """
    dpsi = (psi_axis - psi_bnd) / (n + 1)
    levels = psi_bnd + np.arange(1, n + 1) * dpsi
    return np.sort(levels)
