"""Equilibrium IDS reader for 3D flux projection.

This module provides lightweight helpers for reading an ``equilibrium`` IDS
from an IMAS HDF5 database and extracting the per-timeslice ψ(R, Z) grid into
a frozen dataclass ready for 3D flux projection.

All heavy imports (``imas``, ``scipy``) are deferred to function bodies so
that ``import imas_ink.three_d`` never pulls them eagerly.

EMPTY_DOUBLE sentinel filtering
-------------------------------
The IMAS Access Layer fills missing optional scalars with ``EMPTY_DOUBLE =
-9.0E40``.  Every optional scalar extracted by :func:`extract_slice_2d` is
passed through :func:`imas_ink._sentinel.safe_float`, which returns ``NaN``
for sentinel values.  Downstream code should treat ``NaN`` as "not
available".
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from .._sentinel import safe_float

if TYPE_CHECKING:
    pass


# ---------------------------------------------------------------------------
# Public dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EquilibriumSlice2D:
    """Single time-slice ψ(R, Z) ready for 3D projection.

    All arrays use ``indexing="ij"`` convention (first axis → R, second → Z).
    Optional scalar fields are ``NaN`` when the IDS did not supply them or
    when the IMAS Access Layer filled them with the ``EMPTY_DOUBLE`` sentinel.

    Attributes
    ----------
    time:
        Reconstruction time in seconds.
    psi_axis:
        Poloidal flux at the magnetic axis [Wb/rad].  ``NaN`` if absent.
    psi_boundary:
        Poloidal flux at the plasma boundary [Wb/rad].  ``NaN`` if absent.
    R_1d:
        Radial coordinates of the ψ grid [m], shape ``(nR,)``.
    Z_1d:
        Vertical coordinates of the ψ grid [m], shape ``(nZ,)``.
    psi_2d:
        Poloidal flux on the (R, Z) grid [Wb/rad], shape ``(nR, nZ)``.
    boundary_r:
        R-coordinates of the plasma boundary outline [m].  Empty array if
        not present in the IDS.
    boundary_z:
        Z-coordinates of the plasma boundary outline [m].  Empty array if
        not present in the IDS.
    o_point:
        ``(R, Z)`` of the magnetic axis in metres, or ``None`` if absent or
        sentinel-filled.
    x_points:
        Tuple of ``(R, Z)`` pairs for X-points.  Empty tuple if none are
        present.
    """

    time: float
    psi_axis: float
    psi_boundary: float
    R_1d: np.ndarray  # shape (nR,)
    Z_1d: np.ndarray  # shape (nZ,)
    psi_2d: np.ndarray  # shape (nR, nZ), indexing="ij"
    boundary_r: np.ndarray  # 1D, possibly empty
    boundary_z: np.ndarray  # 1D, possibly empty
    o_point: tuple[float, float] | None  # (R, Z) of magnetic axis
    x_points: tuple[tuple[float, float], ...]  # tuples of (R, Z)


# ---------------------------------------------------------------------------
# Reader
# ---------------------------------------------------------------------------


def read_equilibrium(uri: str, *, dd_version: str | None = None):
    """Open an IMAS database and return the ``equilibrium`` IDS.

    The URI must point to a directory-based IMAS HDF5 database.  For the
    HDF5 backend the trailing ``/`` is required, e.g.::

        "imas:hdf5?path=/path/to/run_set_1/"

    The DD version is resolved via :func:`imas_ink._dd.resolve_dd_version`
    (precedence: *dd_version* kwarg > ``IMAS_VERSION`` env var >
    ``DEFAULT_DD_VERSION``).

    Parameters
    ----------
    uri:
        IMAS URI string.  Must be in the form accepted by
        ``imas.DBEntry``, e.g. ``"imas:hdf5?path=.../"``  The trailing
        ``/`` is required for the HDF5 backend.
    dd_version:
        Override the Data Dictionary version.  When ``None``, the version is
        resolved from the ``IMAS_VERSION`` environment variable or the
        package default ``"4.1.0"``.

    Returns
    -------
    imas.IDSToplevel
        The ``equilibrium`` IDS populated from the database.

    Raises
    ------
    RuntimeError
        If ``imas`` is not importable.
    """
    import imas  # lazy import — do not pull at module level

    from .._dd import resolve_dd_version

    dd = resolve_dd_version(dd_version)
    entry = imas.DBEntry(uri, "r", dd_version=dd)
    try:
        eq_ids = entry.get("equilibrium")
    finally:
        entry.close()
    return eq_ids


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


def extract_slice_2d(eq_ids, time_index: int = 0) -> EquilibriumSlice2D:
    """Extract a single time-slice from an ``equilibrium`` IDS.

    Reads ``profiles_2d[0]`` for the ψ grid and R/Z 1-D coordinate
    arrays (``grid.dim1`` / ``grid.dim2``), ``global_quantities`` for
    psi_axis, psi_boundary and magnetic axis, and ``boundary`` for the
    plasma outline and X-points.

    EMPTY_DOUBLE sentinels (``|v| > 1e9``) are converted to ``NaN`` for
    all optional scalar fields.  Array fields are returned as-is (callers
    should apply their own masks if needed).

    Parameters
    ----------
    eq_ids:
        ``equilibrium`` IDS object.
    time_index:
        Index into ``time_slice`` to extract (default: 0).

    Returns
    -------
    EquilibriumSlice2D
        Frozen dataclass with the extracted data.
    """
    ts = eq_ids.time_slice[time_index]
    p2d = ts.profiles_2d[0]

    r_1d = np.asarray(p2d.grid.dim1, dtype=float)
    z_1d = np.asarray(p2d.grid.dim2, dtype=float)
    psi_2d = np.asarray(p2d.psi, dtype=float)

    # -- global quantities ---------------------------------------------------
    gq = ts.global_quantities
    psi_axis = safe_float(gq.psi_axis)
    psi_boundary = safe_float(gq.psi_boundary)

    try:
        r_axis = safe_float(gq.magnetic_axis.r)
        z_axis = safe_float(gq.magnetic_axis.z)
    except AttributeError:
        r_axis = float("nan")
        z_axis = float("nan")

    if np.isnan(r_axis) or np.isnan(z_axis):
        o_point = None
    else:
        o_point = (r_axis, z_axis)

    # -- time ----------------------------------------------------------------
    try:
        time = float(eq_ids.time[time_index])
    except (AttributeError, IndexError, TypeError):
        time = float("nan")

    # -- boundary outline ----------------------------------------------------
    boundary_r: np.ndarray = np.empty(0)
    boundary_z: np.ndarray = np.empty(0)
    try:
        br = np.asarray(ts.boundary.outline.r, dtype=float)
        bz = np.asarray(ts.boundary.outline.z, dtype=float)
        if br.size > 0 and bz.size > 0:
            boundary_r = br
            boundary_z = bz
    except (AttributeError, IndexError):
        pass

    # -- X-points ------------------------------------------------------------
    x_points: list[tuple[float, float]] = []
    try:
        for xp in ts.boundary.x_point:
            r_x = safe_float(xp.r)
            z_x = safe_float(xp.z)
            if not (np.isnan(r_x) or np.isnan(z_x)):
                x_points.append((r_x, z_x))
    except (AttributeError, IndexError, TypeError):
        pass

    return EquilibriumSlice2D(
        time=time,
        psi_axis=psi_axis,
        psi_boundary=psi_boundary,
        R_1d=r_1d,
        Z_1d=z_1d,
        psi_2d=psi_2d,
        boundary_r=boundary_r,
        boundary_z=boundary_z,
        o_point=o_point,
        x_points=tuple(x_points),
    )


# ---------------------------------------------------------------------------
# Interpolator
# ---------------------------------------------------------------------------


def psi_grid_interpolator(slice_2d: EquilibriumSlice2D):
    """Return a scipy RegularGridInterpolator over ψ(R, Z).

    Points outside the grid are filled with ``NaN`` (``bounds_error=False``,
    ``fill_value=np.nan``).

    Parameters
    ----------
    slice_2d:
        Extracted equilibrium slice.

    Returns
    -------
    scipy.interpolate.RegularGridInterpolator
        Interpolator accepting ``(..., 2)`` arrays of ``[R, Z]`` pairs.
    """
    from scipy.interpolate import RegularGridInterpolator  # lazy import

    return RegularGridInterpolator(
        (slice_2d.R_1d, slice_2d.Z_1d),
        slice_2d.psi_2d,
        method="linear",
        bounds_error=False,
        fill_value=np.nan,
    )
