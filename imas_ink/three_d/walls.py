"""Typed wall-extraction and revolution for first-wall / vessel geometry.

Extracts 2D RZ outlines from the IMAS ``wall`` IDS and revolves them
into closed 3D manifold meshes.  Replaces the mixed wall logic that was
previously embedded in :mod:`imas_ink.three_d.coilset`.

All heavy imports (``pyvista``, ``numpy`` beyond basic) are at function
scope so that ``import imas_ink`` never pulls in VTK.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .._types import VesselShell, WallOutline2D
from ..extract import extract_vessel_shells as extract_vessel_shells
from ..geometry import _offset_polygon

if TYPE_CHECKING:
    import numpy as np
    import pyvista as pv


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class FirstWall(WallOutline2D):
    """Plasma-facing first-wall (limiter) contour."""


# ------------------------------------------------------------------
# Outline closure utility
# ------------------------------------------------------------------


def close_or_reject_outline(
    r,
    z,
    *,
    tol: float = 1e-6,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Close an RZ outline if the gap is within tolerance.

    If the first and last points already coincide (within *tol* times
    the bounding-box diagonal), the outline is returned unchanged with
    ``was_closed=True``.  If the gap is small enough, the first point is
    appended to close the polygon and ``was_closed=True`` is returned.

    Parameters
    ----------
    r, z : array_like
        Vertices of the 2D outline.
    tol : float
        Maximum allowed gap as a fraction of the bounding-box diagonal.
        Defaults to ``1e-6``.

    Returns
    -------
    r_closed, z_closed : np.ndarray
        The (possibly extended) outline arrays.
    was_closed : bool
        ``True`` if the outline is now closed.

    Raises
    ------
    ValueError
        If the gap exceeds the tolerance threshold.
    """
    import numpy as np

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    if r.size < 2:
        raise ValueError("Outline must have at least 2 points")

    gap = float(np.hypot(r[-1] - r[0], z[-1] - z[0]))
    bbox_diag = float(
        np.hypot(r.max() - r.min(), z.max() - z.min())
    )
    threshold = tol * bbox_diag if bbox_diag > 0 else tol

    if gap <= threshold:
        if gap == 0.0:
            return r, z, True
        # Close by appending the first point
        return np.append(r, r[0]), np.append(z, z[0]), True

    raise ValueError(
        f"Outline gap ({gap:.6g} m) exceeds tolerance "
        f"({threshold:.6g} m = {tol} × bbox_diag {bbox_diag:.6g} m). "
        f"First point: ({r[0]:.6g}, {z[0]:.6g}), "
        f"last point: ({r[-1]:.6g}, {z[-1]:.6g})."
    )


# ------------------------------------------------------------------
# IMAS wall extractors
# ------------------------------------------------------------------


def extract_first_wall(wall_ids) -> FirstWall | None:
    """Extract the first-wall (limiter) outline from a ``wall`` IDS.

    Reads ``wall.description_2d[0].limiter.unit[0].outline``.

    Parameters
    ----------
    wall_ids
        ``wall`` IDS object.

    Returns
    -------
    FirstWall or None
        ``None`` if the limiter outline is absent or too small.
    """
    import numpy as np

    try:
        outline = wall_ids.description_2d[0].limiter.unit[0].outline
        r = np.asarray(outline.r, dtype=float)
        z = np.asarray(outline.z, dtype=float)
    except (AttributeError, IndexError, TypeError):
        return None

    if r.size < 3:
        return None

    # Determine closure
    gap = float(np.hypot(r[-1] - r[0], z[-1] - z[0]))
    is_closed = gap < 1e-10

    return FirstWall(r=r, z=z, name="first_wall", is_closed=is_closed)


# ------------------------------------------------------------------
# 3D revolution
# ------------------------------------------------------------------


def revolve_wall_outline(
    outline: WallOutline2D,
    *,
    n_theta: int = 96,
    name: str | None = None,
) -> pv.PolyData:
    """Revolve a 2D wall outline 360° about the Z axis.

    The resulting mesh is validated as a closed manifold via
    :func:`~imas_ink.three_d.manifold.ensure_closed_manifold`.

    Parameters
    ----------
    outline : WallOutline2D
        The 2D RZ polygon to revolve.
    n_theta : int
        Number of azimuthal steps (default 96).
    name : str or None
        Mesh name for error messages.  Defaults to ``outline.name``.

    Returns
    -------
    pyvista.PolyData
        Manifold-validated 3D surface mesh.

    Raises
    ------
    MeshNotManifoldError
        If the revolved mesh cannot be repaired to a closed manifold.
    """
    from .manifold import ensure_closed_manifold
    from .primitives import revolve_polygon

    mesh = revolve_polygon(outline.r, outline.z, n_theta=n_theta)
    label = name or outline.name
    return ensure_closed_manifold(mesh, name=label)


# ------------------------------------------------------------------
# Synthetic vessel shell
# ------------------------------------------------------------------


def synthesize_vessel_shell(
    first_wall: FirstWall,
    *,
    offset: float = 0.4,
    name: str = "synthetic_vessel",
) -> VesselShell:
    """Create a synthetic vessel outline by offsetting the first wall.

    **Demo-only synthetic approximation** — NOT physical truth.  This is
    intended solely for visualisation demos (e.g. ITER datasets that lack
    explicit vessel data).  The offset is a simple 2D outward-normal
    displacement of each vertex; the resulting polygon may self-intersect
    for highly concave outlines.

    Parameters
    ----------
    first_wall : FirstWall
        The plasma-facing contour to offset outward.
    offset : float
        Outward offset distance in metres (default 0.4 m).
    name : str
        Name for the resulting :class:`VesselShell`.

    Returns
    -------
    VesselShell
        Synthetic vessel outline enclosing the first wall.
    """
    r_off, z_off = _offset_polygon(first_wall.r, first_wall.z, offset)

    gap = float(
        __import__("numpy").hypot(r_off[-1] - r_off[0], z_off[-1] - z_off[0])
    )
    is_closed = gap < 1e-10

    return VesselShell(r=r_off, z=z_off, name=name, is_closed=is_closed)
