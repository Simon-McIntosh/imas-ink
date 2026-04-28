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

if TYPE_CHECKING:
    import numpy as np
    import pyvista as pv


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class WallOutline2D:
    """A 2D RZ polygon (or polyline) for one wall component."""

    r: np.ndarray  # 1D
    z: np.ndarray  # 1D
    name: str
    is_closed: bool  # True if last point == first


@dataclass(frozen=True)
class FirstWall(WallOutline2D):
    """Plasma-facing first-wall (limiter) contour."""


@dataclass(frozen=True)
class VesselShell(WallOutline2D):
    """Vacuum vessel inner/outer shell or single-line approximation."""


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


def extract_vessel_shells(wall_ids) -> list[VesselShell]:
    """Extract vessel shell outlines from a ``wall`` IDS.

    Iterates ``wall.description_2d[0].vessel.unit[*]`` and reads:

    - ``annular.centreline`` (r, z) with optional ``thickness`` for
      annular vessel representations.
    - Falls back to ``outline`` if annular data is absent.

    Parameters
    ----------
    wall_ids
        ``wall`` IDS object.

    Returns
    -------
    list[VesselShell]
        One per vessel unit.  Empty list if no vessel data.
    """
    import numpy as np

    shells: list[VesselShell] = []

    try:
        desc = wall_ids.description_2d[0]
        units = desc.vessel.unit
    except (AttributeError, IndexError, TypeError):
        return shells

    for i, unit in enumerate(units):
        r: np.ndarray | None = None
        z: np.ndarray | None = None

        # Try annular centreline first
        try:
            ann = unit.annular
            r_cl = np.asarray(ann.centreline.r, dtype=float)
            z_cl = np.asarray(ann.centreline.z, dtype=float)
            if r_cl.size >= 3:
                thickness = _safe_float(getattr(ann, "thickness", 0.0))
                if thickness > 0:
                    r_outer, z_outer = _offset_polygon(
                        r_cl, z_cl, thickness / 2
                    )
                    r_inner, z_inner = _offset_polygon(
                        r_cl, z_cl, -thickness / 2
                    )
                    r = np.concatenate([r_outer, r_inner[::-1]])
                    z = np.concatenate([z_outer, z_inner[::-1]])
                else:
                    r, z = r_cl, z_cl
        except (AttributeError, TypeError):
            pass

        # Fallback: outline
        if r is None:
            try:
                outline = unit.outline
                r = np.asarray(outline.r, dtype=float)
                z = np.asarray(outline.z, dtype=float)
            except (AttributeError, TypeError):
                continue

        if r is None or r.size < 3:
            continue

        gap = float(np.hypot(r[-1] - r[0], z[-1] - z[0]))
        is_closed = gap < 1e-10

        shells.append(
            VesselShell(
                r=r,
                z=z,
                name=f"vessel_{i}",
                is_closed=is_closed,
            )
        )

    return shells


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


# ------------------------------------------------------------------
# Private helpers (migrated from coilset.py)
# ------------------------------------------------------------------


def _offset_polygon(
    r, z, offset: float
) -> tuple[np.ndarray, np.ndarray]:
    """Naïve inward/outward offset of a 2D polygon.

    Moves each vertex along the local outward normal by *offset*.
    Positive offset = outward, negative = inward.
    """
    import numpy as np

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)
    n = len(r)
    r_off = np.empty(n)
    z_off = np.empty(n)
    for i in range(n):
        i_prev = (i - 1) % n
        i_next = (i + 1) % n
        dr = r[i_next] - r[i_prev]
        dz = z[i_next] - z[i_prev]
        length = np.hypot(dr, dz)
        if length < 1e-12:
            r_off[i] = r[i]
            z_off[i] = z[i]
        else:
            # Outward normal (assuming CCW winding)
            r_off[i] = r[i] + offset * dz / length
            z_off[i] = z[i] - offset * dr / length
    return r_off, z_off


def _safe_float(value, default: float = 0.0) -> float:
    """Convert *value* to float, falling back to *default*."""
    try:
        v = float(value)
        return v if abs(v) < 1e30 else default
    except (TypeError, ValueError):
        return default
