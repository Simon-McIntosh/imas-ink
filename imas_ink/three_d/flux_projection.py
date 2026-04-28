"""Project ψ from a 2D (R, Z) equilibrium onto a 3D cap surface.

Three overlay modes are supported:

- ``contours_and_field`` (default): per-vertex ψ scalar field + contour polylines.
- ``field_only``: per-vertex ψ scalar field, no contour polylines.
- ``contours_only``: contour polylines only, no scalar field.

All overlays should be offset by ε along the cap normal to eliminate
z-fighting between cap face and overlay geometry.  Use
:func:`offset_along_normal` for this.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np

from .._cocos import make_levels

if TYPE_CHECKING:
    import pyvista as pv

    from .equilibrium import EquilibriumSlice2D

FluxMode = Literal["contours_and_field", "field_only", "contours_only"]


@dataclass(frozen=True)
class FluxOverlay:
    """Geometry to render on top of a cap face.

    Attributes
    ----------
    field:
        Per-vertex ψ values on the cap mesh, shape ``(cap.n_points,)``,
        or ``None`` when the mode disables the scalar field.
    contours:
        List of ``pv.PolyData`` polylines (one per level), or empty list.
    levels:
        ψ values used for the contours.
    cap_normal:
        Unit normal of the cap (used for ε offset by the caller).
    """

    field: np.ndarray | None  # shape (cap.n_points,) or None
    contours: list  # list[pv.PolyData] (3D polylines on cap)
    levels: np.ndarray  # 1D array of psi level values
    cap_normal: tuple[float, float, float]


# ---------------------------------------------------------------------------
# ψ sampling on cap
# ---------------------------------------------------------------------------


def sample_psi_on_cap(
    cap_mesh: pv.PolyData,
    slice_2d: EquilibriumSlice2D,
    *,
    axis: str = "y",
) -> np.ndarray:
    """Sample ψ from a 2D equilibrium at every vertex of *cap_mesh*.

    Each cap vertex ``(x, y, z)`` is mapped to cylindrical ``(R, Z)``
    and evaluated via a regular-grid interpolator over the ψ(R, Z) field.
    Out-of-bounds vertices receive ``NaN``.

    Parameters
    ----------
    cap_mesh:
        Cap-face PolyData (e.g. from :func:`~imas_ink.three_d.cutaway.cap_face_of`).
    slice_2d:
        Extracted 2D equilibrium time-slice.
    axis:
        Symmetry axis of the toroidal geometry.  ``"y"`` means the cap lies
        approximately in the ``y ≈ 0`` plane (viewing along ±y).

    Returns
    -------
    np.ndarray
        ψ values at each cap vertex, shape ``(cap_mesh.n_points,)``.

    Raises
    ------
    NotImplementedError
        For axis values other than ``"y"``.
    """
    from .equilibrium import psi_grid_interpolator

    if axis != "y":
        raise NotImplementedError(f"axis={axis!r} is not yet supported")

    pts = np.asarray(cap_mesh.points, dtype=float)  # (N, 3)
    xi, yi, zi = pts[:, 0], pts[:, 1], pts[:, 2]

    R = np.sqrt(xi**2 + yi**2)
    Z = zi

    interp = psi_grid_interpolator(slice_2d)
    return interp(np.column_stack([R, Z]))


# ---------------------------------------------------------------------------
# Contour extraction + cap clipping
# ---------------------------------------------------------------------------


def _cap_boundary_polygon(cap_mesh: pv.PolyData) -> np.ndarray:
    """Extract the 2D (R, Z) boundary polygon of *cap_mesh*.

    Projects cap vertices to ``(R = |x|, Z = z)`` — assumes the cap lies
    approximately in the ``y ≈ 0`` plane — then computes the 2D convex hull
    via :mod:`scipy.spatial`.  Returns the hull vertices as an ``(M, 2)``
    array with columns ``[R, Z]``, ordered counter-clockwise.
    """
    from scipy.spatial import ConvexHull

    pts = np.asarray(cap_mesh.points, dtype=float)
    rz = np.column_stack([np.sqrt(pts[:, 0] ** 2 + pts[:, 1] ** 2), pts[:, 2]])
    hull = ConvexHull(rz)
    return rz[hull.vertices]


def _clip_segments_to_polygon(
    segments: list[np.ndarray],
    polygon_rz: np.ndarray,
) -> list[np.ndarray]:
    """Keep only the parts of (R, Z) segments inside *polygon_rz*.

    Uses :class:`matplotlib.path.Path` for point-in-polygon testing.
    Each segment is filtered to its in-polygon subset; runs of consecutive
    in-polygon points are kept as separate sub-segments (minimum 2 points).
    """
    from matplotlib.path import Path as MplPath

    # Close the polygon for matplotlib
    closed = np.vstack([polygon_rz, polygon_rz[:1]])
    path = MplPath(closed)

    clipped: list[np.ndarray] = []
    for seg in segments:
        if seg.shape[0] < 2:
            continue
        mask = path.contains_points(seg, radius=0.0)
        # Split into runs of consecutive True values
        run_start = None
        for i, inside in enumerate(mask):
            if inside and run_start is None:
                run_start = i
            elif not inside and run_start is not None:
                if i - run_start >= 2:
                    clipped.append(seg[run_start:i])
                run_start = None
        if run_start is not None and len(mask) - run_start >= 2:
            clipped.append(seg[run_start:])
    return clipped


def _segments_to_polydata(
    segments: list[np.ndarray],
    plane_normal: tuple[float, float, float],
) -> list:
    """Convert (R, Z) segments to 3D pv.PolyData polylines on a plane.

    For the default ``plane_normal=(0, 1, 0)`` (y = 0 plane, RHS view),
    each (R, Z) point maps to ``(R, 0, Z)`` in 3D.
    """
    import pyvista as pv

    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)

    polylines: list[pv.PolyData] = []

    for seg in segments:
        if seg.shape[0] < 2:
            continue

        # Build 3D points: for y-normal, (R, Z) → (R, 0, Z)
        pts_3d = np.zeros((seg.shape[0], 3), dtype=float)
        if np.allclose(np.abs(n), [0, 1, 0]):
            pts_3d[:, 0] = seg[:, 0]  # R → x
            pts_3d[:, 1] = 0.0  # on plane
            pts_3d[:, 2] = seg[:, 1]  # Z → z
        else:
            raise NotImplementedError(
                f"plane_normal={plane_normal!r} is not yet supported"
            )

        n_pts = pts_3d.shape[0]
        # Build line cell: [n_pts, 0, 1, 2, ..., n_pts-1]
        lines = np.empty(n_pts + 1, dtype=int)
        lines[0] = n_pts
        lines[1:] = np.arange(n_pts)

        poly = pv.PolyData(pts_3d, lines=lines)
        polylines.append(poly)

    return polylines


def contours_on_cap(
    cap_mesh: pv.PolyData,
    slice_2d: EquilibriumSlice2D,
    *,
    levels: np.ndarray | None = None,
    n_levels: int = 12,
    plane_normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
    cap_polygon_2d: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[list, np.ndarray]:
    """Extract ψ contour polylines clipped to the cap polygon.

    Uses the existing :class:`~imas_ink.contours.ContourExtractor` over
    the (R, Z) ψ grid, clips to the cap boundary, then embeds as 3D
    polylines on the cap plane.

    Parameters
    ----------
    cap_mesh:
        Cap-face PolyData.
    slice_2d:
        Extracted 2D equilibrium time-slice.
    levels:
        Explicit ψ levels.  If ``None``, compute via
        :func:`~imas_ink._cocos.make_levels`.
    n_levels:
        Number of interior levels when *levels* is ``None``.
    plane_normal:
        Unit normal of the cap plane.
    cap_polygon_2d:
        Optional ``(R_array, Z_array)`` defining the 2D clipping polygon.
        When provided, this polygon is used directly for contour clipping
        instead of deriving the boundary from *cap_mesh* via ConvexHull.
        This is essential for **non-convex** caps (e.g. ITER first-wall
        with divertor region) where the convex hull would incorrectly
        include exterior regions.

    Returns
    -------
    tuple[list[pv.PolyData], np.ndarray]
        ``(contour_polylines, levels)`` where each polyline is a
        ``pv.PolyData`` with line cells.
    """
    from ..contours import ContourExtractor

    if levels is None:
        levels = make_levels(slice_2d.psi_axis, slice_2d.psi_boundary, n=n_levels)
    levels = np.asarray(levels, dtype=float)

    # Build 2D meshgrid for ContourExtractor (indexing="ij")
    R_2d, Z_2d = np.meshgrid(slice_2d.R_1d, slice_2d.Z_1d, indexing="ij")
    cx = ContourExtractor(R_2d, Z_2d, slice_2d.psi_2d)

    # Cap boundary polygon for clipping
    if cap_polygon_2d is not None:
        r_poly, z_poly = cap_polygon_2d
        polygon_rz = np.column_stack(
            [np.asarray(r_poly, dtype=float), np.asarray(z_poly, dtype=float)]
        )
    else:
        polygon_rz = _cap_boundary_polygon(cap_mesh)

    all_polylines: list = []
    for lev in levels:
        segments = cx.lines_at(float(lev))
        clipped = _clip_segments_to_polygon(segments, polygon_rz)
        polylines = _segments_to_polydata(clipped, plane_normal)
        all_polylines.extend(polylines)

    return all_polylines, levels


# ---------------------------------------------------------------------------
# Composer
# ---------------------------------------------------------------------------


def build_flux_overlay(
    cap_mesh: pv.PolyData,
    slice_2d: EquilibriumSlice2D,
    *,
    mode: FluxMode = "contours_and_field",
    n_levels: int = 12,
    levels: np.ndarray | None = None,
    plane_normal: tuple[float, float, float] = (0.0, 1.0, 0.0),
    cap_polygon_2d: tuple[np.ndarray, np.ndarray] | None = None,
) -> FluxOverlay:
    """Build a :class:`FluxOverlay` from an equilibrium slice and cap mesh.

    Composes :func:`sample_psi_on_cap` and :func:`contours_on_cap`
    based on *mode*:

    - ``"contours_and_field"``: both field and contours populated.
    - ``"field_only"``: field populated, contours empty.
    - ``"contours_only"``: contours populated, field is ``None``.

    Parameters
    ----------
    cap_mesh:
        Cap-face PolyData.
    slice_2d:
        Extracted 2D equilibrium time-slice.
    mode:
        Overlay mode.
    n_levels:
        Number of interior contour levels when *levels* is ``None``.
    levels:
        Explicit ψ levels for contours.
    plane_normal:
        Unit normal of the cap plane.
    cap_polygon_2d:
        Optional ``(R_array, Z_array)`` for non-convex cap polygon clipping.
        Forwarded to :func:`contours_on_cap`.

    Returns
    -------
    FluxOverlay
    """
    n = np.asarray(plane_normal, dtype=float)
    n = n / np.linalg.norm(n)
    cap_normal = (float(n[0]), float(n[1]), float(n[2]))

    field: np.ndarray | None = None
    contours: list = []
    resolved_levels = np.empty(0)

    if mode in ("contours_and_field", "field_only"):
        field = sample_psi_on_cap(cap_mesh, slice_2d)

    if mode in ("contours_and_field", "contours_only"):
        contours, resolved_levels = contours_on_cap(
            cap_mesh,
            slice_2d,
            levels=levels,
            n_levels=n_levels,
            plane_normal=plane_normal,
            cap_polygon_2d=cap_polygon_2d,
        )
    else:
        # Compute levels even in field_only mode (for metadata)
        if levels is not None:
            resolved_levels = np.asarray(levels, dtype=float)
        else:
            resolved_levels = make_levels(
                slice_2d.psi_axis, slice_2d.psi_boundary, n=n_levels
            )

    return FluxOverlay(
        field=field,
        contours=contours,
        levels=resolved_levels,
        cap_normal=cap_normal,
    )


# ---------------------------------------------------------------------------
# ε offset
# ---------------------------------------------------------------------------


def offset_along_normal(
    geometry: pv.PolyData,
    normal: tuple[float, float, float],
    epsilon: float = 1e-3,
) -> pv.PolyData:
    """Translate *geometry* by ``epsilon * normal`` to eliminate z-fighting.

    Returns a **new** PolyData — the input is not mutated.

    Parameters
    ----------
    geometry:
        PolyData to translate.
    normal:
        Unit normal direction for the offset.
    epsilon:
        Offset distance along *normal* in world units.

    Returns
    -------
    pv.PolyData
        Translated copy.
    """
    n = np.asarray(normal, dtype=float)
    n = n / np.linalg.norm(n)
    offset = epsilon * n

    translated = geometry.copy(deep=True)
    translated.points = np.asarray(translated.points, dtype=float) + offset
    return translated
