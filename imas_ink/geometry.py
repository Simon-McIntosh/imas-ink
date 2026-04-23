"""Geometry processing — backend-neutral, operates on numpy arrays only.

Functions extracted from ``render_sep_gif.py`` and generalised for
arbitrary tokamak geometries. All functions accept and return plain
numpy arrays; no matplotlib or IMAS objects appear in this module.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion
from scipy.ndimage import label as ndlabel


def mask_pfr(
    psi_2d: np.ndarray,
    r_2d: np.ndarray,
    z_2d: np.ndarray,
    psi_axis: float,
    psi_bnd: float,
    r_axis: float,
    z_axis: float,
) -> np.ndarray:
    """Mask the private flux region, keeping only the core plasma.

    Uses morphological erosion to break the 1-2 pixel bridge at the
    X-point saddle, then labels connected components and keeps only the
    region connected to the magnetic axis. Handles limiter, SN, and DN
    configurations identically.

    Parameters
    ----------
    psi_2d : np.ndarray
        2D poloidal flux, shape ``(nR, nZ)``.
    r_2d, z_2d : np.ndarray
        2D meshgrid coordinates, same shape as *psi_2d*.
    psi_axis : float
        Poloidal flux at the magnetic axis.
    psi_bnd : float
        Poloidal flux at the last closed flux surface.
    r_axis, z_axis : float
        Magnetic axis position.

    Returns
    -------
    np.ndarray
        Copy of *psi_2d* with PFR pixels set to ``NaN``.

    Examples
    --------
    >>> masked = mask_pfr(psi_2d, R_2d, Z_2d, psi_ax, psi_bnd, r_ax, z_ax)
    >>> np.isnan(masked).any()
    True
    """
    masked = psi_2d.astype(float, copy=True)
    inside = np.isfinite(psi_2d) & (psi_2d > psi_bnd)
    inside_eroded = binary_erosion(inside, iterations=1)
    labeled, _ = ndlabel(inside_eroded)
    dist_sq = (r_2d - r_axis) ** 2 + (z_2d - z_axis) ** 2
    idx = np.unravel_index(np.argmin(dist_sq), dist_sq.shape)
    axis_label = labeled[idx]
    if axis_label == 0:
        return masked  # fallback: erosion eliminated axis pixel
    core_mask = binary_dilation(labeled == axis_label, iterations=1)
    masked[~(inside & core_mask)] = np.nan
    return masked


def find_xpoints(
    psi_2d: np.ndarray,
    r_2d: np.ndarray,
    z_2d: np.ndarray,
    psi_bnd: float,
    psi_axis: float,
    z_axis: float,
) -> list[tuple[float, float]]:
    r"""Numerically detect X-points as :math:`|\nabla\psi|` minima near psi_boundary.

    Searches above and below *z_axis* independently, returning at most one
    X-point per half-plane. Returns an empty list for limiter plasmas.

    Parameters
    ----------
    psi_2d : np.ndarray
        2D poloidal flux, shape ``(nR, nZ)``.
    r_2d, z_2d : np.ndarray
        2D meshgrid coordinates.
    psi_bnd : float
        Poloidal flux at the LCFS.
    psi_axis : float
        Poloidal flux at the magnetic axis.
    z_axis : float
        Vertical position of the magnetic axis.

    Returns
    -------
    list[tuple[float, float]]
        List of ``(R, Z)`` pairs for detected X-points.

    Examples
    --------
    >>> xpts = find_xpoints(psi_2d, R_2d, Z_2d, psi_bnd, psi_ax, z_ax)
    >>> len(xpts)  # 1 for SN, 2 for DN, 0 for limiter
    1
    """
    r_1d = r_2d[:, 0]
    z_1d = z_2d[0, :]
    dpsi_dr = np.gradient(psi_2d, r_1d, axis=0)
    dpsi_dz = np.gradient(psi_2d, z_1d, axis=1)
    grad_sq = dpsi_dr**2 + dpsi_dz**2
    band = 0.04 * abs(psi_axis - psi_bnd)
    near_bnd = np.abs(psi_2d - psi_bnd) < band
    xpts: list[tuple[float, float]] = []
    for below in (True, False):
        offset = 0.1
        half = near_bnd & (z_2d < z_axis - offset if below else z_2d > z_axis + offset)
        if not np.any(half):
            continue
        g = np.where(half, grad_sq, np.inf)
        idx = np.unravel_index(np.argmin(g), g.shape)
        r_x, z_x = float(r_2d[idx]), float(z_2d[idx])
        if r_x > 0.05:
            xpts.append((r_x, z_x))
    return xpts


def wall_clip_vertices(wall_r: np.ndarray, wall_z: np.ndarray) -> np.ndarray:
    """Return closed polygon vertices ``(N, 2)`` for wall clipping.

    Ensures the polygon is closed (first vertex == last vertex).

    Parameters
    ----------
    wall_r, wall_z : np.ndarray
        1D arrays of wall outline coordinates.

    Returns
    -------
    np.ndarray
        Shape ``(N, 2)`` array with columns ``[R, Z]``.

    Examples
    --------
    >>> verts = wall_clip_vertices(wall_r, wall_z)
    >>> np.allclose(verts[0], verts[-1])
    True
    """
    verts = np.column_stack([wall_r, wall_z])
    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack([verts, verts[0]])
    return verts


def coil_bboxes(pf_ids, empty_threshold: float = 1e9) -> list:
    """Extract coil bounding boxes from a ``pf_active`` IDS.

    Handles both outline and rectangle geometry types. Each coil element
    contributes to the overall bounding box of its parent coil.

    Parameters
    ----------
    pf_ids
        ``pf_active`` IDS object (``imas.imasdef.pf_active``).
    empty_threshold : float
        Threshold for EMPTY sentinel filtering.

    Returns
    -------
    list[CoilRect]
        List of :class:`~efitpp.ink._types.CoilRect` instances.

    Examples
    --------
    >>> coils = coil_bboxes(pf_ids)
    >>> len(coils)
    14
    """
    from ._types import CoilRect

    rects = []
    for coil in pf_ids.coil:
        r_lo: list[float] = []
        r_hi: list[float] = []
        z_lo: list[float] = []
        z_hi: list[float] = []
        for elem in coil.element:
            g = elem.geometry
            outline_r = np.asarray(g.outline.r)
            if outline_r.size > 0:
                outline_z = np.asarray(g.outline.z)
                r_lo.append(outline_r.min())
                r_hi.append(outline_r.max())
                z_lo.append(outline_z.min())
                z_hi.append(outline_z.max())
            else:
                try:
                    rr = float(g.rectangle.r)
                    zr = float(g.rectangle.z)
                    dr = float(g.rectangle.width)
                    dz = float(g.rectangle.height)
                    if abs(rr) < empty_threshold and abs(zr) < empty_threshold:
                        r_lo.append(rr - dr / 2)
                        r_hi.append(rr + dr / 2)
                        z_lo.append(zr - dz / 2)
                        z_hi.append(zr + dz / 2)
                except (TypeError, AttributeError):
                    pass
        if r_lo:
            r0, z0 = min(r_lo), min(z_lo)
            name = getattr(coil, "name", "")
            rects.append(CoilRect(r0, z0, max(r_hi) - r0, max(z_hi) - z0, name))
    return rects


def split_path_segs(vertices: np.ndarray, codes: np.ndarray | None) -> list[np.ndarray]:
    """Split a path into continuous segments by MOVETO / CLOSEPOLY codes.

    A contour path may contain several disconnected sub-paths stitched
    together with ``MOVETO`` (code 1) instructions. Passing raw vertices
    to ``LineCollection`` connects them with straight artefact lines.

    Parameters
    ----------
    vertices : np.ndarray
        Shape ``(N, 2)`` vertex array.
    codes : np.ndarray or None
        Matplotlib-compatible path codes.  If *None*, the entire vertex
        array is returned as a single segment.

    Returns
    -------
    list[np.ndarray]
        List of ``(M, 2)`` vertex arrays, one per continuous sub-path.

    Examples
    --------
    >>> segs = split_path_segs(verts, codes)
    >>> all(s.shape[1] == 2 for s in segs)
    True
    """
    MOVETO = 1
    CLOSEPOLY = 79
    if codes is None:
        return [vertices]
    segs: list[np.ndarray] = []
    start = None
    for j, c in enumerate(codes):
        if c == MOVETO:
            if start is not None and j - start >= 2:
                segs.append(vertices[start:j])
            start = j
        elif c == CLOSEPOLY:
            if start is not None:
                segs.append(vertices[start : j + 1])
            start = None
    if start is not None and len(vertices) - start >= 2:
        segs.append(vertices[start:])
    return segs


def close_polygon(vertices: np.ndarray) -> np.ndarray:
    """Ensure a polygon is closed (first vertex == last vertex).

    Parameters
    ----------
    vertices : np.ndarray
        Shape ``(N, 2)`` vertex array.

    Returns
    -------
    np.ndarray
        Possibly extended array with ``vertices[0]`` appended if not
        already present at the end.

    Examples
    --------
    >>> v = np.array([[0, 0], [1, 0], [1, 1]])
    >>> close_polygon(v)
    array([[0, 0],
           [1, 0],
           [1, 1],
           [0, 0]])
    """
    if len(vertices) < 2:
        return vertices
    if not np.allclose(vertices[0], vertices[-1]):
        return np.vstack([vertices, vertices[0]])
    return vertices


def is_closed_contour(codes: np.ndarray | None) -> bool:
    """Return True if a contourpy path code array ends with CLOSEPOLY (79).

    Parameters
    ----------
    codes : np.ndarray or None
        Matplotlib-compatible path codes from ``contourpy``.

    Returns
    -------
    bool

    Examples
    --------
    >>> is_closed_contour(np.array([1, 2, 2, 79]))
    True
    >>> is_closed_contour(np.array([1, 2, 2]))
    False
    >>> is_closed_contour(None)
    False
    """
    if codes is None or len(codes) == 0:
        return False
    return int(codes[-1]) == 79
