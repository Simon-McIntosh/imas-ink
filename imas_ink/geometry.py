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


def classify_probe_components(
    positions_r: np.ndarray,
    positions_z: np.ndarray,
    angles: np.ndarray,
    *,
    position_tol: float = 0.01,
    angle_tol_deg: float = 20.0,
) -> np.ndarray:
    r"""Label each magnetic probe by its measurement-component family.

    Many tokamaks (WEST, ITER, ...) mount **multi-component** magnetic
    sensors: two or more pickup coils share one ``(R, Z)`` location but
    have distinct coil axes — typically one tangential and one normal to
    the wall, ~90° apart.  Each coil's ``poloidal_angle`` faithfully
    encodes its own sensor-normal (= coil-axis = measured-field)
    direction per the IMAS Data Dictionary
    (``magnetics/b_field_pol_probe/poloidal_angle``: "Angle of the sensor
    normal vector ... clockwise theta-like angle ... zero towards
    increasing major radius").

    Rendering every probe in one colour collapses these distinct
    components, so a tangential B-pol tick is visually indistinguishable
    from a co-located normal-component tick — the figure then *looks* as
    though B-pol probes point normal to the wall.  This function assigns a
    stable per-probe component index so the renderers can colour each
    component distinctly **without** altering the DD-defined angle.

    The discriminator is purely geometric and machine-agnostic: probes
    are grouped by co-location (within *position_tol* metres); within a
    group, probes are partitioned by orientation (axes differing by more
    than *angle_tol_deg*, treated as **undirected** — i.e. modulo 180° —
    because a pickup-coil sensitivity axis is a line, not an arrow).  The
    first orientation encountered in each group is component 0, the next
    distinct orientation is component 1, and so on.  Probes with no
    distinctly-oriented co-located partner are all component 0.  This
    fires on genuine multi-component sensors (WEST 2-component pairs,
    ITER saddle sensors) and does **not** fire on single-orientation
    arrays (AUG, TCV, MAST-U, HL-3) or on toroidally-replicated duplicate
    probes that share one orientation.

    Parameters
    ----------
    positions_r, positions_z : np.ndarray
        Probe ``(R, Z)`` positions, shape ``(N,)``.
    angles : np.ndarray
        Probe ``poloidal_angle`` values in radians, shape ``(N,)``.
        Non-finite (NaN) entries denote probes without an orientation;
        they are always assigned component 0.
    position_tol : float
        Co-location tolerance in metres.  Default ``0.01`` (1 cm).
    angle_tol_deg : float
        Two co-located probes are the *same* component if their axes
        (modulo 180°) differ by at most this many degrees.  Default
        ``20``.

    Returns
    -------
    np.ndarray
        Integer component index per probe, shape ``(N,)``.  ``0`` for the
        primary component (and all singletons); ``1, 2, ...`` for further
        distinct orientations sharing a location.

    Examples
    --------
    >>> r = np.array([3.0, 3.0, 5.0])
    >>> z = np.array([0.0, 0.0, 0.0])
    >>> # two co-located probes 90 deg apart, plus one lone probe
    >>> a = np.array([0.0, np.pi / 2, 0.0])
    >>> classify_probe_components(r, z, a).tolist()
    [0, 1, 0]
    """
    r = np.asarray(positions_r, dtype=float)
    z = np.asarray(positions_z, dtype=float)
    ang = np.asarray(angles, dtype=float)
    n = r.size
    comp = np.zeros(n, dtype=int)
    if n == 0:
        return comp

    tol2 = position_tol * position_tol
    atol = np.radians(angle_tol_deg)
    assigned = np.zeros(n, dtype=bool)

    for i in range(n):
        if assigned[i]:
            continue
        # Collect every probe co-located with i (including i itself).
        same_pos = np.where((r - r[i]) ** 2 + (z - z[i]) ** 2 <= tol2)[0]
        # Within this location, bucket probes by undirected orientation.
        rep_angles: list[float] = []  # one representative angle per component
        for j in same_pos:
            aj = ang[j]
            if not np.isfinite(aj):
                comp[j] = 0  # orientation-less probes: primary bucket
                assigned[j] = True
                continue
            placed = False
            for k, rep in enumerate(rep_angles):
                # Undirected angular distance: |Δ| modulo pi, in [0, pi/2].
                d = abs((aj - rep) % np.pi)
                d = min(d, np.pi - d)
                if d <= atol:
                    comp[j] = k
                    placed = True
                    break
            if not placed:
                rep_angles.append(aj)
                comp[j] = len(rep_angles) - 1
            assigned[j] = True

    return comp


def encloses_point(vertices: np.ndarray, r: float, z: float) -> bool:
    """Test whether a closed polygon *encloses* the point ``(r, z)``.

    Uses the winding-number (ray-casting) test — pure geometry on the
    plotted coordinates.  A contour "encloses the magnetic axis" if and
    only if this returns ``True``.

    This is a **styling** predicate, not a physics computation: it
    determines whether to draw a flux surface in the confined-plasma
    colour vs the SOL / open-field grey.

    Parameters
    ----------
    vertices : np.ndarray
        Shape ``(N, 2)`` array of polygon vertices with columns
        ``[R, Z]``.
    r, z : float
        Point to test.

    Returns
    -------
    bool
        ``True`` if ``(r, z)`` is inside the polygon.

    Examples
    --------
    >>> v = np.array([[0, -1], [1, 0], [0, 1], [-1, 0], [0, -1]])
    >>> encloses_point(v, 0.0, 0.0)
    True
    >>> encloses_point(v, 2.0, 0.0)
    False
    """
    if vertices.shape[0] < 3:
        return False
    # Ray-casting: count crossings of a horizontal ray from (r, z) to +∞
    inside = False
    n = len(vertices)
    xi, yi = float(vertices[0, 0]), float(vertices[0, 1])
    for j in range(1, n + 1):
        xj, yj = float(vertices[j % n, 0]), float(vertices[j % n, 1])
        if ((yi > z) != (yj > z)) and (r < (xj - xi) * (z - yi) / (yj - yi + 1e-300) + xi):
            inside = not inside
        xi, yi = xj, yj
    return inside


def classify_flux_segments(
    level_segs: list[np.ndarray],
    r_axis: float,
    z_axis: float,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Classify contour segments at one psi level into confined vs SOL.

    A segment is *confined* if it is closed (first ≈ last vertex) AND
    encloses the magnetic axis.  All other segments — open lines,
    private-flux lobes that do not enclose the axis, and grid-edge
    artefacts — are classified as SOL.

    This is a **styling** decision: the classification drives the
    visual distinction between blue confined surfaces and thin grey
    SOL/open lines.  It does not compute or invent physics values.

    Parameters
    ----------
    level_segs : list[np.ndarray]
        Segments at one contour level, each ``(N, 2)`` with ``[R, Z]``.
    r_axis, z_axis : float
        Magnetic axis position.

    Returns
    -------
    tuple[list[np.ndarray], list[np.ndarray]]
        ``(confined_segs, sol_segs)``
    """
    confined: list[np.ndarray] = []
    sol: list[np.ndarray] = []
    for seg in level_segs:
        if len(seg) < 3:
            sol.append(seg)
            continue
        # Closed if first ≈ last vertex
        closed = np.allclose(seg[0], seg[-1], atol=1e-6)
        if closed and encloses_point(seg, r_axis, z_axis):
            confined.append(seg)
        else:
            sol.append(seg)
    return confined, sol


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
