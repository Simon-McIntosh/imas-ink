"""Pure geometric primitives for 3D tokamak rendering.

No IMAS dependency — only numpy and pyvista/vtk. All heavy imports
are inside function bodies so ``import imas_ink`` stays cheap.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import pyvista as pv


def cylindrical_to_cartesian(
    r: np.ndarray, phi: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Convert cylindrical (r, φ, z) to Cartesian (x, y, z).

    Parameters
    ----------
    r, phi, z : array_like
        Cylindrical coordinates. *phi* is in radians.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, z)`` arrays with the same shape as the inputs.
    """
    r = np.asarray(r, dtype=float)
    phi = np.asarray(phi, dtype=float)
    z = np.asarray(z, dtype=float)
    x = r * np.cos(phi)
    y = r * np.sin(phi)
    return x, y, z


def revolve_polygon(r: np.ndarray, z: np.ndarray, n_theta: int = 60) -> pv.PolyData:
    """Revolve a 2D (r, z) polygon 360° about the z-axis.

    Parameters
    ----------
    r, z : array_like
        Vertices of the cross-section polygon in the poloidal plane.
    n_theta : int
        Number of azimuthal steps (resolution).

    Returns
    -------
    pyvista.PolyData
        Closed 3D surface mesh.
    """
    import pyvista as pv

    r = np.asarray(r, dtype=float)
    z = np.asarray(z, dtype=float)

    # Build a 2D polygon in the XZ-plane at y=0 (x=r, y=0, z=z)
    n_pts = len(r)
    points_2d = np.column_stack([r, np.zeros(n_pts), z])

    # Polygon face: [n_pts, 0, 1, 2, ..., n_pts-1]
    face = [n_pts, *range(n_pts)]
    poly = pv.PolyData(points_2d, faces=face)

    # extrude_rotate produces a closed revolution surface
    mesh = poly.extrude_rotate(resolution=n_theta, inplace=False)
    return mesh


def ring_from_rectangle(
    r_center: float,
    z_center: float,
    dr: float,
    dz: float,
    n_theta: int = 60,
) -> pv.PolyData:
    """Build an axisymmetric ring from a rectangular cross-section.

    Parameters
    ----------
    r_center, z_center : float
        Centre of the rectangle in the poloidal plane.
    dr, dz : float
        Full width (radial) and full height (vertical) of the rectangle.
    n_theta : int
        Number of azimuthal steps.

    Returns
    -------
    pyvista.PolyData
    """
    hr, hz = dr / 2.0, dz / 2.0
    r = np.array([r_center - hr, r_center + hr, r_center + hr, r_center - hr])
    z = np.array([z_center - hz, z_center - hz, z_center + hz, z_center + hz])
    return revolve_polygon(r, z, n_theta=n_theta)


def _resample_centerline(path: np.ndarray, n: int, closed: bool) -> np.ndarray:
    """Cubic-spline resample a 3D polyline to *n* evenly-spaced samples.

    Parameters
    ----------
    path : (N, 3) array
        Input polyline vertices.
    n : int
        Number of output samples.
    closed : bool
        Whether to treat the path as a closed loop (periodic spline).

    Returns
    -------
    (n, 3) array of resampled points.
    """
    from scipy.interpolate import CubicSpline

    # Cumulative arc length as parameter
    diffs = np.diff(path, axis=0)
    seg = np.linalg.norm(diffs, axis=1)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total <= 0.0:
        return path.copy()

    bc = "periodic" if closed else "not-a-knot"
    if closed and not np.allclose(path[0], path[-1]):
        # Enforce closure for periodic BC
        path = np.vstack([path, path[:1]])
        s = np.concatenate([s, [total + np.linalg.norm(path[-1] - path[-2])]])
        total = s[-1]

    cs = CubicSpline(s, path, bc_type=bc, axis=0)
    u = np.linspace(0.0, total, n, endpoint=not closed)
    return cs(u)


def sweep_section_along_path(
    section_rz: np.ndarray,
    centerline_xyz: np.ndarray,
    frame: str = "frenet",
    resample: int | None = 240,
) -> pv.PolyData:
    """Sweep a 2D cross-section along a 3D centerline path.

    Uses a Rotation-Minimizing Frame (RMF, double-reflection parallel
    transport; Wang et al. ACM TOG 2008) to orient the section at each
    point along the centerline, then builds quad faces between consecutive
    cross-section rings.

    Parameters
    ----------
    section_rz : array_like, shape (M, 2)
        Cross-section polygon vertices as ``(local_x, local_y)`` offsets
        from the centerline. For a rectangle of width *w* and height *h*,
        use ``[[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]``.
    centerline_xyz : array_like, shape (N, 3)
        Ordered 3D points defining the path.
    frame : str
        Frame convention. Only ``"frenet"`` / ``"rmf"`` are accepted;
        both now use the RMF algorithm.
    resample : int or None
        If not ``None``, cubic-spline resample the centerline to this
        many points before sweeping (default 240). Set to ``None`` to
        disable.  Smooths coarse/angular centrelines such as IMAS TF
        conductor element chains.

    Returns
    -------
    pyvista.PolyData
        Swept surface mesh.
    """
    import pyvista as pv

    section_rz = np.asarray(section_rz, dtype=float)
    centerline_xyz = np.asarray(centerline_xyz, dtype=float)

    if len(centerline_xyz) < 2:
        return pv.PolyData()

    if resample is not None and resample > len(centerline_xyz):
        closed = bool(np.allclose(centerline_xyz[0], centerline_xyz[-1]))
        centerline_xyz = _resample_centerline(centerline_xyz, resample, closed)

    n_path = len(centerline_xyz)
    n_sec = len(section_rz)

    if n_path < 2:
        return pv.PolyData()

    # Compute tangent vectors via finite differences
    tangents = np.zeros_like(centerline_xyz)
    tangents[1:-1] = centerline_xyz[2:] - centerline_xyz[:-2]
    tangents[0] = centerline_xyz[1] - centerline_xyz[0]
    tangents[-1] = centerline_xyz[-1] - centerline_xyz[-2]

    # Normalise tangents
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents = tangents / norms

    # Build local frames via Rotation-Minimizing Frame (RMF) using
    # parallel transport (Wang et al., "Computation of Rotation Minimizing
    # Frames", ACM TOG 2008).  This avoids the twist/flip artefacts of the
    # classical Frenet frame near inflection points and when the tangent
    # aligns with the reference axis.
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    # Seed the first frame: pick any vector orthogonal to the first tangent.
    t0 = tangents[0]
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(t0, ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    n0 = np.cross(t0, ref)
    n0 = n0 / max(np.linalg.norm(n0), 1e-12)
    b0 = np.cross(t0, n0)
    b0 = b0 / max(np.linalg.norm(b0), 1e-12)
    normals[0] = n0
    binormals[0] = b0

    # Propagate via double-reflection parallel transport.
    for i in range(1, n_path):
        t_prev = tangents[i - 1]
        t_cur = tangents[i]
        n_prev = normals[i - 1]

        # Reflection 1: reflect n_prev across the bisecting plane of (t_prev, t_cur)
        v1 = t_cur - t_prev
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-18:
            n_cur = n_prev
            t_ref = t_prev
        else:
            n_ref = n_prev - (2.0 / c1) * np.dot(v1, n_prev) * v1
            t_ref = t_prev - (2.0 / c1) * np.dot(v1, t_prev) * v1
            # Reflection 2: reflect n_ref across the plane bisecting (t_ref, t_cur)
            v2 = t_cur - t_ref
            c2 = float(np.dot(v2, v2))
            if c2 < 1e-18:
                n_cur = n_ref
            else:
                n_cur = n_ref - (2.0 / c2) * np.dot(v2, n_ref) * v2

        # Re-orthogonalise against the current tangent (numerical hygiene)
        n_cur = n_cur - np.dot(n_cur, t_cur) * t_cur
        nrm = np.linalg.norm(n_cur)
        if nrm < 1e-12:
            # Fall back to a fresh orthogonal vector
            ref = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(t_cur, ref)) > 0.9:
                ref = np.array([1.0, 0.0, 0.0])
            n_cur = np.cross(t_cur, ref)
            nrm = np.linalg.norm(n_cur)
        n_cur = n_cur / nrm
        b_cur = np.cross(t_cur, n_cur)
        b_cur = b_cur / max(np.linalg.norm(b_cur), 1e-12)

        normals[i] = n_cur
        binormals[i] = b_cur

    # For closed curves, blend the accumulated twist back to the seed frame
    if np.allclose(centerline_xyz[0], centerline_xyz[-1], atol=1e-6) and n_path >= 3:
        # Compute residual angle between transported n at last point and n0
        t_last = tangents[-1]
        n_last = normals[-1]
        # Project n0 into the plane perpendicular to t_last
        n0_in_last = n0 - np.dot(n0, t_last) * t_last
        nrm = np.linalg.norm(n0_in_last)
        if nrm > 1e-12:
            n0_in_last = n0_in_last / nrm
            cos_theta = float(np.clip(np.dot(n_last, n0_in_last), -1.0, 1.0))
            sin_theta = float(np.dot(np.cross(n_last, n0_in_last), t_last))
            twist = np.arctan2(sin_theta, cos_theta)
            # Distribute the counter-rotation linearly along the curve
            for i in range(n_path):
                angle = -twist * (i / (n_path - 1))
                c, s = np.cos(angle), np.sin(angle)
                n = normals[i]
                b = binormals[i]
                normals[i] = c * n + s * b
                binormals[i] = -s * n + c * b

    # Place section at each point along the centerline
    all_points = np.zeros((n_path * n_sec, 3))
    for i in range(n_path):
        c = centerline_xyz[i]
        n = normals[i]
        b = binormals[i]
        for j in range(n_sec):
            u, v = section_rz[j]
            all_points[i * n_sec + j] = c + u * n + v * b

    # Build quad faces connecting consecutive rings
    faces = []
    is_closed = np.allclose(centerline_xyz[0], centerline_xyz[-1], atol=1e-6)
    n_segments = n_path - 1 if not is_closed else n_path

    for i in range(n_segments):
        i_next = (i + 1) % n_path
        for j in range(n_sec):
            j_next = (j + 1) % n_sec
            p0 = i * n_sec + j
            p1 = i * n_sec + j_next
            p2 = i_next * n_sec + j_next
            p3 = i_next * n_sec + j
            faces.extend([4, p0, p1, p2, p3])

    mesh = pv.PolyData(all_points, faces=np.array(faces))
    return mesh
