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


def sweep_section_along_path(
    section_rz: np.ndarray,
    centerline_xyz: np.ndarray,
    frame: str = "frenet",
) -> pv.PolyData:
    """Sweep a 2D cross-section along a 3D centerline path.

    Uses a discrete Frenet frame to orient the section at each point
    along the centerline, then builds quad faces between consecutive
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
        Frame convention. Only ``"frenet"`` is supported.

    Returns
    -------
    pyvista.PolyData
        Swept surface mesh.
    """
    import pyvista as pv

    section_rz = np.asarray(section_rz, dtype=float)
    centerline_xyz = np.asarray(centerline_xyz, dtype=float)

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

    # Build local frames: normal and binormal
    # Use the global z-axis as a reference to bootstrap the initial normal
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    for i in range(n_path):
        t = tangents[i]
        # Choose a reference vector not parallel to the tangent
        ref = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(t, ref)) > 0.9:
            ref = np.array([1.0, 0.0, 0.0])
        n = np.cross(t, ref)
        n_norm = np.linalg.norm(n)
        n = np.array([1.0, 0.0, 0.0]) if n_norm < 1e-12 else n / n_norm
        b = np.cross(t, n)
        b = b / np.linalg.norm(b)
        normals[i] = n
        binormals[i] = b

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
