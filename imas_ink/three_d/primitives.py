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


def _planar_frame(
    centerline_xyz: np.ndarray, tangents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Compute (normal, binormal) frames for a near-planar centerline.

    For a planar D-shape (e.g. a TF coil lying in a poloidal plane), the
    binormal is the **constant out-of-plane direction** found by SVD of
    the centred centerline points (smallest singular vector).  The
    in-plane normal is then ``cross(binormal, tangent)``.

    This avoids the twist artefacts of Frenet/RMF frames on planar
    paths: every section ends up correctly oriented with its first axis
    in-plane (radial) and its second axis out-of-plane (toroidal at the
    coil midplane), matching the IMAS DD convention
    ``cross_section.outline.{normal, binormal}``.

    Parameters
    ----------
    centerline_xyz : (N, 3) array
        Path points.
    tangents : (N, 3) array
        Unit tangent at each point.

    Returns
    -------
    normals, binormals : (N, 3) arrays of unit vectors.
    """
    centred = centerline_xyz - centerline_xyz.mean(axis=0)
    # SVD: smallest singular vector = best-fit-plane normal
    _, _, vh = np.linalg.svd(centred, full_matrices=False)
    plane_normal = vh[-1]
    plane_normal = plane_normal / max(np.linalg.norm(plane_normal), 1e-12)

    n_path = len(tangents)
    binormals = np.broadcast_to(plane_normal, (n_path, 3)).copy()
    # In-plane normal = cross(binormal, tangent), then re-project so it
    # is exactly perpendicular to the tangent.
    normals = np.cross(binormals, tangents)
    nrm = np.linalg.norm(normals, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    normals = normals / nrm
    # Re-derive binormals = cross(tangent, normal) so the triad stays
    # right-handed even where SVD gave a slightly off-plane direction.
    binormals = np.cross(tangents, normals)
    nrm = np.linalg.norm(binormals, axis=1, keepdims=True)
    nrm = np.where(nrm < 1e-12, 1.0, nrm)
    binormals = binormals / nrm
    return normals, binormals


def _rmf_frame(
    centerline_xyz: np.ndarray, tangents: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Rotation-Minimizing Frame via double-reflection parallel transport.

    Wang et al., "Computation of Rotation Minimizing Frames",
    ACM TOG 2008. Use for genuinely non-planar paths (e.g. helical
    coils). For planar TF coils prefer :func:`_planar_frame`.
    """
    n_path = len(tangents)
    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

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

    for i in range(1, n_path):
        t_prev = tangents[i - 1]
        t_cur = tangents[i]
        n_prev = normals[i - 1]
        v1 = t_cur - t_prev
        c1 = float(np.dot(v1, v1))
        if c1 < 1e-18:
            n_cur = n_prev
            t_ref = t_prev
        else:
            n_ref = n_prev - (2.0 / c1) * np.dot(v1, n_prev) * v1
            t_ref = t_prev - (2.0 / c1) * np.dot(v1, t_prev) * v1
            v2 = t_cur - t_ref
            c2 = float(np.dot(v2, v2))
            if c2 < 1e-18:
                n_cur = n_ref
            else:
                n_cur = n_ref - (2.0 / c2) * np.dot(v2, n_ref) * v2
        n_cur = n_cur - np.dot(n_cur, t_cur) * t_cur
        nrm = np.linalg.norm(n_cur)
        if nrm < 1e-12:
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

    return normals, binormals


def _densify_polyline(path: np.ndarray, n_target: int) -> np.ndarray:
    """Linear arc-length resampling of a polyline to *n_target* points.

    Linear interpolation introduces no out-of-plane noise, so this is
    safe to use ahead of the planar-frame sweep: the SVD plane fit is
    invariant to point density along the original polyline.
    """
    if n_target <= len(path):
        return path
    # Cumulative arc length along the input polyline
    seg = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(seg)])
    if s[-1] <= 0.0:
        return path
    s_new = np.linspace(0.0, s[-1], n_target)
    out = np.empty((n_target, 3), dtype=float)
    for k in range(3):
        out[:, k] = np.interp(s_new, s, path[:, k])
    return out


def sweep_section_along_path(
    section_rz: np.ndarray,
    centerline_xyz: np.ndarray,
    frame: str = "planar",
    densify: int | None = None,
    closed_path: bool | None = None,
) -> pv.PolyData:
    """Sweep a 2D cross-section along a 3D centerline path.

    No smoothing or resampling is applied — the centerline is used
    exactly as supplied, preserving the coil shape from the IMAS
    conductor element chain.

    Two frame conventions are supported:

    - ``"planar"`` (default) — best for **planar TF coils**.  The
      out-of-plane direction is determined by SVD of the centerline
      and used as a constant binormal.  The in-plane radial direction
      is ``cross(binormal, tangent)``.  Produces a clean planar sweep
      with no twist; this matches the IMAS DD convention
      ``cross_section.outline.{normal, binormal}`` where ``normal`` is
      in-plane and ``binormal`` is out-of-plane.
    - ``"rmf"`` (also accepts ``"frenet"``) — Rotation-Minimizing
      Frame for genuinely non-planar paths (helical coils).  Uses the
      double-reflection algorithm of Wang et al., ACM TOG 2008.

    For **open paths** (start ≠ end), the start and end faces are
    closed with triangulated endcaps using fan triangulation.  For
    **closed paths** (rings), no caps are needed — the tube wraps
    around and connects back to itself.

    Parameters
    ----------
    section_rz : array_like, shape (M, 2)
        Cross-section polygon vertices as ``(normal, binormal)``
        offsets from the centerline.  For a rectangle with normal
        (radial) extent *w* and binormal (toroidal) extent *h*, use
        ``[[-w/2, -h/2], [w/2, -h/2], [w/2, h/2], [-w/2, h/2]]``.
    centerline_xyz : array_like, shape (N, 3)
        Ordered 3D points defining the path.
    frame : str
        Frame convention: ``"planar"`` (default), ``"rmf"``, or
        ``"frenet"`` (alias for ``"rmf"``).
    densify : int or None
        If given, resample the path to this many points before sweeping.
    closed_path : bool or None
        Whether the path forms a closed loop.  ``None`` (default)
        auto-detects by comparing the first/last point distance against
        the total path length.

    Returns
    -------
    pyvista.PolyData
        Swept surface mesh.  For open paths, the mesh includes start
        and end cap faces that close the tube into a watertight solid.
    """
    import pyvista as pv

    section_rz = np.asarray(section_rz, dtype=float)
    centerline_xyz = np.asarray(centerline_xyz, dtype=float)

    if len(centerline_xyz) < 2:
        return pv.PolyData()

    if densify is not None and densify > len(centerline_xyz):
        centerline_xyz = _densify_polyline(centerline_xyz, densify)

    n_path = len(centerline_xyz)
    n_sec = len(section_rz)

    if n_path < 2:
        return pv.PolyData()

    # Compute tangent vectors via finite differences
    tangents = np.zeros_like(centerline_xyz)
    tangents[1:-1] = centerline_xyz[2:] - centerline_xyz[:-2]
    tangents[0] = centerline_xyz[1] - centerline_xyz[0]
    tangents[-1] = centerline_xyz[-1] - centerline_xyz[-2]

    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    tangents = tangents / norms

    if frame in ("planar", "vector"):
        normals, binormals = _planar_frame(centerline_xyz, tangents)
    elif frame in ("rmf", "frenet"):
        normals, binormals = _rmf_frame(centerline_xyz, tangents)
    else:
        raise ValueError(
            f"Unknown frame {frame!r}; expected 'planar', 'rmf', or 'frenet'"
        )

    # Place section at each point along the centerline
    all_points = np.zeros((n_path * n_sec, 3))
    for i in range(n_path):
        c = centerline_xyz[i]
        n = normals[i]
        b = binormals[i]
        for j in range(n_sec):
            u, v = section_rz[j]
            all_points[i * n_sec + j] = c + u * n + v * b

    # Determine whether the path is a closed loop (ring) or open.
    if closed_path is None:
        end_dist = np.linalg.norm(centerline_xyz[-1] - centerline_xyz[0])
        seg_lengths = np.linalg.norm(np.diff(centerline_xyz, axis=0), axis=1)
        total_length = seg_lengths.sum()
        is_closed = end_dist < 1e-6 or (
            total_length > 0 and end_dist / total_length < 1e-4
        )
    else:
        is_closed = closed_path

    # Build quad faces connecting consecutive section rings
    faces: list[int] = []
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

    # For open paths, close start and end faces with fan triangulation.
    # Fan from vertex 0 of each cap; correct for convex sections.
    # TODO: use ear-clipping for non-convex cross-sections.
    if not is_closed and n_sec >= 3:
        import warnings

        # Heuristic non-convexity check: compare polygon area via
        # shoelace vs convex hull area.  Skipped for now — just warn
        # for sections with many vertices where non-convexity is likely.
        if n_sec > 8:
            warnings.warn(
                f"Fan triangulation of {n_sec}-vertex section may be "
                f"incorrect for non-convex cross-sections",
                stacklevel=2,
            )

        # Start cap (i=0): reversed winding → outward normal along -tangent
        base_start = 0
        for k in range(1, n_sec - 1):
            faces.extend([3, base_start, base_start + k + 1, base_start + k])

        # End cap (i=n_path-1): forward winding → outward normal along +tangent
        base_end = (n_path - 1) * n_sec
        for k in range(1, n_sec - 1):
            faces.extend([3, base_end, base_end + k, base_end + k + 1])

    mesh = pv.PolyData(all_points, faces=np.array(faces))
    return mesh
