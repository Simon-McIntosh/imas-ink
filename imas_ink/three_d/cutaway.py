"""Per-block capped clipping + geometric cap-face extraction + auto camera."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv


# ------------------------------------------------------------------
# Data classes
# ------------------------------------------------------------------


@dataclass(frozen=True)
class ClipPlane:
    """Half-space clip plane in world coordinates."""

    origin: tuple[float, float, float]  # point on plane
    normal: tuple[float, float, float]  # unit normal; geometry on -normal side is removed


@dataclass(frozen=True)
class CappedMesh:
    """A clipped mesh and its cap-face submesh."""

    full: object  # pv.PolyData — clipped result with caps welded in
    cap: object  # pv.PolyData — only the cap face cells (extracted geometrically)
    name: str


# ------------------------------------------------------------------
# Geometric cap-face extraction
# ------------------------------------------------------------------


def cap_face_of(
    mesh: pv.PolyData,
    plane: ClipPlane,
    eps_plane: float = 1e-5,
    eps_angle_deg: float = 2.0,
) -> pv.PolyData:
    """Extract cap-face cells from a mesh clipped by *plane*.

    VTK's ``vtkClipClosedSurface`` does **not** tag the cap cells it
    generates, so cap extraction must be geometric:

    1. For each cell, check that **all** of its vertices lie within
       *eps_plane* (absolute distance) of the clip plane.
    2. Compute the cell's face normal from its first 3 vertices.
    3. Check that the face normal is parallel (within *eps_angle_deg*
       degrees) to ±plane.normal.
    4. Cells passing both checks are cap cells.

    Parameters
    ----------
    mesh : pyvista.PolyData
        The clipped mesh (output of ``vtkClipClosedSurface``).
    plane : ClipPlane
        The clip plane used for clipping.
    eps_plane : float
        Maximum absolute distance from the plane for a vertex to be
        considered "on the plane".
    eps_angle_deg : float
        Maximum angular deviation (in degrees) between a cell normal
        and ±plane.normal for the cell to be considered a cap face.

    Returns
    -------
    pyvista.PolyData
        PolyData containing only the cap-face cells.  May have
        ``n_cells == 0`` if no cap faces are found.
    """
    import numpy as np
    import pyvista as pv

    if mesh.n_cells == 0:
        return pv.PolyData()

    origin = np.asarray(plane.origin, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    cos_threshold = np.cos(np.radians(eps_angle_deg))

    cap_cell_ids: list[int] = []

    for cell_id in range(mesh.n_cells):
        cell = mesh.get_cell(cell_id)
        pts = np.asarray(cell.points, dtype=float)

        # Check all vertices lie on the plane
        dists = np.abs((pts - origin) @ normal)
        if np.any(dists > eps_plane):
            continue

        # Compute face normal from first 3 vertices
        if pts.shape[0] < 3:
            continue
        e1 = pts[1] - pts[0]
        e2 = pts[2] - pts[0]
        face_normal = np.cross(e1, e2)
        fn_len = np.linalg.norm(face_normal)
        if fn_len < 1e-30:
            continue
        face_normal /= fn_len

        # Check alignment with ±plane normal
        cos_angle = abs(float(np.dot(face_normal, normal)))
        if cos_angle >= cos_threshold:
            cap_cell_ids.append(cell_id)

    if not cap_cell_ids:
        return pv.PolyData()

    return mesh.extract_cells(cap_cell_ids)


# ------------------------------------------------------------------
# Capped clip (single PolyData)
# ------------------------------------------------------------------


def capped_clip(
    mesh: pv.PolyData,
    plane: ClipPlane,
    *,
    name: str = "<unnamed>",
    eps_plane: float = 1e-5,
    eps_angle_deg: float = 2.0,
) -> CappedMesh:
    """Clip a single PolyData with a plane and extract cap faces.

    Pipeline:

    1. Validate the input mesh as a closed manifold.
    2. Apply ``vtkClipClosedSurface`` with the supplied plane.
    3. Geometrically extract cap faces via :func:`cap_face_of`.

    ``vtkClipClosedSurface`` only operates on a single PolyData — never
    on a MultiBlock.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh (must be a closed manifold).
    plane : ClipPlane
        Half-space clip plane.
    name : str
        Human-readable mesh name for error messages.
    eps_plane : float
        Plane-proximity tolerance for cap extraction.
    eps_angle_deg : float
        Angular tolerance (degrees) for cap-face normal alignment.

    Returns
    -------
    CappedMesh
        Clipped mesh with its geometrically extracted cap submesh.

    Raises
    ------
    MeshNotManifoldError
        If *mesh* is not a closed manifold.
    """
    import numpy as np
    import pyvista as pv
    import vtk

    from .manifold import ensure_closed_manifold

    # 1. Validate manifold
    manifold_mesh = ensure_closed_manifold(mesh, name=name)

    # 2. Build VTK clipping plane collection
    origin = np.asarray(plane.origin, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    vtk_plane = vtk.vtkPlane()
    vtk_plane.SetOrigin(*origin)
    vtk_plane.SetNormal(*normal)

    plane_collection = vtk.vtkPlaneCollection()
    plane_collection.AddItem(vtk_plane)

    # 3. Apply vtkClipClosedSurface
    clipper = vtk.vtkClipClosedSurface()
    clipper.SetInputData(manifold_mesh)
    clipper.SetClippingPlanes(plane_collection)
    clipper.Update()

    clipped = pv.wrap(clipper.GetOutput())

    if clipped is None or clipped.n_points == 0:
        empty = pv.PolyData()
        return CappedMesh(full=empty, cap=empty, name=name)

    # 4. Geometric cap extraction
    cap = cap_face_of(
        clipped, plane, eps_plane=eps_plane, eps_angle_deg=eps_angle_deg
    )

    return CappedMesh(full=clipped, cap=cap, name=name)


# ------------------------------------------------------------------
# Multi-block convenience
# ------------------------------------------------------------------


def capped_clip_multiblock(
    blocks: dict[str, pv.PolyData],
    plane: ClipPlane,
    **kwargs,
) -> dict[str, CappedMesh]:
    """Apply :func:`capped_clip` per block, skipping outside-halfspace blocks.

    Parameters
    ----------
    blocks : dict[str, pv.PolyData]
        Named meshes to clip.
    plane : ClipPlane
        Clip plane applied to every block.
    **kwargs
        Forwarded to :func:`capped_clip`.

    Returns
    -------
    dict[str, CappedMesh]
        One ``CappedMesh`` per input key.  Blocks entirely on the
        removed side of the plane return an empty ``CappedMesh``.
    """
    import numpy as np
    import pyvista as pv

    origin = np.asarray(plane.origin, dtype=float)
    normal = np.asarray(plane.normal, dtype=float)
    normal = normal / np.linalg.norm(normal)

    results: dict[str, CappedMesh] = {}

    for block_name, mesh in blocks.items():
        if mesh.n_points == 0:
            empty = pv.PolyData()
            results[block_name] = CappedMesh(
                full=empty, cap=empty, name=block_name
            )
            continue

        # Check if the block is entirely on the removed side
        pts = np.asarray(mesh.points, dtype=float)
        signed_dists = (pts - origin) @ normal
        if np.all(signed_dists < 0):
            # Entirely on the removed side (−normal side)
            empty = pv.PolyData()
            results[block_name] = CappedMesh(
                full=empty, cap=empty, name=block_name
            )
            continue

        results[block_name] = capped_clip(
            mesh, plane, name=block_name, **kwargs
        )

    return results


# ------------------------------------------------------------------
# Auto camera
# ------------------------------------------------------------------


def auto_camera(
    mesh_bounds: tuple[float, float, float, float, float, float],
    *,
    view: str = "poloidal_rhs",
    overlay_margin: tuple[float, float] = (0.10, 0.05),
    window_size: tuple[int, int] = (1200, 900),
) -> dict:
    """Compute orthographic camera parameters to fit *mesh_bounds*.

    The camera parameters account for a reserved overlay margin
    (e.g. for a colourbar on the left and title on top) so that the
    mesh fills the **drawable** region of the viewport.

    Parameters
    ----------
    mesh_bounds : tuple
        ``(xmin, xmax, ymin, ymax, zmin, zmax)`` of the scene.
    view : str
        Camera preset.  Currently supported: ``"poloidal_rhs"``.
        Stubs for ``"iso"``, ``"poloidal_lhs"``, ``"toroidal_top"``
        raise ``NotImplementedError``.
    overlay_margin : tuple[float, float]
        ``(left_fraction, top_fraction)`` of window reserved for
        overlays.  Default ``(0.10, 0.05)``.
    window_size : tuple[int, int]
        Render window size ``(width, height)`` in pixels.

    Returns
    -------
    dict
        Keys: ``position``, ``focal_point``, ``view_up``,
        ``parallel_scale``, ``parallel_projection``.  Ready for
        ``plotter.camera_position = [d["position"], ...]``.

    Raises
    ------
    NotImplementedError
        For unimplemented view presets.
    """
    if view not in ("poloidal_rhs",):
        _stubs = {"iso", "poloidal_lhs", "toroidal_top"}
        if view in _stubs:
            raise NotImplementedError(
                f"View preset {view!r} is not yet implemented"
            )
        raise ValueError(f"Unknown view preset {view!r}")

    xmin, xmax, ymin, ymax, zmin, zmax = mesh_bounds

    # For poloidal_rhs: camera looks along +y (toward −y), up = +z.
    # Frames the RHS poloidal cross-section: x ∈ [0, x_max], z ∈ [z_min, z_max].
    # Use x ∈ [0, xmax] because R > 0 always for tokamak geometry.
    frame_xmin = 0.0
    frame_xmax = xmax
    frame_zmin = zmin
    frame_zmax = zmax

    # Add 5% bbox padding
    half_w = (frame_xmax - frame_xmin) / 2.0
    half_h = (frame_zmax - frame_zmin) / 2.0
    cx = (frame_xmin + frame_xmax) / 2.0
    cz = (frame_zmin + frame_zmax) / 2.0

    pad = 0.05
    half_w *= 1.0 + pad
    half_h *= 1.0 + pad

    # Drawable viewport (after reserving overlay margins)
    left_frac, top_frac = overlay_margin
    w_px, h_px = window_size
    drawable_w = w_px * (1.0 - left_frac)
    drawable_h = h_px * (1.0 - top_frac)
    aspect_drawable = drawable_w / drawable_h

    # parallel_scale is the half-height of the visible world in the
    # viewport.  We pick the larger of (half_h, half_w / aspect) so
    # the mesh always fits inside the drawable region.
    parallel_scale = max(half_h, half_w / aspect_drawable)

    # Camera offset: far enough from the mesh along +y to clear it.
    cam_offset = max(half_w, half_h) * 10.0

    return {
        "position": (cx, cam_offset, cz),
        "focal_point": (cx, 0.0, cz),
        "view_up": (0.0, 0.0, 1.0),
        "parallel_scale": parallel_scale,
        "parallel_projection": True,
    }
