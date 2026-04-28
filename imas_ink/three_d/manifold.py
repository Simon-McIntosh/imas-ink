"""Closed-manifold validation and repair for 3D meshes."""

from __future__ import annotations


class MeshNotManifoldError(Exception):
    """Raised when a mesh cannot be repaired to a closed manifold.

    Attributes
    ----------
    name : str
        Human-readable mesh identifier.
    n_open_edges : int
        Number of open (boundary) edges remaining after repair.
    """

    def __init__(self, *, name: str, n_open_edges: int):
        self.name = name
        self.n_open_edges = n_open_edges
        super().__init__(
            f"Mesh {name!r} has {n_open_edges} open edge(s) after repair"
        )


def ensure_closed_manifold(
    mesh,
    *,
    name: str = "<unnamed>",
    max_hole_size: float | None = None,
):
    """Validate and repair a mesh to ensure it is a closed manifold.

    Pipeline: ``triangulate() → clean() → fill_holes()`` (if any
    boundary edges) ``→ compute_normals(consistent + auto-orient)``.

    Parameters
    ----------
    mesh : pyvista.PolyData
        Input mesh.
    name : str
        Human-readable identifier for error messages.
    max_hole_size : float or None
        Maximum hole area for ``fill_holes``.  If *None*, auto-computed
        as twice the mesh bounding-box diagonal — generous enough to
        close any reasonable endcap hole.

    Returns
    -------
    pyvista.PolyData
        Repaired mesh with consistent outward-pointing normals.

    Raises
    ------
    MeshNotManifoldError
        If the mesh still has open edges after repair.
    """
    import pyvista as pv  # noqa: F811 — lazy import

    result = mesh.triangulate()
    result = result.clean()

    # A mesh with no polygon faces (e.g. a bare polyline) can never form
    # a closed manifold.  Detect this early to give a clear error.
    if result.n_cells == 0 or result.n_faces_strict == 0:
        raise MeshNotManifoldError(name=name, n_open_edges=max(result.n_cells, 1))

    # Detect open boundary edges
    boundary = result.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )

    if boundary.n_cells > 0:
        if max_hole_size is None:
            bnd = result.bounds
            diag = (
                (bnd[1] - bnd[0]) ** 2
                + (bnd[3] - bnd[2]) ** 2
                + (bnd[5] - bnd[4]) ** 2
            ) ** 0.5
            # Factor of 2 gives ample margin for endcap-sized holes.
            max_hole_size = diag * 2.0

        # PyVista's fill_holes wraps vtkFillHolesFilter.  It works well
        # for simple planar holes (endcaps) but can be unreliable on
        # complex topology or very small numerical gaps.  If it fails to
        # close all holes, MeshNotManifoldError is raised below.
        # Potential fallback: manual boundary-loop triangulation, but
        # that is left as future work.
        result = result.fill_holes(max_hole_size)

    result = result.compute_normals(
        consistent_normals=True,
        auto_orient_normals=True,
    )

    # Final validation: any remaining open edges?
    boundary = result.extract_feature_edges(
        boundary_edges=True,
        feature_edges=False,
        manifold_edges=False,
        non_manifold_edges=False,
    )
    n_open = boundary.n_cells

    if n_open > 0:
        raise MeshNotManifoldError(name=name, n_open_edges=n_open)

    return result
