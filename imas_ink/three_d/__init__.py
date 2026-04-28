"""3D coilset and vessel rendering — pyvista + vtk.

All heavy imports (``pyvista``, ``vtk``) are deferred to function
bodies in submodules so that ``import imas_ink`` never pulls VTK.
Import from here for the public API::

    from imas_ink.three_d import render_coilset, extract_pf_coils
"""

from .coilset import CoilMesh, extract_pf_coils, extract_tf_coils, extract_wall
from .manifold import MeshNotManifoldError, ensure_closed_manifold
from .primitives import (
    cylindrical_to_cartesian,
    revolve_polygon,
    ring_from_rectangle,
    sweep_section_along_path,
)
from .scene import render_coilset
from .walls import (
    FirstWall,
    VesselShell,
    WallOutline2D,
    close_or_reject_outline,
    extract_first_wall,
    extract_vessel_shells,
    revolve_wall_outline,
    synthesize_vessel_shell,
)

__all__ = [
    "CoilMesh",
    "FirstWall",
    "MeshNotManifoldError",
    "VesselShell",
    "WallOutline2D",
    "close_or_reject_outline",
    "cylindrical_to_cartesian",
    "ensure_closed_manifold",
    "extract_first_wall",
    "extract_pf_coils",
    "extract_tf_coils",
    "extract_vessel_shells",
    "extract_wall",
    "render_coilset",
    "revolve_polygon",
    "revolve_wall_outline",
    "ring_from_rectangle",
    "sweep_section_along_path",
    "synthesize_vessel_shell",
]
