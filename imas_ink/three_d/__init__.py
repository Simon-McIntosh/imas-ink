"""3D coilset, vessel rendering, and equilibrium flux projection — pyvista + vtk.

All heavy imports (``pyvista``, ``vtk``, ``imas``, ``scipy``) are deferred to
function bodies in submodules so that ``import imas_ink`` never pulls them.
Import from here for the public API::

    from imas_ink.three_d import render_coilset, extract_pf_coils
    from imas_ink.three_d import EquilibriumSlice2D, read_equilibrium
"""

from .coilset import CoilMesh, extract_pf_coils, extract_tf_coils, extract_wall
from .equilibrium import (
    EquilibriumSlice2D,
    extract_slice_2d,
    psi_grid_interpolator,
    read_equilibrium,
)
from .flux_projection import (
    FluxMode,
    FluxOverlay,
    build_flux_overlay,
    contours_on_cap,
    offset_along_normal,
    sample_psi_on_cap,
)
from .cutaway import (
    CappedMesh,
    ClipPlane,
    auto_camera,
    cap_face_of,
    capped_clip,
    capped_clip_multiblock,
)
from .manifold import MeshNotManifoldError, ensure_closed_manifold
from .primitives import (
    cylindrical_to_cartesian,
    revolve_polygon,
    ring_from_rectangle,
    sweep_section_along_path,
)
from .scene import render_coilset, render_cutaway_with_flux
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
    "CappedMesh",
    "ClipPlane",
    "CoilMesh",
    "EquilibriumSlice2D",
    "FirstWall",
    "FluxMode",
    "FluxOverlay",
    "MeshNotManifoldError",
    "VesselShell",
    "WallOutline2D",
    "auto_camera",
    "build_flux_overlay",
    "cap_face_of",
    "capped_clip",
    "capped_clip_multiblock",
    "close_or_reject_outline",
    "contours_on_cap",
    "cylindrical_to_cartesian",
    "ensure_closed_manifold",
    "extract_first_wall",
    "extract_pf_coils",
    "extract_slice_2d",
    "extract_tf_coils",
    "extract_vessel_shells",
    "extract_wall",
    "offset_along_normal",
    "psi_grid_interpolator",
    "read_equilibrium",
    "render_coilset",
    "render_cutaway_with_flux",
    "revolve_polygon",
    "revolve_wall_outline",
    "ring_from_rectangle",
    "sample_psi_on_cap",
    "sweep_section_along_path",
    "synthesize_vessel_shell",
]
