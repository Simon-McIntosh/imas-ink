"""imas-ink — IMAS-backed plotting and visualisation for tokamak equilibrium data.

The top-level module exposes the **library API only**.  The MCP server
lives at :mod:`imas_ink.server` and is intentionally not re-exported
from here, so that ``import imas_ink`` does not pull FastMCP.

The :mod:`imas_ink.three_d` subpackage is also not eagerly imported, so
that VTK / pyvista / vedo are only loaded when 3D rendering is actually
requested.

Quick start::

    from imas_ink import extract_slice, ContourExtractor, FluxContours

    sl = extract_slice(eq_ids, 5)
    cx = ContourExtractor(sl.R_2d, sl.Z_2d, sl.psi_2d)
    fc = FluxContours(cx.flux_surfaces(sl.psi_axis, sl.psi_boundary))

Public API is stabilised at v0.1.0. Until then, expect churn.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # source checkout without hatch-vcs run
    __version__ = "0.0.0.dev0"

# -- sentinel ----------------------------------------------------------------
# -- COCOS -------------------------------------------------------------------
from ._cocos import make_levels
from ._dd import DEFAULT_DD_VERSION, resolve_dd_version
from ._sentinel import EMPTY_THRESHOLD, is_empty, safe_float

# -- types -------------------------------------------------------------------
from ._types import (
    CoilRect,
    EquilibriumSlice,
    MachineGeometry,
    RadialProfiles,
    TimeTraces,
    XPoint,
)

# -- altair backend ----------------------------------------------------------
from .alt import render_alt, segments_to_dataframe

# -- animation ---------------------------------------------------------------
from .animate import animate_pulse

# -- components --------------------------------------------------------------
from .components import (
    CoilRects,
    FluxContours,
    FluxLoops,
    LcfsOutline,
    MagneticProbes,
    OPointMarker,
    RadialProfile,
    ScatterPoints,
    Separatrix,
    SolContours,
    TimeLabel,
    TimeSeries,
    WallOutline,
    XPointMarkers,
)

# -- contours ----------------------------------------------------------------
from .contours import ContourExtractor

# -- extractors --------------------------------------------------------------
from .extract import (
    extract_geometry,
    extract_profiles_1d,
    extract_slice,
    extract_time_traces,
)

# -- geometry ----------------------------------------------------------------
from .geometry import (
    classify_flux_segments,
    close_polygon,
    coil_bboxes,
    encloses_point,
    find_xpoints,
    is_closed_contour,
    mask_pfr,
    split_path_segs,
    wall_clip_vertices,
)

# -- I/O utilities -----------------------------------------------------------
from .io import render_to_bytes, save_html, save_png

# -- matplotlib backend ------------------------------------------------------
from .mpl import render_mpl

# -- figure builders ---------------------------------------------------------
from .figures import (
    equilibrium_chart_alt,
    equilibrium_figure_mpl,
    geometry_figure_mpl,
    radial_profile_figure_mpl,
    time_trace_figure_mpl,
)

# -- style -------------------------------------------------------------------
from .style import DEFAULT_STYLE, InkStyle


# -- 3D (lazy) ---------------------------------------------------------------
# MeshNotManifoldError and ensure_closed_manifold are re-exported here but
# lazily loaded: importing imas_ink does NOT pull pyvista/vtk.
def __getattr__(name: str):
    _MANIFOLD_NAMES = {"MeshNotManifoldError", "ensure_closed_manifold"}
    _WALL_NAMES = {
        "WallOutline2D",
        "FirstWall",
        "VesselShell",
        "extract_first_wall",
        "extract_vessel_shells",
        "close_or_reject_outline",
        "revolve_wall_outline",
        "synthesize_vessel_shell",
    }
    _CUTAWAY_NAMES = {
        "ClipPlane",
        "CappedMesh",
        "capped_clip",
        "cap_face_of",
        "capped_clip_multiblock",
        "auto_camera",
    }
    _EQUILIBRIUM_NAMES = {
        "EquilibriumSlice2D",
        "read_equilibrium",
        "extract_slice_2d",
        "psi_grid_interpolator",
    }
    _FLUX_PROJECTION_NAMES = {
        "FluxOverlay",
        "FluxMode",
        "sample_psi_on_cap",
        "contours_on_cap",
        "build_flux_overlay",
        "offset_along_normal",
    }
    _SCENE_NAMES = {
        "render_cutaway_with_flux",
    }
    if name in _MANIFOLD_NAMES:
        from .three_d import manifold

        return getattr(manifold, name)
    if name in _WALL_NAMES:
        from .three_d import walls

        return getattr(walls, name)
    if name in _CUTAWAY_NAMES:
        from .three_d import cutaway

        return getattr(cutaway, name)
    if name in _EQUILIBRIUM_NAMES:
        from .three_d import equilibrium

        return getattr(equilibrium, name)
    if name in _FLUX_PROJECTION_NAMES:
        from .three_d import flux_projection

        return getattr(flux_projection, name)
    if name in _SCENE_NAMES:
        from .three_d import scene

        return getattr(scene, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_DD_VERSION",
    "DEFAULT_STYLE",
    "EMPTY_THRESHOLD",
    "CoilRect",
    "CoilRects",
    # contours
    "ContourExtractor",
    # types
    "EquilibriumSlice",
    # 3D equilibrium (lazy — from imas_ink.three_d.equilibrium)
    "EquilibriumSlice2D",
    # components
    "FluxContours",
    "LcfsOutline",
    "SolContours",
    # 3D flux projection (lazy — from imas_ink.three_d.flux_projection)
    "FluxMode",
    "FluxOverlay",
    # style
    "InkStyle",
    "MachineGeometry",
    # 3D cutaway (lazy — from imas_ink.three_d.cutaway)
    "CappedMesh",
    "ClipPlane",
    # 3D walls (lazy — from imas_ink.three_d.walls)
    "FirstWall",
    # 3D manifold validation (lazy — from imas_ink.three_d)
    "MeshNotManifoldError",
    "OPointMarker",
    "RadialProfile",
    "RadialProfiles",
    "ScatterPoints",
    "Separatrix",
    "TimeLabel",
    "TimeSeries",
    "TimeTraces",
    "WallOutline",
    "XPoint",
    "XPointMarkers",
    # version
    "__version__",
    # animation
    "animate_pulse",
    # 3D cutaway (lazy — from imas_ink.three_d.cutaway)
    "auto_camera",
    # 3D flux projection (lazy — from imas_ink.three_d.flux_projection)
    "build_flux_overlay",
    "cap_face_of",
    "capped_clip",
    "capped_clip_multiblock",
    # 3D walls (lazy — from imas_ink.three_d.walls)
    "close_or_reject_outline",
    "close_polygon",
    "coil_bboxes",
    # 3D flux projection (lazy — from imas_ink.three_d.flux_projection)
    "contours_on_cap",
    # 3D manifold validation (lazy — from imas_ink.three_d)
    "ensure_closed_manifold",
    # 3D walls (lazy — from imas_ink.three_d.walls)
    "extract_first_wall",
    "extract_geometry",
    "extract_profiles_1d",
    # extractors
    "extract_slice",
    # 3D equilibrium (lazy — from imas_ink.three_d.equilibrium)
    "extract_slice_2d",
    "extract_time_traces",
    # 3D walls (lazy — from imas_ink.three_d.walls)
    "extract_vessel_shells",
    "classify_flux_segments",
    "encloses_point",
    "find_xpoints",
    "is_closed_contour",
    # sentinel
    "is_empty",
    # cocos
    "make_levels",
    # geometry
    "mask_pfr",
    # 3D flux projection (lazy — from imas_ink.three_d.flux_projection)
    "offset_along_normal",
    # 3D equilibrium (lazy — from imas_ink.three_d.equilibrium)
    "psi_grid_interpolator",
    # 3D equilibrium (lazy — from imas_ink.three_d.equilibrium)
    "read_equilibrium",
    # 3D scene (lazy — from imas_ink.three_d.scene)
    "render_cutaway_with_flux",
    # altair backend
    "render_alt",
    # mpl backend
    "render_mpl",
    # 3D walls (lazy — from imas_ink.three_d.walls)
    "revolve_wall_outline",
    # DD version
    "resolve_dd_version",
    # I/O
    "render_to_bytes",
    "safe_float",
    # 3D flux projection (lazy — from imas_ink.three_d.flux_projection)
    "sample_psi_on_cap",
    "save_html",
    "save_png",
    "segments_to_dataframe",
    "split_path_segs",
    # 3D walls (lazy — from imas_ink.three_d.walls)
    "synthesize_vessel_shell",
    "VesselShell",
    "WallOutline2D",
    "wall_clip_vertices",
]
