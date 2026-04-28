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
    OPointMarker,
    RadialProfile,
    ScatterPoints,
    Separatrix,
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
    close_polygon,
    coil_bboxes,
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

# -- style -------------------------------------------------------------------
from .style import DEFAULT_STYLE, InkStyle


# -- 3D (lazy) ---------------------------------------------------------------
# MeshNotManifoldError and ensure_closed_manifold are re-exported here but
# lazily loaded: importing imas_ink does NOT pull pyvista/vtk.
def __getattr__(name: str):
    _THREE_D_NAMES = {"MeshNotManifoldError", "ensure_closed_manifold"}
    if name in _THREE_D_NAMES:
        from .three_d import manifold

        return getattr(manifold, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "DEFAULT_STYLE",
    "EMPTY_THRESHOLD",
    "CoilRect",
    "CoilRects",
    # contours
    "ContourExtractor",
    # types
    "EquilibriumSlice",
    # components
    "FluxContours",
    # style
    "InkStyle",
    "MachineGeometry",
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
    "close_polygon",
    "coil_bboxes",
    # 3D manifold validation (lazy — from imas_ink.three_d)
    "ensure_closed_manifold",
    "extract_geometry",
    "extract_profiles_1d",
    # extractors
    "extract_slice",
    "extract_time_traces",
    "find_xpoints",
    "is_closed_contour",
    # sentinel
    "is_empty",
    # cocos
    "make_levels",
    # geometry
    "mask_pfr",
    # altair backend
    "render_alt",
    # mpl backend
    "render_mpl",
    # I/O
    "render_to_bytes",
    "safe_float",
    "save_html",
    "save_png",
    "segments_to_dataframe",
    "split_path_segs",
    "wall_clip_vertices",
]
