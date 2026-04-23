"""High-level 3D scene composition and rendering.

Composes PF coils, TF coils, and the vessel wall into a single
pyvista plotter. All heavy imports are inside function bodies.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv


# Default colour palette
_COLORS = {
    "pf": "#2266cc",
    "cs": "#cc6622",
    "tf": "#888888",
    "vessel": "#9aa0a6",
    "first_wall": "#d7c58f",
    "wall": "#cccccc",  # legacy fallback
}

# Camera presets: (position, focal_point, viewup).  The iso preset sits
# further back than the old 30/20/15 so the full machine fits with room
# for cut-away animations.
_CAMERA_PRESETS: dict[str, tuple[tuple, tuple, tuple]] = {
    "iso": ((42, 30, 22), (0, 0, 0), (0, 0, 1)),
    "iso_close": ((30, 20, 15), (0, 0, 0), (0, 0, 1)),
    "poloidal": ((25, 0, 0), (0, 0, 0), (0, 0, 1)),
    "toroidal": ((0, 0, 30), (0, 0, 0), (0, 1, 0)),
}


def render_coilset(
    uri: str,
    outfile: pathlib.Path | None = None,
    *,
    show_wall: bool = True,
    show_pf: bool = True,
    show_tf: bool = True,
    show_first_wall: bool = True,
    show_vessel: bool = True,
    view: str = "iso",
    off_screen: bool = True,
    clip_normal: tuple[float, float, float] | None = None,
    clip_origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    tf_opacity: float = 1.0,
    vessel_opacity: float = 1.0,
    first_wall_opacity: float = 0.9,
    window_size: tuple[int, int] = (1280, 960),
) -> pv.Plotter:
    """Render a 3D tokamak coilset from an IMAS dataset.

    Opens the IMAS data at *uri*, extracts PF coils, TF coils, vessel
    and first-wall, assembles them into a :class:`pyvista.Plotter`, and
    optionally writes a PNG screenshot.

    Parameters
    ----------
    uri : str
        IMAS URI, e.g. ``"imas:hdf5?path=/path/to/machine/"``.
    outfile : pathlib.Path | None
        If given, save a PNG screenshot to this path.
    show_wall : bool
        Master switch for the vessel + first-wall.  When ``False``,
        neither is rendered regardless of the per-component flags.
    show_pf, show_tf, show_vessel, show_first_wall : bool
        Per-component visibility.
    view : str
        Camera preset: ``"iso"``, ``"iso_close"``, ``"poloidal"``, or
        ``"toroidal"``.
    off_screen : bool
        Run the plotter off-screen (headless). Default ``True``.
    clip_normal : tuple | None
        If given, clip every mesh with a plane of this normal (through
        *clip_origin*), keeping the half-space where ``n · (x-o) >= 0``
        is discarded.  Typical value ``(0, 1, 0)`` removes the ``y>0``
        half of the torus and exposes the poloidal cross-section at
        ``φ = 0`` / ``φ = π``.
    clip_origin : tuple
        Origin of the clipping plane.
    tf_opacity, vessel_opacity, first_wall_opacity : float
        Per-component opacities.  Default 1.0 / 1.0 / 0.9 (fully opaque
        bodies, with the first wall very slightly translucent so the
        plasma-facing side reads clearly against the vessel behind it).
    window_size : tuple
        Render window / screenshot resolution in pixels.

    Returns
    -------
    pyvista.Plotter
        The assembled plotter. If *outfile* was given, the screenshot
        has already been written.
    """
    import imas
    import pyvista as pv

    from .coilset import extract_pf_coils, extract_tf_coils, extract_walls

    entry = imas.DBEntry(uri, "r")
    try:
        pf_active = entry.get("pf_active") if show_pf else None
        tf_ids = entry.get("tf") if show_tf else None
        wall_ids = entry.get("wall") if show_wall else None
    finally:
        entry.close()

    pl = pv.Plotter(off_screen=off_screen, window_size=list(window_size))
    pl.set_background("white")

    def _maybe_clip(mesh: pv.PolyData) -> pv.PolyData:
        if clip_normal is None or mesh.n_points == 0:
            return mesh
        return mesh.clip(normal=clip_normal, origin=clip_origin, invert=False)

    # PF / CS coils
    if pf_active is not None:
        pf_meshes = extract_pf_coils(pf_active)
        for cm in pf_meshes:
            color = _COLORS["cs"] if cm.metadata.get("is_cs_segment") else _COLORS["pf"]
            pl.add_mesh(_maybe_clip(cm.mesh), color=color, opacity=1.0, label=cm.name)

    # TF coils
    if tf_ids is not None:
        tf_meshes = extract_tf_coils(tf_ids)
        for cm in tf_meshes:
            pl.add_mesh(
                _maybe_clip(cm.mesh),
                color=_COLORS["tf"],
                opacity=tf_opacity,
                label=cm.name,
                smooth_shading=True,
            )

    # Wall: vessel (outer) + first wall (plasma-facing).
    if wall_ids is not None:
        for name, mesh in extract_walls(wall_ids):
            if mesh.n_points == 0:
                continue
            if name == "vessel" and not show_vessel:
                continue
            if name == "first_wall" and not show_first_wall:
                continue
            if name == "vessel":
                colour, opacity = _COLORS["vessel"], vessel_opacity
            elif name == "first_wall":
                colour, opacity = _COLORS["first_wall"], first_wall_opacity
            else:
                colour, opacity = _COLORS["wall"], 1.0
            pl.add_mesh(
                _maybe_clip(mesh),
                color=colour,
                opacity=opacity,
                label=name,
                smooth_shading=True,
            )

    # Camera
    pos, focal, up = _CAMERA_PRESETS.get(view, _CAMERA_PRESETS["iso"])
    pl.camera_position = [pos, focal, up]

    if outfile is not None:
        outfile = pathlib.Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(outfile))

    return pl
