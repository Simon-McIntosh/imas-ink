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
    "wall": "#cccccc",
}

# Camera presets: (position, focal_point, viewup)
_CAMERA_PRESETS: dict[str, tuple[tuple, tuple, tuple]] = {
    "iso": ((30, 20, 15), (0, 0, 0), (0, 0, 1)),
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
    view: str = "iso",
    off_screen: bool = True,
) -> pv.Plotter:
    """Render a 3D tokamak coilset from an IMAS dataset.

    Opens the IMAS data at *uri*, extracts PF coils, TF coils, and the
    vessel wall, assembles them into a :class:`pyvista.Plotter`, and
    optionally writes a PNG screenshot.

    Parameters
    ----------
    uri : str
        IMAS URI, e.g. ``"imas:hdf5?path=/path/to/machine/"``.
    outfile : pathlib.Path | None
        If given, save a PNG screenshot to this path.
    show_wall : bool
        Render the vessel / limiter wall.
    show_pf : bool
        Render PF / CS coils.
    show_tf : bool
        Render TF coils.
    view : str
        Camera preset: ``"iso"``, ``"poloidal"``, or ``"toroidal"``.
    off_screen : bool
        Run the plotter off-screen (headless). Default ``True``.

    Returns
    -------
    pyvista.Plotter
        The assembled plotter. If *outfile* was given, the screenshot
        has already been written.
    """
    import imas
    import pyvista as pv

    from .coilset import extract_pf_coils, extract_tf_coils, extract_wall

    entry = imas.DBEntry(uri, "r")
    try:
        pf_active = entry.get("pf_active") if show_pf else None
        tf_ids = entry.get("tf") if show_tf else None
        wall_ids = entry.get("wall") if show_wall else None
    finally:
        entry.close()

    pl = pv.Plotter(off_screen=off_screen)
    pl.set_background("white")

    # PF / CS coils
    if pf_active is not None:
        pf_meshes = extract_pf_coils(pf_active)
        for cm in pf_meshes:
            color = _COLORS["cs"] if cm.metadata.get("is_cs_segment") else _COLORS["pf"]
            pl.add_mesh(cm.mesh, color=color, opacity=1.0, label=cm.name)

    # TF coils
    if tf_ids is not None:
        tf_meshes = extract_tf_coils(tf_ids)
        for cm in tf_meshes:
            pl.add_mesh(cm.mesh, color=_COLORS["tf"], opacity=0.4, label=cm.name)

    # Wall
    if wall_ids is not None:
        wall_mesh = extract_wall(wall_ids)
        if wall_mesh.n_points > 0:
            pl.add_mesh(wall_mesh, color=_COLORS["wall"], opacity=0.25, label="wall")

    # Camera
    pos, focal, up = _CAMERA_PRESETS.get(view, _CAMERA_PRESETS["iso"])
    pl.camera_position = [pos, focal, up]

    if outfile is not None:
        outfile = pathlib.Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(outfile))

    return pl
