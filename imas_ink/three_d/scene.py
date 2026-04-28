"""High-level 3D scene composition and rendering.

Composes PF coils, TF coils, and the vessel wall into a single
pyvista plotter. All heavy imports are inside function bodies.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pyvista as pv


# Tokamak-appropriate palette: muted bronze/copper for PF, warm tungsten
# for FW, slate/gunmetal for TF and vessel.  Slightly desaturated from
# pure CAD tints so the rendered diffuse surfaces don't read as toy-like.
_COLORS = {
    "pf": "#a87a4a",          # muted bronze (PF coils)
    "cs": "#7a5638",          # darker bronze (central solenoid stack)
    "tf": "#6b7480",          # slate (TF case)
    "vessel": "#4a4d52",      # dark gunmetal (VV)
    "first_wall": "#a89a78",  # warm tungsten (FW)
    "wall": "#9aa0a6",        # legacy fallback
}

# Camera presets: (position, focal_point, viewup).
#
# - ``iso``        : far iso view, the full machine fits with room for
#                    cut-away animations.
# - ``iso_close``  : original close-in iso for unclipped renders.
# - ``poloidal``   : looking along +x (one half of the torus).
# - ``toroidal``   : looking down -z onto the equatorial plane.
# - ``cutaway_rhs``: ~normal to the xz cut plane (camera on +y side),
#                    tilted slightly toward +x and slightly above the
#                    midplane.  Use with ``clip_normal=(0,1,0)`` to keep
#                    the y >= 0 half-torus and view straight onto the
#                    RHS poloidal cross-section.
_CAMERA_PRESETS: dict[str, tuple[tuple, tuple, tuple]] = {
    "iso": ((42, 30, 22), (0, 0, 0), (0, 0, 1)),
    "iso_close": ((30, 20, 15), (0, 0, 0), (0, 0, 1)),
    "poloidal": ((25, 0, 0), (0, 0, 0), (0, 0, 1)),
    "toroidal": ((0, 0, 30), (0, 0, 0), (0, 1, 0)),
    # Camera viewed from -y (the cut side, since clip_normal=(0,1,0) keeps
    # y>=0 material): looking back along +y at the y=0 poloidal cross-
    # section.  Camera sits slightly inboard of the major-radius RHS so
    # the right-hand-side first wall + TF cut faces sit centre-frame.
    "cutaway_rhs": ((4, -28, 6), (6, 0, 1), (0, 0, 1)),
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
    # Single primary key light from upper-front (camera-relative high-left)
    # plus a soft ambient term gives a strong directional cue that makes
    # the 3D shapes read prominently — much cleaner than a 3-point kit
    # when components are tightly packed.
    pl.remove_all_lights()
    key = pv.Light(
        position=(20.0, -25.0, 30.0),
        focal_point=(0.0, 0.0, 0.0),
        color="white",
        intensity=1.0,
        light_type="scene light",
    )
    pl.add_light(key)
    # Faint ambient light to lift the shadowed side just enough to keep
    # detail visible without flattening the form.
    fill = pv.Light(
        position=(-15.0, 15.0, 5.0),
        focal_point=(0.0, 0.0, 0.0),
        color="white",
        intensity=0.18,
        light_type="scene light",
    )
    pl.add_light(fill)
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        # Older pyvista may not support ssaa; fall back silently.
        pass

    def _maybe_clip(mesh: pv.PolyData) -> pv.PolyData:
        if clip_normal is None or mesh.n_points == 0:
            return mesh
        return mesh.clip(normal=clip_normal, origin=clip_origin, invert=False)

    def _add(mesh, **kwargs):
        if mesh is None or mesh.n_points == 0:
            return
        pl.add_mesh(mesh, **kwargs)

    # Standard Phong-style shading with low specular — gives a clean
    # matte-engineering look without the metallic glare of PBR.
    _matte = dict(
        ambient=0.25,
        diffuse=0.85,
        specular=0.08,
        specular_power=8,
    )

    # PF / CS coils — bronze, smooth-shaded revolution surfaces.
    if pf_active is not None:
        pf_meshes = extract_pf_coils(pf_active)
        for cm in pf_meshes:
            color = _COLORS["cs"] if cm.metadata.get("is_cs_segment") else _COLORS["pf"]
            _add(
                _maybe_clip(cm.mesh),
                color=color,
                opacity=1.0,
                label=cm.name,
                smooth_shading=True,
                **_matte,
            )

    # TF coils — slate.  Flat shading preserves the rectangular
    # cross-section's edges instead of rounding them off.
    if tf_ids is not None:
        tf_meshes = extract_tf_coils(tf_ids)
        for cm in tf_meshes:
            _add(
                _maybe_clip(cm.mesh),
                color=_COLORS["tf"],
                opacity=tf_opacity,
                label=cm.name,
                smooth_shading=False,
                **_matte,
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
            _add(
                _maybe_clip(mesh),
                color=colour,
                opacity=opacity,
                label=name,
                smooth_shading=True,
                **_matte,
            )

    # Camera
    pos, focal, up = _CAMERA_PRESETS.get(view, _CAMERA_PRESETS["iso"])
    pl.camera_position = [pos, focal, up]

    if outfile is not None:
        outfile = pathlib.Path(outfile)
        outfile.parent.mkdir(parents=True, exist_ok=True)
        pl.screenshot(str(outfile))

    return pl


# ---------------------------------------------------------------------------
# Cutaway with flux projection
# ---------------------------------------------------------------------------


def _build_cutaway_static(
    *,
    wall_ids,
    pf_active,
    tf_ids,
    clip_plane,
    show_pf: bool,
    show_tf: bool,
    show_first_wall: bool,
    show_vessel: bool,
    synthesize_missing_vessel: bool,
):
    """Build clipped static geometry (meshes + caps) for a cutaway render.

    Returns ``(clipped_blocks, first_wall_outline)`` where *clipped_blocks*
    is a ``dict[str, CappedMesh]`` and *first_wall_outline* is a
    :class:`~imas_ink.three_d.walls.FirstWall` or ``None``.

    Separated from the plotter setup so that future animation code can
    cache the static geometry and only update the flux overlay per frame.
    """
    from .coilset import extract_pf_coils, extract_tf_coils
    from .cutaway import ClipPlane, capped_clip_multiblock
    from .walls import (
        extract_first_wall,
        extract_vessel_shells,
        revolve_wall_outline,
        synthesize_vessel_shell,
    )

    blocks: dict = {}

    # PF coils
    if show_pf and pf_active is not None:
        pf_meshes = extract_pf_coils(pf_active)
        for cm in pf_meshes:
            blocks[cm.name] = cm.mesh

    # TF coils
    if show_tf and tf_ids is not None:
        tf_meshes = extract_tf_coils(tf_ids)
        for cm in tf_meshes:
            blocks[cm.name] = cm.mesh

    # First wall
    first_wall = None
    if wall_ids is not None:
        first_wall = extract_first_wall(wall_ids)

    if show_first_wall and first_wall is not None:
        fw_mesh = revolve_wall_outline(first_wall)
        blocks["first_wall"] = fw_mesh

    # Vessel shells
    vessel_shells = []
    if wall_ids is not None:
        vessel_shells = extract_vessel_shells(wall_ids)

    if (
        synthesize_missing_vessel
        and not vessel_shells
        and first_wall is not None
        and show_vessel
    ):
        synth = synthesize_vessel_shell(first_wall)
        vessel_shells = [synth]

    if show_vessel:
        for vs in vessel_shells:
            vs_mesh = revolve_wall_outline(vs)
            blocks[vs.name] = vs_mesh

    # Clip all blocks
    clipped = capped_clip_multiblock(blocks, clip_plane)
    return clipped, first_wall


def render_cutaway_with_flux(
    uri: str,
    *,
    time_index: int = 0,
    clip_plane=None,
    flux_mode: str = "contours_and_field",
    n_levels: int = 12,
    show_pf: bool = True,
    show_tf: bool = True,
    show_first_wall: bool = True,
    show_vessel: bool = True,
    synthesize_missing_vessel: bool = False,
    view: str = "poloidal_rhs",
    overlay_margin: tuple[float, float] = (0.10, 0.05),
    window_size: tuple[int, int] = (1200, 900),
    outfile: str | pathlib.Path = "cutaway.png",
    dd_version: str | None = None,
    background: str = "white",
    show_colorbar: bool = True,
    show_title: bool = True,
) -> pathlib.Path:
    """Render a cutaway view with poloidal flux projected onto the first-wall cap.

    Default clip plane: ``y=0``, ``normal=(0, 1, 0)`` — RHS poloidal
    cross-section retained.  The flux overlay (ψ field and/or contour
    polylines) is projected onto the first-wall cap face produced by the
    capped clip.

    Parameters
    ----------
    uri : str
        IMAS URI pointing to a database with ``equilibrium``, ``wall``,
        and optionally ``pf_active`` / ``tf`` IDSs.
    time_index : int
        Equilibrium time-slice index.
    clip_plane : ClipPlane or None
        Half-space clip plane.  Default: ``ClipPlane((0,0,0), (0,1,0))``.
    flux_mode : str
        One of ``"contours_and_field"``, ``"field_only"``, ``"contours_only"``.
    n_levels : int
        Number of interior ψ contour levels.
    show_pf, show_tf, show_first_wall, show_vessel : bool
        Per-component visibility.
    synthesize_missing_vessel : bool
        If ``True`` and no vessel data exists, synthesise one from the
        first-wall outline (demo only).
    view : str
        Camera preset — currently ``"poloidal_rhs"`` only.
    overlay_margin : tuple[float, float]
        ``(left_fraction, top_fraction)`` reserved for overlays.
    window_size : tuple[int, int]
        Render window size in pixels.
    outfile : str or Path
        Output PNG path.
    dd_version : str or None
        Override the Data Dictionary version for ``imas.DBEntry``.
    background : str
        Background colour.
    show_colorbar : bool
        Add a ψ colourbar.
    show_title : bool
        Add a title with time and ψ range.

    Returns
    -------
    pathlib.Path
        Path to the written PNG file.
    """
    import numpy as np
    import pyvista as pv

    from .._dd import resolve_dd_version
    from .cutaway import ClipPlane, auto_camera
    from .equilibrium import extract_slice_2d
    from .flux_projection import build_flux_overlay, offset_along_normal

    # -- Resolve DD version and open IDS bundle -------------------------
    dd = resolve_dd_version(dd_version)

    import imas

    entry = imas.DBEntry(uri, "r", dd_version=dd)
    try:
        eq_ids = entry.get("equilibrium")
        wall_ids = _try_get(entry, "wall")
        pf_active = _try_get(entry, "pf_active") if show_pf else None
        tf_ids = _try_get(entry, "tf") if show_tf else None
    finally:
        entry.close()

    # -- Extract equilibrium slice --------------------------------------
    slice_2d = extract_slice_2d(eq_ids, time_index=time_index)

    # -- Default clip plane: y=0, keep +y half --------------------------
    if clip_plane is None:
        clip_plane = ClipPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))

    # -- Build static clipped geometry ----------------------------------
    clipped_blocks, first_wall = _build_cutaway_static(
        wall_ids=wall_ids,
        pf_active=pf_active,
        tf_ids=tf_ids,
        clip_plane=clip_plane,
        show_pf=show_pf,
        show_tf=show_tf,
        show_first_wall=show_first_wall,
        show_vessel=show_vessel,
        synthesize_missing_vessel=synthesize_missing_vessel,
    )

    # -- Build flux overlay on first-wall cap ---------------------------
    fw_capped = clipped_blocks.get("first_wall")
    overlay = None
    if fw_capped is not None and fw_capped.cap.n_cells > 0:
        cap_poly_2d = None
        if first_wall is not None:
            cap_poly_2d = (first_wall.r, first_wall.z)

        overlay = build_flux_overlay(
            fw_capped.cap,
            slice_2d,
            mode=flux_mode,
            n_levels=n_levels,
            plane_normal=clip_plane.normal,
            cap_polygon_2d=cap_poly_2d,
        )

    # -- Plotter setup --------------------------------------------------
    pv.OFF_SCREEN = True
    pl = pv.Plotter(off_screen=True, window_size=list(window_size))
    pl.set_background(background)

    # Lighting — same key + fill as render_coilset
    pl.remove_all_lights()
    key = pv.Light(
        position=(20.0, -25.0, 30.0),
        focal_point=(0.0, 0.0, 0.0),
        color="white",
        intensity=1.0,
        light_type="scene light",
    )
    pl.add_light(key)
    fill = pv.Light(
        position=(-15.0, 15.0, 5.0),
        focal_point=(0.0, 0.0, 0.0),
        color="white",
        intensity=0.18,
        light_type="scene light",
    )
    pl.add_light(fill)
    try:
        pl.enable_anti_aliasing("ssaa")
    except Exception:
        pass

    _matte = dict(ambient=0.25, diffuse=0.85, specular=0.08, specular_power=8)

    # -- Add clipped blocks to plotter ----------------------------------
    for block_name, capped in clipped_blocks.items():
        mesh = capped.full
        if mesh.n_points == 0:
            continue

        # Skip first_wall full mesh when we're rendering a flux overlay on it
        # (the cap with field scalars replaces the bare cap face)
        if block_name == "first_wall" and overlay is not None:
            # Add the non-cap body (full minus cap) with first_wall colour
            if overlay.field is not None:
                # Render the full mesh with cap region overridden by scalar
                pl.add_mesh(
                    mesh,
                    color=_COLORS["first_wall"],
                    opacity=0.9,
                    smooth_shading=True,
                    **_matte,
                )
            else:
                pl.add_mesh(
                    mesh,
                    color=_COLORS["first_wall"],
                    opacity=0.9,
                    smooth_shading=True,
                    **_matte,
                )
            continue

        # Colour lookup
        color = _COLORS.get(block_name, _COLORS.get("wall", "#9aa0a6"))
        for prefix, c in [("PF", "pf"), ("CS", "cs"), ("TF", "tf")]:
            if block_name.startswith(prefix):
                color = _COLORS[c]
                break
        if "vessel" in block_name:
            color = _COLORS["vessel"]

        opacity = 1.0
        smooth = True
        if block_name.startswith("TF"):
            smooth = False

        pl.add_mesh(
            mesh, color=color, opacity=opacity, smooth_shading=smooth, **_matte
        )

    # -- Add flux overlay -----------------------------------------------
    if overlay is not None:
        cap = fw_capped.cap
        normal_vec = clip_plane.normal

        # Field (ψ scalar on cap surface)
        if overlay.field is not None:
            cap_with_field = cap.copy(deep=True)
            cap_with_field.point_data["psi"] = overlay.field
            cap_offset = offset_along_normal(
                cap_with_field, normal_vec, epsilon=5e-4
            )
            pl.add_mesh(
                cap_offset,
                scalars="psi",
                cmap="RdBu_r",
                show_scalar_bar=show_colorbar,
                scalar_bar_args=dict(
                    title="ψ [Wb/rad]",
                    n_labels=5,
                    position_x=0.85,
                    position_y=0.1,
                    width=0.08,
                    height=0.7,
                ),
                smooth_shading=True,
                **_matte,
            )

        # Contour polylines
        for polyline in overlay.contours:
            if polyline.n_points < 2:
                continue
            offset_line = offset_along_normal(polyline, normal_vec, epsilon=1e-3)
            pl.add_mesh(
                offset_line,
                color="black",
                line_width=1.5,
                render_lines_as_tubes=False,
            )

    # -- Camera ---------------------------------------------------------
    # Compute combined bounds from all added meshes
    bounds = pl.bounds
    cam = auto_camera(
        bounds,
        view=view,
        overlay_margin=overlay_margin,
        window_size=window_size,
    )
    pl.camera_position = [cam["position"], cam["focal_point"], cam["view_up"]]
    pl.camera.parallel_projection = cam["parallel_projection"]
    pl.camera.parallel_scale = cam["parallel_scale"]

    # -- Title ----------------------------------------------------------
    if show_title and slice_2d is not None:
        title_parts = [f"t = {slice_2d.time:.3f} s"]
        if not np.isnan(slice_2d.psi_axis) and not np.isnan(slice_2d.psi_boundary):
            title_parts.append(
                f"ψ ∈ [{slice_2d.psi_boundary:.2f}, {slice_2d.psi_axis:.2f}] Wb/rad"
            )
        pl.add_text(
            "  ".join(title_parts),
            position="upper_left",
            font_size=10,
            color="black",
        )

    # -- Save -----------------------------------------------------------
    outfile = pathlib.Path(outfile)
    outfile.parent.mkdir(parents=True, exist_ok=True)
    pl.screenshot(str(outfile))
    pl.close()

    return outfile


def _try_get(entry, ids_name: str):
    """Try to get an IDS from an open DBEntry, returning None on failure."""
    try:
        return entry.get(ids_name)
    except Exception:
        return None
