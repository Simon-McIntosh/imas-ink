"""Composed figure builders — one call produces a full plot.

Combines extraction, contouring, component construction, and rendering
into single-call convenience functions. Each function returns a
``(fig, ax)`` pair so the caller can further customise or save.

Architecture
------------
``geometry_figure_mpl`` creates a clean machine cross-section (wall,
coils, probes, flux loops) — no equilibrium data required. Higher-level
builders like ``equilibrium_figure_mpl`` call it and then layer
equilibrium-specific components on top.  Agents can also compose
layers via the persistent REPL using ``render_mpl(ax, component)``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from ._types import EquilibriumSlice, MachineGeometry
from .components import (
    CoilRects,
    FluxContours,
    FluxLoops,
    MagneticProbes,
    OPointMarker,
    Separatrix,
    TimeLabel,
    WallOutline,
    XPointMarkers,
)
from .contours import ContourExtractor
from .geometry import mask_pfr
from .mpl import render_mpl
from .style import DEFAULT_STYLE, InkStyle

if TYPE_CHECKING:
    import altair as alt
    import matplotlib.figure
    from matplotlib.axes import Axes


def time_trace_figure_mpl(
    traces: list,
    style: InkStyle | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[matplotlib.figure.Figure, list[Axes]]:
    """Create a multi-panel time trace figure.

    Parameters
    ----------
    traces : list[TimeSeries]
        List of time series components to render, one per subplot.
    style : InkStyle, optional
        Visual style. Defaults to :data:`DEFAULT_STYLE`.
    figsize : tuple, optional
        Figure size in inches. Defaults to ``(8, 2 * n_panels)``.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The figure and list of axes (one per panel).
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    if style is None:
        style = DEFAULT_STYLE

    n = len(traces)
    if figsize is None:
        figsize = (8, 2.0 * max(n, 1))

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.ravel()

    for ax_i, ts in zip(axes, traces, strict=False):
        render_mpl(ax_i, ts)

    fig.tight_layout()
    return fig, list(axes)


def radial_profile_figure_mpl(
    profiles: list,
    style: InkStyle | None = None,
    figsize: tuple[float, float] | None = None,
) -> tuple[matplotlib.figure.Figure, list[Axes]]:
    """Create a multi-panel radial profile figure.

    Parameters
    ----------
    profiles : list[RadialProfile]
        List of radial profile components, one per subplot.
    style : InkStyle, optional
        Visual style. Defaults to :data:`DEFAULT_STYLE`.
    figsize : tuple, optional
        Figure size in inches.

    Returns
    -------
    tuple[Figure, list[Axes]]
        The figure and list of axes.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    if style is None:
        style = DEFAULT_STYLE

    n = len(profiles)
    if figsize is None:
        figsize = (8, 2.5 * max(n, 1))

    fig, axes = plt.subplots(n, 1, figsize=figsize, sharex=True, squeeze=False)
    axes = axes.ravel()

    for ax_i, rp in zip(axes, profiles, strict=False):
        render_mpl(ax_i, rp)

    fig.tight_layout()
    return fig, list(axes)


def geometry_figure_mpl(
    geom: MachineGeometry,
    style: InkStyle | None = None,
    figsize: tuple[float, float] = (6, 7),
    show_probes: bool = True,
    show_flux_loops: bool = True,
) -> tuple[matplotlib.figure.Figure, Axes]:
    """Build a poloidal cross-section showing only machine geometry.

    Renders wall outline, PF coils, magnetic probes, and flux loops —
    no equilibrium data required.  Returns ``(fig, ax)`` for further
    composition (e.g. layering equilibrium contours on top).

    Parameters
    ----------
    geom : MachineGeometry
        Static machine geometry (wall, coils, probes, flux loops).
    style : InkStyle, optional
        Visual style. Defaults to :data:`DEFAULT_STYLE`.
    figsize : tuple[float, float]
        Figure size in inches ``(width, height)``.
    show_probes : bool
        If *True* and probe positions are available, render probe markers.
    show_flux_loops : bool
        If *True* and flux loop positions are available, render loop markers.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    import matplotlib
    import matplotlib.pyplot as plt

    matplotlib.use("Agg")

    if style is None:
        style = DEFAULT_STYLE

    fig, ax = plt.subplots(figsize=figsize, facecolor=style.figure_facecolor)

    # --- geometry components ---
    coils = CoilRects(geom.coil_rects, style=style)
    wall = WallOutline(geom.wall_r, geom.wall_z, style=style)

    render_mpl(ax, coils)
    render_mpl(ax, wall)
    if show_probes and geom.probe_r.size > 0:
        probes = MagneticProbes(
            positions_r=geom.probe_r,
            positions_z=geom.probe_z,
            angles=geom.probe_angle,
            style=style,
        )
        render_mpl(ax, probes)
    if show_flux_loops and geom.flux_loop_r.size > 0:
        loops = FluxLoops(
            positions_r=geom.flux_loop_r,
            positions_z=geom.flux_loop_z,
            style=style,
        )
        render_mpl(ax, loops)

    # --- viewport ---
    rmin, rmax, zmin, zmax = geom.viewport
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(zmin, zmax)
    ax.set_aspect("equal")
    ax.axis("off")

    fig.tight_layout()
    return fig, ax


def equilibrium_figure_mpl(
    sl: EquilibriumSlice,
    geom: MachineGeometry,
    style: InkStyle | None = None,
    figsize: tuple[float, float] = (6, 7),
    mask_pfr_flag: bool = True,
    show_probes: bool = True,
    show_flux_loops: bool = True,
    show_vacuum_surfaces: bool = False,
) -> tuple[matplotlib.figure.Figure, Axes]:
    """Build a complete poloidal cross-section figure.

    Creates the machine geometry via :func:`geometry_figure_mpl` and
    layers equilibrium-specific components on top: flux contours,
    separatrix, O-point, X-point markers, and time label.

    Parameters
    ----------
    sl : EquilibriumSlice
        Single time-slice equilibrium data.
    geom : MachineGeometry
        Static machine geometry (wall + coils).
    style : InkStyle, optional
        Visual style. Defaults to :data:`DEFAULT_STYLE`.
    figsize : tuple[float, float]
        Figure size in inches ``(width, height)``.
    mask_pfr_flag : bool
        If *True*, mask the private flux region before contouring so
        contour lines do not extend into the PFR.
    show_probes : bool
        If *True* and probe positions are available, render probe markers.
    show_flux_loops : bool
        If *True* and flux loop positions are available, render loop markers.
    show_vacuum_surfaces : bool
        If *True*, render vacuum / SOL flux surface contours in light grey
        outside the LCFS.  Levels span the full psi range on the
        computational grid and are **not** clipped to the wall — this makes
        unconverged frames (LCFS outside the first wall, spurious O-points)
        fully visible.  Uses :attr:`InkStyle.vacuum_color`,
        :attr:`InkStyle.vacuum_linewidth`, and
        :attr:`InkStyle.vacuum_n_levels`.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    from matplotlib.patches import PathPatch
    from matplotlib.path import Path as MplPath

    if style is None:
        style = DEFAULT_STYLE

    # --- base geometry layer ---
    fig, ax = geometry_figure_mpl(
        geom, style=style, figsize=figsize,
        show_probes=show_probes, show_flux_loops=show_flux_loops,
    )

    # --- psi field (optionally mask PFR) ---
    psi = sl.psi_2d
    if mask_pfr_flag:
        psi = mask_pfr(
            psi,
            sl.R_2d,
            sl.Z_2d,
            sl.psi_axis,
            sl.psi_boundary,
            sl.r_axis,
            sl.z_axis,
        )

    # --- contour extraction ---
    cx = ContourExtractor(sl.R_2d, sl.Z_2d, psi)
    n_levels = style.flux_n_levels

    # --- wall clip path ---
    wall_path = MplPath(geom.wall_clip_vertices)
    wall_patch = PathPatch(wall_path, facecolor="none", edgecolor="none")
    ax.add_patch(wall_patch)

    # --- equilibrium components ---
    flux = FluxContours(
        cx.flux_surfaces(sl.psi_axis, sl.psi_boundary, n=n_levels),
        style=style,
    )
    sep = Separatrix(
        cx.separatrix(sl.psi_boundary),
        x_points=sl.x_points,
        style=style,
    )
    opoint = OPointMarker(sl.r_axis, sl.z_axis, style=style)
    xpoints = XPointMarkers(sl.x_points, style=style)
    timelabel = TimeLabel(sl.time, converged=sl.converged, style=style)

    render_mpl(ax, flux, clip_path=wall_patch)
    render_mpl(ax, sep, clip_path=wall_patch)
    render_mpl(ax, opoint)
    render_mpl(ax, xpoints)
    render_mpl(ax, timelabel)

    # --- vacuum surfaces (unclipped, full grid) ---
    if show_vacuum_surfaces:
        import dataclasses

        vac_style = dataclasses.replace(
            style,
            flux_color=style.vacuum_color,
            flux_linewidth=style.vacuum_linewidth,
            flux_linestyle=style.vacuum_linestyle,
        )
        cx_vac = ContourExtractor(sl.R_2d, sl.Z_2d, sl.psi_2d)
        vac_segs = cx_vac.vacuum_surfaces(
            sl.psi_axis, sl.psi_boundary, n=style.vacuum_n_levels
        )
        render_mpl(ax, FluxContours(vac_segs, style=vac_style))

    return fig, ax


def equilibrium_chart_alt(
    sl: EquilibriumSlice,
    geom: MachineGeometry,
    style: InkStyle | None = None,
    mask_pfr_flag: bool = True,
) -> alt.LayerChart:
    """Build a complete Altair equilibrium cross-section chart.

    Assembles flux contours, separatrix, wall outline, PF coils,
    magnetic axis marker, X-point markers, and a time label into a
    single layered Altair chart.

    Parameters
    ----------
    sl : EquilibriumSlice
        Single time-slice equilibrium data.
    geom : MachineGeometry
        Static machine geometry (wall + coils).
    style : InkStyle | None
        Visual style. Falls back to :data:`DEFAULT_STYLE`.
    mask_pfr_flag : bool
        If ``True``, mask the private flux region before contouring.

    Returns
    -------
    alt.LayerChart
        Layered Vega-Lite chart ready for ``.save()`` or notebook display.
    """
    import altair as alt

    from .alt import render_alt

    if style is None:
        style = DEFAULT_STYLE

    # Optionally mask the private flux region.
    psi = sl.psi_2d
    if mask_pfr_flag:
        psi = mask_pfr(
            psi,
            sl.R_2d,
            sl.Z_2d,
            sl.psi_axis,
            sl.psi_boundary,
            sl.r_axis,
            sl.z_axis,
        )

    cx = ContourExtractor(sl.R_2d, sl.Z_2d, psi)

    # Build components.
    fc = FluxContours(
        cx.flux_surfaces(sl.psi_axis, sl.psi_boundary, n=style.flux_n_levels),
        style=style,
    )
    sep = Separatrix(
        cx.separatrix(sl.psi_boundary),
        x_points=sl.x_points,
        style=style,
    )
    wall = WallOutline(geom.wall_r, geom.wall_z, style=style)
    coils = CoilRects(geom.coil_rects, style=style)
    opoint = OPointMarker(sl.r_axis, sl.z_axis, style=style)
    xpoints = XPointMarkers(sl.x_points, style=style)
    timelabel = TimeLabel(sl.time, converged=sl.converged, style=style)

    # Render each layer.
    layers: list = [
        render_alt(fc),
        render_alt(sep),
        render_alt(wall),
        render_alt(coils),
        render_alt(opoint),
        render_alt(xpoints),
        render_alt(timelabel),
    ]

    return alt.layer(*layers).properties(width=style.altair_width, height=style.altair_height)
