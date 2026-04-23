"""Matplotlib rendering backend for ink visual components.

Each visual component type has a private ``_render_*_mpl`` function.
The public :func:`render_mpl` dispatches on ``isinstance`` checks.

No component module imports matplotlib — all mpl usage is confined here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import Rectangle

from .components import (
    CoilRects,
    FluxContours,
    FluxLoops,
    MagneticProbes,
    OPointMarker,
    RadialProfile,
    ScatterPoints,
    Separatrix,
    TimeLabel,
    TimeSeries,
    WallOutline,
    XPointMarkers,
)

if TYPE_CHECKING:
    from matplotlib.axes import Axes
    from matplotlib.patches import Patch


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------


def render_mpl(
    ax: Axes,
    component: object,
    /,
    clip_path: Patch | None = None,
) -> None:
    """Render *component* onto a matplotlib *ax*.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Target axes.
    component : ink component
        One of :class:`FluxContours`, :class:`Separatrix`, etc.
    clip_path : matplotlib.patches.Patch, optional
        Clip artist (typically a wall outline patch).  Applied to flux
        and separatrix line collections so contours do not extend
        outside the vessel.
    """
    if isinstance(component, FluxContours):
        _render_flux_mpl(ax, component, clip_path)
    elif isinstance(component, Separatrix):
        _render_sep_mpl(ax, component, clip_path)
    elif isinstance(component, WallOutline):
        _render_wall_mpl(ax, component)
    elif isinstance(component, CoilRects):
        _render_coils_mpl(ax, component)
    elif isinstance(component, OPointMarker):
        _render_opoint_mpl(ax, component)
    elif isinstance(component, XPointMarkers):
        _render_xpoints_mpl(ax, component)
    elif isinstance(component, MagneticProbes):
        _render_probes_mpl(ax, component)
    elif isinstance(component, FluxLoops):
        _render_fluxloops_mpl(ax, component)
    elif isinstance(component, TimeLabel):
        _render_timelabel_mpl(ax, component)
    elif isinstance(component, TimeSeries):
        _render_timeseries_mpl(ax, component)
    elif isinstance(component, RadialProfile):
        _render_radialprofile_mpl(ax, component)
    elif isinstance(component, ScatterPoints):
        _render_scatter_mpl(ax, component)
    else:
        raise TypeError(f"Unknown component type: {type(component).__name__}")


# ---------------------------------------------------------------------------
# Private renderers
# ---------------------------------------------------------------------------


def _render_flux_mpl(
    ax: Axes,
    fc: FluxContours,
    clip_path: Patch | None = None,
) -> None:
    """Render flux surface contours as LineCollections."""
    s = fc.style
    for level_segs in fc.segments:
        if not level_segs:
            continue
        lc = LineCollection(
            level_segs,
            colors=s.flux_color,
            linewidths=s.flux_linewidth,
            linestyles=s.flux_linestyle,
            zorder=s.zorder_flux,
        )
        if clip_path is not None:
            lc.set_clip_path(clip_path)
        ax.add_collection(lc)


def _render_sep_mpl(
    ax: Axes,
    sep: Separatrix,
    clip_path: Patch | None = None,
) -> None:
    """Render separatrix segments."""
    s = sep.style
    segs = sep.segments

    if not segs:
        return

    lc = LineCollection(
        segs,
        colors=s.sep_color,
        linewidths=s.sep_linewidth,
        linestyles=s.sep_linestyle,
        zorder=s.zorder_sep,
    )
    if clip_path is not None:
        lc.set_clip_path(clip_path)
    ax.add_collection(lc)


def _render_wall_mpl(ax: Axes, wall: WallOutline) -> None:
    """Plot the first-wall outline polygon."""
    s = wall.style
    ax.plot(
        wall.wall_r,
        wall.wall_z,
        color=s.wall_color,
        linewidth=s.wall_linewidth,
        zorder=s.zorder_wall,
    )


def _render_coils_mpl(ax: Axes, coils: CoilRects) -> None:
    """Draw PF coil bounding boxes as Rectangle patches."""
    s = coils.style
    for cr in coils.rects:
        rect = Rectangle(
            (cr.r, cr.z),
            cr.width,
            cr.height,
            edgecolor=s.coil_edgecolor,
            facecolor=s.coil_facecolor,
            linewidth=s.coil_linewidth,
            zorder=s.zorder_coils,
        )
        ax.add_patch(rect)


def _render_opoint_mpl(ax: Axes, opoint: OPointMarker) -> None:
    """Plot the magnetic axis marker."""
    s = opoint.style
    ax.plot(
        opoint.r,
        opoint.z,
        marker=s.axis_marker,
        markersize=s.axis_markersize,
        color=s.axis_color,
        linestyle="none",
        zorder=s.zorder_markers,
    )


def _render_xpoints_mpl(ax: Axes, xpoints: XPointMarkers) -> None:
    """Plot X-point markers."""
    s = xpoints.style
    for r, z in xpoints.points:
        ax.plot(
            r,
            z,
            marker=s.xpt_marker,
            markersize=s.xpt_markersize,
            markeredgewidth=s.xpt_markeredgewidth,
            color=s.xpt_color,
            linestyle="none",
            zorder=s.zorder_markers,
        )


def _render_probes_mpl(ax: Axes, probes: MagneticProbes) -> None:
    """Plot B-pol magnetic probes with optional orientation tick marks."""
    s = probes.style
    r = np.asarray(probes.positions_r)
    z = np.asarray(probes.positions_z)
    ang = np.asarray(probes.angles)

    if r.size == 0:
        return

    ax.scatter(
        r,
        z,
        s=s.probe_markersize**2,
        c=s.probe_color,
        marker="o",
        linewidths=0,
        zorder=s.zorder_probes,
    )

    finite = np.isfinite(ang)
    if np.any(finite):
        L = s.probe_arrow_length
        r_f = r[finite]
        z_f = z[finite]
        a_f = ang[finite]
        segs = [
            [(r_f[i], z_f[i]), (r_f[i] + L * np.cos(a_f[i]), z_f[i] + L * np.sin(a_f[i]))]
            for i in range(r_f.size)
        ]
        lc = LineCollection(
            segs,
            colors=s.probe_color,
            linewidths=s.probe_arrow_linewidth,
            zorder=s.zorder_probes,
        )
        ax.add_collection(lc)


def _render_fluxloops_mpl(ax: Axes, loops: FluxLoops) -> None:
    """Plot flux loop positions."""
    s = loops.style
    r = np.asarray(loops.positions_r)
    z = np.asarray(loops.positions_z)
    if r.size == 0:
        return
    ax.scatter(
        r,
        z,
        s=s.flux_loop_markersize**2,
        c=s.flux_loop_color,
        marker="s",
        linewidths=0,
        zorder=s.zorder_flux_loops,
    )


def _render_timelabel_mpl(ax: Axes, label: TimeLabel) -> None:
    """Add time annotation in the upper-right corner."""
    s = label.style
    text = f"t = {label.time:.4f} s"
    ax.text(
        0.97,
        0.97,
        text,
        transform=ax.transAxes,
        fontsize=s.label_fontsize,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=s.label_bbox,
        zorder=s.zorder_label,
    )


def _render_timeseries_mpl(ax: Axes, ts: TimeSeries) -> None:
    """Render a time series line plot."""
    s = ts.style
    ax.plot(ts.time, ts.values, linewidth=s.trace_linewidth, label=ts.label or None)
    ylabel = f"{ts.ylabel} [{ts.units}]" if ts.units else ts.ylabel
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("Time [s]")
    if ts.label:
        ax.legend(fontsize=s.label_fontsize)
    ax.tick_params(labelsize=s.label_fontsize)


def _render_radialprofile_mpl(ax: Axes, rp: RadialProfile) -> None:
    """Render a 1D radial profile."""
    s = rp.style
    ax.plot(rp.psi_norm, rp.values, linewidth=s.trace_linewidth, label=rp.label or None)
    ylabel = f"{rp.ylabel} [{rp.units}]" if rp.units else rp.ylabel
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_xlabel("ψ_norm")
    if rp.label:
        ax.legend(fontsize=s.label_fontsize)
    ax.tick_params(labelsize=s.label_fontsize)


def _render_scatter_mpl(ax: Axes, sc: ScatterPoints) -> None:
    """Render scatter points (e.g. measured vs fitted)."""
    s = sc.style
    ax.scatter(sc.x, sc.y, s=s.trace_markersize**2, label=sc.label or None)
    if sc.xlabel:
        ax.set_xlabel(sc.xlabel)
    if sc.ylabel:
        ax.set_ylabel(sc.ylabel)
    if sc.label:
        ax.legend(fontsize=s.label_fontsize)
    ax.tick_params(labelsize=s.label_fontsize)
