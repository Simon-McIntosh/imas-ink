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
    DivertorLegs,
    FluxContours,
    FluxLoops,
    LcfsOutline,
    MagneticProbes,
    OPointMarker,
    RadialProfile,
    ReferenceContours,
    ReferenceLcfs,
    ReferenceXPoints,
    ScatterPoints,
    Separatrix,
    SolContours,
    StrikePoints,
    TimeLabel,
    TimeSeries,
    WallOutline,
    XPointMarkers,
)
from .geometry import classify_probe_components

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
    if isinstance(component, ReferenceContours):
        _render_ref_contours_mpl(ax, component)
    elif isinstance(component, ReferenceLcfs):
        _render_ref_lcfs_mpl(ax, component)
    elif isinstance(component, ReferenceXPoints):
        _render_ref_xpoints_mpl(ax, component)
    elif isinstance(component, FluxContours):
        _render_flux_mpl(ax, component, clip_path)
    elif isinstance(component, SolContours):
        _render_sol_mpl(ax, component)
    elif isinstance(component, LcfsOutline):
        _render_lcfs_mpl(ax, component)
    elif isinstance(component, DivertorLegs):
        _render_legs_mpl(ax, component)
    elif isinstance(component, StrikePoints):
        _render_strikes_mpl(ax, component)
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


def _render_sol_mpl(ax: Axes, sol: SolContours) -> None:
    """Render SOL / open-field contours at reduced weight in grey.

    These are surfaces that do not enclose the magnetic axis — open
    field lines, private-flux lobes, and grid-edge artefacts.  They are
    rendered *unclipped* (no wall clip path) so they remain visible as a
    diagnostic even when they extend outside the vessel boundary.
    """
    s = sol.style
    for level_segs in sol.segments:
        if not level_segs:
            continue
        lc = LineCollection(
            level_segs,
            colors=s.sol_color,
            linewidths=s.sol_linewidth,
            linestyles=s.sol_linestyle,
            zorder=s.zorder_flux,
        )
        ax.add_collection(lc)


def _render_ref_contours_mpl(ax: Axes, ref: ReferenceContours) -> None:
    """Render validation-reference flux contours as a faint underlay.

    Draws at ``zorder_ref`` (below everything else) so the reference topology
    is visible without obscuring the primary reconstruction.  Each level group
    is rendered as a single ``LineCollection`` for efficiency.

    A legend entry is added using ``ax.plot([], [], ...)`` with the label
    ``"reference (NAME)"`` so the caller's legend picks it up.
    """
    s = ref.style
    has_segs = False
    for level_segs in ref.segments:
        if not level_segs:
            continue
        lc = LineCollection(
            level_segs,
            colors=s.ref_color,
            linewidths=s.ref_linewidth,
            linestyles=s.ref_linestyle,
            alpha=s.ref_alpha,
            zorder=s.zorder_ref,
        )
        ax.add_collection(lc)
        has_segs = True
    # Proxy artist for legend
    if has_segs:
        label = f"reference ({ref.ref_name})"
        match_note = "" if ref.psi_matched else " [ψ_norm]"
        ax.plot(
            [],
            [],
            color=s.ref_color,
            linewidth=s.ref_linewidth,
            linestyle=s.ref_linestyle,
            alpha=max(s.ref_alpha + 0.2, 0.6),
            label=label + match_note,
            zorder=s.zorder_ref,
        )


def _render_ref_lcfs_mpl(ax: Axes, ref_lcfs: ReferenceLcfs) -> None:
    """Render the validation-reference LCFS as a faint dashed outline.

    Drawn verbatim from ``boundary.outline.r/z``.  When the arrays are empty
    (reference IDS has no boundary data), nothing is rendered.  A legend entry
    is added automatically.
    """
    r = np.asarray(ref_lcfs.r)
    z = np.asarray(ref_lcfs.z)
    if r.size < 2 or z.size < 2:
        return
    s = ref_lcfs.style
    # Ensure closure for rendering
    if not (np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1])):
        r = np.append(r, r[0])
        z = np.append(z, z[0])
    label = f"ref LCFS ({ref_lcfs.ref_name})"
    ax.plot(
        r,
        z,
        color=s.ref_lcfs_color,
        linewidth=s.ref_lcfs_linewidth,
        linestyle=s.ref_lcfs_linestyle,
        alpha=s.ref_lcfs_alpha,
        label=label,
        zorder=s.zorder_ref,
    )


def _render_ref_xpoints_mpl(ax: Axes, ref_xp: ReferenceXPoints) -> None:
    """Render validation-reference X-points as distinct sienna markers.

    The marker shape (``ref_xpt_marker``, default filled plus) is deliberately
    different from the primary's red 'x', so a reference X-point is immediately
    separable.  Empty ``points`` renders nothing (honest absence).  A single
    legend entry is added (label ``ref X-pt (NAME)``).
    """
    pts = ref_xp.points
    if not pts:
        return
    s = ref_xp.style
    r_vals = [p[0] for p in pts]
    z_vals = [p[1] for p in pts]
    ax.plot(
        r_vals,
        z_vals,
        marker=s.ref_xpt_marker,
        markersize=s.ref_xpt_markersize,
        markeredgewidth=s.ref_xpt_markeredgewidth,
        markeredgecolor=s.ref_xpt_markeredgecolor,
        color=s.ref_xpt_color,
        alpha=s.ref_xpt_alpha,
        linestyle="none",
        label=f"ref X-pt ({ref_xp.ref_name})",
        zorder=s.zorder_ref_xpt,
    )


def _render_lcfs_mpl(ax: Axes, lcfs: LcfsOutline) -> None:
    """Render the IDS-derived LCFS outline (boundary.outline.r/z).

    Renders the outline verbatim from the IDS.  No computation is
    performed.  If the arrays are empty, nothing is drawn — honest
    absence beats a computed approximation.
    """
    r = np.asarray(lcfs.r)
    z = np.asarray(lcfs.z)
    if r.size < 2 or z.size < 2:
        return
    s = lcfs.style
    # Ensure the outline is closed for rendering
    if not (np.isclose(r[0], r[-1]) and np.isclose(z[0], z[-1])):
        r = np.append(r, r[0])
        z = np.append(z, z[0])
    ax.plot(
        r,
        z,
        color=s.sep_color,
        linewidth=s.sep_linewidth,
        linestyle=s.sep_linestyle,
        zorder=s.zorder_sep,
    )


def _render_legs_mpl(ax: Axes, legs: DivertorLegs) -> None:
    """Render divertor legs as single continuous lines, IDS-verbatim.

    Each leg is an ``(N, 2)`` ``[R, Z]`` polyline read straight from the
    magnetic-topology levelset (DD PR #243).  Nothing is recomputed.  An
    empty ``legs`` list (limited / vacuum slice, or stock-4.1.0 pulse without
    the multi-segment levelset) renders nothing — graceful degradation.
    """
    s = legs.style
    segs: list[np.ndarray] = []
    for leg in legs.legs:
        arr = np.asarray(leg, dtype=float)
        if arr.ndim == 2 and arr.shape[0] >= 2 and arr.shape[1] == 2:
            segs.append(arr)
    if not segs:
        return
    lc = LineCollection(
        segs,
        colors=s.leg_color,
        linewidths=s.leg_linewidth,
        linestyles=s.leg_linestyle,
        zorder=s.zorder_legs,
    )
    ax.add_collection(lc)


def _render_strikes_mpl(ax: Axes, strikes: StrikePoints) -> None:
    """Plot strike-point markers — IDS-verbatim, nothing computed.

    Empty ``points`` renders nothing (honest absence on limited / vacuum
    slices).
    """
    pts = strikes.points
    if not pts:
        return
    s = strikes.style
    r_vals = [p[0] for p in pts]
    z_vals = [p[1] for p in pts]
    ax.plot(
        r_vals,
        z_vals,
        marker=s.strike_marker,
        markersize=s.strike_markersize,
        markeredgewidth=s.strike_markeredgewidth,
        markeredgecolor=s.strike_markeredgecolor,
        color=s.strike_color,
        linestyle="none",
        zorder=s.zorder_strikes,
    )


def _render_wall_mpl(ax: Axes, wall: WallOutline) -> None:
    """Plot all first-wall unit outlines.

    When ``wall.wall_units`` is populated (new multi-unit path), each unit is
    drawn as a separate line.  Falls back to the single ``wall_r``/``wall_z``
    pair for backward compatibility when ``wall_units`` is empty.
    """
    s = wall.style
    units = getattr(wall, "wall_units", None)
    if units:
        for r_u, z_u in units:
            ax.plot(
                r_u,
                z_u,
                color=s.wall_color,
                linewidth=s.wall_linewidth,
                zorder=s.zorder_wall,
            )
    else:
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
    """Plot B-pol magnetic probes with optional orientation tick marks.

    The direction tick is drawn **along** each probe's IMAS
    ``poloidal_angle`` θ.  Per the Data Dictionary
    (``magnetics/b_field_pol_probe/poloidal_angle``) θ is the angle of the
    *sensor normal vector* — the vector parallel to the coil axis, which
    for a pickup coil is the direction of the magnetic-field component the
    probe measures — taken clockwise (COCOS-11/17 θ-like) from +R̂, with
    θ = 0 pointing towards increasing major radius.  In (R, Z) screen
    space that unit vector is n = (cos θ, -sin θ) (the -sin maps the DD's
    clockwise convention onto matplotlib's CCW axes).  EFIT consumes the
    identical vector (``src/imas.cpp``: ``n = cos θ R̂ - sin θ Ẑ``), so the
    tick shows the probe's true sensing axis with no offset.

    The sensitivity axis is *undirected* (a coil measures ±B along its
    axis equally), so the tick is centred on the probe rather than drawn
    one-sided.  Co-located multi-component sensors (e.g. WEST/ITER
    tangential+normal pairs sharing one location, ~90° apart) are coloured
    by component so a tangential B-pol tick is visually distinct from a
    co-located normal-component tick.
    """
    s = probes.style
    r = np.asarray(probes.positions_r)
    z = np.asarray(probes.positions_z)
    ang = np.asarray(probes.angles)

    if r.size == 0:
        return

    comp = classify_probe_components(r, z, ang)
    palette = [s.probe_color, s.probe_secondary_color]

    def _comp_color(c: int) -> str:
        return palette[c] if c < len(palette) else s.probe_secondary_color

    ax.scatter(
        r,
        z,
        s=s.probe_markersize**2,
        c=[_comp_color(int(c)) for c in comp],
        marker="o",
        linewidths=0,
        zorder=s.zorder_probes,
    )

    finite = np.isfinite(ang)
    if np.any(finite):
        half = s.probe_arrow_length / 2.0
        idx = np.where(finite)[0]
        # n = (cos θ, -sin θ): DD sensor-normal / coil-axis / measured-field
        # direction (clockwise θ from +R).  Centre the (undirected) tick.
        segs = [
            [
                (r[i] - half * np.cos(ang[i]), z[i] + half * np.sin(ang[i])),
                (r[i] + half * np.cos(ang[i]), z[i] - half * np.sin(ang[i])),
            ]
            for i in idx
        ]
        lc = LineCollection(
            segs,
            colors=[_comp_color(int(comp[i])) for i in idx],
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
