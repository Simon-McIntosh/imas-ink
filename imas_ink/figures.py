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
    LcfsOutline,
    MagneticProbes,
    OPointMarker,
    ReferenceContours,
    ReferenceLcfs,
    ReferenceXPoints,
    Separatrix,
    SolContours,
    TimeLabel,
    WallOutline,
    XPointMarkers,
)
from .contours import ContourExtractor
from .geometry import classify_flux_segments, mask_pfr
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
    wall = WallOutline(
        geom.wall_r,
        geom.wall_z,
        wall_units=geom.wall_units if geom.wall_units else [],
        style=style,
    )

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


def _render_containment_annotation(ax: Axes, containment: dict) -> None:
    """Render wall-containment annotation text onto *ax*.

    Drawn UNCLIPPED (axes-transform) so it is always visible regardless of
    the wall clip path.  Primary signal is ``lcfs_outside`` (geometric vertex
    count); ``frac`` is secondary and always shown caveated.

    Styling:
      - ``lcfs_outside > 0`` → red text (boundary exits wall).
      - ``lcfs_outside == 0`` → grey text (wall contained).
      - ``frac`` line is always grey (secondary, may be > 1 on healthy diverted
        plasmas due to private-flux wall vertices — not a bug indicator).
    """
    lcfs_outside = containment.get("lcfs_outside", 0)
    frac = containment.get("frac", float("nan"))
    boundary_type = containment.get("boundary_type")
    n_xpoints = containment.get("n_xpoints")

    # Primary line: lcfs_outside count (geometric signal)
    primary_color = "red" if lcfs_outside > 0 else "gray"
    primary_text = f"lcfs_outside={lcfs_outside}"

    # Secondary line: psi-frac (caveated — may be >1 on diverted machines)
    try:
        frac_val = float(frac)
        frac_str = f"psi-frac={frac_val:.2f} (secondary)"
    except (TypeError, ValueError):
        frac_str = "psi-frac=N/A (secondary)"

    # Optional topology info
    extra_parts = []
    if boundary_type is not None:
        _btype_name = {1: "limited", 2: "diverted"}.get(boundary_type, f"type={boundary_type}")
        extra_parts.append(_btype_name)
    if n_xpoints is not None:
        extra_parts.append(f"{n_xpoints} X-pt")
    extra_str = "  ".join(extra_parts)

    lines = [primary_text, frac_str]
    if extra_str:
        lines.append(extra_str)
    annotation = "\n".join(lines)

    ax.text(
        0.03,
        0.03,
        annotation,
        transform=ax.transAxes,
        fontsize=7,
        verticalalignment="bottom",
        horizontalalignment="left",
        color=primary_color,
        bbox={
            "boxstyle": "round,pad=0.2",
            "facecolor": "white",
            "alpha": 0.7,
            "edgecolor": primary_color,
        },
        zorder=20,
    )


def equilibrium_figure_mpl(
    sl: EquilibriumSlice,
    geom: MachineGeometry,
    style: InkStyle | None = None,
    figsize: tuple[float, float] = (6, 7),
    mask_pfr_flag: bool = True,
    show_probes: bool = True,
    show_flux_loops: bool = True,
    show_vacuum_surfaces: bool = False,
    containment_result: dict | None = None,
    reference_slice: EquilibriumSlice | None = None,
    reference_name: str = "reference",
) -> tuple[matplotlib.figure.Figure, Axes]:
    """Build a complete poloidal cross-section figure.

    Creates the machine geometry via :func:`geometry_figure_mpl` and
    layers equilibrium-specific components on top: flux contours,
    separatrix, O-point, X-point markers, and time label.

    An optional *reference_slice* adds a validation-reference underlay
    in faint sienna behind the primary reconstruction.  The reference
    contours are drawn at the **same absolute psi levels** as the primary
    confined surfaces (computed once from the primary's psi_axis /
    psi_boundary so the level array is identical).  If the psi frames of
    the two IDSs are incompatible (opposite-signed or irreconcilably
    different scales), the function falls back to normalised levels
    (psi_norm) and adds a small note to the figure title.

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
    containment_result : dict or None
        Optional dict from ``efit.wall_containment.compute_psi_frac``.
        When provided, a containment annotation is drawn in the lower-left
        corner of the axes.  Keys used:

        - ``lcfs_outside`` (int) — primary signal: LCFS vertices outside
          the wall polygon (styled red when > 0, grey otherwise).
        - ``frac`` (float) — secondary ψ-fraction metric (shown caveated).
        - ``boundary_type`` (int or None) — IMAS topology code.
        - ``n_xpoints`` (int or None) — number of X-points.

        Pass ``None`` (default) to suppress the annotation entirely
        (backward-compatible behaviour).
    reference_slice : EquilibriumSlice or None
        Optional second equilibrium slice from a validation reference code
        (e.g. DINA, NICE, EPM, LIUQE).  When provided, faint sienna contours
        from the reference psi field are drawn *behind* the primary
        reconstruction at the same absolute psi levels.  The reference LCFS
        (from ``boundary.outline``) is drawn as a faint dashed line.
        Pass ``None`` (default) to suppress the underlay (backward-compatible).
    reference_name : str
        Short label for the reference code, used in legend entries
        (e.g. ``"DINA"``, ``"NICE"``, ``"EPM"``).  Default: ``"reference"``.

    Returns
    -------
    tuple[Figure, Axes]
        The matplotlib figure and axes objects.
    """
    if style is None:
        style = DEFAULT_STYLE

    # --- base geometry layer ---
    fig, ax = geometry_figure_mpl(
        geom,
        style=style,
        figsize=figsize,
        show_probes=show_probes,
        show_flux_loops=show_flux_loops,
    )

    # --- psi field (optionally mask PFR) for confined-surface extraction ---
    psi_confined = sl.psi_2d
    if mask_pfr_flag:
        psi_confined = mask_pfr(
            sl.psi_2d,
            sl.R_2d,
            sl.Z_2d,
            sl.psi_axis,
            sl.psi_boundary,
            sl.r_axis,
            sl.z_axis,
        )

    # --- Interior (confined) flux surface extraction ---
    # Uses masked psi so that private-flux lobes are excluded from the
    # confined-surface set (they are classified as SOL by enclosure test).
    cx_confined = ContourExtractor(sl.R_2d, sl.Z_2d, psi_confined)
    n_levels = style.flux_n_levels
    interior_level_segs = cx_confined.flux_surfaces(sl.psi_axis, sl.psi_boundary, n=n_levels)

    confined_by_level: list[list] = []
    sol_from_interior: list[list] = []
    for level_segs in interior_level_segs:
        confined, sol = classify_flux_segments(level_segs, sl.r_axis, sl.z_axis)
        confined_by_level.append(confined)
        sol_from_interior.append(sol)

    # --- Exterior (SOL / vacuum) flux surface extraction ---
    # Uses the FULL (unmasked) psi grid to capture open-field contours,
    # private-flux lobes, and grid-edge artefacts — all surfaces that do
    # NOT enclose the magnetic axis.  Rendered in thin grey.
    cx_full = ContourExtractor(sl.R_2d, sl.Z_2d, sl.psi_2d)
    vac_segs_by_level = cx_full.vacuum_surfaces(
        sl.psi_axis, sl.psi_boundary, n=style.vacuum_n_levels
    )
    sol_by_level = sol_from_interior + vac_segs_by_level

    flux_confined = FluxContours(confined_by_level, style=style)
    flux_sol = SolContours(sol_by_level, style=style)

    # --- LCFS: read verbatim from IDS boundary.outline (plot-only rule) ---
    # If boundary.outline is absent (solver did not write it), render
    # nothing — honest absence beats a computed approximation.
    lcfs: LcfsOutline | None = None
    if sl.boundary_r is not None and sl.boundary_z is not None:
        br = sl.boundary_r
        bz = sl.boundary_z
        if br.size >= 2:
            lcfs = LcfsOutline(br, bz, style=style)

    # --- X-points: read from IDS only, never computed ---
    # If boundary.x_point was absent from the IDS (current state while
    # the solver-side topology-write work is in flight), render no
    # X markers.  Honest absence is correct.
    xpoints = XPointMarkers(sl.x_points, style=style)

    # --- Other markers ---
    opoint = OPointMarker(sl.r_axis, sl.z_axis, style=style)
    timelabel = TimeLabel(sl.time, converged=sl.converged, style=style)

    # --- Reference (validation) underlay ---
    # Extract contours from the reference IDS at IDENTICAL absolute psi levels
    # (derived once from the primary's psi_axis / psi_boundary so the level
    # array is bit-for-bit identical).  The reference field is contoured on its
    # own (R, Z) grid — no regridding required.
    #
    # Frame compatibility check: both IDS must be in the same psi sign convention
    # (COCOS-17: psi_axis < psi_boundary in absolute value direction, more negative
    # at axis).  If signs disagree, fall back to normalised levels.
    ref_contours: ReferenceContours | None = None
    ref_lcfs_comp: ReferenceLcfs | None = None
    ref_xpoints_comp: ReferenceXPoints | None = None
    if reference_slice is not None:
        ref = reference_slice
        import math as _math

        import numpy as _np

        from ._cocos import make_levels as _make_levels

        ref_psi_bnd = ref.psi_boundary
        ref_psi_ax = ref.psi_axis

        # Does the reference carry a usable 2D ψ field?  Some references
        # (e.g. NICE) write only boundary + global quantities, no profiles_2d.
        # In that case we still draw the reference LCFS + X-points (the boundary
        # mismatch remains visible) but skip the field-contour underlay.
        _ref_psi2d = getattr(ref, "psi_2d", None)
        ref_has_field = (
            _ref_psi2d is not None
            and getattr(_ref_psi2d, "size", 0) > 2
            and getattr(ref, "R_2d", None) is not None
            and getattr(ref.R_2d, "size", 0) > 2
        )

        # Some reference codes (e.g. DINA) leave global_quantities.psi_axis as
        # an IMAS sentinel (extracted here as NaN).  Infer it from the 2D field:
        # the axis ψ is the field extremum in the inboard (more-negative for
        # COCOS-17) direction relative to the boundary.  This recovers a usable
        # absolute psi_axis without violating the plot-only rule (the value
        # comes straight from the reference's own psi_2d, not a recomputation).
        _sentinel_threshold = 1e10
        ref_axis_valid = (
            ref_has_field
            and ref_psi_bnd is not None
            and not _math.isnan(float(ref_psi_bnd))
            and abs(float(ref_psi_bnd)) < _sentinel_threshold
        )
        if ref_axis_valid and (
            ref_psi_ax is None
            or _math.isnan(float(ref_psi_ax))
            or abs(float(ref_psi_ax)) >= _sentinel_threshold
        ):
            # Infer axis from the field extremum.  We only reach the
            # absolute-matching path when primary and reference share the same
            # ψ sign at the boundary (COCOS-17 here: ψ increases OUTWARD, so the
            # axis is the most-NEGATIVE value of the field).  Use the field
            # minimum — this is correct for the same-sign case we gate on and
            # avoids picking the grid far-field corner (the field max).
            ref_psi_ax = float(_np.nanmin(ref.psi_2d))

        # Absolute-level matching is valid when both equilibria share the same
        # ψ sign convention at the boundary AND we have a usable reference axis.
        use_absolute = (
            ref_axis_valid
            and ref_psi_ax is not None
            and not _math.isnan(float(ref_psi_ax))
            and (_np.sign(float(sl.psi_boundary)) == _np.sign(float(ref_psi_bnd)))
        )

        if not ref_has_field:
            # Reference carries no 2D ψ field (e.g. NICE writes boundary only).
            # Draw no field contours; the reference LCFS + X-points below still
            # convey the boundary/topology mismatch.  ref_contours stays None.
            pass
        elif use_absolute:
            # IDENTICAL absolute psi levels: compute the level array ONCE from
            # the PRIMARY's psi_axis/psi_boundary, then evaluate the REFERENCE
            # field at those exact level values.
            primary_levels = _make_levels(sl.psi_axis, sl.psi_boundary, n=n_levels)
            cx_ref = ContourExtractor(ref.R_2d, ref.Z_2d, ref.psi_2d)
            ref_segs_by_level = [cx_ref.lines_at(lev) for lev in primary_levels]
            ref_contours = ReferenceContours(
                ref_segs_by_level,
                ref_name=reference_name,
                psi_matched=True,
                style=style,
            )
        else:
            # Frames irreconcilable (opposite-sign ψ).  Fall back to MATCHED
            # NORMALISED levels (psi_norm): map the primary's fractional levels
            # onto the reference's own ψ range.
            primary_levels = _make_levels(sl.psi_axis, sl.psi_boundary, n=n_levels)
            denom = sl.psi_axis - sl.psi_boundary
            if abs(denom) < 1e-30:
                denom = 1.0
            psi_norm = (primary_levels - sl.psi_boundary) / denom
            if ref_psi_ax is None or _math.isnan(float(ref_psi_ax)):
                ref_psi_ax = float(_np.nanmin(ref.psi_2d))
            ref_levels = float(ref_psi_bnd) + psi_norm * (float(ref_psi_ax) - float(ref_psi_bnd))
            cx_ref = ContourExtractor(ref.R_2d, ref.Z_2d, ref.psi_2d)
            ref_segs_by_level = [cx_ref.lines_at(lev) for lev in ref_levels]
            ref_contours = ReferenceContours(
                ref_segs_by_level,
                ref_name=reference_name,
                psi_matched=False,
                style=style,
            )
            # Small caption note that normalised fallback was used
            ax.set_title(
                "reference at matched ψ_norm (absolute ψ frames differ)",
                fontsize=6,
                loc="left",
                color=style.ref_color,
            )

        # Reference LCFS — IDS-verbatim boundary.outline
        if ref.boundary_r is not None and ref.boundary_z is not None:
            rbr = _np.asarray(ref.boundary_r)
            rbz = _np.asarray(ref.boundary_z)
            if rbr.size >= 2:
                ref_lcfs_comp = ReferenceLcfs(rbr, rbz, ref_name=reference_name, style=style)

        # Reference X-points — IDS-verbatim (extract_slice reads these from
        # contour_tree / constraints / boundary).  These are the headline
        # topology-mismatch signal (e.g. a reference lower X-point behind a
        # limited reconstruction).  Empty list → nothing drawn (honest absence).
        ref_xp = getattr(ref, "x_points", None)
        if ref_xp:
            ref_xpoints_comp = ReferenceXPoints(list(ref_xp), ref_name=reference_name, style=style)

    # Render order: reference underlay FIRST (zorder=1, below everything),
    # then SOL (grey, zorder=2), confined flux (blue, zorder=2), LCFS (red,
    # zorder=5), O-point, X-points, time label.  Reference X-points render at
    # zorder_ref_xpt (≈primary-marker level) so the mismatch reads clearly.
    # All are UNCLIPPED — no wall clip path applied.
    if ref_lcfs_comp is not None:
        render_mpl(ax, ref_lcfs_comp)
    if ref_contours is not None:
        render_mpl(ax, ref_contours)
    render_mpl(ax, flux_sol)
    render_mpl(ax, flux_confined)
    if lcfs is not None:
        render_mpl(ax, lcfs)
    render_mpl(ax, opoint)
    render_mpl(ax, xpoints)
    if ref_xpoints_comp is not None:
        render_mpl(ax, ref_xpoints_comp)
    render_mpl(ax, timelabel)

    # --- containment annotation (lower-left, UNCLIPPED) ---
    if containment_result is not None:
        _render_containment_annotation(ax, containment_result)

    # Note: show_vacuum_surfaces is deprecated and has no effect —
    # vacuum / SOL surfaces are now always rendered via the grey-thin
    # SolContours path above (enclosure classification + vacuum_surfaces
    # on the full psi grid).  The parameter is retained for API
    # compatibility but ignored.

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
    wall = WallOutline(
        geom.wall_r,
        geom.wall_z,
        wall_units=geom.wall_units if geom.wall_units else [],
        style=style,
    )
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
