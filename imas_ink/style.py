"""Default style configuration — single source of visual constants.

No hardcoded magic numbers anywhere else in the library. All rendering
functions accept an optional ``style`` parameter; if omitted,
:data:`DEFAULT_STYLE` is used.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class InkStyle:
    """Visual style for ink renderings.

    Every visual constant — colours, linewidths, marker sizes, z-order
    values — is stored here. Renderers read from an ``InkStyle`` instance
    rather than hard-coding values.

    To customise, create a new instance via :func:`dataclasses.replace`::

        from dataclasses import replace
        my_style = replace(DEFAULT_STYLE, sep_color="#00cc00")
    """

    # Flux contours — confined (closed, encloses magnetic axis)
    flux_color: str = "#3366cc"
    flux_linewidth: float = 0.7
    flux_linestyle: str = "solid"
    flux_n_levels: int = 6

    # SOL / open contours — surfaces that do not enclose the magnetic axis
    # (open field lines, private-flux lobes, or contours at the grid edge).
    # Rendered at reduced weight in grey so they are visible but visually
    # subordinate to the confined plasma region.
    sol_color: str = "#999999"
    sol_linewidth: float = 0.35
    sol_linestyle: str = "solid"

    # Separatrix
    sep_color: str = "#cc0000"
    sep_linewidth: float = 1.5
    sep_linestyle: str = "solid"

    # Wall
    wall_color: str = "#000000"
    wall_linewidth: float = 1.0

    # Coils
    coil_edgecolor: str = "#888888"
    coil_facecolor: str = "none"
    coil_linewidth: float = 0.4

    # Magnetic axis marker
    axis_marker: str = "."
    axis_markersize: float = 6.0
    axis_color: str = "#cc0000"

    # X-point markers
    xpt_marker: str = "x"
    xpt_markersize: float = 3.0
    xpt_markeredgewidth: float = 0.8
    xpt_color: str = "#cc0000"

    # Magnetic probes (B-pol)
    probe_color: str = "#888888"
    # Second colour for the secondary orientation component of co-located
    # multi-component sensors (e.g. WEST/ITER 2-component probe pairs).
    # The direction tick is drawn along each probe's DD poloidal_angle
    # (sensor-normal = coil-axis = measured-field axis); the two distinct
    # axes at one location are coloured differently so a tangential B-pol
    # tick is not confused with a co-located normal-component tick.
    probe_secondary_color: str = "#cc7722"
    probe_markersize: float = 2.5
    probe_arrow_length: float = 0.12
    probe_arrow_linewidth: float = 0.6

    # Flux loops
    flux_loop_color: str = "#666666"
    flux_loop_markersize: float = 3.0

    # Time label
    label_fontsize: float = 8.0
    label_bbox: dict = field(
        default_factory=lambda: {
            "facecolor": "white",
            "alpha": 0.9,
            "edgecolor": "none",
            "pad": 2,
        }
    )

    # Figure
    figure_facecolor: str = "white"
    figure_dpi: int = 120

    # Z-order map
    zorder_coils: int = 3
    zorder_wall: int = 4
    zorder_flux: int = 2
    zorder_sep: int = 5
    zorder_markers: int = 6
    zorder_probes: int = 6
    zorder_flux_loops: int = 6
    zorder_label: int = 7

    # Vacuum / SOL contours (outside LCFS, full grid, no wall clip)
    vacuum_color: str = "#aaaaaa"
    vacuum_linewidth: float = 0.4
    vacuum_linestyle: str = "solid"
    vacuum_n_levels: int = 10

    # Reference (validation) underlay — sienna contours behind everything,
    # drawn at the same absolute psi levels as the primary confined surfaces so
    # level-to-level comparison is direct.  The LCFS is a dashed line from
    # boundary.outline (IDS-verbatim, no recomputation).  Styling is faint but
    # legible: the reference is subordinate to the primary, yet the topology
    # MISMATCH (e.g. reference X-point / divertor leg vs a limited primary) must
    # remain clearly visible — that visibility is the whole purpose.
    ref_color: str = "#b85c38"  # sienna — distinct from primary blue/grey/red
    ref_alpha: float = 0.45  # subordinate to primary but clearly legible
    ref_linewidth: float = 0.6  # thin
    ref_linestyle: str = "solid"
    ref_lcfs_color: str = "#a02c00"  # deeper sienna/rust for the reference LCFS
    ref_lcfs_alpha: float = 0.75  # prominent — the boundary mismatch signal
    ref_lcfs_linewidth: float = 1.6
    ref_lcfs_linestyle: str = "dashed"
    zorder_ref: int = 1  # below primary (zorder_flux=2)
    # Reference X-point marker — a filled sienna plus, clearly distinct from the
    # primary's red 'x' X-markers.  This is the headline topology-mismatch
    # signal (e.g. DINA lower X-point behind a limited / upper-X reconstruction).
    ref_xpt_marker: str = "P"  # filled plus — distinct shape from primary 'x'
    ref_xpt_color: str = "#a02c00"
    ref_xpt_markersize: float = 8.0
    ref_xpt_markeredgewidth: float = 1.0
    ref_xpt_markeredgecolor: str = "white"
    ref_xpt_alpha: float = 0.9  # prominent
    zorder_ref_xpt: int = 5  # at primary-marker level so it reads clearly

    # 1D plots
    trace_linewidth: float = 1.2
    trace_markersize: float = 3.0

    # Altair
    altair_width: int = 500
    altair_height: int = 600


DEFAULT_STYLE = InkStyle()
