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

    # 1D plots
    trace_linewidth: float = 1.2
    trace_markersize: float = 3.0

    # Altair
    altair_width: int = 500
    altair_height: int = 600


DEFAULT_STYLE = InkStyle()
