"""Altair (Vega-Lite) rendering backend for ink components.

All ``altair`` and ``pandas`` imports are lazy — this module can be
imported even when the ``[altair]`` optional dependency is not installed.
The actual import happens at first render call.

Examples
--------
>>> fc = FluxContours(cx.flux_surfaces(psi_axis, psi_bnd))
>>> chart = render_alt(fc)
>>> chart.save("flux.html")
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

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

if TYPE_CHECKING:
    import altair as alt
    import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def segments_to_dataframe(
    segments: list[np.ndarray],
    level: int = 0,
) -> pd.DataFrame:
    """Convert a list of (N, 2) segment arrays to a single DataFrame.

    Each segment receives a unique ``seg_id`` so Altair encodes them as
    separate paths (preventing stray connecting lines between
    disconnected contour pieces).

    Parameters
    ----------
    segments : list[np.ndarray]
        Each entry is an ``(N, 2)`` array with columns ``[R, Z]``.
    level : int
        Integer level index attached to every row (default ``0``).

    Returns
    -------
    pd.DataFrame
        Columns: ``['r', 'z', 'level', 'seg_id']``.
    """
    import pandas as pd

    rows: list[dict] = []
    for seg_idx, seg in enumerate(segments):
        for pt in seg:
            rows.append(
                {
                    "r": float(pt[0]),
                    "z": float(pt[1]),
                    "level": level,
                    "seg_id": f"{level}_{seg_idx}",
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Public dispatch
# ---------------------------------------------------------------------------

_RENDERERS: dict[type, str] = {
    FluxContours: "_render_flux_alt",
    Separatrix: "_render_sep_alt",
    WallOutline: "_render_wall_alt",
    CoilRects: "_render_coils_alt",
    OPointMarker: "_render_opoint_alt",
    XPointMarkers: "_render_xpoints_alt",
    TimeLabel: "_render_timelabel_alt",
    TimeSeries: "_render_timeseries_alt",
    RadialProfile: "_render_radialprofile_alt",
    ScatterPoints: "_render_scatter_alt",
}


def render_alt(component) -> alt.Chart | alt.LayerChart:
    """Render an ink component to an Altair chart.

    Parameters
    ----------
    component
        Any ink visual component (e.g. :class:`FluxContours`,
        :class:`Separatrix`, etc.).

    Returns
    -------
    alt.Chart | alt.LayerChart

    Raises
    ------
    TypeError
        If *component* is not a recognised ink component.
    """
    for cls, func_name in _RENDERERS.items():
        if isinstance(component, cls):
            return globals()[func_name](component)
    raise TypeError(f"No Altair renderer for {type(component).__name__}")


# ---------------------------------------------------------------------------
# Component renderers
# ---------------------------------------------------------------------------


def _render_flux_alt(fc: FluxContours) -> alt.Chart:
    """Render :class:`FluxContours` — interior flux surface isolines."""
    import altair as alt
    import pandas as pd

    style = fc.style

    # Flatten all level segments into a single DataFrame.
    frames: list[pd.DataFrame] = []
    for level_idx, level_segs in enumerate(fc.segments):
        if level_segs:
            frames.append(segments_to_dataframe(level_segs, level=level_idx))

    if not frames:
        # Nothing to render — return an empty chart.
        return alt.Chart(pd.DataFrame({"r": [], "z": []})).mark_point()

    df = pd.concat(frames, ignore_index=True)

    return (
        alt.Chart(df)
        .mark_line(strokeWidth=style.flux_linewidth)
        .encode(
            x=alt.X("r:Q", title="R [m]"),
            y=alt.Y("z:Q", title="Z [m]"),
            detail="seg_id:N",
            color=alt.Color("level:N", legend=None),
        )
        .properties(width=style.altair_width, height=style.altair_height)
    )


def _render_sep_alt(sep: Separatrix) -> alt.Chart | alt.LayerChart:
    """Render :class:`Separatrix` — LCFS + optional X-point markers."""
    import altair as alt
    import pandas as pd

    style = sep.style
    df = segments_to_dataframe(sep.segments, level=0)

    if df.empty:
        return alt.Chart(pd.DataFrame({"r": [], "z": []})).mark_point()

    line = (
        alt.Chart(df)
        .mark_line(color=style.sep_color, strokeWidth=style.sep_linewidth)
        .encode(
            x=alt.X("r:Q", title="R [m]"),
            y=alt.Y("z:Q", title="Z [m]"),
            detail="seg_id:N",
        )
    )

    if sep.x_points:
        xdf = pd.DataFrame([{"r": float(p[0]), "z": float(p[1])} for p in sep.x_points])
        xpts = (
            alt.Chart(xdf)
            .mark_point(
                shape="cross",
                color=style.xpt_color,
                size=style.xpt_markersize**2,
            )
            .encode(x="r:Q", y="z:Q")
        )
        return alt.layer(line, xpts)

    return line


def _render_wall_alt(wall: WallOutline) -> alt.Chart:
    """Render :class:`WallOutline` — first wall polygon."""
    import altair as alt
    import pandas as pd

    style = wall.style
    df = pd.DataFrame({"r": wall.wall_r, "z": wall.wall_z, "seg_id": 0})

    return (
        alt.Chart(df)
        .mark_line(color=style.wall_color, strokeWidth=style.wall_linewidth)
        .encode(
            x=alt.X("r:Q", title="R [m]"),
            y=alt.Y("z:Q", title="Z [m]"),
            detail="seg_id:N",
        )
    )


def _render_coils_alt(coils: CoilRects) -> alt.Chart:
    """Render :class:`CoilRects` — PF coil bounding boxes."""
    import altair as alt
    import pandas as pd

    style = coils.style
    rows: list[dict] = []
    for c in coils.rects:
        rows.append(
            {
                "r": float(c.r),
                "z": float(c.z),
                "r2": float(c.r + c.width),
                "z2": float(c.z + c.height),
                "name": c.name,
            }
        )

    if not rows:
        return alt.Chart(pd.DataFrame({"r": [], "z": []})).mark_point()

    df = pd.DataFrame(rows)

    return (
        alt.Chart(df)
        .mark_rect(stroke=style.coil_edgecolor, strokeWidth=style.coil_linewidth)
        .encode(
            x=alt.X("r:Q", title="R [m]"),
            y=alt.Y("z:Q", title="Z [m]"),
            x2="r2:Q",
            y2="z2:Q",
            color=alt.value(
                style.coil_facecolor if style.coil_facecolor != "none" else "transparent"
            ),
        )
    )


def _render_opoint_alt(opoint: OPointMarker) -> alt.Chart:
    """Render :class:`OPointMarker` — magnetic axis marker."""
    import altair as alt
    import pandas as pd

    style = opoint.style
    df = pd.DataFrame({"r": [opoint.r], "z": [opoint.z]})

    return (
        alt.Chart(df)
        .mark_point(color=style.axis_color, size=style.axis_markersize**2)
        .encode(
            x=alt.X("r:Q", title="R [m]"),
            y=alt.Y("z:Q", title="Z [m]"),
        )
    )


def _render_xpoints_alt(xpoints: XPointMarkers) -> alt.Chart:
    """Render :class:`XPointMarkers` — X-point cross markers."""
    import altair as alt
    import pandas as pd

    style = xpoints.style

    if not xpoints.points:
        return alt.Chart(pd.DataFrame({"r": [], "z": []})).mark_point()

    df = pd.DataFrame([{"r": float(p[0]), "z": float(p[1])} for p in xpoints.points])

    return (
        alt.Chart(df)
        .mark_point(
            shape="cross",
            color=style.xpt_color,
            size=style.xpt_markersize**2,
        )
        .encode(
            x=alt.X("r:Q", title="R [m]"),
            y=alt.Y("z:Q", title="Z [m]"),
        )
    )


def _render_timelabel_alt(label: TimeLabel) -> alt.Chart:
    """Render :class:`TimeLabel` — time annotation text."""
    import altair as alt
    import pandas as pd

    style = label.style
    tag = "✓" if label.converged else "✗"
    text = f"t = {label.time:.4f} s  {tag}"

    df = pd.DataFrame({"text": [text]})

    return (
        alt.Chart(df)
        .mark_text(
            align="left",
            baseline="top",
            fontSize=style.label_fontsize,
            dx=5,
            dy=5,
        )
        .encode(text="text:N")
    )


# ---------------------------------------------------------------------------
# 1D renderers (time series, radial profiles, scatter)
# ---------------------------------------------------------------------------


def _render_timeseries_alt(ts: TimeSeries) -> alt.LayerChart:
    """Render :class:`TimeSeries` — time vs scalar with interactive tooltip."""
    import altair as alt
    import pandas as pd

    style = ts.style
    ylabel = f"{ts.ylabel} [{ts.units}]" if ts.units else ts.ylabel
    df = pd.DataFrame({"time": ts.time, "value": ts.values})

    nearest = alt.selection_point(
        nearest=True,
        on="pointerover",
        fields=["time"],
        empty=False,
    )

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=style.trace_linewidth)
        .encode(
            x=alt.X("time:Q", title="Time [s]"),
            y=alt.Y("value:Q", title=ylabel),
        )
    )

    sel_points = (
        line.mark_point()
        .encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        .add_params(nearest)
    )

    rule = alt.Chart(df).mark_rule(color="gray").encode(x="time:Q").transform_filter(nearest)

    return alt.layer(line, sel_points, rule).properties(width=style.altair_width, height=300)


def _render_radialprofile_alt(rp: RadialProfile) -> alt.LayerChart:
    """Render :class:`RadialProfile` — ψ_N vs quantity with interactive tooltip."""
    import altair as alt
    import pandas as pd

    style = rp.style
    ylabel = f"{rp.ylabel} [{rp.units}]" if rp.units else rp.ylabel
    df = pd.DataFrame({"psi_norm": rp.psi_norm, "value": rp.values})

    nearest = alt.selection_point(
        nearest=True,
        on="pointerover",
        fields=["psi_norm"],
        empty=False,
    )

    line = (
        alt.Chart(df)
        .mark_line(strokeWidth=style.trace_linewidth)
        .encode(
            x=alt.X("psi_norm:Q", title="ψ_N"),
            y=alt.Y("value:Q", title=ylabel),
        )
    )

    sel_points = (
        line.mark_point()
        .encode(opacity=alt.condition(nearest, alt.value(1), alt.value(0)))
        .add_params(nearest)
    )

    rule = alt.Chart(df).mark_rule(color="gray").encode(x="psi_norm:Q").transform_filter(nearest)

    return alt.layer(line, sel_points, rule).properties(width=style.altair_width, height=300)


def _render_scatter_alt(sc: ScatterPoints) -> alt.Chart:
    """Render :class:`ScatterPoints` — x vs y scatter with tooltips on hover."""
    import altair as alt
    import pandas as pd

    style = sc.style
    df = pd.DataFrame({"x": sc.x, "y": sc.y})

    return (
        alt.Chart(df)
        .mark_point(size=style.trace_markersize**2)
        .encode(
            x=alt.X("x:Q", title=sc.xlabel or "x"),
            y=alt.Y("y:Q", title=sc.ylabel or "y"),
            tooltip=["x:Q", "y:Q"],
        )
        .properties(width=style.altair_width, height=300)
    )
