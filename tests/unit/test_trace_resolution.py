"""Trace figures are bounded to the resolution they are read at.

A trace sampled far finer than a figure's pixel width adds file size and no
visible detail.  Matplotlib bounds a drawn path at save time through the style's
``path_simplify_threshold``; Altair has no such simplification, so a long series
is drawn through a min/max envelope instead.  Both bounds are exercised here on
a series long enough to matter.
"""

from __future__ import annotations

import numpy as np
import pytest

from imas_ink import TimeSeries, render_to_bytes, time_trace_figure_mpl
from imas_ink.alt import _render_timeseries_alt
from imas_ink.style import DEFAULT_STYLE

_SAMPLES = 300_000
_BYTE_LIMIT = 100 * 1024


def _noisy_series(n: int = _SAMPLES) -> TimeSeries:
    """A trace with sample-scale detail, so path simplification has work to do."""
    rng = np.random.default_rng(0)
    time = np.linspace(0.0, 1.0, n)
    values = rng.standard_normal(n)
    return TimeSeries(time, values, ylabel="Ip", units="MA")


def test_long_trace_svg_is_bounded():
    """A 300,000-sample trace renders to an SVG under 100 KB."""
    ts = _noisy_series()
    fig, _ = time_trace_figure_mpl([ts])
    svg = render_to_bytes(fig, format="svg")
    assert len(svg) < _BYTE_LIMIT


def test_altair_bounds_long_series_but_keeps_extremes():
    """The Altair renderer emits at most ``trace_max_points`` points, extremes intact."""
    ts = _noisy_series()
    chart = _render_timeseries_alt(ts)
    df = chart.data
    assert len(df) <= DEFAULT_STYLE.trace_max_points
    assert df["value"].min() == pytest.approx(ts.values.min())
    assert df["value"].max() == pytest.approx(ts.values.max())


def test_altair_keeps_short_series_unchanged():
    """A series at or below the limit is drawn with all its samples."""
    ts = _noisy_series(n=500)
    chart = _render_timeseries_alt(ts)
    assert len(chart.data) == ts.values.size