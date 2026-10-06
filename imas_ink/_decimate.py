"""Down-sampling of long traces for backends without path simplification.

Matplotlib simplifies a drawn path to display resolution at save time; Altair
has no equivalent and its data transformer caps a chart's rows, so a series
sampled far beyond the figure's pixel width must be reduced before it reaches
Altair.  The reduction keeps the trace's extremes: a min/max envelope reads the
same at display resolution as the full series.
"""

from __future__ import annotations

import numpy as np


def _minmax_envelope(
    time: np.ndarray,
    values: np.ndarray,
    max_points: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a series to at most *max_points* points, preserving extremes.

    The series is split into ``max_points / 2`` consecutive equal-count bins.
    Each bin contributes its minimum and maximum sample, ordered by time, so
    every spike, extreme and the noise-band width survive the reduction.

    Parameters
    ----------
    time, values : np.ndarray
        Sample coordinates and values, same length.
    max_points : int
        Bound on the returned point count.  A series already at or below it is
        returned unchanged.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(time, values)`` of the envelope.
    """
    t = np.asarray(time, dtype=float)
    v = np.asarray(values, dtype=float)
    n = v.size
    n_bins = max_points // 2
    if n <= max_points or n_bins < 1:
        return t, v

    t_out: list[float] = []
    v_out: list[float] = []
    for chunk in np.array_split(np.arange(n), n_bins):
        cv = v[chunk]
        lo = int(chunk[np.argmin(cv)])
        hi = int(chunk[np.argmax(cv)])
        first, second = (lo, hi) if lo <= hi else (hi, lo)
        t_out.append(t[first])
        v_out.append(v[first])
        t_out.append(t[second])
        v_out.append(v[second])
    return np.asarray(t_out), np.asarray(v_out)
