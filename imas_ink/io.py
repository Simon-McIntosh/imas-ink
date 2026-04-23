"""I/O utilities for saving and exporting ink figures.

Handles matplotlib → PNG bytes, file saving, and (future) Altair HTML
export. All figure-closing is handled here so callers don't leak memory.
"""

from __future__ import annotations

import io
from pathlib import Path
from typing import TYPE_CHECKING

from .style import DEFAULT_STYLE

if TYPE_CHECKING:
    import matplotlib.figure


def render_to_bytes(
    fig: matplotlib.figure.Figure,
    dpi: int | None = None,
    format: str = "png",
) -> bytes:
    """Render a matplotlib figure to raw image bytes.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to render.
    dpi : int, optional
        Dots-per-inch. Defaults to :attr:`InkStyle.figure_dpi` (120).
    format : str
        Image format (``'png'``, ``'svg'``, ``'pdf'``).

    Returns
    -------
    bytes
        Raw image data.
    """
    import matplotlib.pyplot as plt

    if dpi is None:
        dpi = DEFAULT_STYLE.figure_dpi
    buf = io.BytesIO()
    fig.savefig(buf, format=format, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return buf.read()


def save_png(
    fig: matplotlib.figure.Figure,
    path: str | Path,
    dpi: int | None = None,
) -> None:
    """Save a matplotlib figure to a PNG file.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to save.
    path : str or Path
        Output file path.
    dpi : int, optional
        Dots-per-inch. Defaults to :attr:`InkStyle.figure_dpi`.
    """
    import matplotlib.pyplot as plt

    if dpi is None:
        dpi = DEFAULT_STYLE.figure_dpi
    fig.savefig(str(path), dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def save_html(chart: object, path: str | Path) -> None:
    """Save an Altair chart to a standalone HTML file.

    Parameters
    ----------
    chart
        An Altair chart object (``alt.Chart`` or ``alt.LayerChart``).
    path : str or Path
        Output HTML file path.

    Raises
    ------
    ImportError
        If ``altair`` is not installed.
    """
    try:
        import altair  # noqa: F401
    except ImportError as exc:
        raise ImportError("altair is required for HTML export: pip install altair") from exc
    chart.save(str(path))  # type: ignore[union-attr]
