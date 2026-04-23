"""GIF animation of time-evolving equilibrium cross-sections.

Uses matplotlib for frame rendering and Pillow for GIF encoding.
"""

from __future__ import annotations

import io


def animate_pulse(
    eq_ids,
    geom,
    style=None,
    figsize: tuple[float, float] = (6, 7),
    duration_s: float = 10.0,
    dpi: int = 90,
    mask_pfr_flag: bool = True,
) -> bytes:
    """Render a full-pulse GIF animation of poloidal cross-sections.

    Iterates over all time slices in the equilibrium IDS, renders each
    frame with equilibrium_figure_mpl(), and encodes as a GIF via Pillow.

    Parameters
    ----------
    eq_ids : equilibrium IDS
        Equilibrium IDS with time_slice array.
    geom : MachineGeometry
        Static machine geometry (wall + coils).
    style : InkStyle, optional
        Visual style. Defaults to DEFAULT_STYLE.
    figsize : tuple
        Figure size in inches.
    duration_s : float
        Total GIF duration in seconds.
    dpi : int
        DPI for each frame.
    mask_pfr_flag : bool
        Whether to apply PFR masking.

    Returns
    -------
    bytes
        GIF-encoded animation.

    Example
    -------
    >>> gif_bytes = animate_pulse(eq_ids, geom, duration_s=8.0)
    >>> with open("pulse.gif", "wb") as f:
    ...     f.write(gif_bytes)
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from PIL import Image

    from .extract import extract_slice
    from .figures import equilibrium_figure_mpl
    from .style import DEFAULT_STYLE

    if style is None:
        style = DEFAULT_STYLE

    n_slices = len(eq_ids.time_slice)
    if n_slices == 0:
        raise ValueError("Equilibrium IDS has no time slices")

    frame_duration_ms = int(duration_s * 1000 / n_slices)
    frame_duration_ms = max(frame_duration_ms, 20)  # minimum 20ms per frame

    frames: list[Image.Image] = []
    for i in range(n_slices):
        try:
            sl = extract_slice(eq_ids, i)
        except (IndexError, AttributeError):
            continue

        fig, _ = equilibrium_figure_mpl(
            sl,
            geom,
            style=style,
            figsize=figsize,
            mask_pfr_flag=mask_pfr_flag,
        )
        fig.set_dpi(dpi)

        buf = io.BytesIO()
        fig.savefig(
            buf, format="png", dpi=dpi, bbox_inches="tight", facecolor=style.figure_facecolor
        )
        plt.close(fig)
        buf.seek(0)
        frames.append(Image.open(buf).convert("RGBA"))

    if not frames:
        raise ValueError("No frames could be rendered")

    # Encode as GIF
    gif_buf = io.BytesIO()
    frames[0].save(
        gif_buf,
        format="GIF",
        save_all=True,
        append_images=frames[1:],
        duration=frame_duration_ms,
        loop=0,
    )
    gif_buf.seek(0)
    return gif_buf.read()
