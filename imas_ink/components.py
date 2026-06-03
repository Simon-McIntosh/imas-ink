"""Visual components — data + style, no rendering logic.

Components hold the data needed to draw a visual element. They are
passed to ``render_mpl()`` or ``render_alt()`` for backend-specific
rendering.

Components are **not** frozen because users may want to swap the
``style`` attribute after construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .style import DEFAULT_STYLE, InkStyle


@dataclass
class FluxContours:
    """Interior flux surface contours.

    Each level is a list of (N, 2) segment arrays produced by
    :class:`ContourExtractor`.

    Examples
    --------
    >>> cx = ContourExtractor(sl.R_2d, sl.Z_2d, sl.psi_2d)
    >>> fc = FluxContours(cx.flux_surfaces(sl.psi_axis, sl.psi_boundary))
    """

    segments: list[list[np.ndarray]]  # [level][segment] of (N, 2) arrays
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class Separatrix:
    """Last closed flux surface (LCFS) contour segments.

    Examples
    --------
    >>> sep = Separatrix(cx.separatrix(sl.psi_boundary))
    """

    segments: list[np.ndarray]  # list of (N, 2) arrays
    x_points: list[tuple[float, float]] = field(default_factory=list)
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class WallOutline:
    """First wall outline polygon (one or more units).

    ``wall_r`` / ``wall_z`` hold the first unit for backward compatibility.
    ``wall_units`` holds all units as a list of ``(r_array, z_array)`` pairs.
    When non-empty, renderers iterate ``wall_units`` to draw every unit.

    Examples
    --------
    >>> wall = WallOutline(geom.wall_r, geom.wall_z, wall_units=geom.wall_units)
    """

    wall_r: np.ndarray
    wall_z: np.ndarray
    wall_units: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class CoilRects:
    """PF coil bounding rectangles.

    Examples
    --------
    >>> coils = CoilRects(geom.coil_rects)
    """

    rects: list  # list of CoilRect
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class OPointMarker:
    """Magnetic axis (O-point) marker."""

    r: float
    z: float
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class XPointMarkers:
    """X-point markers (1 for SN, 2 for DN, 0 for limiter)."""

    points: list[tuple[float, float]]
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class TimeLabel:
    """Time annotation in figure corner."""

    time: float  # [s]
    converged: bool = True
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class TimeSeries:
    """1D time series data for line plot.

    Examples
    --------
    >>> ts = TimeSeries(tt.time, tt.ip / 1e6, label="Ip", ylabel="Ip", units="MA")
    """

    time: np.ndarray
    values: np.ndarray
    label: str = ""
    ylabel: str = ""
    units: str = ""
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class RadialProfile:
    """1D radial profile data.

    Examples
    --------
    >>> rp_comp = RadialProfile(rp.psi_norm, rp.pressure / 1e3, ylabel="p", units="kPa")
    """

    psi_norm: np.ndarray
    values: np.ndarray
    label: str = ""
    ylabel: str = ""
    units: str = ""
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class ScatterPoints:
    """2D scatter data (e.g. measured vs fitted diagnostics)."""

    x: np.ndarray
    y: np.ndarray
    xlabel: str = ""
    ylabel: str = ""
    label: str = ""
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class MagneticProbes:
    """B-pol magnetic probe positions with optional orientation angles.

    ``angles`` are poloidal angles in radians. Non-finite (NaN) entries
    indicate the probe has no orientation vector and should be drawn as
    a position-only marker.
    """

    positions_r: np.ndarray
    positions_z: np.ndarray
    angles: np.ndarray
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)


@dataclass
class FluxLoops:
    """Flux loop positions (no orientation)."""

    positions_r: np.ndarray
    positions_z: np.ndarray
    style: InkStyle = field(default_factory=lambda: DEFAULT_STYLE)
