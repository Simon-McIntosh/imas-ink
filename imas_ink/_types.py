"""Typed data containers for ink components.

All fields use SI units. COCOS 17 convention: psi_axis > psi_boundary.
EMPTY sentinel values (|v| > 1e9) are filtered at extraction time
and never appear in these dataclasses.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class EquilibriumSlice:
    """Single time-slice equilibrium data.

    Holds the 2D poloidal flux map, 1D grid vectors, global scalar
    quantities, and optional boundary / X-point geometry extracted
    from an ``equilibrium`` IDS.

    Examples
    --------
    >>> sl = extract_slice(eq_ids, 5)
    >>> sl.psi_2d.shape
    (65, 65)
    >>> sl.converged
    True
    """

    psi_2d: np.ndarray  # shape (nR, nZ)
    r_grid: np.ndarray  # shape (nR,)
    z_grid: np.ndarray  # shape (nZ,)
    psi_axis: float
    psi_boundary: float
    r_axis: float
    z_axis: float
    ip: float  # plasma current [A]
    time: float  # time [s]
    converged: bool
    x_points: list[tuple[float, float]] = field(default_factory=list)
    boundary_r: np.ndarray | None = None
    boundary_z: np.ndarray | None = None
    beta_pol: float | None = None
    li_3: float | None = None
    q95: float | None = None

    @property
    def R_2d(self) -> np.ndarray:
        """2D R meshgrid, shape (nR, nZ), indexing='ij'."""
        return np.meshgrid(self.r_grid, self.z_grid, indexing="ij")[0]

    @property
    def Z_2d(self) -> np.ndarray:
        """2D Z meshgrid, shape (nR, nZ), indexing='ij'."""
        return np.meshgrid(self.r_grid, self.z_grid, indexing="ij")[1]


@dataclass(frozen=True)
class CoilRect:
    """Bounding box of a PF coil element.

    Parameters
    ----------
    r : float
        Lower-left R coordinate [m].
    z : float
        Lower-left Z coordinate [m].
    width : float
        Extent in R (delta-R) [m].
    height : float
        Extent in Z (delta-Z) [m].
    name : str
        Human-readable coil name (e.g. ``"PF1"``).
    """

    r: float  # lower-left R
    z: float  # lower-left Z
    width: float  # delta-R
    height: float  # delta-Z
    name: str = ""


@dataclass(frozen=True)
class XPoint:
    """X-point location in the poloidal plane.

    Parameters
    ----------
    r : float
        Major radius [m].
    z : float
        Vertical position [m].
    """

    r: float
    z: float


@dataclass(frozen=True)
class MachineGeometry:
    """Static machine geometry — wall outline and PF coils.

    Examples
    --------
    >>> geom = extract_geometry(wall_ids, pf_ids)
    >>> geom.viewport
    (0.17, 2.05, -1.83, 1.83)
    """

    wall_r: np.ndarray  # (N,) wall R coords
    wall_z: np.ndarray  # (N,) wall Z coords
    coil_rects: list[CoilRect]  # PF coil bounding boxes
    wall_clip_vertices: np.ndarray  # (M, 2) closed polygon
    probe_r: np.ndarray = field(default_factory=lambda: np.array([]))
    probe_z: np.ndarray = field(default_factory=lambda: np.array([]))
    probe_angle: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_loop_r: np.ndarray = field(default_factory=lambda: np.array([]))
    flux_loop_z: np.ndarray = field(default_factory=lambda: np.array([]))

    @property
    def viewport(self) -> tuple[float, float, float, float]:
        """``(rmin, rmax, zmin, zmax)`` bounding box including coils + margin."""
        all_r = [self.wall_r.min(), self.wall_r.max()]
        all_z = [self.wall_z.min(), self.wall_z.max()]
        for c in self.coil_rects:
            all_r.extend([c.r, c.r + c.width])
            all_z.extend([c.z, c.z + c.height])
        margin = 0.03
        return (
            min(all_r) - margin,
            max(all_r) + margin,
            min(all_z) - margin,
            max(all_z) + margin,
        )


@dataclass(frozen=True)
class TimeTraces:
    """Global scalar time traces from an equilibrium IDS.

    Examples
    --------
    >>> tt = extract_time_traces(eq_ids)
    >>> tt.time.shape
    (150,)
    """

    time: np.ndarray  # (N,)
    ip: np.ndarray  # (N,) [A]
    beta_pol: np.ndarray  # (N,)
    li_3: np.ndarray  # (N,)
    q95: np.ndarray  # (N,)
    converged: np.ndarray  # (N,) bool
    chi_squared: np.ndarray  # (N,) total chi-squared
    n_iterations: np.ndarray  # (N,) iteration count


@dataclass(frozen=True)
class RadialProfiles:
    """1D radial profiles at a single time slice.

    Examples
    --------
    >>> rp = extract_profiles_1d(eq_ids, 5)
    >>> rp.psi_norm.shape
    (129,)
    """

    psi_norm: np.ndarray  # normalised psi in [0, 1]
    pressure: np.ndarray  # [Pa]
    j_tor: np.ndarray  # toroidal current density [A/m2]
    q: np.ndarray  # safety factor
    pprime: np.ndarray  # dp/dpsi
    ffprime: np.ndarray  # FF'
    time: float
