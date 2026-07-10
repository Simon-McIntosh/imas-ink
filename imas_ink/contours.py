"""Backend-neutral contour extraction from a 2D psi field.

Wraps :func:`contourpy.contour_generator` and provides methods that
return lists of (R, Z) coordinate arrays — consumable by any renderer.

Examples
--------
>>> sl = extract_slice(eq_ids, 5)
>>> cx = ContourExtractor(sl.R_2d, sl.Z_2d, sl.psi_2d)
>>> segs = cx.flux_surfaces(sl.psi_axis, sl.psi_boundary, n=6)
>>> len(segs)
6
"""

from __future__ import annotations

from dataclasses import dataclass

import contourpy
import numpy as np

from ._cocos import make_levels


@dataclass
class ContourExtractor:
    """Backend-neutral contour extraction from a 2D psi field.

    Wraps ``contourpy.contour_generator()`` and provides methods that
    return lists of ``(R, Z)`` coordinate arrays — consumable by any
    renderer.

    Parameters
    ----------
    r_2d : np.ndarray
        2D major radius grid, shape ``(nR, nZ)``.
    z_2d : np.ndarray
        2D vertical position grid, shape ``(nR, nZ)``.
    psi_2d : np.ndarray
        2D poloidal flux, shape ``(nR, nZ)``.

    Examples
    --------
    >>> cx = ContourExtractor(sl.R_2d, sl.Z_2d, sl.psi_2d)
    >>> lines = cx.lines_at(0.5)
    >>> all(seg.shape[1] == 2 for seg in lines)
    True
    """

    r_2d: np.ndarray
    z_2d: np.ndarray
    psi_2d: np.ndarray

    def __post_init__(self) -> None:
        self._gen = contourpy.contour_generator(
            self.r_2d,
            self.z_2d,
            self.psi_2d,
            line_type="SeparateCode",
            quad_as_tri=True,
        )

    def lines_at(self, psi: float) -> list[np.ndarray]:
        """Return isoline segments at a single *psi* value.

        Each segment is an ndarray of shape ``(N, 2)`` with columns
        ``[R, Z]``. Disconnected contour pieces are separate arrays.

        Parameters
        ----------
        psi : float
            Flux level at which to extract isolines.

        Returns
        -------
        list[np.ndarray]
            One ``(N, 2)`` array per disconnected segment.
        """
        points, _codes = self._gen.lines(psi)
        return [seg for seg in points if len(seg) >= 2]

    def is_closed(self, psi: float, seg_index: int = 0) -> bool:
        """Check if segment *seg_index* at level *psi* is a closed contour.

        Parameters
        ----------
        psi : float
            Flux level.
        seg_index : int
            Index of the segment to check (default first).

        Returns
        -------
        bool
        """
        _points, codes = self._gen.lines(psi)
        if seg_index >= len(codes):
            return False
        return codes[seg_index][-1] == 79  # CLOSEPOLY

    def flux_surfaces(self, psi_axis: float, psi_bnd: float, n: int = 6) -> list[list[np.ndarray]]:
        """Extract *n* interior flux surface isolines.

        Returns a list of *n* level-groups, each a list of segments.
        Uses COCOS 17 ordering: levels between *psi_bnd* and *psi_axis*.

        Parameters
        ----------
        psi_axis : float
            Poloidal flux at the magnetic axis.
        psi_bnd : float
            Poloidal flux at the LCFS.
        n : int
            Number of interior levels.

        Returns
        -------
        list[list[np.ndarray]]
            ``segments[level_index]`` is a list of ``(N, 2)`` arrays.
        """
        levels = make_levels(psi_axis, psi_bnd, n)
        return [self.lines_at(lev) for lev in levels]

    def vacuum_surfaces(
        self,
        psi_axis: float,
        psi_bnd: float,
        n: int = 10,
    ) -> list[list[np.ndarray]]:
        """Extract *n* vacuum / SOL flux surface isolines outside the LCFS.

        Levels span from just outside the LCFS to the extremum of the psi
        field on the computational grid — automatically covering the full
        vacuum region regardless of COCOS sign convention.  Contours are
        **not** clipped so that unconverged slices (where the LCFS may lie
        outside the first wall) remain fully visible.

        The level spacing is computed from the vacuum psi range
        ``(psi_bnd, psi_grid_edge)`` so that *n* contours are distributed
        evenly across the entire vacuum field rather than being cramped near
        the LCFS.

        Parameters
        ----------
        psi_axis : float
            Poloidal flux at the magnetic axis (used only to determine the
            sign convention / vacuum direction).
        psi_bnd : float
            Poloidal flux at the last closed flux surface.
        n : int
            Number of vacuum levels.  Default 10.

        Returns
        -------
        list[list[np.ndarray]]
            ``segments[level_index]`` is a list of ``(N, 2)`` arrays.
        """
        # Grid extremum in the vacuum direction (works for either COCOS sign)
        if psi_axis < psi_bnd:
            # Vacuum psi is LARGER than psi_bnd (e.g. COCOS-3 / WEST DDv3)
            psi_grid_edge = float(self.psi_2d.max())
        else:
            # Vacuum psi is SMALLER than psi_bnd (e.g. COCOS-17 / DDv4)
            psi_grid_edge = float(self.psi_2d.min())
        # n evenly-spaced levels between LCFS and grid edge (endpoints excluded)
        levels = np.linspace(psi_bnd, psi_grid_edge, n + 2)[1:-1]
        return [self.lines_at(lev) for lev in levels]

    def uniform_step_levels(
        self,
        psi_axis: float,
        psi_bnd: float,
        n_interior: int = 6,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Contour levels at a SINGLE uniform step across the whole grid.

        The step is set once by the confined region —
        ``dpsi = |psi_axis - psi_bnd| / (n_interior + 1)`` — and the
        **same** step is used for every level from one corner of the psi
        grid to the other.  Levels are generated across the entire grid
        psi range ``[psi_2d.min(), psi_2d.max()]`` (corner-to-corner, no
        clipping to the wall or LCFS) and partitioned into those strictly
        inside the LCFS and those outside it.

        Vacuum / non-plasma slices — where ``psi_axis`` and/or ``psi_bnd``
        are IMAS EMPTY sentinels (extracted as NaN or ``|v| > 1e10``) or
        the two coincide — have no confined region.  The step then falls
        back to a grid-range spacing (``(psi_max - psi_min) / (n+1)``) and
        **all** levels are returned as exterior, so the full flux map is
        still contoured in the non-plasma style.

        Parameters
        ----------
        psi_axis : float
            Poloidal flux at the magnetic axis (may be a sentinel/NaN).
        psi_bnd : float
            Poloidal flux at the LCFS (may be a sentinel/NaN).
        n_interior : int
            Number of contour intervals used to size the uniform step.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(interior_levels, exterior_levels)`` — sorted level arrays
            strictly inside and strictly outside the LCFS respectively.
            For a vacuum slice ``interior_levels`` is empty.
        """
        finite = np.isfinite(self.psi_2d)
        if not finite.any():
            return np.array([]), np.array([])
        psi_min = float(np.nanmin(self.psi_2d))
        psi_max = float(np.nanmax(self.psi_2d))
        if not np.isfinite(psi_min) or not np.isfinite(psi_max) or psi_max <= psi_min:
            return np.array([]), np.array([])

        sentinel = 1e10

        def _usable(v: float) -> bool:
            return v is not None and np.isfinite(float(v)) and abs(float(v)) < sentinel

        have_lcfs = (
            _usable(psi_axis) and _usable(psi_bnd) and abs(float(psi_axis) - float(psi_bnd)) > 0.0
        )

        if have_lcfs:
            dpsi = abs(float(psi_axis) - float(psi_bnd)) / (n_interior + 1)
            anchor = float(psi_bnd)
        else:
            # Vacuum slice: no confined region — size the step from the grid.
            dpsi = (psi_max - psi_min) / (n_interior + 1)
            anchor = psi_min

        if dpsi <= 0:
            return np.array([]), np.array([])

        # Generate a uniform ladder of levels anchored at the LCFS (or the
        # grid floor for a vacuum slice) spanning the full grid range.
        k_lo = int(np.floor((psi_min - anchor) / dpsi))
        k_hi = int(np.ceil((psi_max - anchor) / dpsi))
        levels = anchor + np.arange(k_lo, k_hi + 1) * dpsi
        # Keep strictly inside the grid range (endpoints carry no contour).
        levels = levels[(levels > psi_min) & (levels < psi_max)]

        if not have_lcfs:
            return np.array([]), np.sort(levels)

        lo, hi = sorted((float(psi_axis), float(psi_bnd)))
        interior_mask = (levels > lo) & (levels < hi)
        interior = np.sort(levels[interior_mask])
        exterior = np.sort(levels[~interior_mask])
        return interior, exterior

    def separatrix(self, psi_bnd: float) -> list[np.ndarray]:
        """Extract the separatrix (LCFS) contour segments.

        Parameters
        ----------
        psi_bnd : float
            Poloidal flux at the last closed flux surface.

        Returns
        -------
        list[np.ndarray]
            List of ``(N, 2)`` segment arrays.
        """
        return self.lines_at(psi_bnd)
