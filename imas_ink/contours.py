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
