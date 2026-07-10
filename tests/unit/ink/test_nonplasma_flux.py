"""Tests for corner-to-corner non-plasma flux rendering.

The equilibrium figure must contour the ENTIRE psi grid at a single uniform
step: confined surfaces inside the LCFS in the plasma style (blue) and every
surface outside the LCFS in the non-plasma style (light grey, thinner).  A
vacuum slice (no LCFS, sentinel psi_axis/psi_boundary) must render through the
same path without error, producing only the non-plasma family.
"""

from __future__ import annotations

import numpy as np
from matplotlib.collections import LineCollection

from imas_ink.style import DEFAULT_STYLE


def _synthetic_eq_slice(
    psi_axis: float = -50.0,
    psi_bnd: float = 2.0,
    n_r: int = 48,
    n_z: int = 64,
    r_min: float = 4.0,
    r_max: float = 8.0,
    z_min: float = -4.0,
    z_max: float = 4.0,
):
    """Elliptic paraboloid psi field: axis at the centroid, increasing out."""
    r_ax = (r_min + r_max) / 2.0
    z_ax = (z_min + z_max) / 2.0
    r_1d = np.linspace(r_min, r_max, n_r)
    z_1d = np.linspace(z_min, z_max, n_z)
    R, Z = np.meshgrid(r_1d, z_1d, indexing="ij")
    dr = (r_max - r_min) / 2.0
    dz = (z_max - z_min) / 2.0
    dist_sq = ((R - r_ax) / dr) ** 2 + ((Z - z_ax) / dz) ** 2
    psi_2d = psi_axis + (psi_bnd - psi_axis) * dist_sq
    return R, Z, psi_2d, psi_axis, psi_bnd, r_ax, z_ax


def _geom():
    from imas_ink._types import MachineGeometry

    r = np.array([3.5, 8.5, 8.5, 3.5, 3.5])
    z = np.array([-4.5, -4.5, 4.5, 4.5, -4.5])
    return MachineGeometry(
        wall_r=r,
        wall_z=z,
        coil_rects=[],
        wall_clip_vertices=np.column_stack([r, z]),
        wall_units=[(r, z)],
        probe_r=np.array([]),
        probe_z=np.array([]),
        probe_angle=np.array([]),
        flux_loop_r=np.array([]),
        flux_loop_z=np.array([]),
    )


def _slice(psi_axis, psi_bnd, sentinel=False, boundary=True):
    from imas_ink._types import EquilibriumSlice

    # A vacuum slice carries a real (non-flat) psi map from the coils but has
    # sentinel/NaN global quantities — the field shape must not depend on the
    # reported scalars, so build it from fixed values.
    R, Z, psi_2d, pa, pb, ra, za = _synthetic_eq_slice(psi_axis=-50.0, psi_bnd=2.0)
    br = bz = None
    if boundary:
        theta = np.linspace(0, 2 * np.pi, 60)
        br = ra + 1.2 * np.cos(theta)
        bz = za + 1.6 * np.sin(theta)
    return EquilibriumSlice(
        psi_2d=psi_2d,
        r_grid=R[:, 0],
        z_grid=Z[0, :],
        psi_axis=(float("nan") if sentinel else pa),
        psi_boundary=(float("nan") if sentinel else pb),
        r_axis=(float("nan") if sentinel else ra),
        z_axis=(float("nan") if sentinel else za),
        ip=(1.0e3 if sentinel else 1.0e6),
        time=0.5,
        converged=(not sentinel),
        x_points=[],
        boundary_r=br,
        boundary_z=bz,
    )


def _line_collections(ax):
    return [c for c in ax.collections if isinstance(c, LineCollection)]


def _color_hex(c):
    from matplotlib.colors import to_hex

    return to_hex(c)


class TestNonPlasmaFlux:
    def setup_method(self):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.close("all")

    def teardown_method(self):
        import matplotlib.pyplot as plt

        plt.close("all")

    def test_diverted_like_slice_has_both_styles(self):
        """A normal plasma slice produces both plasma-blue confined contours
        and light-grey thin non-plasma contours, sharing one step."""
        from imas_ink.figures import equilibrium_figure_mpl

        sl = _slice(-50.0, 2.0)
        _fig, ax = equilibrium_figure_mpl(sl, _geom())
        lcs = _line_collections(ax)
        colors = {_color_hex(lc.get_edgecolor()[0]) for lc in lcs if len(lc.get_edgecolor())}
        assert _color_hex(DEFAULT_STYLE.flux_color) in colors, (
            "expected plasma-blue confined contours"
        )
        assert _color_hex(DEFAULT_STYLE.sol_color) in colors, (
            "expected light-grey non-plasma contours"
        )

    def test_nonplasma_is_thinner_than_plasma(self):
        """Non-plasma lines are thinner than the confined plasma lines."""
        assert DEFAULT_STYLE.sol_linewidth < DEFAULT_STYLE.flux_linewidth

    def test_vacuum_slice_renders_without_error(self):
        """Sentinel psi_axis/boundary, no boundary outline → renders through the
        same path, producing only non-plasma contours, no confined blue."""
        from imas_ink.figures import equilibrium_figure_mpl

        sl = _slice(0.0, 0.0, sentinel=True, boundary=False)
        _fig, ax = equilibrium_figure_mpl(sl, _geom())
        lcs = _line_collections(ax)
        colors = {_color_hex(lc.get_edgecolor()[0]) for lc in lcs if len(lc.get_edgecolor())}
        assert _color_hex(DEFAULT_STYLE.sol_color) in colors, (
            "vacuum slice must still contour the grid"
        )
        assert _color_hex(DEFAULT_STYLE.flux_color) not in colors, (
            "vacuum slice has no confined region"
        )

    def test_full_grid_contoured_corner_to_corner(self):
        """The non-plasma contours reach the grid extrema (not clipped to the
        wall or the LCFS bbox)."""
        from imas_ink.figures import equilibrium_figure_mpl

        sl = _slice(-50.0, 2.0)
        _fig, ax = equilibrium_figure_mpl(sl, _geom())
        grey = _color_hex(DEFAULT_STYLE.sol_color)
        pts = []
        for lc in _line_collections(ax):
            if not len(lc.get_edgecolor()):
                continue
            if _color_hex(lc.get_edgecolor()[0]) != grey:
                continue
            for seg in lc.get_segments():
                pts.append(seg)
        assert pts, "expected non-plasma segments"
        allpts = np.vstack(pts)
        # Contours extend well outside the confined ~1.2 m minor radius,
        # reaching toward the grid edge (R spans 4..8, Z spans -4..4).
        assert allpts[:, 0].max() > 7.0
        assert allpts[:, 0].min() < 5.0
