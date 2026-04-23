"""Tests for matplotlib rendering of MagneticProbes and FluxLoops."""

from __future__ import annotations

import matplotlib
import numpy as np
from numpy.testing import assert_allclose

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection, PathCollection

from imas_ink.components import FluxLoops, MagneticProbes
from imas_ink.mpl import render_mpl
from imas_ink.style import DEFAULT_STYLE


class TestRenderProbesMpl:
    """_render_probes_mpl — B-pol probe markers and direction ticks."""

    def test_empty_no_error(self):
        """Rendering empty probes must not error."""
        fig, ax = plt.subplots()
        mp = MagneticProbes(
            positions_r=np.array([]),
            positions_z=np.array([]),
            angles=np.array([]),
        )
        render_mpl(ax, mp)  # must not raise
        plt.close(fig)

    def test_scatter_artists(self):
        """Probe scatter produces a PathCollection artist."""
        fig, ax = plt.subplots()
        n = 5
        mp = MagneticProbes(
            positions_r=np.linspace(3, 7, n),
            positions_z=np.linspace(-2, 2, n),
            angles=np.full(n, np.nan),  # no direction ticks
        )
        render_mpl(ax, mp)
        # Should have at least one PathCollection (scatter)
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) >= 1
        # The scatter should have n offsets
        assert len(scatters[0].get_offsets()) == n
        plt.close(fig)

    def test_direction_arrows(self):
        """Probes with finite angles produce a LineCollection for direction ticks."""
        fig, ax = plt.subplots()
        n = 4
        angles = np.array([0.0, np.pi / 2, np.pi, 3 * np.pi / 2])
        mp = MagneticProbes(
            positions_r=np.array([4.0, 5.0, 6.0, 7.0]),
            positions_z=np.array([0.0, 0.0, 0.0, 0.0]),
            angles=angles,
        )
        render_mpl(ax, mp)
        line_colls = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_colls) >= 1
        # LineCollection should have n segments (one per probe with finite angle)
        segs = line_colls[0].get_segments()
        assert len(segs) == n
        plt.close(fig)

    def test_mixed_nan_angles(self):
        """Only finite-angle probes get direction ticks."""
        fig, ax = plt.subplots()
        angles = np.array([0.0, np.nan, np.pi / 4, np.nan, np.pi])
        mp = MagneticProbes(
            positions_r=np.ones(5) * 5.0,
            positions_z=np.linspace(-1, 1, 5),
            angles=angles,
        )
        render_mpl(ax, mp)
        line_colls = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_colls) >= 1
        segs = line_colls[0].get_segments()
        assert len(segs) == 3  # only 3 finite angles
        plt.close(fig)

    def test_arrow_direction_accuracy(self):
        """Direction tick endpoints must align with the declared angle within 1e-6."""
        fig, ax = plt.subplots()
        angle = np.pi / 3  # 60 degrees
        r0, z0 = 5.0, 0.0
        mp = MagneticProbes(
            positions_r=np.array([r0]),
            positions_z=np.array([z0]),
            angles=np.array([angle]),
        )
        render_mpl(ax, mp)

        line_colls = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(line_colls) >= 1
        seg = line_colls[0].get_segments()[0]
        # seg is [[r_start, z_start], [r_end, z_end]]
        dr = seg[1, 0] - seg[0, 0]
        dz = seg[1, 1] - seg[0, 1]
        # The direction should match the declared angle
        actual_angle = np.arctan2(dz, dr)
        assert_allclose(actual_angle, angle, atol=1e-6)

        # Length should equal probe_arrow_length
        length = np.sqrt(dr**2 + dz**2)
        assert_allclose(length, DEFAULT_STYLE.probe_arrow_length, atol=1e-6)
        plt.close(fig)


class TestRenderFluxLoopsMpl:
    """_render_fluxloops_mpl — flux loop markers."""

    def test_empty_no_error(self):
        """Rendering empty flux loops must not error."""
        fig, ax = plt.subplots()
        fl = FluxLoops(positions_r=np.array([]), positions_z=np.array([]))
        render_mpl(ax, fl)  # must not raise
        plt.close(fig)

    def test_scatter_artists(self):
        """Flux loop scatter produces a PathCollection artist."""
        fig, ax = plt.subplots()
        n = 8
        fl = FluxLoops(
            positions_r=np.linspace(3, 7, n),
            positions_z=np.linspace(-2, 2, n),
        )
        render_mpl(ax, fl)
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        assert len(scatters) >= 1
        assert len(scatters[0].get_offsets()) == n
        plt.close(fig)

    def test_correct_count(self):
        """Each flux loop produces exactly one marker."""
        fig, ax = plt.subplots()
        n = 12
        fl = FluxLoops(
            positions_r=np.linspace(3, 7, n),
            positions_z=np.zeros(n),
        )
        render_mpl(ax, fl)
        scatters = [c for c in ax.collections if isinstance(c, PathCollection)]
        total = sum(len(s.get_offsets()) for s in scatters)
        assert total == n
        plt.close(fig)
