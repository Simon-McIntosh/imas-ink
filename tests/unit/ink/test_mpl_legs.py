"""Tests for matplotlib rendering of DivertorLegs and StrikePoints.

These components carry IDS-verbatim topology geometry (DD PR #243 levelset
legs + constraints strike points).  The renderers must draw them faithfully
and degrade gracefully (no artists, no error) when the IDS carried none.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from imas_ink.components import DivertorLegs, StrikePoints
from imas_ink.mpl import render_mpl


class TestRenderLegsMpl:
    """_render_legs_mpl — single continuous lines, no re-contouring."""

    def test_empty_no_error_no_artist(self):
        """No legs (stock 4.1.0 / limited / vacuum) → nothing drawn."""
        fig, ax = plt.subplots()
        render_mpl(ax, DivertorLegs([]))  # must not raise
        assert len(ax.collections) == 0
        plt.close(fig)

    def test_two_legs_one_collection(self):
        """Two legs render as a single LineCollection with two segments."""
        fig, ax = plt.subplots()
        leg0 = np.array([[2.21, -0.59], [2.15, -0.62]])
        leg1 = np.array([[2.21, -0.59], [2.24, -0.65]])
        render_mpl(ax, DivertorLegs([leg0, leg1]))
        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(lcs) == 1
        assert len(lcs[0].get_segments()) == 2
        plt.close(fig)

    def test_degenerate_leg_skipped(self):
        """A single-point leg is too short to draw → skipped, no error."""
        fig, ax = plt.subplots()
        good = np.array([[2.21, -0.59], [2.15, -0.62]])
        bad = np.array([[2.21, -0.59]])  # only one point
        render_mpl(ax, DivertorLegs([bad, good]))
        lcs = [c for c in ax.collections if isinstance(c, LineCollection)]
        assert len(lcs) == 1
        assert len(lcs[0].get_segments()) == 1  # only the good leg
        plt.close(fig)


class TestRenderStrikesMpl:
    """_render_strikes_mpl — IDS-verbatim strike markers."""

    def test_empty_no_error(self):
        fig, ax = plt.subplots()
        render_mpl(ax, StrikePoints([]))  # must not raise
        plt.close(fig)

    def test_strikes_plotted(self):
        fig, ax = plt.subplots()
        render_mpl(ax, StrikePoints([(2.16, -0.62), (2.23, -0.65)]))
        # One Line2D with two marker points
        assert len(ax.lines) == 1
        xdata = ax.lines[0].get_xdata()
        assert len(xdata) == 2
        plt.close(fig)
