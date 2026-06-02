"""Tests for magnetic-probe component classification and orientation rendering.

Background
----------
The IMAS Data Dictionary defines
``magnetics/b_field_pol_probe/poloidal_angle`` as the angle of the *sensor
normal vector* (the coil axis, i.e. the direction of the magnetic-field
component the probe measures), clockwise from +R̂.  EFIT consumes the
identical vector n = (cos θ, −sin θ).  The direction tick must therefore be
drawn along θ with that exact convention (no +90° offset, no sign flip).

Multi-component sensors (WEST, ITER) mount two pickup coils at one location
with axes ~90° apart — one tangential, one normal to the wall.  Rendering
both in one colour makes a tangential B-pol tick indistinguishable from a
co-located normal-component tick.  :func:`classify_probe_components` assigns
each probe a component index so the renderers can colour them distinctly,
without altering the DD-defined angle.
"""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from imas_ink.components import MagneticProbes
from imas_ink.geometry import classify_probe_components
from imas_ink.mpl import render_mpl
from imas_ink.style import DEFAULT_STYLE


class TestClassifyProbeComponents:
    """classify_probe_components — co-located orthogonal-pair detection."""

    def test_empty(self):
        out = classify_probe_components(np.array([]), np.array([]), np.array([]))
        assert out.shape == (0,)

    def test_single_probe_is_primary(self):
        out = classify_probe_components(np.array([3.0]), np.array([0.0]), np.array([0.5]))
        assert out.tolist() == [0]

    def test_distinct_positions_all_primary(self):
        """Probes at different locations are all component 0 (singletons)."""
        r = np.array([3.0, 4.0, 5.0])
        z = np.array([0.0, 0.5, 1.0])
        a = np.array([0.0, 1.0, 2.0])  # arbitrary, but not co-located
        out = classify_probe_components(r, z, a)
        assert out.tolist() == [0, 0, 0]

    def test_colocated_orthogonal_pair_split(self):
        """Two co-located probes 90 deg apart → two distinct components."""
        r = np.array([3.0, 3.0])
        z = np.array([0.0, 0.0])
        a = np.array([0.0, np.pi / 2])
        out = classify_probe_components(r, z, a)
        assert out[0] == 0
        assert out[1] == 1
        assert out[0] != out[1]

    def test_colocated_same_orientation_one_component(self):
        """Co-located probes with the SAME axis (toroidal duplicates) → one component."""
        # e.g. ITER MLF-3xxx/6xxx/9xxx: same (R,Z), same poloidal_angle
        r = np.array([3.215, 3.215, 3.215])
        z = np.array([-3.304, -3.304, -3.304])
        a = np.full(3, np.radians(270.0))
        out = classify_probe_components(r, z, a)
        assert out.tolist() == [0, 0, 0]

    def test_undirected_axis_180_is_same_component(self):
        """θ and θ+180° describe the same (undirected) sensing axis → one component."""
        r = np.array([3.0, 3.0])
        z = np.array([0.0, 0.0])
        a = np.array([0.3, 0.3 + np.pi])
        out = classify_probe_components(r, z, a)
        assert out.tolist() == [0, 0]

    def test_nan_angle_is_primary(self):
        """Probes without an orientation are always component 0."""
        r = np.array([3.0, 3.0])
        z = np.array([0.0, 0.0])
        a = np.array([np.nan, np.pi / 2])
        out = classify_probe_components(r, z, a)
        assert out[0] == 0

    def test_mixed_pairs_and_singletons(self):
        """A co-located pair plus a lone probe at a third location."""
        r = np.array([3.0, 3.0, 5.0])
        z = np.array([0.0, 0.0, 0.0])
        a = np.array([0.0, np.pi / 2, 0.0])
        out = classify_probe_components(r, z, a)
        assert out.tolist() == [0, 1, 0]


class TestProbeOrientationRendering:
    """_render_probes_mpl — DD-correct sensor-normal direction + family colour."""

    def test_tick_drawn_along_poloidal_angle(self):
        """Tick axis must align with n = (cos θ, −sin θ) — the DD sensor normal."""
        fig, ax = plt.subplots()
        theta = np.pi / 3  # 60 deg, clockwise from +R
        r0, z0 = 5.0, 0.0
        mp = MagneticProbes(
            positions_r=np.array([r0]),
            positions_z=np.array([z0]),
            angles=np.array([theta]),
        )
        render_mpl(ax, mp)
        lc = [c for c in ax.collections if isinstance(c, LineCollection)][0]
        seg = lc.get_segments()[0]
        dr = seg[1, 0] - seg[0, 0]
        dz = seg[1, 1] - seg[0, 1]
        # Axis orientation (undirected): atan2(dz, dr) ≡ −θ (mod π).
        axis_ang = np.arctan2(dz, dr)
        diff = (axis_ang - (-theta)) % np.pi
        diff = min(diff, np.pi - diff)
        assert diff < 1e-6, "tick must lie along DD sensor-normal axis (cos θ, −sin θ)"
        # Centred tick: midpoint is the probe position.
        mid_r = 0.5 * (seg[0, 0] + seg[1, 0])
        mid_z = 0.5 * (seg[0, 1] + seg[1, 1])
        assert abs(mid_r - r0) < 1e-9
        assert abs(mid_z - z0) < 1e-9
        # Full length equals the configured arrow length.
        length = np.hypot(dr, dz)
        np.testing.assert_allclose(length, DEFAULT_STYLE.probe_arrow_length, atol=1e-9)
        plt.close(fig)

    def test_colocated_pair_rendered_in_two_colours(self):
        """A co-located orthogonal pair must get two distinct tick colours."""
        fig, ax = plt.subplots()
        mp = MagneticProbes(
            positions_r=np.array([3.0, 3.0]),
            positions_z=np.array([0.0, 0.0]),
            angles=np.array([0.0, np.pi / 2]),
        )
        render_mpl(ax, mp)
        lc = [c for c in ax.collections if isinstance(c, LineCollection)][0]
        colors = lc.get_colors()
        assert len(colors) == 2
        assert not np.allclose(colors[0], colors[1]), (
            "the two orthogonal components must be drawn in different colours"
        )
        plt.close(fig)

    def test_single_orientation_array_one_colour(self):
        """A non-paired probe array stays a single colour (no spurious split)."""
        fig, ax = plt.subplots()
        n = 4
        mp = MagneticProbes(
            positions_r=np.array([3.0, 4.0, 5.0, 6.0]),
            positions_z=np.array([0.0, 0.5, 1.0, 1.5]),
            angles=np.array([0.1, 0.2, 0.3, 0.4]),
        )
        render_mpl(ax, mp)
        lc = [c for c in ax.collections if isinstance(c, LineCollection)][0]
        colors = lc.get_colors()
        # All ticks the same (primary) colour.
        assert all(np.allclose(colors[0], c) for c in colors)
        assert len(lc.get_segments()) == n
        plt.close(fig)
