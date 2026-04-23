"""Tests for 3D geometric primitives — no IMAS dependency."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.render
class TestCylindricalToCartesian:
    def test_identity_at_phi_zero(self):
        from imas_ink.three_d.primitives import cylindrical_to_cartesian

        r = np.array([1.0, 2.0])
        phi = np.array([0.0, 0.0])
        z = np.array([3.0, 4.0])
        x, y, z_out = cylindrical_to_cartesian(r, phi, z)
        np.testing.assert_allclose(x, r)
        np.testing.assert_allclose(y, 0.0, atol=1e-15)
        np.testing.assert_allclose(z_out, z)

    def test_quarter_turn(self):
        from imas_ink.three_d.primitives import cylindrical_to_cartesian

        r = np.array([1.0])
        phi = np.array([np.pi / 2])
        z = np.array([0.0])
        x, y, _z_out = cylindrical_to_cartesian(r, phi, z)
        np.testing.assert_allclose(x, 0.0, atol=1e-15)
        np.testing.assert_allclose(y, 1.0, atol=1e-15)

    def test_round_trip(self):
        from imas_ink.three_d.primitives import cylindrical_to_cartesian

        r_in = np.array([5.0, 3.0, 7.0])
        phi_in = np.array([0.3, 1.2, 2.8])
        z_in = np.array([1.0, -2.0, 0.5])
        x, y, z_out = cylindrical_to_cartesian(r_in, phi_in, z_in)
        r_back = np.sqrt(x**2 + y**2)
        phi_back = np.arctan2(y, x)
        np.testing.assert_allclose(r_back, r_in, atol=1e-12)
        np.testing.assert_allclose(phi_back, phi_in, atol=1e-12)
        np.testing.assert_allclose(z_out, z_in)


@pytest.mark.render
class TestRevolvePolygon:
    def test_rectangle_bounds(self):
        from imas_ink.three_d.primitives import revolve_polygon

        # Rectangle at r=5, z=0, width=1, height=2
        r = np.array([4.5, 5.5, 5.5, 4.5])
        z = np.array([-1.0, -1.0, 1.0, 1.0])
        mesh = revolve_polygon(r, z, n_theta=30)

        assert mesh.n_points > 0
        assert mesh.n_cells > 0

        bounds = mesh.bounds  # (xmin, xmax, ymin, ymax, zmin, zmax)
        # After revolution about z-axis, the mesh spans [-5.5, 5.5] in x and y
        assert bounds[0] < -4.0  # xmin
        assert bounds[1] > 4.0  # xmax
        assert bounds[4] <= -1.0  # zmin
        assert bounds[5] >= 1.0  # zmax

    def test_triangle(self):
        from imas_ink.three_d.primitives import revolve_polygon

        r = np.array([3.0, 4.0, 3.5])
        z = np.array([0.0, 0.0, 1.0])
        mesh = revolve_polygon(r, z, n_theta=20)
        assert mesh.n_points > 0


@pytest.mark.render
class TestRingFromRectangle:
    def test_centroid_height(self):
        from imas_ink.three_d.primitives import ring_from_rectangle

        mesh = ring_from_rectangle(r_center=6.0, z_center=3.0, dr=1.0, dz=0.5, n_theta=30)
        assert mesh.n_points > 0
        bounds = mesh.bounds
        # z bounds should bracket z_center ± dz/2
        assert bounds[4] <= 3.0 - 0.25 + 0.01
        assert bounds[5] >= 3.0 + 0.25 - 0.01


@pytest.mark.render
class TestSweepSectionAlongPath:
    def test_straight_line_prism(self):
        from imas_ink.three_d.primitives import sweep_section_along_path

        # Sweep a unit square along a straight line in x
        section = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
        path = np.array([[0, 0, 0], [10, 0, 0]], dtype=float)
        mesh = sweep_section_along_path(section, path, frame="frenet")

        assert mesh.n_points > 0
        bounds = mesh.bounds
        # Should span from x=0 to x=10
        assert bounds[0] <= 0.1
        assert bounds[1] >= 9.9

    def test_empty_path(self):
        from imas_ink.three_d.primitives import sweep_section_along_path

        section = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float)
        path = np.array([[0, 0, 0]], dtype=float)
        mesh = sweep_section_along_path(section, path, frame="frenet")
        assert mesh.n_points == 0
