"""Tests for closed-manifold validation and the sweep endcap fix."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.render
class TestEnsureClosedManifold:
    """Tests for :func:`ensure_closed_manifold`."""

    def test_synthetic_open_swept_tube_repaired(self):
        """Sweep along an open path → repair succeeds with 0 open edges."""
        from imas_ink.three_d.manifold import ensure_closed_manifold
        from imas_ink.three_d.primitives import sweep_section_along_path

        section = np.array(
            [[-0.2, -0.1], [0.2, -0.1], [0.2, 0.1], [-0.2, 0.1]]
        )
        path = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 1]], dtype=float)
        mesh = sweep_section_along_path(section, path, frame="frenet")

        # Should not raise — the sweep already closes endcaps, and
        # ensure_closed_manifold validates + normalises.
        repaired = ensure_closed_manifold(mesh, name="open_tube")

        boundary = repaired.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        assert boundary.n_cells == 0

    def test_synthetic_closed_swept_ring_already_closed(self):
        """Sweep along a closed ring path → repair is a no-op."""
        from imas_ink.three_d.manifold import ensure_closed_manifold
        from imas_ink.three_d.primitives import sweep_section_along_path

        section = np.array(
            [[-0.1, -0.1], [0.1, -0.1], [0.1, 0.1], [-0.1, 0.1]]
        )
        theta = np.linspace(0, 2 * np.pi, 33)
        path = np.column_stack(
            [3.0 * np.cos(theta), 3.0 * np.sin(theta), np.zeros_like(theta)]
        )
        mesh = sweep_section_along_path(section, path, frame="frenet")

        repaired = ensure_closed_manifold(mesh, name="ring")

        boundary = repaired.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        assert boundary.n_cells == 0

    def test_torus_already_manifold(self):
        """pv.Torus() is already a closed manifold — no modification."""
        import pyvista as pv

        from imas_ink.three_d.manifold import ensure_closed_manifold

        torus = pv.ParametricTorus()
        repaired = ensure_closed_manifold(torus, name="torus")

        boundary = repaired.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        assert boundary.n_cells == 0

    def test_open_polyline_unrepairable_raises(self):
        """A bare polyline (no faces) cannot be repaired → raises."""
        import pyvista as pv

        from imas_ink.three_d.manifold import (
            MeshNotManifoldError,
            ensure_closed_manifold,
        )

        points = np.array([[0, 0, 0], [1, 0, 0], [2, 1, 0]], dtype=float)
        lines = np.array([3, 0, 1, 2])
        mesh = pv.PolyData(points, lines=lines)

        with pytest.raises(MeshNotManifoldError) as exc_info:
            ensure_closed_manifold(mesh, name="polyline")

        assert exc_info.value.n_open_edges > 0
        assert exc_info.value.name == "polyline"


@pytest.mark.render
class TestSweptOpenPathNowCapped:
    """Regression test: sweep along an open path produces a capped mesh."""

    def test_swept_open_path_now_capped(self):
        """Open-path sweep has zero open edges WITHOUT ensure_closed_manifold."""
        from imas_ink.three_d.primitives import sweep_section_along_path

        section = np.array(
            [[-0.3, -0.2], [0.3, -0.2], [0.3, 0.2], [-0.3, 0.2]]
        )
        path = np.array(
            [[0, 0, 0], [2, 0, 0], [4, 0, 0.5], [6, 0, 1]], dtype=float
        )
        mesh = sweep_section_along_path(section, path, frame="planar")

        # Triangulate + clean for edge extraction (matching manifold pipeline)
        tri = mesh.triangulate().clean()
        boundary = tri.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        assert boundary.n_cells == 0, (
            f"Expected 0 open edges from capped sweep, got {boundary.n_cells}"
        )
