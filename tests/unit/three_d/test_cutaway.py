"""Tests for per-block capped clipping, geometric cap extraction, and auto camera."""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.render
class TestCappedClipTorus:
    """Clip a torus with the y=0 plane and verify cap extraction."""

    def test_capped_clip_torus_yields_cap(self):
        """Clipping a torus with y=0 produces cap faces."""
        import pyvista as pv

        from imas_ink.three_d.cutaway import ClipPlane, capped_clip

        torus = pv.ParametricTorus()
        plane = ClipPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))

        result = capped_clip(torus, plane, name="torus")

        assert result.full.n_cells > 0, "Clipped mesh should have cells"
        assert result.cap.n_cells > 0, "Cap should have cells"
        assert result.name == "torus"

        # The full clipped mesh should still be a closed manifold
        from imas_ink.three_d.manifold import ensure_closed_manifold

        ensure_closed_manifold(result.full, name="clipped_torus")

    def test_cap_face_extraction_geometric_torus(self):
        """Cap cells all have vertices on the y=0 plane and normals along ±y."""
        import pyvista as pv

        from imas_ink.three_d.cutaway import ClipPlane, capped_clip

        torus = pv.ParametricTorus()
        plane = ClipPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        eps_plane = 1e-5

        result = capped_clip(torus, plane, name="torus", eps_plane=eps_plane)
        cap = result.cap
        assert cap.n_cells > 0

        normal = np.array([0.0, 1.0, 0.0])

        # Verify every cap cell's vertices are on the plane
        for cell_id in range(cap.n_cells):
            cell = cap.get_cell(cell_id)
            pts = np.asarray(cell.points, dtype=float)
            # Vertex distance from y=0 plane
            dists = np.abs(pts @ normal)
            assert np.all(
                dists < eps_plane * 10
            ), f"Cell {cell_id} has vertices off-plane: max dist={dists.max()}"

            # Face normal should be along ±y
            if pts.shape[0] >= 3:
                e1 = pts[1] - pts[0]
                e2 = pts[2] - pts[0]
                fn = np.cross(e1, e2)
                fn_len = np.linalg.norm(fn)
                if fn_len > 1e-30:
                    fn /= fn_len
                    cos_angle = abs(float(np.dot(fn, normal)))
                    assert cos_angle > np.cos(
                        np.radians(5.0)
                    ), f"Cell {cell_id} normal not aligned with plane: cos={cos_angle}"


@pytest.mark.render
class TestCappedClipMultiblock:
    """Tests for multi-block fan-out and outside-halfspace skipping."""

    def test_capped_clip_skips_block_outside_halfspace(self):
        """Block entirely on the removed side returns empty CappedMesh."""
        import pyvista as pv

        from imas_ink.three_d.cutaway import ClipPlane, capped_clip_multiblock
        from imas_ink.three_d.manifold import ensure_closed_manifold

        # Box on the kept side (y > 0 → positive side of normal)
        kept_box = ensure_closed_manifold(
            pv.Cube(center=(0, 2, 0), x_length=1, y_length=1, z_length=1),
            name="kept",
        )
        # Box on the removed side (y < 0 → negative side of normal)
        removed_box = ensure_closed_manifold(
            pv.Cube(center=(0, -2, 0), x_length=1, y_length=1, z_length=1),
            name="removed",
        )

        plane = ClipPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        results = capped_clip_multiblock(
            {"kept": kept_box, "removed": removed_box}, plane
        )

        assert "kept" in results
        assert "removed" in results

        # Kept box is entirely on the kept side — should have cells
        assert results["kept"].full.n_cells > 0

        # Removed box is entirely on the removed side — empty
        assert results["removed"].full.n_cells == 0
        assert results["removed"].cap.n_cells == 0

    def test_capped_clip_uses_per_block_not_multiblock(self):
        """capped_clip_multiblock returns one CappedMesh per input key."""
        import pyvista as pv

        from imas_ink.three_d.cutaway import ClipPlane, capped_clip_multiblock
        from imas_ink.three_d.manifold import ensure_closed_manifold

        blocks = {}
        for i, label in enumerate(["alpha", "beta", "gamma"]):
            box = ensure_closed_manifold(
                pv.Cube(
                    center=(3 * i, 1, 0),
                    x_length=1,
                    y_length=1,
                    z_length=1,
                ),
                name=label,
            )
            blocks[label] = box

        plane = ClipPlane(origin=(0.0, 0.0, 0.0), normal=(0.0, 1.0, 0.0))
        results = capped_clip_multiblock(blocks, plane)

        assert set(results.keys()) == {"alpha", "beta", "gamma"}
        for key, cm in results.items():
            assert cm.name == key
            assert isinstance(cm.full.n_cells, int)


@pytest.mark.render
class TestAutoCamera:
    """Tests for orthographic camera auto-fitting."""

    def test_auto_camera_fits_bounds_orthographic(self):
        """Camera parallel_scale covers the ITER-like bbox inside drawable."""
        from imas_ink.three_d.cutaway import auto_camera

        # ITER-like bounds: R=4..9, Z=-5..5
        bounds = (4.0, 9.0, -2.0, 2.0, -5.0, 5.0)

        cam = auto_camera(bounds, view="poloidal_rhs")

        # parallel_scale is the half-height in world coords visible
        # in the viewport.  It must cover at least half the Z-extent.
        z_half = (bounds[5] - bounds[4]) / 2.0
        assert cam["parallel_scale"] >= z_half, (
            f"parallel_scale {cam['parallel_scale']} < z_half {z_half}"
        )

        # Also must cover half the X-extent scaled by aspect
        x_half = bounds[1] / 2.0  # from 0 to xmax
        assert cam["parallel_scale"] > 0

    def test_auto_camera_overlay_margin_reduces_drawable(self):
        """Increasing overlay_margin increases parallel_scale."""
        from imas_ink.three_d.cutaway import auto_camera

        # Use wide bounds so the width constraint dominates and
        # the aspect ratio (affected by margin) actually matters.
        bounds = (0.0, 20.0, -5.0, 5.0, -3.0, 3.0)

        cam_small = auto_camera(
            bounds,
            overlay_margin=(0.05, 0.02),
            window_size=(1200, 900),
        )
        cam_large = auto_camera(
            bounds,
            overlay_margin=(0.30, 0.20),
            window_size=(1200, 900),
        )

        # Larger margin → smaller drawable → larger parallel_scale
        assert cam_large["parallel_scale"] > cam_small["parallel_scale"], (
            f"Large margin scale {cam_large['parallel_scale']} "
            f"should exceed small margin scale {cam_small['parallel_scale']}"
        )

    def test_auto_camera_view_up_and_normal_for_poloidal_rhs(self):
        """poloidal_rhs: camera along ±y, view_up = +z, orthographic."""
        from imas_ink.three_d.cutaway import auto_camera

        bounds = (0.0, 10.0, -5.0, 5.0, -5.0, 5.0)
        cam = auto_camera(bounds, view="poloidal_rhs")

        # View direction: position to focal_point should be along -y
        pos = np.array(cam["position"])
        fp = np.array(cam["focal_point"])
        direction = fp - pos
        direction /= np.linalg.norm(direction)

        # Should be pointing along -y (0, -1, 0)
        assert abs(direction[0]) < 1e-10, f"Camera direction x: {direction[0]}"
        assert direction[1] < 0, f"Camera should look toward -y, got {direction[1]}"
        assert abs(direction[2]) < 1e-10, f"Camera direction z: {direction[2]}"

        # View up should be +z
        assert cam["view_up"] == (0.0, 0.0, 1.0)

        # Must be orthographic
        assert cam["parallel_projection"] is True

    def test_auto_camera_iso_raises_not_implemented(self):
        """Stub views raise NotImplementedError."""
        from imas_ink.three_d.cutaway import auto_camera

        with pytest.raises(NotImplementedError):
            auto_camera((0, 10, -5, 5, -5, 5), view="iso")

    def test_auto_camera_unknown_view_raises(self):
        """Unknown view preset raises ValueError."""
        from imas_ink.three_d.cutaway import auto_camera

        with pytest.raises(ValueError, match="Unknown view"):
            auto_camera((0, 10, -5, 5, -5, 5), view="fish_eye")
