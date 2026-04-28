"""Tests for render_cutaway_with_flux — the Phase 7 composer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

pv = pytest.importorskip("pyvista", reason="pyvista not installed")


# ---------------------------------------------------------------------------
# VTK offscreen guard
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _offscreen():
    """Force pyvista off-screen rendering for CI compatibility."""
    pv.OFF_SCREEN = True


# ---------------------------------------------------------------------------
# Synthetic helpers
# ---------------------------------------------------------------------------


def _make_slice_2d():
    """Build a synthetic EquilibriumSlice2D."""
    from imas_ink.three_d.equilibrium import EquilibriumSlice2D

    R_1d = np.linspace(4.0, 8.5, 12)
    Z_1d = np.linspace(-4.5, 4.5, 12)
    R_2d, Z_2d = np.meshgrid(R_1d, Z_1d, indexing="ij")
    psi_2d = (R_2d - 6.2) ** 2 + Z_2d**2

    return EquilibriumSlice2D(
        time=1.0,
        psi_axis=0.0,
        psi_boundary=float(np.max(psi_2d)),
        R_1d=R_1d,
        Z_1d=Z_1d,
        psi_2d=psi_2d,
        boundary_r=np.empty(0),
        boundary_z=np.empty(0),
        o_point=(6.2, 0.0),
        x_points=(),
    )


def _make_capped_mesh(name="first_wall"):
    """Create a synthetic CappedMesh from a simple quad for testing."""
    from imas_ink.three_d.cutaway import CappedMesh

    # A simple planar quad (cap face) in the y=0 plane
    points = np.array(
        [[4.0, 0.0, -3.5], [8.5, 0.0, -3.5], [8.5, 0.0, 3.5], [4.0, 0.0, 3.5]],
        dtype=float,
    )
    faces = np.array([4, 0, 1, 2, 3])
    cap = pv.PolyData(points, faces)

    # For the full mesh, just triangulate the cap
    full = cap.triangulate()

    return CappedMesh(full=full, cap=cap, name=name)


def _make_first_wall():
    """Build a synthetic FirstWall dataclass."""
    from imas_ink.three_d.walls import FirstWall

    theta = np.linspace(0, 2 * np.pi, 40, endpoint=False)
    return FirstWall(
        name="first_wall",
        r=6.2 + 2.0 * np.cos(theta),
        z=0.0 + 3.5 * np.sin(theta),
        is_closed=True,
    )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_build_static():
    """Patch _build_cutaway_static to return synthetic geometry."""
    capped = _make_capped_mesh("first_wall")
    fw = _make_first_wall()

    def _fake_build(**kwargs):
        blocks = {"first_wall": capped}
        return blocks, fw

    with patch(
        "imas_ink.three_d.scene._build_cutaway_static",
        side_effect=_fake_build,
    ) as m:
        yield m


@pytest.fixture
def mock_imas_entry():
    """Patch imas.DBEntry and extract_slice_2d to avoid real I/O."""
    slice_2d = _make_slice_2d()

    mock_entry = MagicMock()
    mock_entry.get.return_value = MagicMock()  # eq_ids stub
    mock_entry.close = MagicMock()

    # Patch imas module at import site and extract_slice_2d at its origin
    fake_imas = MagicMock()
    fake_imas.DBEntry.return_value = mock_entry

    with patch.dict("sys.modules", {"imas": fake_imas}), patch(
        "imas_ink.three_d.equilibrium.extract_slice_2d",
        return_value=slice_2d,
    ):
        yield mock_entry


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestRenderCutawaySynthetic:
    def test_writes_png(self, mock_imas_entry, mock_build_static, tmp_path):
        """render_cutaway_with_flux writes a non-empty PNG."""
        from imas_ink.three_d.scene import render_cutaway_with_flux

        outfile = tmp_path / "cutaway.png"
        result = render_cutaway_with_flux(
            "imas:hdf5?path=synthetic/",
            outfile=outfile,
            show_pf=False,
            show_tf=False,
            show_vessel=False,
            n_levels=4,
            show_colorbar=False,
            show_title=False,
            window_size=(400, 300),
        )

        assert result == outfile
        assert outfile.exists()
        assert outfile.stat().st_size > 1024, "PNG should be > 1 KB"

    def test_field_only_mode(self, mock_imas_entry, mock_build_static, tmp_path):
        """flux_mode='field_only' renders without contour errors."""
        from imas_ink.three_d.scene import render_cutaway_with_flux

        outfile = tmp_path / "field_only.png"
        render_cutaway_with_flux(
            "imas:hdf5?path=synthetic/",
            outfile=outfile,
            flux_mode="field_only",
            show_pf=False,
            show_tf=False,
            show_vessel=False,
            n_levels=4,
            show_colorbar=False,
            show_title=False,
            window_size=(400, 300),
        )

        assert outfile.exists()
        assert outfile.stat().st_size > 1024

    def test_contours_only_mode(self, mock_imas_entry, mock_build_static, tmp_path):
        """flux_mode='contours_only' renders without field errors."""
        from imas_ink.three_d.scene import render_cutaway_with_flux

        outfile = tmp_path / "contours_only.png"
        render_cutaway_with_flux(
            "imas:hdf5?path=synthetic/",
            outfile=outfile,
            flux_mode="contours_only",
            show_pf=False,
            show_tf=False,
            show_vessel=False,
            n_levels=4,
            show_colorbar=False,
            show_title=False,
            window_size=(400, 300),
        )

        assert outfile.exists()
        assert outfile.stat().st_size > 1024

    def test_build_static_called_once(self, mock_imas_entry, mock_build_static, tmp_path):
        """_build_cutaway_static is called exactly once per render."""
        from imas_ink.three_d.scene import render_cutaway_with_flux

        outfile = tmp_path / "count.png"
        render_cutaway_with_flux(
            "imas:hdf5?path=synthetic/",
            outfile=outfile,
            show_pf=False,
            show_tf=False,
            show_vessel=False,
            n_levels=4,
            show_colorbar=False,
            show_title=False,
            window_size=(400, 300),
        )

        mock_build_static.assert_called_once()

    def test_no_wall_renders_without_overlay(self, mock_imas_entry, tmp_path):
        """When _build_cutaway_static returns no first_wall, overlay is skipped."""
        from imas_ink.three_d.scene import render_cutaway_with_flux

        def _no_wall_build(**kwargs):
            return {}, None

        with patch(
            "imas_ink.three_d.scene._build_cutaway_static",
            side_effect=_no_wall_build,
        ):
            outfile = tmp_path / "no_wall.png"
            render_cutaway_with_flux(
                "imas:hdf5?path=synthetic/",
                outfile=outfile,
                show_pf=False,
                show_tf=False,
                show_vessel=False,
                n_levels=4,
                show_colorbar=False,
                show_title=False,
                window_size=(400, 300),
            )

            assert outfile.exists()

