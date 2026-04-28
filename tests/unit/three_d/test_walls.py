"""Tests for typed wall extraction, closure, revolution, and synthesis."""

from __future__ import annotations

import types

import numpy as np
import pytest


def _ns(**kwargs):
    """Shorthand for ``types.SimpleNamespace``."""
    return types.SimpleNamespace(**kwargs)


# ------------------------------------------------------------------
# close_or_reject_outline
# ------------------------------------------------------------------


class TestCloseOrRejectOutline:
    def test_already_closed(self):
        """Input where last == first: returned unchanged, was_closed=True."""
        from imas_ink.three_d.walls import close_or_reject_outline

        r = np.array([1.0, 2.0, 2.0, 1.0, 1.0])
        z = np.array([0.0, 0.0, 1.0, 1.0, 0.0])
        r_out, z_out, was_closed = close_or_reject_outline(r, z)

        assert was_closed is True
        np.testing.assert_array_equal(r_out, r)
        np.testing.assert_array_equal(z_out, z)

    def test_small_gap_closes(self):
        """Small gap appends first point, returns was_closed=True."""
        from imas_ink.three_d.walls import close_or_reject_outline

        r = np.array([1.0, 2.0, 2.0, 1.0])
        z = np.array([0.0, 0.0, 1.0, 1.0])
        # Gap = sqrt((1-1)^2 + (1-0)^2) = 1.0
        # bbox_diag = sqrt(1^2 + 1^2) ≈ 1.414
        # threshold = 1e-6 * 1.414 ≈ 1.4e-6
        # Gap of 1.0 >> threshold → should raise at default tol

        # Use a generous tolerance that makes the gap acceptable
        r_out, z_out, was_closed = close_or_reject_outline(r, z, tol=1.0)

        assert was_closed is True
        assert len(r_out) == len(r) + 1
        assert r_out[-1] == r[0]
        assert z_out[-1] == z[0]

    def test_large_gap_raises(self):
        """Gap larger than tolerance → ValueError."""
        from imas_ink.three_d.walls import close_or_reject_outline

        r = np.array([1.0, 2.0, 3.0])
        z = np.array([0.0, 0.0, 5.0])

        with pytest.raises(ValueError, match="exceeds tolerance"):
            close_or_reject_outline(r, z, tol=1e-6)


# ------------------------------------------------------------------
# revolve_wall_outline → manifold
# ------------------------------------------------------------------


class TestRevolveWallOutline:
    def test_revolve_synthetic_outline_manifold(self):
        """Rectangular RZ outline → revolved mesh is a closed manifold."""
        from imas_ink.three_d.walls import WallOutline2D, revolve_wall_outline

        # Small rectangle in RZ plane
        r = np.array([4.0, 5.0, 5.0, 4.0, 4.0])
        z = np.array([-0.5, -0.5, 0.5, 0.5, -0.5])
        outline = WallOutline2D(r=r, z=z, name="test_rect", is_closed=True)

        mesh = revolve_wall_outline(outline, n_theta=32)

        assert mesh.n_cells > 0
        boundary = mesh.extract_feature_edges(
            boundary_edges=True,
            feature_edges=False,
            manifold_edges=False,
            non_manifold_edges=False,
        )
        assert boundary.n_cells == 0, "Mesh has open edges — not a closed manifold"


# ------------------------------------------------------------------
# synthesize_vessel_shell
# ------------------------------------------------------------------


class TestSynthesizeVesselShell:
    def test_outward_offset(self):
        """Synthesised vessel encloses the first wall (bounding box check)."""
        from imas_ink.three_d.walls import FirstWall, synthesize_vessel_shell

        # Ellipse-ish first wall
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        r_fw = 6.0 + 1.5 * np.cos(theta)
        z_fw = 2.0 * np.sin(theta)
        fw = FirstWall(r=r_fw, z=z_fw, name="first_wall", is_closed=False)

        vessel = synthesize_vessel_shell(fw, offset=0.4)

        # Vessel bounding box must enclose the first-wall bounding box
        assert vessel.r.min() < r_fw.min()
        assert vessel.r.max() > r_fw.max()
        assert vessel.z.min() < z_fw.min()
        assert vessel.z.max() > z_fw.max()
        assert vessel.name == "synthetic_vessel"

    def test_custom_name_and_offset(self):
        """Custom name and offset are respected."""
        from imas_ink.three_d.walls import FirstWall, synthesize_vessel_shell

        theta = np.linspace(0, 2 * np.pi, 32, endpoint=False)
        r_fw = 5.0 + 1.0 * np.cos(theta)
        z_fw = 1.0 * np.sin(theta)
        fw = FirstWall(r=r_fw, z=z_fw, name="fw", is_closed=False)

        vessel = synthesize_vessel_shell(fw, offset=0.8, name="my_vessel")

        assert vessel.name == "my_vessel"
        # Larger offset → larger bounding box
        assert vessel.r.max() - vessel.r.min() > r_fw.max() - r_fw.min()


# ------------------------------------------------------------------
# IMAS extractors (synthetic IDS)
# ------------------------------------------------------------------


class TestExtractFirstWall:
    def test_extracts_limiter(self):
        """Extracts first-wall outline from synthetic wall IDS."""
        from imas_ink.three_d.walls import extract_first_wall

        theta = np.linspace(0, 2 * np.pi, 20)
        r = 6.0 + 1.5 * np.cos(theta)
        z = 1.5 * np.sin(theta)

        wall_ids = _ns(
            description_2d=[
                _ns(
                    limiter=_ns(unit=[_ns(outline=_ns(r=r, z=z))]),
                    vessel=_ns(unit=[]),
                )
            ]
        )

        fw = extract_first_wall(wall_ids)
        assert fw is not None
        assert fw.name == "first_wall"
        assert len(fw.r) == len(r)

    def test_returns_none_when_absent(self):
        """Returns None when no limiter data present."""
        from imas_ink.three_d.walls import extract_first_wall

        wall_ids = _ns(description_2d=[_ns(limiter=_ns(unit=[]))])
        assert extract_first_wall(wall_ids) is None

    def test_returns_none_for_empty_ids(self):
        """Returns None for an IDS with no description_2d."""
        from imas_ink.three_d.walls import extract_first_wall

        wall_ids = _ns(description_2d=[])
        assert extract_first_wall(wall_ids) is None


class TestExtractVesselShells:
    def test_extracts_annular_vessel(self):
        """Extracts vessel shells from synthetic wall IDS."""
        from imas_ink.three_d.walls import extract_vessel_shells

        theta = np.linspace(0, 2 * np.pi, 20)
        r_cl = 6.0 + 2.0 * np.cos(theta)
        z_cl = 2.0 * np.sin(theta)

        wall_ids = _ns(
            description_2d=[
                _ns(
                    vessel=_ns(
                        unit=[
                            _ns(
                                annular=_ns(
                                    centreline=_ns(r=r_cl, z=z_cl),
                                    thickness=0.1,
                                ),
                            ),
                        ]
                    ),
                    limiter=_ns(unit=[]),
                )
            ]
        )

        shells = extract_vessel_shells(wall_ids)
        assert len(shells) == 1
        assert shells[0].name == "vessel_0"
        # With thickness, the combined outline is inner + outer reversed
        assert len(shells[0].r) == 2 * len(r_cl)

    def test_returns_empty_when_no_vessel(self):
        """Returns empty list when no vessel data."""
        from imas_ink.three_d.walls import extract_vessel_shells

        wall_ids = _ns(description_2d=[_ns(limiter=_ns(unit=[]))])
        assert extract_vessel_shells(wall_ids) == []
