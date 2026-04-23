"""Tests for IMAS → 3D mesh extraction using synthetic IDS data.

Uses ``types.SimpleNamespace`` to mock IMAS IDS structures, avoiding
any dependency on real IMAS HDF5 files or imas-python.
"""

from __future__ import annotations

import types

import numpy as np
import pytest


def _ns(**kwargs):
    """Shorthand for ``types.SimpleNamespace``."""
    return types.SimpleNamespace(**kwargs)


def _make_pf_active():
    """Build a synthetic pf_active IDS with 2 coils.

    Coil 0: "PF1" — rectangle geometry (r=6, z=3, dr=0.5, dz=0.8).
    Coil 1: "CS1" — outline geometry (hexagon at r=2, z=0).
    """
    # Rectangle coil
    rect = _ns(r=6.0, z=3.0, width=0.5, height=0.8)
    geom0 = _ns(geometry_type=2, rectangle=rect, outline=_ns(r=[], z=[]))
    elem0 = _ns(geometry=geom0)
    coil0 = _ns(name="PF1", element=[elem0])

    # Outline coil (hexagon)
    angles = np.linspace(0, 2 * np.pi, 7)[:-1]
    hex_r = 2.0 + 0.3 * np.cos(angles)
    hex_z = 0.0 + 0.3 * np.sin(angles)
    outline = _ns(r=hex_r, z=hex_z)
    geom1 = _ns(geometry_type=1, outline=outline, rectangle=_ns(r=0, z=0, width=0, height=0))
    elem1 = _ns(geometry=geom1)
    coil1 = _ns(name="CS1", element=[elem1])

    return _ns(coil=[coil0, coil1])


def _make_tf():
    """Build a synthetic tf IDS with one 8-point centerline coil."""
    # Simple circular-ish centerline in cylindrical coords
    theta = np.linspace(0, 2 * np.pi, 9)  # 8 segments + close
    r_cyl = 5.0 + 2.0 * np.cos(theta)
    phi_cyl = np.zeros_like(theta)  # all at phi=0
    z_cyl = 3.0 * np.sin(theta)

    start_pts = _ns(r=r_cyl[:-1], phi=phi_cyl[:-1], z=z_cyl[:-1])
    end_pts = _ns(r=r_cyl[1:], phi=phi_cyl[1:], z=z_cyl[1:])
    mid_pts = _ns(r=np.array([]), phi=np.array([]), z=np.array([]))

    elements = _ns(start_points=start_pts, intermediate_points=mid_pts, end_points=end_pts)
    cross_section = _ns(delta_r=0.4, delta_phi=0.3)
    conductor = _ns(elements=elements, cross_section=cross_section)
    coil = _ns(name="TF1", conductor=[conductor])

    return _ns(coil=[coil], coils_n=18)


def _make_wall():
    """Build a synthetic wall IDS with a simple annular vessel."""
    # Circular vessel centreline
    theta = np.linspace(0, 2 * np.pi, 20)
    r_cl = 6.0 + 1.5 * np.cos(theta)
    z_cl = 1.5 * np.sin(theta)

    centreline = _ns(r=r_cl, z=z_cl)
    annular = _ns(centreline=centreline, thickness=0.1)
    unit = _ns(annular=annular)
    vessel = _ns(unit=[unit])

    limiter_outline = _ns(r=r_cl, z=z_cl)
    limiter_unit = _ns(outline=limiter_outline)
    limiter = _ns(unit=[limiter_unit])

    desc = _ns(vessel=vessel, limiter=limiter)
    return _ns(description_2d=[desc])


@pytest.mark.render
class TestExtractPfCoils:
    def test_count(self):
        from imas_ink.three_d.coilset import extract_pf_coils

        pf = _make_pf_active()
        coils = extract_pf_coils(pf)
        assert len(coils) == 2

    def test_names(self):
        from imas_ink.three_d.coilset import extract_pf_coils

        pf = _make_pf_active()
        coils = extract_pf_coils(pf)
        names = {c.name for c in coils}
        assert "PF1" in names
        assert "CS1" in names

    def test_meshes_nonempty(self):
        from imas_ink.three_d.coilset import extract_pf_coils

        pf = _make_pf_active()
        for cm in extract_pf_coils(pf):
            assert cm.mesh.n_points > 0, f"{cm.name} has empty mesh"

    def test_metadata_fields(self):
        from imas_ink.three_d.coilset import extract_pf_coils

        pf = _make_pf_active()
        for cm in extract_pf_coils(pf):
            assert "coil_index" in cm.metadata
            assert "element_count" in cm.metadata
            assert "is_cs_segment" in cm.metadata

    def test_cs_detection(self):
        from imas_ink.three_d.coilset import extract_pf_coils

        pf = _make_pf_active()
        coils = extract_pf_coils(pf)
        cs_coils = [c for c in coils if c.metadata["is_cs_segment"]]
        pf_coils = [c for c in coils if not c.metadata["is_cs_segment"]]
        assert len(cs_coils) == 1
        assert len(pf_coils) == 1


@pytest.mark.render
class TestExtractTfCoils:
    def test_single_coil(self):
        from imas_ink.three_d.coilset import extract_tf_coils

        tf = _make_tf()
        # Override to just get the stored coil, not replicated
        tf.coils_n = 0
        coils = extract_tf_coils(tf, n_coils=1)
        assert len(coils) == 1
        assert coils[0].mesh.n_points > 0

    def test_replicated_coils(self):
        from imas_ink.three_d.coilset import extract_tf_coils

        tf = _make_tf()
        coils = extract_tf_coils(tf, n_coils=4)
        assert len(coils) == 4
        for cm in coils:
            assert cm.mesh.n_points > 0


@pytest.mark.render
class TestExtractWall:
    def test_annular_wall(self):
        from imas_ink.three_d.coilset import extract_wall

        wall = _make_wall()
        mesh = extract_wall(wall)
        assert mesh.n_points > 0

    def test_empty_wall(self):
        from imas_ink.three_d.coilset import extract_wall

        wall = _ns(description_2d=[])
        mesh = extract_wall(wall)
        assert mesh.n_points == 0
