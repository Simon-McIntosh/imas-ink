"""TDD red-green tests for multi-unit wall extraction in extract_geometry.

Tests:
  1. Synthetic wall IDS with N>1 units → extract_geometry returns all N outlines
     (fails today: only unit[0] is read).
  2. WEST-like 2-description fixture → instance selection picks desc[0]
     (typed beats untyped — mirrors selectDesc2d scoring).
  3. Mobile-outline nearest-in-time selection (WEST desc[1] has mobile units).
  4. Overlay test: figure carries the containment annotation artists.
"""

from __future__ import annotations

import types

import numpy as np
import pytest

from imas_ink.extract import extract_geometry
from imas_ink._types import MachineGeometry


# ---------------------------------------------------------------------------
# Mock helpers
# ---------------------------------------------------------------------------
def _ns(**kw):
    return types.SimpleNamespace(**kw)


def _make_unit(r, z):
    """Return a wall limiter unit with outline.r/z."""
    return _ns(outline=_ns(r=np.asarray(r, dtype=float),
                           z=np.asarray(z, dtype=float)))


def _make_pf_ids(n_coils=1):
    """Minimal pf_active IDS with no coils (tests don't need them)."""
    coils = []
    for i in range(n_coils):
        elem = _ns(geometry=_ns(outline=_ns(r=np.array([4.0, 4.5, 4.5, 4.0]),
                                            z=np.array([0.0, 0.0, 0.5, 0.5])),
                                rectangle=_ns(r=4.2, z=0.25, width=0.5, height=0.5)))
        coils.append(_ns(element=[elem], name=f"PF{i+1}"))
    return _ns(coil=coils)


def _make_wall_single_desc_n_units(n_units=3):
    """Wall IDS with a single description_2d containing N limiter units."""
    r_base = np.linspace(2.0, 3.0, 10)
    z_base = np.linspace(-1.0, 1.0, 10)
    units = []
    for i in range(n_units):
        offset = i * 0.5
        units.append(_make_unit(r_base + offset, z_base))
    limiter = _ns(unit=units, type=_ns(index=-999999999))
    desc = _ns(limiter=limiter)
    return _ns(description_2d=[desc])


def _make_wall_two_descs_typed_first():
    """Wall IDS with 2 description_2d entries.

    desc[0] has type.index=1 (typed, 1 unit) — should win.
    desc[1] has type.index=SENTINEL (untyped, 2 units).
    """
    unit0 = _make_unit([2.0, 3.0, 3.0, 2.0], [0.0, 0.0, 1.0, 1.0])
    unit1a = _make_unit([1.5, 4.0, 4.0, 1.5], [-0.5, -0.5, 1.5, 1.5])
    unit1b = _make_unit([1.8, 3.5, 3.5, 1.8], [-0.3, -0.3, 1.2, 1.2])

    limiter0 = _ns(unit=[unit0], type=_ns(index=1))   # typed wins
    limiter1 = _ns(unit=[unit1a, unit1b], type=_ns(index=-999999999))  # untyped

    desc0 = _ns(limiter=limiter0)
    desc1 = _ns(limiter=limiter1)
    return _ns(description_2d=[desc0, desc1])


def _make_wall_two_descs_untyped_wins_by_units():
    """Wall IDS where desc[0] is typed=0, desc[1] is typed=0 but 3 units.

    Both have same effective type → tie broken by unit count → desc[1] wins.
    """
    units_a = [_make_unit([2.0, 3.0], [0.0, 1.0])]
    units_b = [
        _make_unit([2.0, 3.0], [0.0, 1.0]),
        _make_unit([2.5, 3.5], [0.5, 1.5]),
        _make_unit([3.0, 4.0], [1.0, 2.0]),
    ]
    limiter_a = _ns(unit=units_a, type=_ns(index=0))
    limiter_b = _ns(unit=units_b, type=_ns(index=0))
    desc_a = _ns(limiter=limiter_a)
    desc_b = _ns(limiter=limiter_b)
    return _ns(description_2d=[desc_a, desc_b])


# ---------------------------------------------------------------------------
# Test 1: multi-unit extraction — wall_units contains all N units
# ---------------------------------------------------------------------------
class TestMultiUnitExtraction:
    """extract_geometry returns all limiter units of the selected description_2d."""

    def test_n1_unit_backward_compat(self):
        """Single unit: wall_r/wall_z preserved + wall_units has 1 entry."""
        wall = _make_wall_single_desc_n_units(n_units=1)
        pf = _make_pf_ids()
        geom = extract_geometry(wall, pf)
        assert isinstance(geom, MachineGeometry)
        # backward compat: wall_r/wall_z non-empty
        assert len(geom.wall_r) > 0
        assert len(geom.wall_z) > 0
        # new: wall_units has exactly 1 entry
        assert hasattr(geom, "wall_units"), "MachineGeometry must have wall_units field"
        assert len(geom.wall_units) == 1

    def test_n3_units_all_returned(self):
        """Three units: wall_units has 3 (r, z) pairs."""
        wall = _make_wall_single_desc_n_units(n_units=3)
        pf = _make_pf_ids()
        geom = extract_geometry(wall, pf)
        assert hasattr(geom, "wall_units"), "MachineGeometry must have wall_units field"
        assert len(geom.wall_units) == 3, (
            f"Expected 3 wall units, got {len(geom.wall_units)}"
        )

    def test_wall_units_are_array_pairs(self):
        """Each entry in wall_units is an (r_array, z_array) pair."""
        wall = _make_wall_single_desc_n_units(n_units=2)
        pf = _make_pf_ids()
        geom = extract_geometry(wall, pf)
        for r_arr, z_arr in geom.wall_units:
            assert isinstance(r_arr, np.ndarray)
            assert isinstance(z_arr, np.ndarray)
            assert len(r_arr) == len(z_arr)
            assert len(r_arr) > 0

    def test_wall_r_z_compat_equals_first_unit(self):
        """wall_r/wall_z must equal the first unit in wall_units for backward compat."""
        wall = _make_wall_single_desc_n_units(n_units=2)
        pf = _make_pf_ids()
        geom = extract_geometry(wall, pf)
        r0, z0 = geom.wall_units[0]
        np.testing.assert_array_equal(geom.wall_r, r0)
        np.testing.assert_array_equal(geom.wall_z, z0)


# ---------------------------------------------------------------------------
# Test 2: description_2d instance selection — typed beats untyped
# ---------------------------------------------------------------------------
class TestDescriptionSelection:
    """select_description_2d logic: typed instance beats untyped."""

    def test_typed_first_wins(self):
        """When desc[0] is typed (index=1) and desc[1] is untyped (SENTINEL),
        desc[0] should be selected even though it has fewer units.
        """
        wall = _make_wall_two_descs_typed_first()
        pf = _make_pf_ids()
        geom = extract_geometry(wall, pf)
        # desc[0] has 1 unit; desc[1] has 2 units
        # typed desc[0] wins → 1 unit expected
        assert len(geom.wall_units) == 1, (
            f"Expected typed desc[0] (1 unit) to win, got {len(geom.wall_units)}"
        )

    def test_tie_broken_by_unit_count(self):
        """Two descriptions with same effective type → more units wins."""
        wall = _make_wall_two_descs_untyped_wins_by_units()
        pf = _make_pf_ids()
        geom = extract_geometry(wall, pf)
        # Both type=0, desc[1] has 3 units → desc[1] wins
        assert len(geom.wall_units) == 3, (
            f"Expected tie-broken desc[1] (3 units), got {len(geom.wall_units)}"
        )


# ---------------------------------------------------------------------------
# Test 3: mobile-outline nearest-in-time selection
# ---------------------------------------------------------------------------
def _make_wall_with_mobile(times, r_arr, z_arr, static_r, static_z):
    """Wall IDS with desc[0]=typed(static+mobile) matching real DD structure.

    Real DD structure::
        description_2d.mobile.unit[i].outline[j].r/z/time

    times: sequence of mobile outline times
    r_arr, z_arr: shape (N_times, N_pts) arrays for each mobile snapshot
    static_r, static_z: coordinates of the single static limiter unit
    """
    # desc[0]: 1 static limiter unit + 1 mobile PFC unit with N time-snapshots
    # type=2 — ensures it wins over any untyped desc
    static_unit = _make_unit(static_r, static_z)

    # Build mobile.unit[0].outline[j].r/z/time for each time snapshot
    outlines = []
    for i, t in enumerate(times):
        ol = _ns(
            r=np.asarray(r_arr[i], dtype=float),
            z=np.asarray(z_arr[i], dtype=float),
            time=float(t),
        )
        outlines.append(ol)
    mobile_unit0 = _ns(outline=outlines)
    mobile = _ns(unit=[mobile_unit0])

    limiter0 = _ns(unit=[static_unit], type=_ns(index=2))
    desc0 = _ns(limiter=limiter0, mobile=mobile)

    return _ns(description_2d=[desc0])


class TestMobileOutlineSelection:
    """Mobile outline nearest-in-time selection."""

    def test_mobile_nearest_time_selected(self):
        """When desc has .mobile with time-tagged units, nearest time is picked."""
        times = [10.0, 20.0, 30.0]
        r_arr = [
            np.array([2.0, 3.0, 3.0, 2.0]),
            np.array([2.1, 3.1, 3.1, 2.1]),
            np.array([2.2, 3.2, 3.2, 2.2]),
        ]
        z_arr = [
            np.array([0.0, 0.0, 1.0, 1.0]),
            np.array([0.1, 0.1, 1.1, 1.1]),
            np.array([0.2, 0.2, 1.2, 1.2]),
        ]
        static_r = np.array([1.0, 4.0, 4.0, 1.0])
        static_z = np.array([-1.0, -1.0, 2.0, 2.0])
        wall = _make_wall_with_mobile(times, r_arr, z_arr, static_r, static_z)
        pf = _make_pf_ids()

        # desc[1] type=2 beats desc[0] type=1 → selected
        # request time nearest to 22.0 → should pick t=20.0 (index 1)
        geom = extract_geometry(wall, pf, time=22.0)
        assert hasattr(geom, "wall_units")
        # desc[1] is selected (type=2 > type=1)
        # nearest to t=22.0 is t=20.0 → r_arr[1] = [2.1, 3.1, 3.1, 2.1]
        assert len(geom.wall_units) >= 1
        r0, z0 = geom.wall_units[0]
        np.testing.assert_allclose(r0, r_arr[1], atol=1e-10)

    def test_no_time_returns_limiter_units(self):
        """When time=None (default), limiter units are returned (mobile not filtered)."""
        times = [10.0, 20.0]
        r_arr = [np.array([2.0, 3.0]), np.array([2.1, 3.1])]
        z_arr = [np.array([0.0, 1.0]), np.array([0.1, 1.1])]
        static_r = np.array([1.0, 4.0])
        static_z = np.array([-1.0, 2.0])
        wall = _make_wall_with_mobile(times, r_arr, z_arr, static_r, static_z)
        pf = _make_pf_ids()
        geom = extract_geometry(wall, pf)  # no time — fallback to limiter units
        # desc[0] is the only desc (type=2), limiter has 1 static unit
        assert len(geom.wall_units) == 1


# ---------------------------------------------------------------------------
# Test 4: figure carries containment annotation artists
# ---------------------------------------------------------------------------
class TestContainmentAnnotationInFigure:
    """equilibrium_figure_mpl with containment_result renders annotation text."""

    def _make_minimal_eq_slice(self):
        """Return a minimal EquilibriumSlice-like object via extract_slice."""
        from imas_ink.extract import extract_slice
        import types

        n_r, n_z = 33, 33
        r_grid = np.linspace(4.0, 8.0, n_r)
        z_grid = np.linspace(-2.0, 2.0, n_z)
        r_2d, z_2d = np.meshgrid(r_grid, z_grid, indexing="ij")
        r0, z0 = 6.0, 0.0
        psi_2d = 10.0 - ((r_2d - r0) ** 2 + (z_2d - z0) ** 2)
        p2d = types.SimpleNamespace(
            grid=types.SimpleNamespace(dim1=r_grid, dim2=z_grid), psi=psi_2d
        )
        theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
        br = r0 + 0.5 * np.cos(theta)
        bz = z0 + 0.5 * np.sin(theta)
        gq = types.SimpleNamespace(
            psi_axis=10.0, psi_boundary=9.0,
            magnetic_axis=types.SimpleNamespace(r=r0, z=z0),
            ip=1e6, beta_pol=0.5, li_3=1.0, q95=3.5,
        )
        boundary = types.SimpleNamespace(
            outline=types.SimpleNamespace(r=br, z=bz), x_point=[]
        )
        ts = types.SimpleNamespace(
            profiles_2d=[p2d], global_quantities=gq, boundary=boundary
        )
        eq = types.SimpleNamespace(time_slice=[ts], time=np.array([0.5]))
        return extract_slice(eq, 0)

    def _make_minimal_geom(self):
        """Return a minimal MachineGeometry with 2 wall units."""
        wall = _make_wall_single_desc_n_units(n_units=2)
        pf = _make_pf_ids()
        return extract_geometry(wall, pf)

    def test_figure_rendered_without_error(self):
        """equilibrium_figure_mpl with containment_result must not raise."""
        import matplotlib
        matplotlib.use("Agg")
        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_minimal_eq_slice()
        geom = self._make_minimal_geom()
        containment = {
            "lcfs_outside": 5,
            "frac": 1.2,
            "boundary_type": 2,
            "n_xpoints": 1,
            "psi_axis": 10.0,
            "psi_limiter": 9.5,
            "psi_boundary": 9.0,
        }
        # Should not raise:
        fig, ax = equilibrium_figure_mpl(sl, geom, containment_result=containment)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_containment_text_in_figure_when_outside(self):
        """When lcfs_outside>0, annotation text must appear in the axes."""
        import matplotlib
        matplotlib.use("Agg")
        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_minimal_eq_slice()
        geom = self._make_minimal_geom()
        containment = {
            "lcfs_outside": 7,
            "frac": 1.3,
            "boundary_type": 2,
            "n_xpoints": 1,
            "psi_axis": 10.0,
            "psi_limiter": 9.5,
            "psi_boundary": 9.0,
        }
        fig, ax = equilibrium_figure_mpl(sl, geom, containment_result=containment)
        # Collect all text strings from ax
        texts = [t.get_text() for t in ax.texts]
        combined = " ".join(texts)
        assert "lcfs_outside=7" in combined, (
            f"Expected 'lcfs_outside=7' in annotation text; got: {combined!r}"
        )
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_no_annotation_when_containment_none(self):
        """When containment_result=None, figure renders without annotation."""
        import matplotlib
        matplotlib.use("Agg")
        from imas_ink.figures import equilibrium_figure_mpl

        sl = self._make_minimal_eq_slice()
        geom = self._make_minimal_geom()
        # Default: no containment_result → backward compat
        fig, ax = equilibrium_figure_mpl(sl, geom)
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_all_wall_units_rendered(self):
        """geometry_figure_mpl renders all wall units in geom.wall_units."""
        import matplotlib
        matplotlib.use("Agg")
        from imas_ink.figures import geometry_figure_mpl

        geom = self._make_minimal_geom()
        assert len(geom.wall_units) == 2, "Fixture must have 2 units"
        fig, ax = geometry_figure_mpl(geom)
        # Count lines in axes — each unit should add at least 1 line
        n_lines = len(ax.lines)
        assert n_lines >= 2, (
            f"Expected at least 2 wall outlines plotted, got {n_lines} lines"
        )
        import matplotlib.pyplot as plt
        plt.close(fig)
