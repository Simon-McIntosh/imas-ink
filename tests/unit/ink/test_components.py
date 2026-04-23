"""Tests for efit.ink.components — visual component dataclasses."""

from __future__ import annotations

import numpy as np
import pytest

from imas_ink.components import (
    CoilRects,
    FluxContours,
    FluxLoops,
    MagneticProbes,
    OPointMarker,
    RadialProfile,
    ScatterPoints,
    Separatrix,
    TimeLabel,
    TimeSeries,
    WallOutline,
    XPointMarkers,
)
from imas_ink.style import DEFAULT_STYLE, InkStyle


class TestFluxContours:
    def test_construct(self):
        segs = [[np.array([[1, 2], [3, 4]])]]
        fc = FluxContours(segments=segs)
        assert len(fc.segments) == 1

    def test_default_style(self):
        fc = FluxContours(segments=[])
        assert fc.style is DEFAULT_STYLE

    def test_custom_style(self):
        custom = InkStyle(flux_color="#ff0000")
        fc = FluxContours(segments=[], style=custom)
        assert fc.style.flux_color == "#ff0000"

    def test_mutable(self):
        fc = FluxContours(segments=[])
        fc.segments = [[np.array([[0, 0]])]]
        assert len(fc.segments) == 1


class TestSeparatrix:
    def test_construct(self):
        segs = [np.array([[1, 2], [3, 4]])]
        sep = Separatrix(segments=segs)
        assert len(sep.segments) == 1
        assert len(sep.x_points) == 0

    def test_with_xpoints(self):
        sep = Separatrix(segments=[], x_points=[(5.5, -1.0)])
        assert len(sep.x_points) == 1

    def test_default_style(self):
        sep = Separatrix(segments=[])
        assert sep.style is DEFAULT_STYLE


class TestWallOutline:
    def test_construct(self):
        r = np.array([1.0, 2.0])
        z = np.array([0.0, 1.0])
        wall = WallOutline(wall_r=r, wall_z=z)
        assert len(wall.wall_r) == 2

    def test_default_style(self):
        wall = WallOutline(wall_r=np.array([]), wall_z=np.array([]))
        assert wall.style is DEFAULT_STYLE


class TestCoilRects:
    def test_construct(self):
        cr = CoilRects(rects=[])
        assert len(cr.rects) == 0

    def test_default_style(self):
        cr = CoilRects(rects=[])
        assert cr.style is DEFAULT_STYLE


class TestOPointMarker:
    def test_construct(self):
        m = OPointMarker(r=6.0, z=0.0)
        assert m.r == 6.0
        assert m.z == 0.0

    def test_default_style(self):
        m = OPointMarker(r=0.0, z=0.0)
        assert m.style is DEFAULT_STYLE


class TestXPointMarkers:
    def test_construct(self):
        xm = XPointMarkers(points=[(5.5, -1.0), (5.5, 1.0)])
        assert len(xm.points) == 2

    def test_default_style(self):
        xm = XPointMarkers(points=[])
        assert xm.style is DEFAULT_STYLE


class TestTimeLabel:
    def test_construct(self):
        tl = TimeLabel(time=0.5)
        assert tl.time == 0.5
        assert tl.converged is True

    def test_not_converged(self):
        tl = TimeLabel(time=1.0, converged=False)
        assert tl.converged is False

    def test_default_style(self):
        tl = TimeLabel(time=0.0)
        assert tl.style is DEFAULT_STYLE


class TestTimeSeries:
    def test_construct(self):
        t = np.array([0.0, 0.1, 0.2])
        v = np.array([1.0, 2.0, 3.0])
        ts = TimeSeries(time=t, values=v, label="Ip", ylabel="Ip", units="MA")
        assert ts.label == "Ip"
        assert ts.units == "MA"

    def test_default_style(self):
        ts = TimeSeries(time=np.array([]), values=np.array([]))
        assert ts.style is DEFAULT_STYLE


class TestRadialProfile:
    def test_construct(self):
        psi = np.linspace(0, 1, 10)
        vals = np.zeros(10)
        rp = RadialProfile(psi_norm=psi, values=vals, ylabel="p", units="kPa")
        assert rp.ylabel == "p"

    def test_default_style(self):
        rp = RadialProfile(psi_norm=np.array([]), values=np.array([]))
        assert rp.style is DEFAULT_STYLE


class TestScatterPoints:
    def test_construct(self):
        sp = ScatterPoints(
            x=np.array([1, 2]),
            y=np.array([3, 4]),
            xlabel="R",
            ylabel="Z",
        )
        assert sp.xlabel == "R"

    def test_default_style(self):
        sp = ScatterPoints(x=np.array([]), y=np.array([]))
        assert sp.style is DEFAULT_STYLE


class TestAllComponentsMutable:
    """All components should NOT be frozen (mutable dataclasses)."""

    @pytest.mark.parametrize(
        "cls, kwargs",
        [
            (FluxContours, {"segments": []}),
            (Separatrix, {"segments": []}),
            (WallOutline, {"wall_r": np.array([]), "wall_z": np.array([])}),
            (CoilRects, {"rects": []}),
            (
                MagneticProbes,
                {"positions_r": np.array([]), "positions_z": np.array([]), "angles": np.array([])},
            ),
            (FluxLoops, {"positions_r": np.array([]), "positions_z": np.array([])}),
            (OPointMarker, {"r": 0.0, "z": 0.0}),
            (XPointMarkers, {"points": []}),
            (TimeLabel, {"time": 0.0}),
            (TimeSeries, {"time": np.array([]), "values": np.array([])}),
            (RadialProfile, {"psi_norm": np.array([]), "values": np.array([])}),
            (ScatterPoints, {"x": np.array([]), "y": np.array([])}),
        ],
    )
    def test_not_frozen(self, cls, kwargs):
        """Components are mutable — style can be reassigned."""
        obj = cls(**kwargs)
        custom = InkStyle(flux_color="#123456")
        obj.style = custom  # should not raise
        assert obj.style.flux_color == "#123456"
