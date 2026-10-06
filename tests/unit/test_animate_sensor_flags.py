"""Sensor-visibility flags on the pulse animation.

``animate_pulse`` renders every frame with ``equilibrium_figure_mpl`` and must
forward that figure's ``show_probes`` and ``show_flux_loops`` unchanged, so a
caller can leave the magnetic probes and flux loops out of an equilibrium
animation. The MCP ``animate_pulse`` tool carries the same two flags and
forwards them to the Python entry point.
"""

from __future__ import annotations

import asyncio
import base64
import types

import numpy as np
import pytest

from imas_ink._types import MachineGeometry
from imas_ink.animate import animate_pulse


def _ns(**kwargs):
    return types.SimpleNamespace(**kwargs)


def _make_eq_ids(n_slices: int = 2):
    """A mock equilibrium IDS with *n_slices* time slices extract_slice can read."""
    n_r = n_z = 33
    r_grid = np.linspace(4.0, 8.0, n_r)
    z_grid = np.linspace(-2.0, 2.0, n_z)
    r_2d, z_2d = np.meshgrid(r_grid, z_grid, indexing="ij")
    r0, z0 = 6.0, 0.0
    psi_2d = 10.0 - ((r_2d - r0) ** 2 + (z_2d - z0) ** 2)
    theta = np.linspace(0.0, 2.0 * np.pi, 64, endpoint=False)

    slices = []
    for _i in range(n_slices):
        grid = _ns(dim1=r_grid, dim2=z_grid)
        p2d = _ns(grid=grid, psi=psi_2d)
        gq = _ns(
            psi_axis=10.0,
            psi_boundary=2.0,
            magnetic_axis=_ns(r=r0, z=z0),
            ip=1e6,
            beta_pol=0.8,
            li_3=1.2,
            q95=3.5,
        )
        boundary = _ns(outline=_ns(r=r0 + np.cos(theta), z=z0 + np.sin(theta)), x_point=[])
        slices.append(_ns(profiles_2d=[p2d], global_quantities=gq, boundary=boundary))

    return _ns(time_slice=slices, time=np.linspace(0.5, 0.6, n_slices))


def _geom() -> MachineGeometry:
    r = np.array([3.5, 8.5, 8.5, 3.5, 3.5])
    z = np.array([-4.5, -4.5, 4.5, 4.5, -4.5])
    return MachineGeometry(
        wall_r=r,
        wall_z=z,
        coil_rects=[],
        wall_clip_vertices=np.column_stack([r, z]),
        wall_units=[(r, z)],
        probe_r=np.array([4.0, 5.0]),
        probe_z=np.array([0.0, 0.0]),
        probe_angle=np.array([0.0, np.pi / 2]),
        flux_loop_r=np.array([4.5]),
        flux_loop_z=np.array([1.0]),
    )


@pytest.fixture
def frame_calls(monkeypatch):
    """Record every equilibrium_figure_mpl call's sensor flags, then draw for real."""
    from imas_ink import figures

    recorded: list[tuple[bool, bool]] = []
    real = figures.equilibrium_figure_mpl

    def spy(sl, geom, **kwargs):
        recorded.append((kwargs.get("show_probes", True), kwargs.get("show_flux_loops", True)))
        return real(sl, geom, **kwargs)

    monkeypatch.setattr(figures, "equilibrium_figure_mpl", spy)
    return recorded


class TestAnimatePulseForwardsFlags:
    def test_both_false_on_every_frame(self, frame_calls):
        """show_probes=False and show_flux_loops=False reach every frame."""
        gif = animate_pulse(_make_eq_ids(2), _geom(), show_probes=False, show_flux_loops=False)

        assert len(frame_calls) == 2, "one equilibrium_figure_mpl call per time slice"
        assert all(
            probes is False and loops is False for probes, loops in frame_calls
        ), f"not every frame drew with both flags false: {frame_calls}"
        assert gif[:6] in (b"GIF87a", b"GIF89a"), "output is not a GIF"

    def test_default_call_draws_them(self, frame_calls):
        """The default call still draws probes and flux loops."""
        animate_pulse(_make_eq_ids(2), _geom())

        assert len(frame_calls) == 2
        assert all(
            probes is True and loops is True for probes, loops in frame_calls
        ), f"default did not draw both sensor families: {frame_calls}"


class TestMcpToolForwardsFlags:
    def test_mcp_animate_pulse_forwards_both_flags(self, monkeypatch):
        """The MCP tool passes show_probes and show_flux_loops to animate_pulse."""
        import imas_ink.animate as animate_mod
        import imas_ink.extract as extract_mod
        import imas_ink.server.mcp as mcp_mod

        sentinel_geom = object()
        captured: dict = {}

        class _FakeEntry:
            def get(self, _name):
                return None

            def close(self):
                pass

        def fake_animate(eq, geom, **kwargs):
            captured.update(kwargs)
            captured["geom"] = geom
            return b"GIF89a-fake"

        monkeypatch.setattr(mcp_mod, "_open_entry", lambda uri: _FakeEntry())
        monkeypatch.setattr(extract_mod, "extract_geometry", lambda *a, **k: sentinel_geom)
        monkeypatch.setattr(animate_mod, "animate_pulse", fake_animate)

        provider = mcp_mod.PlotProvider()
        out = asyncio.run(
            provider.animate_pulse(
                "imas:hdf5?path=fake", show_probes=False, show_flux_loops=False
            )
        )

        assert captured["geom"] is sentinel_geom
        assert captured.get("show_probes") is False
        assert captured.get("show_flux_loops") is False
        assert base64.b64decode(out) == b"GIF89a-fake"
