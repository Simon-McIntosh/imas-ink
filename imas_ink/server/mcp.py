"""MCP tool registration for imas-ink.

Provides high-level plotting tools as MCP server endpoints. Each tool
opens IMAS data, extracts what is needed, renders, and returns
base64-encoded output (PNG) or an HTML string.

Run the server::

    uv run imas-ink serve          # via console entry point
    python -m imas_ink.server.mcp  # via __main__

Tools registered
----------------
- ``plot_equilibrium``     — poloidal cross-section (single frame)
- ``plot_geometry``        — machine geometry only (wall, coils, probes, flux loops)
- ``plot_time_traces``     — Ip, beta_pol, li_3, q95 time traces
- ``plot_convergence``     — convergence status bar chart
- ``animate_pulse``        — full-pulse GIF animation
- ``plot_radial_profiles`` — 1D radial profiles (p, q, j_tor, …)
- ``plot_coilset_3d``      — 3D coilset + vessel render (requires [3d])
- ``repl``                 — stateful, namespaced Python REPL
"""

from __future__ import annotations

import base64


class InkServer:
    """Standalone MCP server for imas-ink plotting tools.

    Wraps a :class:`fastmcp.FastMCP` instance with all ink tools
    pre-registered. Use :meth:`run` to start the stdio transport.

    Parameters
    ----------
    name : str
        Server name (default ``"imas-ink"``). FastMCP uses this as a
        namespace prefix so tools appear as ``imas-ink-*``.

    Examples
    --------
    >>> server = InkServer()
    >>> list(server.tool_names())  # doctest: +NORMALIZE_WHITESPACE
    ['plot_equilibrium', 'plot_geometry', 'plot_time_traces',
     'plot_convergence', 'animate_pulse', 'plot_radial_profiles',
     'plot_coilset_3d', 'repl']
    """

    def __init__(self, name: str = "imas-ink") -> None:
        from fastmcp import FastMCP

        self._mcp = FastMCP(name)
        PlotProvider().register(self._mcp)
        _register_repl(self._mcp)

    @property
    def mcp(self):
        """The underlying :class:`~fastmcp.FastMCP` instance."""
        return self._mcp

    def tool_names(self) -> list[str]:
        """Return the names of all registered tools."""
        import asyncio

        tools = asyncio.run(self._mcp.list_tools())
        return [t.name for t in tools]

    def run(self) -> None:
        """Start the MCP server (stdio transport)."""
        self._mcp.run()


def serve() -> None:
    """Entry point for ``imas-ink serve``."""
    InkServer().run()


def _open_entry(uri: str):
    """Open an IMAS DBEntry, letting imas-python auto-detect the DD version.

    Parameters
    ----------
    uri : str
        IMAS URI, e.g. ``"imas:hdf5?path=/path/to/data/"``.

    Returns
    -------
    imas.DBEntry
        Open entry in read mode. Caller is responsible for closing it.
    """
    import imas

    return imas.DBEntry(uri, "r")


class PlotProvider:
    """Registers imas-ink plotting tools with a FastMCP server.

    Each public method becomes an MCP tool. Methods are ``async`` so
    the server can handle concurrent requests without blocking.

    Examples
    --------
    >>> from fastmcp import FastMCP
    >>> mcp = FastMCP("test")
    >>> PlotProvider().register(mcp)
    """

    def register(self, mcp) -> None:
        """Register all plotting tools on *mcp*."""
        mcp.tool()(self.plot_equilibrium)
        mcp.tool()(self.plot_geometry)
        mcp.tool()(self.plot_time_traces)
        mcp.tool()(self.plot_convergence)
        mcp.tool()(self.animate_pulse)
        mcp.tool()(self.plot_radial_profiles)
        mcp.tool()(self.plot_coilset_3d)

    async def plot_equilibrium(
        self,
        uri: str,
        time_index: int = 0,
        n_levels: int = 6,
        show_xpoints: bool = True,
        backend: str = "mpl",
    ) -> str:
        """Render a single-frame poloidal cross-section.

        Parameters
        ----------
        uri : str
            IMAS URI, e.g. ``"imas:hdf5?path=output/ITER/run1/data/"``
        time_index : int
            Time slice index.
        n_levels : int
            Number of interior flux surface contours.
        show_xpoints : bool
            Whether to show X-point markers.
        backend : str
            ``"mpl"`` for PNG, ``"alt"`` for HTML.

        Returns
        -------
        str
            Base64-encoded PNG (mpl) or HTML string (alt).
        """
        import matplotlib

        matplotlib.use("Agg")

        from ..extract import extract_geometry, extract_slice
        from ..figures import equilibrium_chart_alt, equilibrium_figure_mpl
        from ..io import render_to_bytes
        from ..style import DEFAULT_STYLE, InkStyle

        entry = _open_entry(uri)
        try:
            eq = entry.get("equilibrium")
            wall = entry.get("wall")
            pf = entry.get("pf_active")
            magnetics = entry.get("magnetics")
        finally:
            entry.close()

        sl = extract_slice(eq, time_index)
        geom = extract_geometry(wall, pf, magnetics)
        style = (
            InkStyle(flux_n_levels=n_levels) if n_levels != DEFAULT_STYLE.flux_n_levels else None
        )

        if backend == "alt":
            chart = equilibrium_chart_alt(sl, geom, style=style)
            return chart.to_html()
        else:
            fig, _ax = equilibrium_figure_mpl(sl, geom, style=style)
            png_bytes = render_to_bytes(fig)
            return base64.b64encode(png_bytes).decode("ascii")

    async def plot_geometry(
        self,
        uri: str,
        show_probes: bool = True,
        show_flux_loops: bool = True,
    ) -> str:
        """Render machine geometry (wall, coils, probes, flux loops).

        No equilibrium data required — useful for validating sensor
        positions and machine layout before running a solver.

        Parameters
        ----------
        uri : str
            IMAS URI pointing to machine description data containing
            ``wall``, ``pf_active``, and optionally ``magnetics`` IDSs.
        show_probes : bool
            Whether to show magnetic probe markers and orientation ticks.
        show_flux_loops : bool
            Whether to show flux loop position markers.

        Returns
        -------
        str
            Base64-encoded PNG.
        """
        import matplotlib

        matplotlib.use("Agg")

        from ..extract import extract_geometry
        from ..figures import geometry_figure_mpl
        from ..io import render_to_bytes

        entry = _open_entry(uri)
        try:
            wall = entry.get("wall")
            pf = entry.get("pf_active")
            magnetics = entry.get("magnetics")
        finally:
            entry.close()

        geom = extract_geometry(wall, pf, magnetics)
        fig, _ax = geometry_figure_mpl(
            geom, show_probes=show_probes, show_flux_loops=show_flux_loops,
        )
        png_bytes = render_to_bytes(fig)
        return base64.b64encode(png_bytes).decode("ascii")

    async def plot_time_traces(
        self,
        uri: str,
        quantities: list[str] | None = None,
        backend: str = "mpl",
    ) -> str:
        """Plot Ip, beta_pol, li_3, q95 time traces.

        Parameters
        ----------
        uri : str
            IMAS URI.
        quantities : list[str], optional
            Subset of ``["ip", "beta_pol", "li_3", "q95"]``. Default: all four.
        backend : str
            ``"mpl"`` for PNG, ``"alt"`` for HTML.

        Returns
        -------
        str
            Base64-encoded PNG or HTML string.
        """
        import matplotlib

        matplotlib.use("Agg")

        from ..alt import render_alt
        from ..components import TimeSeries
        from ..extract import extract_time_traces
        from ..figures import time_trace_figure_mpl
        from ..io import render_to_bytes

        entry = _open_entry(uri)
        try:
            eq = entry.get("equilibrium")
        finally:
            entry.close()

        tt = extract_time_traces(eq)

        available: dict[str, tuple[str, str, str, float]] = {
            "ip": ("ip", "Ip", "MA", 1e-6),
            "beta_pol": ("beta_pol", "βp", "", 1.0),
            "li_3": ("li_3", "li", "", 1.0),
            "q95": ("q95", "q95", "", 1.0),
        }
        if quantities is None:
            quantities = list(available.keys())

        traces = []
        for q in quantities:
            if q not in available:
                continue
            attr, ylabel, units, scale = available[q]
            values = getattr(tt, attr) * scale
            traces.append(TimeSeries(tt.time, values, label=ylabel, ylabel=ylabel, units=units))

        if backend == "alt":
            import altair as alt

            charts = [render_alt(ts) for ts in traces]
            combined = alt.vconcat(*charts) if len(charts) > 1 else charts[0]
            return combined.to_html()
        else:
            fig, _axes = time_trace_figure_mpl(traces)
            return base64.b64encode(render_to_bytes(fig)).decode("ascii")

    async def plot_convergence(self, uri: str) -> str:
        """Render convergence status bar chart.

        Returns base64-encoded PNG showing converged (green) vs
        non-converged (red) time slices.

        Parameters
        ----------
        uri : str
            IMAS URI.

        Returns
        -------
        str
            Base64-encoded PNG.
        """
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np

        from ..extract import extract_time_traces
        from ..io import render_to_bytes

        entry = _open_entry(uri)
        try:
            eq = entry.get("equilibrium")
        finally:
            entry.close()

        tt = extract_time_traces(eq)
        colors = ["#22cc44" if c else "#cc2222" for c in tt.converged]

        fig, ax = plt.subplots(figsize=(10, 2))
        ax.bar(
            tt.time,
            np.ones_like(tt.time),
            color=colors,
            width=np.diff(tt.time, append=tt.time[-1] + 0.01).clip(min=0.001),
        )
        ax.set_xlabel("Time [s]")
        ax.set_yticks([])
        ax.set_title("Convergence status")
        fig.tight_layout()
        return base64.b64encode(render_to_bytes(fig)).decode("ascii")

    async def animate_pulse(
        self,
        uri: str,
        duration_s: float = 10.0,
        n_levels: int = 6,
    ) -> str:
        """Render a full-pulse GIF animation. Returns base64-encoded GIF.

        Parameters
        ----------
        uri : str
            IMAS URI.
        duration_s : float
            Total GIF duration in seconds.
        n_levels : int
            Number of interior flux surface contours.

        Returns
        -------
        str
            Base64-encoded GIF.
        """
        from ..animate import animate_pulse as _animate
        from ..extract import extract_geometry
        from ..style import DEFAULT_STYLE, InkStyle

        entry = _open_entry(uri)
        try:
            eq = entry.get("equilibrium")
            wall = entry.get("wall")
            pf = entry.get("pf_active")
            magnetics = entry.get("magnetics")
        finally:
            entry.close()

        geom = extract_geometry(wall, pf, magnetics)
        style = (
            InkStyle(flux_n_levels=n_levels) if n_levels != DEFAULT_STYLE.flux_n_levels else None
        )
        gif_bytes = _animate(eq, geom, style=style, duration_s=duration_s)
        return base64.b64encode(gif_bytes).decode("ascii")

    async def plot_radial_profiles(
        self,
        uri: str,
        time_index: int = 0,
        quantities: list[str] | None = None,
        backend: str = "mpl",
    ) -> str:
        """Plot 1D radial profiles (p, q, j_tor, pprime, ffprime).

        Parameters
        ----------
        uri : str
            IMAS URI.
        time_index : int
            Time slice index.
        quantities : list[str], optional
            Subset of ``["pressure", "q", "j_tor", "pprime", "ffprime"]``.
            Default: all.
        backend : str
            ``"mpl"`` or ``"alt"``.

        Returns
        -------
        str
            Base64-encoded PNG or HTML.
        """
        import matplotlib

        matplotlib.use("Agg")

        from ..alt import render_alt
        from ..components import RadialProfile
        from ..extract import extract_profiles_1d
        from ..figures import radial_profile_figure_mpl
        from ..io import render_to_bytes

        entry = _open_entry(uri)
        try:
            eq = entry.get("equilibrium")
        finally:
            entry.close()

        rp = extract_profiles_1d(eq, time_index)

        available: dict[str, tuple[str, str, str]] = {
            "pressure": ("pressure", "p", "Pa"),
            "q": ("q", "q", ""),
            "j_tor": ("j_tor", "j_tor", "A/m²"),
            "pprime": ("pprime", "p'", "Pa/Wb"),
            "ffprime": ("ffprime", "FF'", "T²m²/Wb"),
        }
        if quantities is None:
            quantities = list(available.keys())

        profiles = []
        for q in quantities:
            if q not in available:
                continue
            attr, ylabel, units = available[q]
            values = getattr(rp, attr)
            profiles.append(
                RadialProfile(rp.psi_norm, values, label=ylabel, ylabel=ylabel, units=units)
            )

        if backend == "alt":
            import altair as alt

            charts = [render_alt(p) for p in profiles]
            combined = alt.vconcat(*charts) if len(charts) > 1 else charts[0]
            return combined.to_html()
        else:
            fig, _axes = radial_profile_figure_mpl(profiles)
            return base64.b64encode(render_to_bytes(fig)).decode("ascii")

    async def plot_coilset_3d(
        self,
        uri: str,
        outfile: str,
        show_wall: bool = True,
        show_pf: bool = True,
        show_tf: bool = True,
        view: str = "iso",
    ) -> str:
        """Render the 3D coilset for an IMAS dataset to a PNG file on disk.

        Requires the ``[3d]`` extra (pyvista, vtk). If not installed,
        raises :class:`RuntimeError` with the install command.

        The rendered PNG is written directly to *outfile* — no base64
        payload is returned over the wire.  For animations or chained
        renders, call ``imas-ink-repl`` and drive pyvista directly; the
        REPL exposes ``ink`` (``imas_ink``) so you can
        ``from imas_ink.three_d.scene import render_coilset``.

        Parameters
        ----------
        uri : str
            IMAS URI, e.g. ``"imas:hdf5?path=/path/to/machine/"``.
        outfile : str
            Absolute path to the PNG file to write.  Parent directories
            are created if needed.
        show_wall : bool
            Include vessel + first wall in the render.
        show_pf : bool
            Include PF / CS coils.
        show_tf : bool
            Include TF coils.
        view : str
            Camera preset: ``"iso"``, ``"iso_close"``, ``"poloidal"``,
            or ``"toroidal"``.

        Returns
        -------
        str
            Absolute path to the written PNG file.
        """
        try:
            from ..three_d.scene import render_coilset
        except ImportError as exc:
            raise RuntimeError(
                "The [3d] extra is required for 3D rendering. "
                'Install with: pip install "imas-ink[3d]"'
            ) from exc

        import pathlib

        outpath = pathlib.Path(outfile).expanduser().resolve()
        render_coilset(
            uri,
            outfile=outpath,
            show_wall=show_wall,
            show_pf=show_pf,
            show_tf=show_tf,
            view=view,
            off_screen=True,
        )
        return str(outpath)


def _register_repl(mcp) -> None:
    """Register the namespaced stateful REPL tool on *mcp*."""
    from .repl import repl as _repl

    repl_description = (
        "Execute Python in a persistent REPL with pre-imported imas-ink helpers.\n\n"
        "State (variables, imports, results) persists across calls within the same\n"
        "namespace for the life of the server process.\n\n"
        "Pre-loaded names:\n"
        "  ink  — imas_ink package\n"
        "  np   — numpy\n"
        "  plt  — matplotlib.pyplot (lazy, Agg backend)\n\n"
        "Parameters:\n"
        "  code (str): Python source. Multi-line supported. Use print() for output;\n"
        "      a trailing bare expression has its repr printed.\n"
        "  namespace (str): Isolated state bucket (default: 'default').\n"
        "  reset (bool): If True, wipe the namespace before executing code.\n\n"
        "Returns:\n"
        "  str: Captured stdout, repr of last expression if nothing printed,\n"
        "       or a traceback on error."
    )

    @mcp.tool(description=repl_description)
    def repl(code: str, namespace: str = "default", reset: bool = False) -> str:
        return _repl(code, namespace=namespace, reset=reset)
