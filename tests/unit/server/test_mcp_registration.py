"""Unit tests for MCP server tool registration.

Imports the server object, introspects registered tool names, and
verifies the server name — without starting the stdio transport.
"""

from __future__ import annotations

import pytest

EXPECTED_TOOLS = {
    "plot_equilibrium",
    "plot_time_traces",
    "plot_convergence",
    "animate_pulse",
    "plot_radial_profiles",
    "repl",
}


@pytest.fixture(scope="module")
def ink_server():
    """Instantiate the InkServer without starting the stdio loop."""
    from imas_ink.server.mcp import InkServer

    return InkServer()


class TestServerName:
    def test_server_name_is_imas_ink(self, ink_server):
        assert ink_server.mcp.name == "imas-ink"


class TestToolRegistration:
    def test_all_expected_tools_registered(self, ink_server):
        names = set(ink_server.tool_names())
        assert names >= EXPECTED_TOOLS, (
            f"Missing tools: {EXPECTED_TOOLS - names}\nRegistered: {names}"
        )

    def test_plot_equilibrium_registered(self, ink_server):
        assert "plot_equilibrium" in ink_server.tool_names()

    def test_plot_time_traces_registered(self, ink_server):
        assert "plot_time_traces" in ink_server.tool_names()

    def test_plot_convergence_registered(self, ink_server):
        assert "plot_convergence" in ink_server.tool_names()

    def test_animate_pulse_registered(self, ink_server):
        assert "animate_pulse" in ink_server.tool_names()

    def test_plot_radial_profiles_registered(self, ink_server):
        assert "plot_radial_profiles" in ink_server.tool_names()

    def test_repl_registered(self, ink_server):
        assert "repl" in ink_server.tool_names()

    def test_no_3d_tools(self, ink_server):
        """3D coilset tools must not appear — Phase 4 only."""
        names = ink_server.tool_names()
        assert "plot_coilset_3d" not in names

    def test_no_efit_imports(self):
        """The server module must not import anything from efit.*."""
        import importlib
        import sys

        # Reload to catch import-time side effects
        if "imas_ink.server.mcp" in sys.modules:
            mod = sys.modules["imas_ink.server.mcp"]
        else:
            mod = importlib.import_module("imas_ink.server.mcp")

        # None of the module's imports should be from efit
        source_file = getattr(mod, "__file__", "") or ""
        if source_file:
            with open(source_file) as fh:
                content = fh.read()
            assert "from efit" not in content
            assert "import efit" not in content
