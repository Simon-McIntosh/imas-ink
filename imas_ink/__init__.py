"""imas-ink — IMAS-backed plotting and visualisation for tokamak equilibrium data.

The top-level module exposes the **library API only**. The MCP server
lives at :mod:`imas_ink.server` and is intentionally not re-exported from
here, so that ``import imas_ink`` does not pull FastMCP.

The :mod:`imas_ink.three_d` subpackage is also not eagerly imported, so
that VTK / pyvista / vedo are only loaded when 3D rendering is actually
requested.

Public API is stabilised at v0.1.0. Until then, expect churn.
"""

from __future__ import annotations

try:
    from ._version import __version__
except ImportError:  # source checkout without hatch-vcs run
    __version__ = "0.0.0.dev0"

__all__ = ["__version__"]
