"""imas-ink command-line interface.

Dispatches sub-commands:

- ``serve``   — start the MCP server (stdio transport, FastMCP).
- ``demo``    — 3-D coilset demo (Phase 4, not yet implemented).
- ``version`` — print the package version and exit.

The ``fastmcp`` dependency is imported lazily inside the ``serve`` branch
so that ``imas-ink --version`` does not pull in server dependencies.
"""

from __future__ import annotations

import sys


def main(argv: list[str] | None = None) -> int:
    """Entry point for ``imas-ink`` / ``uv run imas-ink``.

    Parameters
    ----------
    argv
        Argument vector (default: ``sys.argv[1:]``).

    Returns
    -------
    int
        Process exit code (0 on success, 2 on error/unimplemented).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "imas-ink — IMAS plotting library + MCP server\n"
            "\n"
            "Usage:\n"
            "  imas-ink serve                 Start MCP server (stdio)\n"
            "  imas-ink demo iter-coilset     Render ITER coilset (Phase 4)\n"
            "  imas-ink version               Print version and exit\n",
        )
        return 0
    if argv[0] in {"version", "--version", "-V"}:
        from . import __version__

        print(__version__)
        return 0
    if argv[0] == "serve":
        # Lazy import so version / help don't load server dependencies.
        from .server.mcp import serve

        serve()
        return 0
    if argv[0] == "demo":
        print("imas-ink demo: 3D coilset demo lands in Phase 4.", file=sys.stderr)
        return 2
    print(f"imas-ink: unknown command {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
