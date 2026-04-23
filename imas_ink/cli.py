"""imas-ink command-line interface.

Currently a placeholder — the full CLI lands in Phase 3 (server) and
Phase 4 (3D demo). The ``imas-ink`` console entry point dispatches here.
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
        Process exit code.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "imas-ink — IMAS plotting library + MCP server\n"
            "\n"
            "Usage:\n"
            "  imas-ink serve                 Start MCP server (Phase 3)\n"
            "  imas-ink demo iter-coilset     Render ITER coilset (Phase 4)\n"
            "  imas-ink version               Print version and exit\n",
        )
        return 0
    if argv[0] == "version":
        from . import __version__

        print(__version__)
        return 0
    if argv[0] == "serve":
        print("imas-ink serve: MCP server lands in Phase 3.", file=sys.stderr)
        return 2
    if argv[0] == "demo":
        print("imas-ink demo: demos land in Phase 4.", file=sys.stderr)
        return 2
    print(f"imas-ink: unknown command {argv[0]!r}", file=sys.stderr)
    return 2


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
