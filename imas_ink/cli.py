"""imas-ink command-line interface.

Dispatches sub-commands:

- ``serve``                     — start the MCP server (stdio transport).
- ``demo iter-coilset``         — render a 3D ITER coilset PNG.
- ``version`` / ``--version``   — print the package version and exit.

Heavy dependencies (``fastmcp``, ``pyvista``) are imported lazily inside
the relevant sub-command so that ``imas-ink --version`` stays light.
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
        Process exit code (0 on success, 1 on user error, 2 on unknown command).
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "imas-ink — IMAS plotting library + MCP server\n"
            "\n"
            "Usage:\n"
            "  imas-ink serve                 Start MCP server (stdio)\n"
            "  imas-ink demo iter-coilset     Render ITER coilset\n"
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
        return _demo(argv[1:])
    print(f"imas-ink: unknown command {argv[0]!r}", file=sys.stderr)
    return 2


def _demo(argv: list[str]) -> int:
    """Dispatch ``imas-ink demo <subcommand>``."""
    if not argv or argv[0] in {"-h", "--help"}:
        print(
            "imas-ink demo — demonstration renderers\n"
            "\n"
            "Sub-commands:\n"
            "  iter-coilset   Render ITER coilset to PNG\n"
            "\n"
            "Options (iter-coilset):\n"
            "  --uri URI          IMAS URI (required)\n"
            "  --outfile PATH     Output PNG path (default: iter-coilset.png)\n"
            "  --no-wall          Hide vessel wall\n"
            "  --no-tf            Hide TF coils\n"
            "  --no-pf            Hide PF / CS coils\n"
            "  --view VIEW        Camera preset: iso, poloidal, toroidal\n",
            file=sys.stderr,
        )
        return 1

    if argv[0] == "iter-coilset":
        return _demo_iter_coilset(argv[1:])

    print(f"imas-ink demo: unknown sub-command {argv[0]!r}", file=sys.stderr)
    return 1


def _demo_iter_coilset(argv: list[str]) -> int:
    """Handle ``imas-ink demo iter-coilset [OPTIONS]``."""
    import pathlib

    # Simple argument parsing (no external dep required)
    uri: str | None = None
    outfile: pathlib.Path = pathlib.Path("iter-coilset.png")
    show_wall = True
    show_pf = True
    show_tf = True
    view = "iso"

    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--uri" and i + 1 < len(argv):
            uri = argv[i + 1]
            i += 2
        elif arg == "--outfile" and i + 1 < len(argv):
            outfile = pathlib.Path(argv[i + 1])
            i += 2
        elif arg == "--no-wall":
            show_wall = False
            i += 1
        elif arg == "--no-tf":
            show_tf = False
            i += 1
        elif arg == "--no-pf":
            show_pf = False
            i += 1
        elif arg == "--view" and i + 1 < len(argv):
            view = argv[i + 1]
            i += 2
        else:
            print(f"imas-ink demo iter-coilset: unknown option {arg!r}", file=sys.stderr)
            return 1

    if uri is None:
        print(
            "imas-ink demo iter-coilset: --uri is required\n"
            "Example: imas-ink demo iter-coilset --uri "
            '"imas:hdf5?path=/path/to/machine/"',
            file=sys.stderr,
        )
        return 1

    try:
        # Lazy-import three_d — VTK must not load for --version
        from .three_d.scene import render_coilset

        render_coilset(
            uri,
            outfile=outfile,
            show_wall=show_wall,
            show_pf=show_pf,
            show_tf=show_tf,
            view=view,
            off_screen=True,
        )
        print(f"Saved: {outfile}")
        return 0
    except Exception as exc:
        print(f"imas-ink demo iter-coilset: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
