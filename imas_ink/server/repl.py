"""Namespaced stateful Python REPL for the imas-ink MCP server.

Each namespace is an isolated globals dict. State persists across
tool calls for the life of the server process. A single global lock
serialises execution — REPL calls are rare and the lock window is short.

Example
-------
>>> from imas_ink.server.repl import repl
>>> repl("x = 42", namespace="agent-a")
''
>>> repl("x + 1", namespace="agent-a")
'43\n'
>>> repl("x", namespace="agent-b")  # doctest: +ELLIPSIS
"NameError: name 'x' is not defined..."
>>> repl("", namespace="agent-a", reset=True)
''
"""

from __future__ import annotations

import contextlib
import io
import threading
import traceback
from typing import Any

# Per-namespace globals dicts, keyed by namespace name.
_namespaces: dict[str, dict[str, Any]] = {}

# One global lock is sufficient — REPL calls are rare.
_lock = threading.Lock()


class _LazyModule:
    """Thin lazy-import proxy.

    The module is imported once on first attribute access and cached.
    This avoids loading heavy dependencies (matplotlib) at namespace
    seeding time when the namespace may never be used for plotting.

    Parameters
    ----------
    module_name : str
        Fully-qualified module name to import.
    setup_fn : callable, optional
        Called before the first import (e.g. to set the backend).
    """

    def __init__(self, module_name: str, setup_fn=None) -> None:
        # Use object.__setattr__ to bypass __setattr__ override if needed.
        object.__setattr__(self, "_module_name", module_name)
        object.__setattr__(self, "_setup_fn", setup_fn)
        object.__setattr__(self, "_module", None)

    def _load(self) -> Any:
        mod = object.__getattribute__(self, "_module")
        if mod is None:
            setup = object.__getattribute__(self, "_setup_fn")
            if setup is not None:
                setup()
            import importlib

            mod = importlib.import_module(object.__getattribute__(self, "_module_name"))
            object.__setattr__(self, "_module", mod)
        return mod

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __repr__(self) -> str:  # pragma: no cover
        mod = object.__getattribute__(self, "_module")
        if mod is None:
            return (
                f"<LazyModule {object.__getattribute__(self, '_module_name')!r} (not yet imported)>"
            )
        return repr(mod)


def _make_repl_help() -> Any:
    """Return a ``repl_help`` callable bound to a fixed help string."""

    def repl_help() -> str:
        """Print pre-loaded names and REPL usage."""
        return (
            "imas-ink REPL — pre-loaded names\n"
            "=================================\n"
            "  ink   — imas_ink package (extract, plot, components, …)\n"
            "  np    — numpy\n"
            "  plt   — matplotlib.pyplot (lazy; Agg backend)\n"
            "\n"
            "Usage:\n"
            "  repl('x = 1')                          # set a variable\n"
            "  repl('x + 1')                          # evaluate expression\n"
            "  repl(code, namespace='myns')           # isolated workspace\n"
            "  repl('', namespace='myns', reset=True) # wipe namespace\n"
            "\n"
            "Multi-line code:\n"
            "  repl('def f(x):\\n    return x*2\\nf(21)')\n"
        )

    return repl_help


def _seed_namespace(ns: dict[str, Any]) -> None:
    """Populate a fresh globals dict with useful defaults.

    Imports are performed lazily where possible to keep the first call
    fast even when heavy libraries (matplotlib) are not yet in memory.

    Parameters
    ----------
    ns : dict
        The globals dict to populate in-place.
    """
    import numpy as np

    import imas_ink as ink

    def _plt_setup() -> None:
        import matplotlib

        matplotlib.use("Agg")

    ns["__name__"] = "__repl__"
    ns["__builtins__"] = __builtins__
    ns["ink"] = ink
    ns["np"] = np
    ns["plt"] = _LazyModule("matplotlib.pyplot", setup_fn=_plt_setup)
    ns["repl_help"] = _make_repl_help()


def _get_namespace(namespace: str) -> dict[str, Any]:
    """Return the globals dict for *namespace*, creating it if absent."""
    if namespace not in _namespaces:
        ns: dict[str, Any] = {}
        _seed_namespace(ns)
        _namespaces[namespace] = ns
    return _namespaces[namespace]


def repl(code: str, namespace: str = "default", reset: bool = False) -> str:
    """Execute Python in a persistent, per-namespace REPL.

    Parameters
    ----------
    code : str
        Python source. Multi-line supported. Use ``print()`` for output;
        a trailing bare expression has its ``repr`` printed automatically.
    namespace : str
        Isolated state bucket (default: ``"default"``). Per-caller
        workspaces never share variables.
    reset : bool
        If *True*, wipe this namespace's globals before executing *code*.

    Returns
    -------
    str
        Captured stdout; repr of last expression if nothing was printed;
        or a formatted traceback on error.

    Examples
    --------
    >>> repl("x = 10")
    ''
    >>> repl("x * 3")
    '30\n'
    >>> repl("x", namespace="other")  # isolated  # doctest: +ELLIPSIS
    "NameError: ..."
    """
    with _lock:
        if reset and namespace in _namespaces:
            del _namespaces[namespace]

        ns = _get_namespace(namespace)

        stdout_capture = io.StringIO()
        stderr_capture = io.StringIO()

        try:
            with (
                contextlib.redirect_stdout(stdout_capture),
                contextlib.redirect_stderr(stderr_capture),
            ):
                try:
                    result = eval(code, ns)
                    if result is not None:
                        ns["_"] = result
                        print(repr(result))
                except SyntaxError:
                    exec(code, ns)

            return stdout_capture.getvalue()

        except Exception:
            return traceback.format_exc()
