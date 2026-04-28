"""DD version pinning helper for imas-python entrypoints.

The precedence order is:
1. Explicit ``dd_version`` kwarg if not ``None``
2. ``IMAS_VERSION`` environment variable if set
3. :data:`DEFAULT_DD_VERSION`
"""

from __future__ import annotations

import os

DEFAULT_DD_VERSION: str = "4.1.0"  # matches IMAS-Cpp/5.5.0 module on SDCC


def resolve_dd_version(explicit: str | None = None) -> str:
    """Resolve the Data Dictionary version to use for an imas-python entry.

    Precedence (highest to lowest):

    1. *explicit* kwarg — returned as-is when not ``None``.
    2. ``IMAS_VERSION`` environment variable — used when set and non-empty.
    3. :data:`DEFAULT_DD_VERSION` — fallback, matches the SDCC IMAS-Cpp module.

    Parameters
    ----------
    explicit:
        Caller-supplied version string (e.g. ``"4.1.0"``).  When not
        ``None`` this is returned without further inspection.

    Returns
    -------
    str
        A non-empty DD version string.  Never ``None``.

    Examples
    --------
    >>> resolve_dd_version("3.42.0")
    '3.42.0'
    >>> import os; os.environ["IMAS_VERSION"] = "4.0.0"
    >>> resolve_dd_version()
    '4.0.0'
    >>> del os.environ["IMAS_VERSION"]
    >>> resolve_dd_version()
    '4.1.0'
    """
    if explicit is not None:
        return explicit
    env = os.environ.get("IMAS_VERSION", "").strip()
    if env:
        return env
    return DEFAULT_DD_VERSION
