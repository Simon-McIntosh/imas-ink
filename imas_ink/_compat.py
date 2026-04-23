"""DD-version compatibility shims for imas_ink.

Narrow, explicit helpers for known field renames between Data Dictionary
major versions (DD3 → DD4).  Each helper tries the newer name first and
falls back to the older name.

Rules
-----
- Return ``None`` only when the field is *genuinely absent* from the IDS
  object (i.e. neither name exists as an attribute).
- Never return ``None`` for a field that is present but holds an IMAS
  EMPTY sentinel value — let the sentinel propagate to :func:`safe_float`
  so the caller can handle it uniformly.
- Do NOT add a generic ``getattr_fallback`` here; each helper must have a
  specific docstring explaining which rename it addresses.
"""

from __future__ import annotations

_MISSING = object()  # sentinel distinguishing "absent" from "None value"


def resolve_q95(gq) -> object:
    """Return the q95 safety factor from ``global_quantities``.

    Handles the rename between DD3 (``q95``) and DD4 (``q_95``).

    Parameters
    ----------
    gq
        ``global_quantities`` sub-structure from an equilibrium time-slice.

    Returns
    -------
    object
        The raw field value (may be an IMAS EMPTY sentinel — callers
        must pass it through :func:`~imas_ink._sentinel.safe_float`).
        Returns ``None`` only if *neither* ``q_95`` nor ``q95`` exists.
    """
    for attr in ("q_95", "q95"):
        val = getattr(gq, attr, _MISSING)
        if val is not _MISSING:
            return val
    return None
