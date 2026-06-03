"""IDS to dataclass extractor functions.

Each function reads an IMAS IDS object and returns a frozen dataclass.
All EMPTY sentinel filtering happens here — downstream code never sees
sentinel values.
"""

from __future__ import annotations

import numpy as np

from ._compat import resolve_q95
from ._sentinel import is_empty, safe_float
from ._types import (
    EquilibriumSlice,
    MachineGeometry,
    RadialProfiles,
    TimeTraces,
)
from .geometry import coil_bboxes, find_xpoints, wall_clip_vertices


def extract_slice(eq_ids, time_index: int) -> EquilibriumSlice:
    """Extract a single time-slice from an equilibrium IDS.

    Reads ``profiles_2d[0]`` grid, ``global_quantities``, and boundary
    data. Numerically detects X-points if ``boundary.x_point`` is empty.

    Parameters
    ----------
    eq_ids
        ``equilibrium`` IDS object.
    time_index : int
        Index into ``time_slice`` to extract.

    Returns
    -------
    EquilibriumSlice

    Examples
    --------
    >>> sl = extract_slice(eq_ids, 0)
    >>> sl.time
    0.1
    """
    ts = eq_ids.time_slice[time_index]
    p2d = ts.profiles_2d[0]

    r_grid = np.asarray(p2d.grid.dim1)
    z_grid = np.asarray(p2d.grid.dim2)
    psi_2d = np.asarray(p2d.psi)

    gq = ts.global_quantities
    psi_axis = safe_float(gq.psi_axis)
    psi_bnd = safe_float(gq.psi_boundary)
    r_axis = safe_float(gq.magnetic_axis.r)
    z_axis = safe_float(gq.magnetic_axis.z)
    ip = safe_float(gq.ip)
    time = float(eq_ids.time[time_index])
    converged = True  # default; refine from convergence info if available

    # X-points: try IDS first, fall back to numerical detection
    x_points: list[tuple[float, float]] = []
    try:
        for xp in ts.boundary.x_point:
            r_x = safe_float(xp.r)
            z_x = safe_float(xp.z)
            if not (np.isnan(r_x) or np.isnan(z_x)):
                x_points.append((r_x, z_x))
    except (AttributeError, IndexError):
        pass
    if not x_points and not np.isnan(psi_bnd) and not np.isnan(z_axis):
        r_2d, z_2d = np.meshgrid(r_grid, z_grid, indexing="ij")
        x_points = find_xpoints(psi_2d, r_2d, z_2d, psi_bnd, psi_axis, z_axis)

    # Boundary shape
    boundary_r = boundary_z = None
    try:
        br = np.asarray(ts.boundary.outline.r)
        bz = np.asarray(ts.boundary.outline.z)
        if br.size > 0 and not np.any(is_empty(br)):
            boundary_r, boundary_z = br, bz
    except (AttributeError, IndexError):
        pass

    # Optional global quantities
    beta_pol = safe_float(getattr(gq, "beta_pol", None))
    li_3 = safe_float(getattr(gq, "li_3", None))
    q95 = safe_float(resolve_q95(gq))

    return EquilibriumSlice(
        psi_2d=psi_2d,
        r_grid=r_grid,
        z_grid=z_grid,
        psi_axis=psi_axis,
        psi_boundary=psi_bnd,
        r_axis=r_axis,
        z_axis=z_axis,
        ip=ip,
        time=time,
        converged=converged,
        x_points=x_points,
        boundary_r=boundary_r,
        boundary_z=boundary_z,
        beta_pol=beta_pol if not np.isnan(beta_pol) else None,
        li_3=li_3 if not np.isnan(li_3) else None,
        q95=q95 if not np.isnan(q95) else None,
    )


def _select_description_2d(wall_ids):
    """Select the richest description_2d entry from the wall IDS.

    Mirrors ``efit.wall_containment.select_description_2d`` (and
    ``src/imas.cpp::selectDesc2d``) exactly:
      1. If only one entry, return it.
      2. Untyped (type.index == IMAS_INT_SENTINEL or < 0) scores -1.
      3. Among entries, higher effective type wins.
      4. Tie on type → higher unit count wins.
      5. i==0 seeds the default.

    Note: imas_ink must NOT import efit (circular dependency), so this
    mirrors the logic natively.
    """
    _SENTINEL = -999999999

    def _eff_type(desc) -> int:
        try:
            idx = int(desc.limiter.type.index)
            return -1 if (idx == _SENTINEL or idx < 0) else idx
        except Exception:
            return -1

    descs = wall_ids.description_2d
    n = len(descs)
    if n <= 1:
        return descs[0]

    best_idx = 0
    best_type = _eff_type(descs[0])
    best_count = len(descs[0].limiter.unit)

    for i in range(1, n):
        eff_type = _eff_type(descs[i])
        unit_count = len(descs[i].limiter.unit)
        better = eff_type > best_type or (
            eff_type == best_type and unit_count > best_count
        )
        if better:
            best_idx = i
            best_type = eff_type
            best_count = unit_count

    return descs[best_idx]


_MOBILE_TIME_SENTINEL: float = -1e30
"""Times more negative than this are IMAS EMPTY sentinels — skip for time selection."""


def _select_mobile_unit(desc_2d, time: float | None) -> list:
    """Return the list of wall units from desc_2d, with mobile-outline nearest-time selection.

    IMAS wall DD structure for mobile components::

        description_2d.mobile.unit[i]       — one mobile PFC
            .outline[j]                     — time-indexed snapshot
                .r / .z  (FLT_1D)          — outline coordinates
                .time    (FLT_0D)           — snapshot time [s]

    When ``desc_2d`` has a ``.mobile`` attribute with at least one ``unit``
    AND a ``time`` value is provided AND at least one snapshot has a valid
    (non-sentinel) time, the per-unit nearest-time outline is picked.

    Returns a synthetic list of one-shot unit proxies with ``outline.r/z``
    set to the nearest-time snapshot.  Falls through to all limiter units when:
    - ``time`` is None
    - no mobile attribute
    - all mobile outline times are sentinels (e.g. WEST static mobile)

    Parameters
    ----------
    desc_2d :
        Selected wall description_2d entry.
    time : float or None
        Slice time for mobile-outline selection.  None → return all limiter units.

    Returns
    -------
    list of unit objects (from limiter or synthetic mobile proxies)
    """
    import types as _types

    mobile = getattr(desc_2d, "mobile", None)
    if mobile is None or time is None:
        return list(desc_2d.limiter.unit)

    try:
        mobile_units = list(mobile.unit)
    except (AttributeError, TypeError):
        return list(desc_2d.limiter.unit)

    if not mobile_units:
        return list(desc_2d.limiter.unit)

    # Try to select nearest-time outline per mobile unit
    selected: list = []
    for mu in mobile_units:
        try:
            outlines = list(mu.outline)
        except (AttributeError, TypeError):
            continue
        best_ol = None
        best_dt = float("inf")
        for ol in outlines:
            try:
                t_ol = float(ol.time)
                if t_ol < _MOBILE_TIME_SENTINEL:
                    continue  # sentinel — skip
                dt = abs(t_ol - time)
                if dt < best_dt:
                    best_dt = dt
                    best_ol = ol
            except (AttributeError, TypeError, ValueError):
                continue
        if best_ol is not None:
            # Build a synthetic unit proxy with .outline.r/.outline.z
            proxy = _types.SimpleNamespace(
                outline=_types.SimpleNamespace(r=best_ol.r, z=best_ol.z)
            )
            selected.append(proxy)

    # If no valid mobile outlines found (all sentinels), fall back to limiter units
    if not selected:
        return list(desc_2d.limiter.unit)
    return selected


def extract_geometry(wall_ids, pf_ids, magnetics_ids=None, time: float | None = None) -> MachineGeometry:
    """Extract static machine geometry from wall, pf_active, and optionally magnetics IDSs.

    Parameters
    ----------
    wall_ids
        ``wall`` IDS object.
    pf_ids
        ``pf_active`` IDS object.
    magnetics_ids
        Optional ``magnetics`` IDS. When provided, B-pol probe and flux
        loop positions are extracted (with DDv3/DDv4 field-name fallback).
    time : float, optional
        Slice time for mobile-outline selection (WEST description_2d[1].mobile).
        When provided, mobile units with a ``.time`` attribute are filtered to
        the nearest time.  Defaults to None (return all units).

    Returns
    -------
    MachineGeometry
    """
    # Select the richest description_2d (mirrors selectDesc2d in solver)
    desc_2d = _select_description_2d(wall_ids)

    # Collect all (or nearest-time mobile) units
    units = _select_mobile_unit(desc_2d, time)

    # Build wall_units: list of (r_array, z_array) per unit
    wall_units: list[tuple[np.ndarray, np.ndarray]] = []
    for unit in units:
        try:
            r_u = np.asarray(unit.outline.r)
            z_u = np.asarray(unit.outline.z)
            if r_u.size > 0 and r_u.size == z_u.size:
                wall_units.append((r_u, z_u))
        except (AttributeError, TypeError):
            continue

    # Backward compat: wall_r/wall_z are the first unit
    if wall_units:
        wall_r, wall_z = wall_units[0]
    else:
        # Fallback: original single-unit path (handles malformed IDS gracefully)
        wall_r = np.asarray(wall_ids.description_2d[0].limiter.unit[0].outline.r)
        wall_z = np.asarray(wall_ids.description_2d[0].limiter.unit[0].outline.z)
        wall_units = [(wall_r, wall_z)]

    clip_verts = wall_clip_vertices(wall_r, wall_z)
    coils = coil_bboxes(pf_ids)

    probe_r_list: list[float] = []
    probe_z_list: list[float] = []
    probe_angle_list: list[float] = []
    flux_loop_r_list: list[float] = []
    flux_loop_z_list: list[float] = []

    if magnetics_ids is not None:
        # DDv4: b_field_pol_probe; DDv3: bpol_probe
        probes = getattr(magnetics_ids, "b_field_pol_probe", None)
        if probes is None or len(probes) == 0:
            probes = getattr(magnetics_ids, "bpol_probe", [])
        for p in probes:
            try:
                r_p = safe_float(p.position.r)
                z_p = safe_float(p.position.z)
            except (AttributeError, IndexError):
                continue
            if np.isnan(r_p) or np.isnan(z_p):
                continue
            try:
                a_p = safe_float(p.poloidal_angle)
            except (AttributeError, IndexError):
                a_p = float("nan")
            probe_r_list.append(r_p)
            probe_z_list.append(z_p)
            probe_angle_list.append(a_p)

        loops = getattr(magnetics_ids, "flux_loop", [])
        for fl in loops:
            try:
                pos = fl.position[0]
                r_l = safe_float(pos.r)
                z_l = safe_float(pos.z)
            except (AttributeError, IndexError):
                continue
            if np.isnan(r_l) or np.isnan(z_l):
                continue
            flux_loop_r_list.append(r_l)
            flux_loop_z_list.append(z_l)

    return MachineGeometry(
        wall_r=wall_r,
        wall_z=wall_z,
        coil_rects=coils,
        wall_clip_vertices=clip_verts,
        wall_units=wall_units,
        probe_r=np.asarray(probe_r_list, dtype=float),
        probe_z=np.asarray(probe_z_list, dtype=float),
        probe_angle=np.asarray(probe_angle_list, dtype=float),
        flux_loop_r=np.asarray(flux_loop_r_list, dtype=float),
        flux_loop_z=np.asarray(flux_loop_z_list, dtype=float),
    )


def extract_time_traces(eq_ids) -> TimeTraces:
    """Extract global scalar time traces from an equilibrium IDS.

    Iterates all ``time_slice`` entries, extracting global quantities
    into aligned 1D arrays.

    Parameters
    ----------
    eq_ids
        ``equilibrium`` IDS object.

    Returns
    -------
    TimeTraces

    Examples
    --------
    >>> tt = extract_time_traces(eq_ids)
    >>> tt.ip.shape == tt.time.shape
    True
    """
    n = len(eq_ids.time_slice)
    time = np.asarray(eq_ids.time[:n])
    ip = np.full(n, np.nan)
    beta_pol = np.full(n, np.nan)
    li_3 = np.full(n, np.nan)
    q95 = np.full(n, np.nan)
    converged = np.zeros(n, dtype=bool)
    chi_squared = np.full(n, np.nan)
    n_iterations = np.zeros(n, dtype=int)

    for i, ts in enumerate(eq_ids.time_slice):
        gq = ts.global_quantities
        ip[i] = safe_float(gq.ip)
        beta_pol[i] = safe_float(gq.beta_pol)
        li_3[i] = safe_float(gq.li_3)
        q95[i] = safe_float(resolve_q95(gq))

    return TimeTraces(
        time=time,
        ip=ip,
        beta_pol=beta_pol,
        li_3=li_3,
        q95=q95,
        converged=converged,
        chi_squared=chi_squared,
        n_iterations=n_iterations,
    )


def extract_profiles_1d(eq_ids, time_index: int) -> RadialProfiles:
    """Extract 1D radial profiles at a single time slice.

    Reads ``profiles_1d`` and normalises psi if it appears to be in
    physical (non-normalised) units.

    Parameters
    ----------
    eq_ids
        ``equilibrium`` IDS object.
    time_index : int
        Index into ``time_slice``.

    Returns
    -------
    RadialProfiles

    Examples
    --------
    >>> rp = extract_profiles_1d(eq_ids, 5)
    >>> 0.0 <= rp.psi_norm[0] <= rp.psi_norm[-1] <= 1.0
    True
    """
    ts = eq_ids.time_slice[time_index]
    p1d = ts.profiles_1d
    psi_norm = np.asarray(p1d.psi)
    if psi_norm.max() > 1.5:
        psi_ax = safe_float(ts.global_quantities.psi_axis)
        psi_bnd = safe_float(ts.global_quantities.psi_boundary)
        denom = psi_ax - psi_bnd
        if not np.isnan(psi_ax) and not np.isnan(psi_bnd) and abs(denom) > 1e-10:
            psi_norm = (psi_norm - psi_bnd) / denom

    def _safe_array(attr: str) -> np.ndarray:
        try:
            arr = np.asarray(getattr(p1d, attr))
            arr = arr.astype(float, copy=True)
            arr[is_empty(arr)] = np.nan
            return arr
        except (AttributeError, TypeError):
            return np.full_like(psi_norm, np.nan)

    return RadialProfiles(
        psi_norm=psi_norm,
        pressure=_safe_array("pressure"),
        j_tor=_safe_array("j_tor"),
        q=_safe_array("q"),
        pprime=_safe_array("dpressure_dpsi"),
        ffprime=_safe_array("f_df_dpsi"),
        time=float(eq_ids.time[time_index]),
    )
