# Changelog

All notable changes to `imas-ink` are documented in this file. The format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the
project adheres to semantic versioning.

## [Unreleased]

### Added
- Project bootstrap: pyproject, AGENTS.md, CI scaffolding.
- Extraction of `efit.ink` 2D plotting library is in progress; public API
  stabilises at v0.1.0.
- Namespaced REPL tool (`repl(code, namespace=...)`) for the MCP server.
- `three_d` subpackage for 3D coilset rendering (pyvista + vedo).
- `geometry.classify_probe_components()` — labels magnetic probes by
  measurement component so co-located multi-component sensors (e.g. WEST
  and ITER tangential+normal probe pairs sharing one location, ~90° apart)
  can be rendered distinctly. Machine-agnostic: groups by co-location and
  partitions by undirected orientation; does not fire on single-orientation
  arrays (AUG, TCV, MAST-U, HL-3) or on toroidally-replicated duplicates.
- `style.probe_secondary_color` — colour for the secondary orientation
  component of co-located probe pairs.

### Fixed
- Magnetic-probe direction ticks now render each probe's true sensing axis
  and distinguish co-located multi-component sensors by colour. The tick is
  drawn along the IMAS `b_field_pol_probe/poloidal_angle`, which the Data
  Dictionary defines as the **sensor normal vector** (the coil axis, i.e.
  the direction of the field component the probe measures), clockwise from
  +R̂ — screen-space vector n = (cos θ, −sin θ), matching EFIT's forward
  model exactly. The previous single-colour rendering collapsed WEST's
  tangential B-pol and co-located normal-component probes, making B-pol
  ticks appear normal to the wall. The angle convention is unchanged (no
  offset/flip is applied; the DD vector is correct). The Altair backend
  (`_render_probes_alt`) is brought in line with matplotlib: it had used a
  counter-clockwise (cos θ, +sin θ) tick, now corrected to the DD-grounded
  clockwise (cos θ, −sin θ) and centred on the probe.
