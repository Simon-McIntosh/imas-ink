# imas-ink

IMAS-backed plotting and visualisation library for tokamak equilibrium
data. Provides a library API (2D + 3D), a command-line entry point, and
an MCP server so LLM agents can render equilibria and coilsets directly
from IMAS HDF5 datasets.

## Status

Pre-release (`v0` series). The 2D API is ported from `efit.ink` in the
ITER EFIT codebase; the `three_d` subpackage renders coilsets, TF coils,
and vessel geometry using pyvista with vedo primitives for axisymmetric
extrusion and swept cross-sections.

## Design principles

- **Library-first.** Everything is importable as plain Python. The MCP
  server is a thin wrapper over the library.
- **IMAS-native.** Reads data via
  [`imas-python`](https://github.com/iterorganization/imas-python). Trusts
  imas-python to auto-detect the on-disk DD version; no forced pinning.
  A tiny `_compat` layer handles known field renames between DD major
  versions.
- **No heavy deps in the core.** Base install pulls matplotlib +
  contourpy + altair + imas-python. VTK/pyvista/vedo land in the `[3d]`
  extra. FastMCP lands in `[server]`. Lazy-imported inside subpackages
  so `import imas_ink` stays cheap.
- **Fast rendering.** `three_d` targets off-screen PNG / HTML export,
  reusing geometry primitives patterned after the nova codebase (ITER
  GIP). No runtime dependency on nova.

## Install

```bash
# From source (during development)
pip install -e '.[server,3d,test]'

# Or via uv
uv sync --extra server --extra 3d --extra test
```

## Quick start

```python
import imas_ink as ink

slice_ = ink.extract_slice("imas:hdf5?path=/path/to/iter/135013/", 5)
fig = ink.equilibrium_figure_mpl(slice_)
fig.savefig("equilibrium.png", dpi=150)
```

### MCP server

```bash
uv run imas-ink serve
```

Registers tools `imas-ink-plot_equilibrium`, `imas-ink-plot_time_traces`,
`imas-ink-plot_convergence`, `imas-ink-animate_pulse`,
`imas-ink-plot_radial_profiles`, `imas-ink-plot_coilset_3d`, and a
stateful, namespaced REPL.

### 3D coilset demo

```bash
uv run imas-ink demo iter-coilset --uri "imas:hdf5?path=/path/to/iter/machine/"
```

## License

ITER GIP. See [`LICENSE`](./LICENSE).
