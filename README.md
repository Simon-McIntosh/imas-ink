# imas-ink

IMAS-backed plotting and visualisation library for tokamak equilibrium
data. Provides a library API (2D + 3D), a command-line entry point, and
an MCP server so LLM agents can render equilibria and coilsets directly
from IMAS HDF5 datasets.

## Status

**Alpha — API under active development.** The 2D library, MCP server,
namespaced REPL, and 3D coilset renderer are all shipped and functional.
The API stabilises at v0.1.0.

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

## Install (developer, pre-release)

```bash
# From source (during development)
pip install -e '.[server,3d,test]'

# Or via uv
uv sync --extra server --extra 3d --extra test
```

PyPI publication is wired up (trusted publishing on `v*.*.*` tags) but
no release has been cut yet — install from source until the first
`v0.1.0` tag.

## MCP Server

The MCP server is shipped and functional. Start it with:

```bash
uv run imas-ink serve   # stdio transport — ready for Copilot CLI / Claude Desktop
```

Registered tools:

| Tool | Description |
|------|-------------|
| `imas-ink-plot_equilibrium` | Poloidal cross-section (single frame) |
| `imas-ink-plot_time_traces` | Ip, beta_pol, li_3, q95 over time |
| `imas-ink-plot_convergence` | Convergence status bar chart |
| `imas-ink-animate_pulse` | Full-pulse GIF animation |
| `imas-ink-plot_radial_profiles` | 1D radial profiles (p, q, j_tor, …) |
| `imas-ink-plot_coilset_3d` | 3D coilset + vessel render (requires `[3d]`) |
| `imas-ink-repl` | Stateful, namespaced Python REPL |

The REPL keeps a per-namespace globals dict so concurrent callers do not
step on each other:

```python
repl("x = 1", namespace="agent-a")          # sets x in agent-a
repl("x + 1", namespace="agent-a")          # → 2
repl("x", namespace="agent-b")              # NameError — isolated
repl("", namespace="agent-a", reset=True)   # clear that namespace
```

Pre-loaded names in every namespace: `ink` (imas_ink), `np` (numpy),
`plt` (matplotlib.pyplot, lazy).

## API

The following surfaces are shipped. The API stabilises at v0.1.0.

### Library (Phase 2 — shipped)

```python
import imas_ink as ink

slice_ = ink.extract_slice("imas:hdf5?path=/path/to/iter/135013/", 5)
```

### MCP server (shipped — see above)

### 3D coilset demo (shipped)

```bash
imas-ink demo iter-coilset --uri "imas:hdf5?path=/path/to/iter/machine/" --outfile iter-coilset.png
```

## License

ITER GIP — **proprietary, not OSI-approved open source**. The repository
is published publicly for collaboration but the licence does not grant
general open-source rights. See [`LICENSE`](./LICENSE) for the full
terms.

## Releases

Tags matching `vX.Y.Z` on `main` trigger the `release` workflow, which
publishes sdist + wheel to PyPI via the [trusted-publishing flow][tp].
The workflow rejects non-semver tags and tags that are not ancestors of
`main`. The PyPI project and a `pypi` GitHub environment must be
pre-configured by a maintainer before the first release.

[tp]: https://docs.pypi.org/trusted-publishers/
