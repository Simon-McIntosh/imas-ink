# Agent Guidelines

Use terminal for direct operations (`rg`, `fd`, `git`), `uv run` for tests/CLI,
MCP `repl()` for chained data exploration when the server is registered.
Conventional commits. **CRITICAL: Always commit and push when files have been
modified — no confirmation, no asking, just do it. Every response that modifies
files MUST end with `git add`, `git commit`, and `git push origin main`.**

**Never use `vscode_askQuestions` or any interactive popup dialog tool** —
present every question inline in the chat so the user can answer in one reply.

## Git Workflow — One Remote

This repository uses a **single-remote workflow**. There is no fork / upstream
split — `origin` is the canonical `Simon-McIntosh/imas-ink` repo on GitHub.

- All work happens on the `main` branch.
- Always **merge** on pull — never rebase.
- Never create feature branches without explicit user approval. Keep history
  linear on `main` unless a PR is specifically requested.
- Release tags follow `vMAJOR.MINOR.PATCH` semver.

### New Clone Setup

Run once after cloning on any machine:

```bash
git config --local pull.rebase false      # merge on pull, never rebase
git config --local rebase.autoStash false  # don't silently stash — make dirty worktree visible
git config --local merge.ff true           # allow fast-forward merges
```

**Why this matters:** different machines may have different global git configs
(ITER SDCC sets `pull.rebase=true` globally, for example). Local config
overrides global/system, ensuring consistent behaviour everywhere.

### Session Workflow

1. **Session start:** `git pull origin main` before any work.
2. **Before push:** `git pull origin main && git push origin main` — never
   push without pulling first.
3. **Dirty worktree:** Commit or stash **your own files** before pulling. Never
   `git stash` (no paths) — that stashes every agent's unstaged work. Only
   stash your files: `git stash push -- <path1> <path2>`.
4. **Conflict resolution:** resolve and commit. Never force-push without user
   approval.

## Commit Workflow — Check-Only Hooks

**CRITICAL — pre-commit hooks are check-only and MUST NOT modify files.** The
pre-commit framework's stash/restore cycle is catastrophic in multi-agent
environments: when a modifying hook (e.g. `ruff --fix`, `ruff-format` in
write mode) touches a file while another agent has unstaged work on any file,
pre-commit stashes that unstaged work to `.cache/pre-commit/` before running
the hook. If the hook then rolls back on conflict, the other agent's edits
are off-disk for a window during which any crash, abort, or rebase causes
silent data loss.

All modifying hooks here have been converted to check-only. Run format and
fix **BEFORE** staging:

```bash
uv run ruff check --fix .                 # lint + autofix (explicit)
uv run ruff format .                      # format (explicit)
git add <file1> <file2> ...               # stage specific files (never git add -A)
git commit -m "type(scope): concise summary"
git pull --no-rebase origin main          # merge peer changes first
git push origin main
```

If a check-only hook reports a violation, run the corresponding `--fix` /
`format` command manually, `git add` the updated files, and re-commit. **Never
re-enable `--fix` or write-mode formatting in `.pre-commit-config.yaml`** —
that reintroduces the multi-agent corruption risk.

### Conventional Commit Format

```
<type>(<optional scope>): <summary>
```

| Type | Purpose |
|------|---------|
| feat | New feature |
| fix | Bug fix |
| refactor | Code restructuring (no behaviour change) |
| docs | Documentation only |
| test | Test changes |
| chore | Build / CI / tooling / maintenance |
| perf | Performance improvement |

Breaking changes use a `BREAKING CHANGE:` footer, not a `type!:` suffix.

### Commit Message Rules

- **No AI co-authorship trailers.** Never add `Co-authored-by: Copilot`,
  `Co-authored-by: Claude`, or any AI-assistant trailer. Commits are authored
  solely by the human developer.
- **No plan / phase / step labels.** Never prefix commit titles with
  `Phase 1:`, `(P3)`, `Step 2:`, plan filenames, or internal task IDs. Each
  commit describes **what changed**, not which step of a plan it belongs to.
  Planning context belongs in session artefacts, not permanent git history.
- **No `git add -A` or `git add .`** — always stage explicit file paths.

## Parallel Agents — Safety Rules

Multiple agents may be editing this repository simultaneously. Assume another
agent could be writing to a file **right now**.

### Verify Before Modifying

- **Re-read files before editing.** Your in-memory view of any file may be
  hours old. Another agent may have renamed functions, added features, or
  restructured code since you last read it. Always `view` / `cat` the file
  from disk before making changes. **If the file looks different from what
  you expect, stop and re-read** — do not "fix" it back to what you remember.
- **Check recent git history** before modifying shared files:
  `git log --oneline -5 -- <file>`. If you see commits you don't recognise,
  read the file fresh and understand the current state.
- **If you see unfamiliar symbols or patterns, assume they are deliberate.**
  Another agent put them there intentionally. Do not revert them.

### Destructive Git Commands — BANNED Against Other Agents' Work

The following commands are **banned** against any file the current agent did
not create or modify in this session:

| Command | Effect | Scope |
|---------|--------|-------|
| `git restore <path>` | Wipes unstaged + staged changes | Files this agent didn't touch |
| `git checkout -- <path>` | Same (legacy syntax) | Same |
| `git checkout <ref> -- <path>` | Overwrites with version from another ref | Same |
| `git clean -f` / `git clean -fd` | Deletes untracked files | **Always banned** on the worktree |
| `git reset --hard` | Resets index + worktree | **Always banned** without explicit user approval |
| `git stash` (no paths) | Stashes every dirty file | Use `git stash push -- <path>` only |
| `git stash pop` / `apply` without age+stat check | May silently restore stale code | See below |

**`git stash pop` / `git stash apply` rules:** Stale stashes from prior
sessions can silently overwrite the worktree with old code, reverting
committed renames and improvements. Always verify first:
`git stash show stash@{N} --stat` and
`git log -1 --format='%ci' stash@{N}`. If the stash is more than a day old,
**drop it** — the code has moved on.

If you encounter unexpected file state (content missing, your prior edit is
gone, a file doesn't compile), you MUST:

1. **Stop.** Do not try to "restore" to a known-good state.
2. Run `git status`, `git log --since="2 hours ago"`, `git reflog | head -20`.
3. Run `git log --since="2 hours ago" -- <file>` to see recent commits on the
   affected file.
4. **Surface the anomaly to the user** before any destructive action.
5. Only proceed after user authorisation, with an explicit revert target SHA.

### Pre-Edit Discipline for Cross-Cutting Files

For AGENTS.md, pyproject.toml, shared modules, CI config — the window between
edit and push is the only window where another agent's `git restore` can
destroy the work. **Close that window on every coherent change.** Do not
batch multiple cross-cutting edits across turns while holding them
uncommitted.

```bash
git fetch origin main && git log --since="1 hour ago" -- <path>
git pull --no-rebase origin main
# ... edit ...
git add <path> && git commit -m "..." && git push origin main  # IMMEDIATELY
```

### Fleet Dispatch File Scoping

When an orchestrating agent dispatches multiple sub-agents in parallel, it
MUST allocate **non-overlapping file sets** per sub-agent and refuse to
dispatch two agents that would edit the same file. Overlapping scopes
produce lost work regardless of how careful each agent is individually.

### Session Hygiene

- **Close sessions when done** — `ctrl+d`, `/exit`, or `/quit`. Idle agent
  processes with stale context are the #1 cause of regressions.
- **Audit periodically:** `ps aux | grep copilot` — kill any process older
  than the current session.
- **Avoid long-lived `--yolo` sessions.** Auto-approve combined with stale
  context is the most dangerous failure mode. Start fresh sessions for new
  tasks.

## Root Cause Discipline

- **Always fix root causes.** Never add workarounds, fallback paths, or
  conditional branches to make tests pass. If a test fails because of wrong
  data, fix the data. If it fails because of a code bug, fix the code.
- **Test data is not a dumping ground for test fixtures.** Production code
  should remain unaware of test infrastructure.
- **Investigate deeply before fixing.** Read logs, inspect data, trace the
  code path. Surface fixes that suppress symptoms will be reverted.

### Never Adjust Test Criteria as a First Response

Tests measure the code. When a test fails, fix the code — not the test.

The following are all forbidden as a first-line response to a failing test:
loosening numeric tolerances, skipping assertions, marking tests
`xfail`/`skip` without a documented issue, deleting failing tests, widening
physics-gate thresholds.

Adjusting test criteria is available as a **last resort** only, with an
explicit justification in a plan document and explicit user approval.

## Project Philosophy

Greenfield project under active development. No backwards compatibility
yet — the API stabilises at v0.1.0.

- Breaking changes are expected; remove deprecated code decisively.
- Avoid "enhanced", "new", "refactored" in names — just use the good name.
- When patterns change, update all usages — do not leave old patterns
  alongside new.
- **Stale context kills.** If your session is more than a few hours old, your
  memory of file contents may be wrong. Re-read any file before modifying it.
- Prefer explicit over clever — future agents will read this code.
- Library first, MCP server second. Everything must be importable as plain
  Python. MCP tools are thin wrappers over the library.
- **Build on common infrastructure.** Before implementing functionality,
  search for existing utilities. When a pattern is needed by multiple
  modules, extract it to a shared location and import from there.

## Code Style

- Python ≥ 3.11 type syntax: `list[str]`, `X | Y`, `isinstance(e, ValueError | TypeError)`.
- Exception chaining: `raise NewError("msg") from e`.
- `dataclasses` (frozen) for value types; `pydantic` only when validation is
  needed at a trust boundary.
- `uv run` for all Python commands. Never manually activate the venv.
- Never use `git add -A`.
- The `.env` file holds secrets — never expose or commit it.

### Naming

**Never name files after implementation plans.** File names must be
understandable without knowledge of any plan document. Once a plan is
deleted (per project rules), names like `test_capability_gaps` become
meaningless. Instead, name files after what they test or implement:
`test_extract.py`, `test_server_repl.py`, `test_coilset.py`.

## Testing

Follow `~/.agents/AGENTS.md` "Development Environment". Use the existing root
`.venv`; missing test or 3D extras are blockers. In a detached worktree, set
`UV_PROJECT_ENVIRONMENT=/home/ITER/mcintos/Code/imas-ink/.venv` and
`PYTHONPATH="$PWD"` before running:

```bash
uv run --no-sync pytest                     # default: non-render tests
uv run --no-sync pytest -m render           # render-marked tests (VTK off-screen)
uv run --no-sync pytest --cov=imas_ink      # with coverage
```

The `render` marker gates tests that construct a `pyvista.Plotter`. These
run in a separate CI job (`.github/workflows/ci-3d.yml`) and must use
`off_screen=True` — **never call `pv.start_xvfb()`** in default test
environments.

## IMAS Data

Test fixtures requiring IMAS HDF5 data are **not vendored** in this
repository. Unit tests use synthetic mocks. Integration / demo tests read
from a path supplied via the `IMAS_INK_ITER_DATA` environment variable
and `pytest.skip` if it is unset.

```bash
# Example: point at an ITER dataset copied from the efit project
export IMAS_INK_ITER_DATA=/path/to/efitpp/tests/data/imas/ITER/135013
uv run pytest tests/test_geometry_magnetics.py
```

**DD version handling:** `imas_ink.extract` passes `dd_version=None` to
`imas.DBEntry`, letting `imas-python` auto-detect the on-disk version.
A small `_compat` module handles known field renames (e.g. `q95` vs
`q_95`, probe-name fallbacks) between DD major versions so callers never
have to pin a version themselves.

## MCP Server

```bash
# Stdio transport (for Copilot CLI, Claude Desktop, etc.)
uv run --no-sync imas-ink serve
```

Register in `~/.copilot/mcp-config.json`:

```json
{
  "mcp-servers": {
    "imas-ink": {
      "command": "uv",
      "args": ["--directory", "/path/to/imas-ink", "run", "--no-sync", "imas-ink", "serve"]
    }
  }
}
```

Tools exposed:

- `imas-ink-plot_equilibrium` — poloidal flux contour plot
- `imas-ink-plot_time_traces` — Ip, Wp, beta, li over time
- `imas-ink-plot_convergence` — residual history
- `imas-ink-plot_radial_profiles` — pressure, q, jphi
- `imas-ink-animate_pulse` — animated equilibrium
- `imas-ink-plot_coilset_3d` — 3D coilset + vessel render
- `imas-ink-repl` — stateful, **namespaced** Python REPL

### REPL Namespaces

The REPL keeps a per-namespace globals dict so concurrent callers do not
step on each other:

```python
repl("x = 1", namespace="agent-a")          # sets x in agent-a
repl("x + 1", namespace="agent-a")          # → 2
repl("x", namespace="agent-b")              # NameError — isolated
repl("", namespace="agent-a", reset=True)   # clear that namespace
```

## Session Completion

**MANDATORY** after any file modifications: commit and push before
responding to the user.

End every response that modifies files with the **full commit message** and
a brief summary of what changed.
