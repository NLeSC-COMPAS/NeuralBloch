# Development

Environment setup, tooling, and code style for working on this repository.
See [README.md](README.md) for what the project does and how to run it.

## Environment setup

Python dependencies are managed with [uv](https://docs.astral.sh/uv/) and specified in `pyproject.toml`:

```bash
uv sync
```

This installs the runtime dependencies together with the `dev` group. Use `uv sync --no-dev` to skip the development tools.

## Julia dependencies

Julia is used for Bloch simulations through [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). Julia itself and the required packages are installed automatically on first use, based on `juliapkg.json`.

## Code style

Linting and formatting are handled by [Ruff](https://docs.astral.sh/ruff/), configured under `[tool.ruff]` in `pyproject.toml`.

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
```

Two conventions are worth knowing before adding rules:

- **MR-STAT notation is exempt from `pep8-naming`.** `N803`, `N806`, and `N813` are ignored so that domain names (`T1`, `T2`, `B1`, `dT1`, `dT2`, `NTR`) and the `from juliacall import Main as jl` alias keep the spelling used in the literature and in `data_generation.jl`. Class and function naming (`N801`, `N802`) is still enforced.
- **Some docstring rules are deferred, not disabled.** `D100`, `D104`, `D105`, and `D107` are ignored; docstrings on public functions, classes, and methods (`D101`, `D102`, `D103`) are required. Drop entries from the `ignore` list as the remaining docstrings get written.

## Git hooks

Git hooks are managed with [prek](https://prek.j178.dev/). The hooks are defined in `prek.toml`. Install them once per clone:

```bash
uv run prek install          # install the git hooks (one-off)
uv run prek run              # run the hooks on staged files
uv run prek run --all-files  # run the hooks on all files
uv run prek list             # show the configured hooks
```

Once installed, the hooks run automatically on every commit.

## Continuous integration

`.github/workflows/lint.yml` runs the same hooks on every push to `main` and on every pull request, via the [`j178/prek-action`](https://github.com/j178/prek-action) action. Because CI and local commits share `prek.toml`, the two cannot drift.
