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

## Type checking

Type checking is handled by [ty](https://docs.astral.sh/ty/), configured under `[tool.ty]` in `pyproject.toml`. The files it checks are listed there (`tool.ty.src`), so no arguments are needed:

```bash
uv run ty check
```

## Tests

The tests use [pytest](https://docs.pytest.org/). The pytest configuration is in the `[tool.pytest]` table of `pyproject.toml`.

To run the tests, use these commands:

```bash
uv run pytest                        # run all the tests
uv run pytest tests/test_bundle.py   # run the tests in one file
uv run pytest -k rf_table            # run the tests that have "rf_table" in their names
uv run pytest -x                     # stop after the first test that fails
uv run pytest --lf                   # run only the tests that failed the last time
```

To see which lines the tests run, add `--cov`:

```bash
uv run pytest --cov   # the coverage of each file, and the lines that no test runs
```

The coverage settings are in the `[tool.coverage]` tables of `pyproject.toml`. The coverage includes the branches, and the report also shows the files that no test imports.

Before the first test, pytest starts Julia, because `data/data_loading.py` imports `juliacall`.

The git hooks and CI do not run the tests. Run the tests before you open a pull request.

Write new tests as pytest functions. Do not write them as `unittest.TestCase` classes.

## Git hooks

Git hooks are managed with [prek](https://prek.j178.dev/). The hooks are defined in `prek.toml`. Install them once per clone:

```bash
uv run prek install          # install the git hooks (one-off)
uv run prek uninstall        # uninstall the git hooks (one-off)
```

Once installed, the hooks run automatically on every commit — except ty, which runs on `git push` instead.


> [!IMPORTANT]
> You must run `uv run prek install` to ensure the git hooks work automatically as expected.


You can also run the hooks manually:

```
uv run prek run              # run the hooks on staged files (only pre-commit stage)
uv run prek run --all-files  # run the hooks on all files
```

To run ty together with the other hooks manually, ask for the stage explicitly:

```bash
uv run prek run --all-files --stage pre-push
```

## Continuous integration

`.github/workflows/lint.yml` runs the same hooks on every push to `main` and on every pull request, via the [`j178/prek-action`](https://github.com/j178/prek-action) action. Because CI and local commits share `prek.toml`, the two cannot drift.
