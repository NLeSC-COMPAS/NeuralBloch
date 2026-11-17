# Development

This file tells you how to set up the development environment of this repository.
See [README.md](README.md) for what the project does and how to run it.

## Environment setup

The project needs Python 3.12 or later. [uv](https://docs.astral.sh/uv/) manages the Python dependencies. To install them, use this command:

```bash
uv sync
```

This command installs the runtime dependencies and the `dev` group. To skip the development tools, use `uv sync --no-dev`.

## Julia dependencies

The Bloch simulations use Julia through [JuliaCall](https://juliapy.github.io/PythonCall.jl/stable/juliacall/). At the first use, JuliaCall installs Julia and the Julia packages of `src/neuralbloch/juliapkg.json`. The simulations need an NVIDIA GPU with CUDA.

## Code style

Linting and formatting are handled by [Ruff](https://docs.astral.sh/ruff/), configured under `[tool.ruff]` in `pyproject.toml`.

```bash
uv run ruff check .    # lint
uv run ruff format .   # format
```

## Type checking

[ty](https://docs.astral.sh/ty/) does the type checking. The settings are in `[tool.ty]` in `pyproject.toml`.

```bash
uv run ty check
```

## Tests

The tests use [pytest](https://docs.pytest.org/). The settings are in `[tool.pytest]` in `pyproject.toml`. The tests are in two directories:

- `tests/unit/`: Each test checks one function or one class, with small inputs.
- `tests/integration/`: Each test checks modules together, or a full command. It can use real external components.

In new tests, use the fixtures of `tests/conftest.py`. Write new tests as pytest functions, not as `unittest.TestCase` classes.

To run the tests, use these commands:

```bash
uv run pytest                             # run all the tests
uv run pytest tests/unit                  # run only the unit tests, during development
uv run pytest tests/unit/test_bundle.py   # run the tests in one file
uv run pytest -k rf_table                 # run the tests that have "rf_table" in their names
uv run pytest -x                          # stop after the first test that fails
uv run pytest --lf                        # run only the tests that failed the last time
uv run pytest --cov                       # also show the coverage
```

### GPU and Julia

A test that needs a CUDA GPU has the marker `@has_gpu` (from `tests/helpers.py`). Without a GPU, pytest skips it. To skip these tests on a machine with a GPU, set `CUDA_VISIBLE_DEVICES=""`.

The import of `juliacall` starts Julia. If torch loads before `juliacall`, Julia can crash ([pytorch#78829](https://github.com/pytorch/pytorch/issues/78829)). Obey these rules:

- In `neuralbloch.cli`, do not import `juliacall` at module level. Do not import torch, or a module that loads torch, at module level. Each command calls `_start()` first, and `_start()` imports `juliacall`. Then the command imports the modules that load torch.
- In the other modules, do not import `juliacall` at module level. Import it in the function that runs Julia.
- A test without `@has_gpu` must not run Julia. To replace Julia, use the fixture `julia_batch_mock` or `git_and_julia_in_export_mock`.


## Git hooks

[prek](https://prek.j178.dev/) manages the git hooks. `prek.toml` defines them. Install the hooks one time for each clone:

```bash
uv run prek install
```

When you commit, all hooks except `ty` run. When you push, all hooks run, including `ty`. The hooks do not run the tests.


> [!IMPORTANT]
> You must run `uv run prek install` to ensure the git hooks work automatically as expected.


You can also run the hooks manually:

```bash
uv run prek run --all-files                    # all hooks except ty
uv run prek run --all-files --stage pre-push   # all hooks, including ty
```

## Documentation

The documentation uses [MkDocs](https://www.mkdocs.org/). To install the tools of the `docs` group, use this command:

```bash
uv sync --group docs
```

## Continuous integration

`.github/workflows/ci.yml` has two jobs:

- `lint` runs the hooks of `prek.toml`, via the [`j178/prek-action`](https://github.com/j178/prek-action) action.
- `test` runs the unit and integration tests with coverage, on Python 3.12, 3.13, and 3.14. The runners have no GPU, so the Julia tests are skipped.
