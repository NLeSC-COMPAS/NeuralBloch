"""Import juliacall before torch when the Julia tests run.

If torch is imported first, Julia can crash
(https://github.com/pytorch/pytorch/issues/78829). The conftest files import
torch, so pyproject.toml loads this plugin with -p, before the conftest files.
"""

import pytest
from _pytest.mark.expression import Expression


@pytest.hookimpl(tryfirst=True)
def pytest_load_initial_conftests(early_config: pytest.Config) -> None:
    # The import starts Julia, so do it only when -m selects the Julia tests
    markexpr = early_config.known_args_namespace.markexpr
    if markexpr and Expression.compile(markexpr).evaluate(
        lambda name, **_: name == "julia"
    ):
        import juliacall  # noqa: F401, PLC0415
