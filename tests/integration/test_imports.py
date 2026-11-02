import subprocess
import sys
from pathlib import Path
import pytest


def test_no_module_starts_julia_at_import(pytestconfig: pytest.Config) -> None:
    # The import of juliacall starts Julia, and installs it at the first use.
    # Only run.py imports it at module level, before torch.
    root: Path = pytestconfig.rootpath
    modules = sorted(
        ".".join(path.relative_to(root).with_suffix("").parts)
        for directory in ("src", "scripts", "data")
        for path in (root / directory).glob("*.py")
    )
    assert "data.data_loading" in modules
    assert "scripts.train" in modules
    # A new process, because another test of this run can import juliacall.
    # There, an import of juliacall raises ImportError, and the traceback
    # names the module that imports it.
    code = (
        "import importlib, sys\n"
        "sys.modules['juliacall'] = None\n"
        f"for name in {modules!r}:\n"
        "    importlib.import_module(name)\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
