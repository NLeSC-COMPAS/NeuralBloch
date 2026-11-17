import subprocess
import sys
from pathlib import Path
import pytest


def test_no_module_starts_julia_at_import(pytestconfig: pytest.Config) -> None:
    # The import of juliacall starts Julia, and installs it at the first use.
    # Only neuralbloch.cli imports it, in main(), before torch.
    root: Path = pytestconfig.rootpath
    src = root / "src"
    modules = sorted(
        ".".join(path.relative_to(src).with_suffix("").parts)
        for path in (src / "neuralbloch").rglob("*.py")
        if path.stem != "__init__"
    )
    assert "neuralbloch.data.data_loading" in modules
    assert "neuralbloch.scripts.train" in modules
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
