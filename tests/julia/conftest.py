"""Fixtures of the tests that run Julia.

These tests run Julia in a new Python process that imports juliacall before
torch, as run.py does. The pytest process has already imported torch.
"""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
import pytest
import yaml
from src.config import load_config


@pytest.fixture
def julia_config_dir(tmp_path: Path, pytestconfig: pytest.Config) -> Path:
    """A copy of config/ for a small data generation, in tmp_path.

    The Julia code reads the configuration keys by name, so the copy keeps all
    keys and values of config/. Only the sizes are smaller, and the datasets
    and the outputs are in tmp_path. The RF tables stay the tables of the
    repository, with paths relative to the repository root.
    """
    config = load_config(pytestconfig.rootpath / "config")
    config["experiment"].update(
        DATASET_UNSPLIT_PATH=str(tmp_path / "dataset_unsplit.h5"),
        DATASET_FOR_VALIDATION_PATH=str(tmp_path / "dataset_for_validation.h5"),
        DATASET_FOR_TESTING_PATH=str(tmp_path / "dataset_for_testing.h5"),
        PT_FILE_PATH=str(tmp_path / "model.pt"),
        LOG_FILE_PATH=str(tmp_path / "log"),
    )
    # A grid of 5 values of T1, T2 and B1
    config["data"]["N_SAMPLES"] = 125
    # 100 samples of the online training data
    config["training"].update(ONLINE_BATCH_SIZE=50, REGEN_EVERY=2)

    directory = tmp_path / "config"
    directory.mkdir()
    for name, section in config.items():
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(section))
    return directory


@pytest.fixture
def run_python(
    pytestconfig: pytest.Config,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a function that runs Python in the repository root.

    The function takes the arguments of the Python command (a script, or -c
    and the code) and fails the test with the output of a failed run.

    juliapkg finds the juliapkg.json of the repository only in sys.path. With
    -c, or with a script in the repository root, sys.path has the root. With a
    script in another directory, juliapkg changes the Julia environment of the
    virtual environment to one without BlochSimulators.
    """

    def run(*args: str) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, *args],
            cwd=pytestconfig.rootpath,
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-5000:]
        return result

    return run
