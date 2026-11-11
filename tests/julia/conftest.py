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
from neuralbloch.config import load_config


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
        dataset_unsplit_path=str(tmp_path / "dataset_unsplit.h5"),
        dataset_for_validation_path=str(tmp_path / "dataset_for_validation.h5"),
        dataset_for_testing_path=str(tmp_path / "dataset_for_testing.h5"),
        pt_file_path=str(tmp_path / "model.pt"),
        log_dir=str(tmp_path / "log"),
    )
    # A grid of 5 values of T1, T2 and B1
    config["data"]["n_samples"] = 125
    # 100 samples of the online training data
    config["training"].update(online_samples_per_epoch=50, regen_every_n_epochs=2)

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
    and the code) and fails the test with the output of a failed run. With
    succeeds=False, it fails the test with the output of a run that succeeds.
    A run that takes more than 30 minutes fails: the first run in a new
    virtual environment installs the Julia packages (a few minutes), but a
    run that hangs, for example on the GPU, must not stop the tests.
    """

    def run(*args: str, succeeds: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, *args],
            cwd=pytestconfig.rootpath,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        output = result.stdout[-2000:] + result.stderr[-5000:]
        assert (result.returncode == 0) == succeeds, output
        return result

    return run
