"""The pipeline from the training to the model bundle.

The other export tests write the checkpoint with write_checkpoint. This test
exports the checkpoint that the training writes, so a change of the checkpoint
on one side only fails it. Only Julia and git are mocks (julia_batch_mock,
git_and_julia_in_export_mock).
"""

from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import pytest
import torch
from neuralbloch.utils import file_sha256
from scripts.export_bundle import check_bundle
from scripts.export_bundle import export_bundle
from scripts.export_onnx import export_onnx
from scripts.train import train


@pytest.fixture
def pipeline_config(
    config: dict[str, Any],
    export_config: dict[str, Any],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> dict[str, Any]:
    """config with a validation file of 8 samples (NTR = 5) and the export."""
    rng = np.random.default_rng(1)
    params = np.column_stack(
        [
            rng.uniform(1.0, 2.0, 8),
            rng.uniform(0.05, 0.5, 8),
            rng.uniform(0.5, 1.5, 8),
        ]
    )
    targets = (
        rng.standard_normal((8, 5)) + 1j * rng.standard_normal((8, 5)) for _ in range(3)
    )
    path = tmp_path / "validation.h5"
    write_dataset(path, params, *targets)
    config["experiment"]["DATASET_FOR_VALIDATION_PATH"] = str(path)
    config["model"]["basis_initialization"] = "learned"
    config["export"] = export_config
    return config


@pytest.mark.usefixtures("julia_batch_mock", "git_and_julia_in_export_mock")
def test_the_bundle_describes_the_checkpoint_of_the_training(
    pipeline_config: dict[str, Any],
    capsys: pytest.CaptureFixture[str],
) -> None:
    config = pipeline_config

    # The tasks of run.py: train, export_onnx and export_bundle
    train(config)
    export_onnx(config)
    export_bundle(config)

    card = check_bundle(config["export"]["BUNDLE_DIR"])
    model = card["model"]
    assert (model["family"], model["version"], model["ntr"], model["rank"]) == (
        "fisp5-rank3",
        "0.1.0",
        5,
        3,
    )
    checkpoint = torch.load(config["experiment"]["PT_FILE_PATH"])
    assert card["provenance"]["checkpoint_epoch"] == checkpoint["epoch"]
    for target, paths in config["export"]["TARGETS"].items():
        assert model[target]["sha256"] == file_sha256(paths["ONNX_FILE_PATH"])
    # export_bundle goes on with a warning for a checkpoint without rf_sha256
    assert "WARNING" not in capsys.readouterr().out
