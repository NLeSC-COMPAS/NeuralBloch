from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import pytest
import torch
from scripts.orthogonalize_basis import load_checkpoint
from scripts.orthogonalize_basis import postprocess_basis
from src.basis import worst_relative_output_difference
from src.model import FCModel


FIT_PARAMS = torch.tensor(
    [[0.4, 0.04, 0.8], [0.8, 0.07, 1.0], [1.2, 0.1, 0.9], [2.0, 0.2, 1.1]]
)
HELD_OUT = torch.tensor([[0.55, 0.05, 1.15], [3.0, 0.3, 0.85]])


@pytest.fixture
def model_config(derived_config: dict[str, Any]) -> dict[str, Any]:
    """A model of rank 4."""
    return {**derived_config, "NB_signal": 4, "hidden_size_signal": 8}


@pytest.fixture
def model(
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    make_model: Callable[..., FCModel],
) -> FCModel:
    torch.manual_seed(42)
    return make_model(5, model_config, data_config).eval()


def test_command_writes_new_checkpoints_and_keeps_input(
    model_config: dict[str, Any],
    data_config: dict[str, Any],
    model: FCModel,
    write_checkpoint: Callable[..., None],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> None:
    checkpoint_path = tmp_path / "trained.pt"
    dataset_path = tmp_path / "validation.h5"
    output_path = tmp_path / "ordered.pt"
    write_checkpoint(checkpoint_path, model, model_config, data_config, epoch=10)
    original_bytes = checkpoint_path.read_bytes()
    zeros = np.zeros((len(FIT_PARAMS), 5), dtype=np.complex64)
    write_dataset(dataset_path, FIT_PARAMS.numpy(), zeros, zeros, zeros)

    written = postprocess_basis(checkpoint_path, dataset_path, output_path, [2], "cpu")
    assert written == [output_path, tmp_path / "ordered_r2.pt"]
    assert checkpoint_path.read_bytes() == original_bytes
    ordered, _ = load_checkpoint(output_path, "cpu")
    smaller, saved = load_checkpoint(tmp_path / "ordered_r2.pt", "cpu")
    assert worst_relative_output_difference(model, ordered, HELD_OUT) < 1e-5
    assert smaller.U_signal[0].in_features == 2
    assert saved["model_config"]["NB_signal"] == 2

    with pytest.raises(ValueError, match="must differ from --checkpoint"):
        postprocess_basis(checkpoint_path, dataset_path, checkpoint_path, [], "cpu")
