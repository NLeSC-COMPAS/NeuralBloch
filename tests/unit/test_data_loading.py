import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any
import numpy as np
import pytest
import torch
from data.data_loading import TensorBatchLoader
from data.data_loading import generate_val_loader


@pytest.fixture
def validation_arrays() -> dict[str, np.ndarray]:
    """The arguments of write_dataset for a dataset of 6 samples, NTR = 5."""
    rng = np.random.default_rng(0)
    return {
        "params": rng.uniform(0.5, 1.5, (6, 3)),
        **{
            key: rng.standard_normal((6, 5)) + 1j * rng.standard_normal((6, 5))
            for key in ("signal", "dT1", "dT2")
        },
    }


def test_a_loader_without_samples_is_refused() -> None:
    # Its epochs have no step, and their mean loss is 0 / 0
    tensors = (torch.zeros(0, 3), *(torch.zeros(0, 5, 2) for _ in range(3)))

    with pytest.raises(ValueError, match="at least one sample"):
        TensorBatchLoader(tensors, 4, shuffle=False)


def test_a_finite_validation_dataset_is_loaded(
    config: dict[str, Any],
    validation_arrays: dict[str, np.ndarray],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.h5"
    write_dataset(path, **validation_arrays)
    config["experiment"]["DATASET_FOR_VALIDATION_PATH"] = str(path)

    loader, NTR = generate_val_loader(config)

    assert loader.n_samples == 6
    assert NTR == 5


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("params", math.nan),
        ("signal", complex(math.nan, 0.0)),
        ("dT1", complex(0.0, math.inf)),
        ("dT2", complex(-math.inf, 0.0)),
    ],
)
def test_a_validation_dataset_with_values_that_are_not_finite_is_refused(
    key: str,
    value: complex,
    config: dict[str, Any],
    validation_arrays: dict[str, np.ndarray],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> None:
    # They give a validation loss that is not finite, the error of a training
    # that diverges. The error must name the file instead
    path = tmp_path / "validation.h5"
    validation_arrays[key][2, 1] = value
    write_dataset(path, **validation_arrays)
    config["experiment"]["DATASET_FOR_VALIDATION_PATH"] = str(path)

    with pytest.raises(ValueError, match=rf"{re.escape(str(path))}.*not finite"):
        generate_val_loader(config)
