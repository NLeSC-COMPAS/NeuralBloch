import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
import numpy as np
import pytest
import torch
from data.data_loading import TensorBatchLoader
from data.data_loading import generate_train_loader
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


@pytest.fixture
def julia_batch(
    config: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> SimpleNamespace:
    """Put a fake juliacall module in sys.modules; return its batch.

    generate_data of the fake gives this batch: 6 samples with T1 > T2 and
    NTR = 5, with the targets in the layout of the Julia arrays, (NTR, N). A
    test can change the batch. config gets ONLINE_BATCH_SIZE, a key of the
    Julia data generation.
    """
    rng = np.random.default_rng(0)
    batch = SimpleNamespace(
        T1=rng.uniform(1.0, 2.0, 6),
        T2=rng.uniform(0.05, 0.5, 6),
        B1=rng.uniform(0.5, 1.5, 6),
        **{
            key: rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))
            for key in ("signal", "dM_dT1", "dM_dT2")
        },
    )
    juliacall = ModuleType("juliacall")
    juliacall.__dict__["Main"] = SimpleNamespace(
        generate_data=lambda config, for_training: batch
    )
    monkeypatch.setitem(sys.modules, "juliacall", juliacall)
    config["training"]["ONLINE_BATCH_SIZE"] = 2
    return batch


def test_a_loader_without_samples_is_refused() -> None:
    # Its epochs have no batch, and their mean loss is 0 / 0 = NaN
    tensors = (torch.zeros(0, 3), *(torch.zeros(0, 5, 2) for _ in range(3)))

    with pytest.raises(ValueError, match="at least one sample"):
        TensorBatchLoader(tensors, 4, shuffle=False)


def test_a_loader_of_one_sample_is_accepted() -> None:
    tensors = (torch.zeros(1, 3), *(torch.zeros(1, 5, 2) for _ in range(3)))

    loader = TensorBatchLoader(tensors, 4, shuffle=False)

    assert len(loader) == 1


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


def test_a_validation_dataset_without_samples_is_refused(
    config: dict[str, Any],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> None:
    # split_dataset writes it for a dataset of 1 sample
    path = tmp_path / "validation.h5"
    targets = np.zeros((0, 5), dtype=np.complex64)
    write_dataset(path, np.zeros((0, 3)), targets, targets, targets)
    config["experiment"]["DATASET_FOR_VALIDATION_PATH"] = str(path)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The validation dataset {path} "
            "(experiment.DATASET_FOR_VALIDATION_PATH) has no samples"
        ),
    ):
        generate_val_loader(config)


@pytest.mark.parametrize(
    ("key", "name", "value"),
    [
        ("params", "T1, T2 or B1", math.nan),
        ("signal", "signal", complex(math.nan, 0.0)),
        ("dT1", "deriv_T1", complex(0.0, math.inf)),
        ("dT2", "deriv_T2", complex(-math.inf, 0.0)),
    ],
)
def test_a_validation_dataset_with_values_that_are_not_finite_is_refused(
    key: str,
    name: str,
    value: complex,
    config: dict[str, Any],
    validation_arrays: dict[str, np.ndarray],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> None:
    # These values give a NaN validation loss, and then the error of a
    # training that diverges. The error must name the file and the array
    path = tmp_path / "validation.h5"
    validation_arrays[key][2, 1] = value
    write_dataset(path, **validation_arrays)
    config["experiment"]["DATASET_FOR_VALIDATION_PATH"] = str(path)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The validation dataset {path} "
            "(experiment.DATASET_FOR_VALIDATION_PATH) has values that are not "
            f"finite in {name}: 1 of 6 samples, the first is sample 2 (T1, T2, B1"
        ),
    ):
        generate_val_loader(config)


@pytest.mark.parametrize(("column", "value"), [(0, 0.0), (1, -0.1)])
def test_a_validation_dataset_with_t1_or_t2_that_are_not_positive_is_refused(
    column: int,
    value: float,
    config: dict[str, Any],
    validation_arrays: dict[str, np.ndarray],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> None:
    # The model takes the logarithm of T1 and T2: these finite values give a
    # NaN validation loss
    path = tmp_path / "validation.h5"
    validation_arrays["params"][4, column] = value
    write_dataset(path, **validation_arrays)
    config["experiment"]["DATASET_FOR_VALIDATION_PATH"] = str(path)

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"The validation dataset {path} "
            "(experiment.DATASET_FOR_VALIDATION_PATH) has values of T1 or T2 "
            "that are not positive: 1 of 6 samples, the first is sample 4"
        ),
    ):
        generate_val_loader(config)


def test_finite_julia_data_are_loaded(
    config: dict[str, Any],
    julia_batch: SimpleNamespace,
) -> None:
    loader = generate_train_loader(config)

    assert loader.n_samples == 6


def test_julia_data_without_samples_are_refused(
    config: dict[str, Any],
    julia_batch: SimpleNamespace,
) -> None:
    # The filter T1 > T2 of data/data_generation.jl can remove all samples
    julia_batch.T1 = julia_batch.T2 = julia_batch.B1 = np.zeros(0)
    for key in ("signal", "dM_dT1", "dM_dT2"):
        setattr(julia_batch, key, np.zeros((5, 0), dtype=np.complex64))

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The training data of the Julia simulation has no samples. "
            "data/data_generation.jl keeps only the samples with T1 > T2 of the "
            "200 that it samples"
        ),
    ):
        generate_train_loader(config)


@pytest.mark.parametrize(
    ("key", "name"),
    [
        ("T2", "T1, T2 or B1"),
        ("signal", "signal"),
        ("dM_dT1", "dM_dT1"),
        ("dM_dT2", "dM_dT2"),
    ],
)
def test_julia_data_with_values_that_are_not_finite_are_refused(
    key: str,
    name: str,
    config: dict[str, Any],
    julia_batch: SimpleNamespace,
) -> None:
    # Without the check, they give the error of a training that diverges
    getattr(julia_batch, key)[..., 3] = math.nan  # sample 3

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The training data of the Julia simulation has values that are not "
            f"finite in {name}: 1 of 6 samples, the first is sample 3 (T1, T2, B1"
        ),
    ):
        generate_train_loader(config)
