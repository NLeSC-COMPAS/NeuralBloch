import math
import re
import sys
from collections.abc import Callable
from pathlib import Path
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


def indexed_tensors(n_samples: int) -> tuple[torch.Tensor, ...]:
    """The four tensors of a loader, NTR = 5: each value is its sample index."""
    index = torch.arange(n_samples, dtype=torch.float32)
    return (
        index[:, None].repeat(1, 3),
        *(index[:, None, None].repeat(1, 5, 2) for _ in range(3)),
    )


def sample_indices(batch: tuple[torch.Tensor, ...]) -> list[list[int]]:
    """The sample index of each row, for each tensor of a batch."""
    return [tensor.flatten(start_dim=1)[:, 0].long().tolist() for tensor in batch]


def test_a_loader_without_samples_is_refused() -> None:
    # Its epochs have no batch, and their mean loss is 0 / 0 = NaN
    tensors = (torch.zeros(0, 3), *(torch.zeros(0, 5, 2) for _ in range(3)))

    with pytest.raises(ValueError, match="at least one sample"):
        TensorBatchLoader(tensors, 4, shuffle=False)


def test_a_loader_of_one_sample_is_accepted() -> None:
    tensors = (torch.zeros(1, 3), *(torch.zeros(1, 5, 2) for _ in range(3)))

    loader = TensorBatchLoader(tensors, 4, shuffle=False)

    assert len(loader) == 1


def test_a_loader_gives_the_samples_in_order_with_a_smaller_last_batch() -> None:
    loader = TensorBatchLoader(indexed_tensors(10), 4, shuffle=False)

    batches = [sample_indices(batch) for batch in loader]

    assert len(loader) == len(batches) == 3
    assert batches == [
        [indices] * 4 for indices in ([0, 1, 2, 3], [4, 5, 6, 7], [8, 9])
    ]


@pytest.mark.parametrize(
    ("tensors", "batch_size", "error"),
    [
        ((), 4, "At least one tensor is required"),
        (indexed_tensors(10), 0, "batch_size must be positive"),
        (
            (torch.zeros(10, 3), torch.zeros(9, 5, 2)),
            4,
            "All tensors must contain the same number of samples",
        ),
    ],
)
def test_a_loader_refuses_tensors_that_do_not_give_batches(
    tensors: tuple[torch.Tensor, ...],
    batch_size: int,
    error: str,
) -> None:
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
        TensorBatchLoader(tensors, batch_size, shuffle=False)


def test_the_shuffle_gives_each_epoch_a_new_order_of_all_samples() -> None:
    torch.manual_seed(0)
    loader = TensorBatchLoader(indexed_tensors(20), 8, shuffle=True)

    orders = []
    for _ in range(2):
        epoch = [sample_indices(batch) for batch in loader]
        # All tensors have the same order. Another order for each tensor
        # gives each input the targets of another sample, without an error
        for batch in epoch:
            assert all(indices == batch[0] for indices in batch)
        orders.append([index for batch in epoch for index in batch[0]])

    for order in orders:
        assert sorted(order) == list(range(20))
    assert orders[0] != list(range(20))
    assert orders[0] != orders[1]


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


def test_the_validation_loader_has_the_samples_of_the_file_in_order(
    config: dict[str, Any],
    validation_arrays: dict[str, np.ndarray],
    write_dataset: Callable[..., None],
    tmp_path: Path,
) -> None:
    path = tmp_path / "validation.h5"
    write_dataset(path, **validation_arrays)
    config["experiment"]["DATASET_FOR_VALIDATION_PATH"] = str(path)

    loader, _ = generate_val_loader(config)

    params, *targets = (tensor.numpy() for tensor in loader.tensors)
    np.testing.assert_array_equal(
        params, validation_arrays["params"].astype(np.float32)
    )
    # The complex values become pairs [real, imag] on a last axis
    for target, key in zip(targets, ("signal", "dT1", "dT2"), strict=True):
        values = validation_arrays[key].astype(np.complex64)
        np.testing.assert_array_equal(
            target, np.stack([values.real, values.imag], axis=-1)
        )
    assert not loader.shuffle
    assert loader.batch_size == config["training"]["VALIDATION_BATCH_SIZE"]


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
    julia_batch_mock: SimpleNamespace,
) -> None:
    loader = generate_train_loader(config)

    assert loader.n_samples == 6


def test_the_julia_arrays_become_the_samples_of_the_training_loader(
    config: dict[str, Any],
    julia_batch_mock: SimpleNamespace,
) -> None:
    loader = generate_train_loader(config)

    # New samples of the online training, not the grid of a dataset
    jl = sys.modules["juliacall"].Main
    jl.generate_data.assert_called_once_with(config, for_training=True)
    params, *targets = (tensor.numpy() for tensor in loader.tensors)
    np.testing.assert_array_equal(
        params,
        np.column_stack(
            [julia_batch_mock.T1, julia_batch_mock.T2, julia_batch_mock.B1]
        ).astype(np.float32),
    )
    # The Julia arrays are (NTR, N): sample i, readout t of a target is
    # [real, imag] of the array at [t, i]
    for target, key in zip(targets, ("signal", "dM_dT1", "dM_dT2"), strict=True):
        values = getattr(julia_batch_mock, key).astype(np.complex64).T
        np.testing.assert_array_equal(
            target, np.stack([values.real, values.imag], axis=-1)
        )
    assert loader.shuffle
    assert loader.batch_size == config["training"]["TRAIN_BATCH_SIZE"]


def test_julia_data_without_samples_are_refused(
    config: dict[str, Any],
    julia_batch_mock: SimpleNamespace,
) -> None:
    # data/data_generation.jl raises an error before it gives 0 samples (a
    # test with the marker julia checks it). This check is for the data that
    # Python gets
    julia_batch_mock.T1 = julia_batch_mock.T2 = julia_batch_mock.B1 = np.zeros(0)
    for key in ("signal", "dM_dT1", "dM_dT2"):
        setattr(julia_batch_mock, key, np.zeros((5, 0), dtype=np.complex64))

    error = "The training data of the Julia simulation has no samples."
    with pytest.raises(ValueError, match=f"^{re.escape(error)}$"):
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
    julia_batch_mock: SimpleNamespace,
) -> None:
    # Without the check, they give the error of a training that diverges
    getattr(julia_batch_mock, key)[..., 3] = math.nan  # sample 3

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The training data of the Julia simulation has values that are not "
            f"finite in {name}: 1 of 6 samples, the first is sample 3 (T1, T2, B1"
        ),
    ):
        generate_train_loader(config)
