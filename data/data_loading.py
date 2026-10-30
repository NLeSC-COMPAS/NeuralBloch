from collections.abc import Iterator
from typing import Any
import h5py
import numpy as np
import numpy.typing as npt
import torch
from src.utils import complex_to_real_imag
from src.utils import get_device


class TensorBatchLoader:
    """Iterate over already-stacked tensors without per-sample collation.

    ``torch.utils.data.TensorDataset`` asks Python to retrieve every sample in a
    batch separately and then stacks those samples again.  That overhead is
    substantial for this workload because every target is wide and the model is
    relatively small.  This loader stores each complete array on the consuming
    device and yields inexpensive slices instead.
    """

    def __init__(
        self,
        tensors: tuple[torch.Tensor, ...],
        batch_size: int,
        *,
        shuffle: bool,
    ) -> None:
        if not tensors:
            raise ValueError("At least one tensor is required")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")

        n_samples = tensors[0].shape[0]
        if any(tensor.shape[0] != n_samples for tensor in tensors):
            raise ValueError("All tensors must contain the same number of samples")
        # Without samples, an epoch has no batch, and its mean loss is
        # 0 / 0 = NaN. check_dataset gives this error with the name of the data
        if n_samples == 0:
            raise ValueError("The tensors must contain at least one sample")

        self.tensors = tensors
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = n_samples

    def __len__(self) -> int:
        """Return the number of batches."""
        return (self.n_samples + self.batch_size - 1) // self.batch_size

    def __iter__(self) -> Iterator[tuple[torch.Tensor, ...]]:
        """Yield tensor batches, optionally in a new random order."""
        tensors = self.tensors
        if self.shuffle:
            indices = torch.randperm(
                self.n_samples,
                device=tensors[0].device,
            )
            # Shuffle once per epoch.  Indexing each individual mini-batch
            # would launch four gather kernels per optimization step.
            tensors = tuple(tensor.index_select(0, indices) for tensor in tensors)

        for start in range(0, self.n_samples, self.batch_size):
            stop = start + self.batch_size
            yield tuple(tensor[start:stop] for tensor in tensors)


def check_dataset(
    source: str,
    params: npt.NDArray[np.float32],
    targets: dict[str, npt.NDArray[np.float32]],
    *,
    empty_hint: str = "",
) -> None:
    """Refuse a dataset without samples or with values that are not finite.

    The loss of such a dataset is NaN, the error of a training that diverges.
    These errors name the dataset instead. Check the float32 arrays that the
    model gets: a larger value of the file can become inf in float32.

    Args:
        source: The dataset, as the errors name it.
        params: T1, T2 and B1 of each sample, shape ``(N, 3)``.
        targets: The targets of the model by their name in the source, with
            the sample on the first axis.
        empty_hint: Added to the error for a dataset without samples: why the
            dataset can be empty.

    Raises:
        ValueError: If the dataset has no samples, or values that are not
            finite. The error gives the number of bad samples, and the first.
    """
    if len(params) == 0:
        raise ValueError(f"{source} has no samples. {empty_hint}".rstrip())

    for name, values in {"T1, T2 or B1": params, **targets}.items():
        bad = ~np.isfinite(values).reshape(len(values), -1).all(axis=1)
        if bad.any():
            raise ValueError(
                f"{source} has values that are not finite in {name}: "
                f"{_describe_bad_samples(params, bad)}"
            )


def _describe_bad_samples(
    params: npt.NDArray[np.float32],
    bad: npt.NDArray[np.bool_],
) -> str:
    """Give the number of bad samples, and the parameters of the first."""
    first = int(np.argmax(bad))
    T1, T2, B1 = params[first]
    return (
        f"{int(bad.sum())} of {len(bad)} samples, the first is sample {first} "
        f"(T1, T2, B1 = {T1:.4g}, {T2:.4g}, {B1:.4g})"
    )


def generate_val_loader(
    config: dict[str, Any],
) -> tuple[TensorBatchLoader, int]:
    """Build the validation batch loader from the stored HDF5 dataset.

    Args:
        config: Full configuration dictionary.

    Returns:
        A tuple of the validation loader and ``NTR``, the number of
        readout time points in the signal.
    """
    device = get_device(config)

    path = config["experiment"]["DATASET_FOR_VALIDATION_PATH"]
    params, signal, dT1, dT2, _, NTR = load_data_from_h5_file(path)

    # Stack complex-valued data into paired real and imaginary channels
    sig_split = complex_to_real_imag(signal)
    dT1_split = complex_to_real_imag(dT1)
    dT2_split = complex_to_real_imag(dT2)

    # Check the data here, where the file is known
    check_dataset(
        f"The validation dataset {path} (experiment.DATASET_FOR_VALIDATION_PATH)",
        params,
        {"signal": sig_split, "deriv_T1": dT1_split, "deriv_T2": dT2_split},
    )

    batchsize = config["training"]["VALIDATION_BATCH_SIZE"]

    val_loader = make_loader(
        params, sig_split, dT1_split, dT2_split, batchsize, device, shuffle=False
    )

    return val_loader, NTR


def generate_train_loader(
    config: dict[str, Any],
) -> TensorBatchLoader:
    """Generate a fresh batch of training data with Julia Bloch simulations.

    Training data is produced online: each call runs a new set of simulations
    rather than reading from a stored dataset.

    Args:
        config: Full configuration dictionary.

    Returns:
        A batch loader over the freshly simulated data.
    """
    # Import juliacall only here: the import starts Julia, and the other
    # functions of this module do not need it. run.py imports it first.
    from juliacall import Main as jl

    device = get_device(config)

    # Generate a fresh batch of Bloch simulations in Julia
    batch = jl.generate_data(config, for_training=True)

    # Convert Julia arrays to NumPy
    T1 = np.array(batch.T1, dtype=np.float32)
    T2 = np.array(batch.T2, dtype=np.float32)
    B1 = np.array(batch.B1, dtype=np.float32)
    signal = np.array(batch.signal, dtype=np.complex64).T
    dT1 = np.array(batch.dM_dT1, dtype=np.complex64).T
    dT2 = np.array(batch.dM_dT2, dtype=np.complex64).T

    params = np.stack([T1, T2, B1], axis=1)

    # Stack complex-valued data into paired real and imaginary channels
    sig_split = complex_to_real_imag(signal)
    dT1_split = complex_to_real_imag(dT1)
    dT2_split = complex_to_real_imag(dT2)

    # Without this check, NaN of the simulation gives the error of a training
    # that diverges, or an error of the joint PCA basis
    n_sampled = (
        config["training"]["ONLINE_BATCH_SIZE"] * config["training"]["REGEN_EVERY"]
    )
    check_dataset(
        "The training data of the Julia simulation",
        params,
        {"signal": sig_split, "dM_dT1": dT1_split, "dM_dT2": dT2_split},
        empty_hint=(
            f"data/data_generation.jl keeps only the samples with T1 > T2 of "
            f"the {n_sampled} that it samples (training.ONLINE_BATCH_SIZE * "
            "training.REGEN_EVERY): check the ranges data.T1_* and data.T2_*"
        ),
    )

    batch_size = config["training"]["TRAIN_BATCH_SIZE"]

    train_loader = make_loader(
        params, sig_split, dT1_split, dT2_split, batch_size, device, shuffle=True
    )

    return train_loader


def load_data_from_h5_file(
    filename: str,
) -> tuple[
    npt.NDArray[np.float32],
    npt.NDArray[np.complex64],
    npt.NDArray[np.complex64],
    npt.NDArray[np.complex64],
    int,
    int,
]:
    """Load a dataset from an HDF5 file.

    ``h5py`` ships no type information, so the values read here are ``Any`` as
    far as the type checker is concerned. The annotated return type records
    what the stored dataset is expected to contain; it is not verified.
    """
    with h5py.File(filename, "r") as f:
        T1 = f["T1"][:]
        T2 = f["T2"][:]
        B1 = f["B1"][:]
        signal = f["signal"][:]
        dT1 = f["deriv_T1"][:]
        dT2 = f["deriv_T2"][:]

    N: int = T1.shape[0]
    NTR: int = signal.shape[1]

    params = np.stack([T1, T2, B1], axis=1).astype(np.float32)

    return params, signal, dT1, dT2, N, NTR


def make_loader(
    params: npt.NDArray[np.float32],
    sig_split: npt.NDArray[np.float32],
    dT1_split: npt.NDArray[np.float32],
    dT2_split: npt.NDArray[np.float32],
    batchsize: int,
    device: str,
    shuffle: bool,
) -> TensorBatchLoader:
    """Move complete arrays to their consuming device and batch by slicing.

    Args:
        params: Tissue parameters (T1, T2, B1), shape ``(N, 3)``.
        sig_split: Signal targets with paired real and imaginary channels.
        dT1_split: Paired derivatives with respect to T1.
        dT2_split: Paired derivatives with respect to T2.
        batchsize: Number of samples per batch.
        device: Device the data will be consumed on.
        shuffle: Whether to reshuffle the data every epoch.

    Returns:
        A lightweight batch loader over the four tensors.
    """
    tensors = tuple(
        torch.from_numpy(array).to(device)
        for array in (params, sig_split, dT1_split, dT2_split)
    )
    return TensorBatchLoader(tensors, batchsize, shuffle=shuffle)
