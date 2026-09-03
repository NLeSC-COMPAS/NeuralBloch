from collections.abc import Iterator
from typing import Any
import h5py
import numpy as np
import numpy.typing as npt
import torch
from juliacall import Main as jl
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

    params, signal, dT1, dT2, _, NTR = load_data_from_h5_file(
        config["experiment"]["DATASET_FOR_VALIDATION_PATH"]
    )

    # Stack complex-valued data into paired real and imaginary channels
    sig_split = complex_to_real_imag(signal)
    dT1_split = complex_to_real_imag(dT1)
    dT2_split = complex_to_real_imag(dT2)

    batchsize = config["training"].get(
        "VALIDATION_BATCH_SIZE",
        config["training"]["TRAIN_BATCH_SIZE"],
    )

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
