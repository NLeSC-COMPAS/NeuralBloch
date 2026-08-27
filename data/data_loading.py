from typing import Any
import h5py
import numpy as np
import numpy.typing as npt
import torch
from juliacall import Main as jl
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
from src.utils import complex_to_real_imag
from src.utils import get_device


def generate_val_loader(
    config: dict[str, Any],
) -> tuple[DataLoader[tuple[torch.Tensor, ...]], int]:
    """Build the validation DataLoader from the stored HDF5 dataset.

    Args:
        config: Full configuration dictionary.

    Returns:
        A tuple of the validation ``DataLoader`` and ``NTR``, the number of
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

    batchsize = config["training"]["TRAIN_BATCH_SIZE"]

    val_loader = make_loader(
        params, sig_split, dT1_split, dT2_split, batchsize, device, shuffle=False
    )

    return val_loader, NTR


def generate_train_loader(
    config: dict[str, Any],
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Generate a fresh batch of training data with Julia Bloch simulations.

    Training data is produced online: each call runs a new set of simulations
    rather than reading from a stored dataset.

    Args:
        config: Full configuration dictionary.

    Returns:
        A ``DataLoader`` over the freshly simulated batch.
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
) -> DataLoader[tuple[torch.Tensor, ...]]:
    """Wrap tissue parameters and signal targets in a PyTorch ``DataLoader``.

    Args:
        params: Tissue parameters (T1, T2, B1), shape ``(N, 3)``.
        sig_split: Signal targets with paired real and imaginary channels.
        dT1_split: Paired derivatives with respect to T1.
        dT2_split: Paired derivatives with respect to T2.
        batchsize: Number of samples per batch.
        device: Device the data will be consumed on.
        shuffle: Whether to reshuffle the data every epoch.

    Returns:
        A ``DataLoader`` over the four stacked tensors.
    """
    dataset = TensorDataset(
        torch.tensor(params),
        torch.tensor(sig_split),
        torch.tensor(dT1_split),
        torch.tensor(dT2_split),
    )
    # `pin_memory` is a bool. It previously received `device` directly, which is
    # a non-empty string in both branches of `get_device` and therefore always
    # truthy, so memory was pinned even on CPU.
    return DataLoader(
        dataset, batchsize, shuffle=shuffle, pin_memory=(device == "cuda")
    )
