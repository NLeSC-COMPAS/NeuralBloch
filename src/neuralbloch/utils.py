import hashlib
import random
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
import numpy.typing as npt
import torch


if TYPE_CHECKING:
    from neuralbloch.model import FCModel


def complex_to_real_imag(
    arr: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.float32]:
    """Stack a complex array into paired real and imaginary components.

    The returned array has one additional final axis ordered as ``[real, imag]``.
    """
    return np.stack((arr.real, arr.imag), axis=-1).astype(np.float32)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(config: dict[str, Any]) -> str:
    """Select computation device."""
    device: str = config["experiment"]["device"]
    return device if torch.cuda.is_available() else "cpu"


def sample_parameters(config: dict[str, Any], n_samples: int) -> torch.Tensor:
    """Sample tissue parameters uniformly over the training ranges.

    T1 and T2 are sampled log-uniformly and returned in physical units. B1 is
    sampled uniformly, matching the public model input format ``[T1, T2, B1]``.

    Args:
        config: Full configuration dictionary.
        n_samples: Number of parameter samples to draw.

    Returns:
        A float32 tensor of shape ``(n_samples, 3)``.
    """
    ranges = config["data"]["parameter_ranges"]
    T1, T2, B1_range = ranges["T1_s"], ranges["T2_s"], ranges["B1"]

    log_T1 = torch.empty(n_samples).uniform_(np.log(T1["lower"]), np.log(T1["upper"]))
    log_T2 = torch.empty(n_samples).uniform_(np.log(T2["lower"]), np.log(T2["upper"]))
    B1 = torch.empty(n_samples).uniform_(B1_range["lower"], B1_range["upper"])

    return torch.stack([log_T1.exp(), log_T2.exp(), B1], dim=1)


def load_model(
    config: dict[str, Any],
    device: str,
) -> tuple["FCModel", dict[str, Any]]:
    """Load a trained surrogate model from a checkpoint.

    Returns:
        The model (``model_from_checkpoint``) and the checkpoint.
    """
    checkpoint = load_checkpoint(config, device)
    return model_from_checkpoint(checkpoint, device), checkpoint


def load_checkpoint(config: dict[str, Any], device: str) -> dict[str, Any]:
    """Load the checkpoint file that ``experiment.pt_file_path`` gives."""
    checkpoint: dict[str, Any] = torch.load(
        config["experiment"]["pt_file_path"],
        map_location=device,
    )
    return checkpoint


def model_from_checkpoint(checkpoint: dict[str, Any], device: str) -> "FCModel":
    """Create the model of a checkpoint and load its saved weights.

    The model comes from the configuration in the checkpoint, not from
    ``config/``.
    """
    from neuralbloch.model import FCModel

    if (
        "parameter_ranges" not in checkpoint["data_config"]
        or "signal" not in checkpoint["model_config"]
    ):
        raise ValueError(
            "The checkpoint has the config keys of an earlier version of "
            "NeuralBloch. Train the model again."
        )

    NTR = checkpoint["NTR"]

    model = FCModel(NTR, checkpoint["model_config"], checkpoint["data_config"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model


def file_sha256(path: str | Path) -> str:
    """Return the sha256 of the file contents, in hexadecimal."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# The keys of the data configuration that give the RF tables of the training
# sequence (data_generation.jl)
RF_TABLE_KEYS = ("rf_angle_file", "rf_phase_file")


def rf_sha256(data_config: Mapping[str, Any]) -> dict[str, str]:
    """Return the sha256 of each RF table that a data configuration gives.

    Returns:
        The sha256 by key of the data configuration (``RF_TABLE_KEYS``).
    """
    return {key: file_sha256(data_config[key]) for key in RF_TABLE_KEYS}
