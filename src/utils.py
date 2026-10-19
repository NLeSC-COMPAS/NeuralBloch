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
    from src.model import FCModel


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
    device: str = config["experiment"]["DEVICE"]
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
    data = config["data"]

    log_T1 = torch.empty(n_samples).uniform_(
        np.log(data["T1_lower"]), np.log(data["T1_upper"])
    )
    log_T2 = torch.empty(n_samples).uniform_(
        np.log(data["T2_lower"]), np.log(data["T2_upper"])
    )
    B1 = torch.empty(n_samples).uniform_(data["B1_lower"], data["B1_upper"])

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
    return model_from_checkpoint(checkpoint, config, device), checkpoint


def load_checkpoint(config: dict[str, Any], device: str) -> dict[str, Any]:
    """Load the checkpoint file that ``experiment.PT_FILE_PATH`` gives."""
    checkpoint: dict[str, Any] = torch.load(
        config["experiment"]["PT_FILE_PATH"],
        map_location=device,
    )
    return checkpoint


def model_from_checkpoint(
    checkpoint: dict[str, Any],
    config: dict[str, Any],
    device: str,
) -> "FCModel":
    """Create the model of a checkpoint and load its saved weights.

    The model architecture comes from the configurations in the checkpoint.
    Old checkpoints do not contain them. For these checkpoints, the function
    uses the configurations in ``config``.
    """
    from src.model import FCModel

    NTR = checkpoint["NTR"]

    model_config = checkpoint.get("model_config", config["model"])
    data_config = checkpoint.get("data_config", config["data"])
    model = FCModel(NTR, model_config, data_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model


def file_sha256(path: str | Path) -> str:
    """Return the sha256 of the file contents, in hexadecimal."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


# The keys of the data configuration that give the RF tables of the training
# sequence (data/data_generation.jl)
RF_TABLE_KEYS = ("RF_ANGLE_FILE", "RF_PHASE_FILE")


def rf_sha256(data_config: Mapping[str, Any]) -> dict[str, str]:
    """Return the sha256 of each RF table that a data configuration gives.

    Returns:
        The sha256 by key of the data configuration (``RF_TABLE_KEYS``).
    """
    return {key: file_sha256(data_config[key]) for key in RF_TABLE_KEYS}
