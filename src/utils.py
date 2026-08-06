import random
from typing import TYPE_CHECKING
from typing import Any
import numpy as np
import numpy.typing as npt
import torch


if TYPE_CHECKING:
    from src.model import FCModel


def split_real_imag(
    arr: npt.NDArray[np.complexfloating],
) -> npt.NDArray[np.float32]:
    """Convert complex-valued arrays into real-valued arrays."""
    return np.concatenate([arr.real, arr.imag], axis=1).astype(np.float32)


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


def load_model(
    config: dict[str, Any],
    device: str,
) -> tuple["FCModel", dict[str, Any]]:
    """Load a trained surrogate model from a checkpoint.

    The model architecture is created from the configuration file,
    after which the saved weights are loaded.
    """
    checkpoint: dict[str, Any] = torch.load(
        config["experiment"]["PT_FILE_PATH"],
        map_location=device,
    )

    from src.model import FCModel

    NTR = checkpoint["NTR"]

    model = FCModel(NTR, config["model"])
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()

    return model, checkpoint
