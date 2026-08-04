import random
import numpy as np
import torch


def split_real_imag(arr):
    """Convert complex-valued arrays into real-valued arrays."""
    return np.concatenate([arr.real, arr.imag], axis=1).astype(np.float32)


def set_seed(seed):
    """Set random seeds for reproducible experiments."""
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(config):
    """Select computation device."""
    device = config["experiment"]["DEVICE"]
    return device if torch.cuda.is_available() else "cpu"


def load_model(config, device):
    """Load a trained surrogate model from a checkpoint.

    The model architecture is created from the configuration file,
    after which the saved weights are loaded.
    """
    checkpoint = torch.load(
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
