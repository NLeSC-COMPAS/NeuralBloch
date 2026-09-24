from typing import Any
from juliacall import Main as jl
from data.data_loading import generate_val_loader
from src.model import FCModel
from src.trainer import Trainer


def train(config: dict[str, Any]) -> None:
    """Build the model and run the training loop.

    Args:
        config: Full configuration dictionary.
    """
    # Create validation DataLoader and determine the number of time points
    val_loader, NTR = generate_val_loader(config)

    # Initialize the model
    model = FCModel(NTR, config["model"], config["data"])

    # Print the number of trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Create the trainer
    trainer = Trainer(model, val_loader, config)

    # Load Julia data generation functions (used for online training data)
    jl.include("data/data_generation.jl")

    # Train the model
    trainer.train()
