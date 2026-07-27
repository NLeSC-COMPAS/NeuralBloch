from src.model import FCModel
from data.data_loading import generate_val_loader
from src.trainer import Trainer
from juliacall import Main as jl


def train(config):

    # Create validation DataLoader and determine the number of time points
    val_loader, NTR = generate_val_loader(
        config
    )

    # Initialize the model
    model = FCModel(
        NTR,
        config["model"]
    )

    # Print the number of trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Create the trainer
    trainer = Trainer(
        model,
        val_loader,
        config
    )

    # Load Julia data generation functions (used for online training data)
    jl.include("data/data_generation.jl")

    # Train the model
    trainer.train()