import time
from typing import Any
import torch
from data.data_loading import TensorBatchLoader
from data.data_loading import generate_train_loader
from data.data_loading import generate_val_loader
from src.model import FCModel
from src.trainer import Trainer


def initialize_joint_pca_basis(
    model: FCModel,
    train_loader: TensorBatchLoader,
) -> None:
    """Initialize the shared decoder from signal and derivative targets.

    Every target row is normalized before accumulating the covariance, matching
    the squared relative-L2 objective and giving signal, dT1, and dT2 equal
    weight. The leading eigenvectors of this covariance are the right singular
    vectors of the stacked normalized targets: for a fixed rank, their span
    minimizes squared reconstruction error over this initialization batch.
    The basis remains trainable afterward.
    """
    decoder = model.U_signal[0]
    if not isinstance(decoder, torch.nn.Linear):
        raise TypeError("Expected the first signal decoder module to be Linear")

    _, signal, dT1, dT2 = train_loader.tensors
    output_dim = signal[0].numel()
    covariance = torch.zeros(
        (output_dim, output_dim), dtype=torch.float32, device=signal.device
    )
    for target in (signal, dT1, dT2):
        rows = target.flatten(start_dim=1)
        rows = rows / torch.linalg.vector_norm(rows, dim=1, keepdim=True).clamp_min(
            1e-4
        )
        covariance.addmm_(rows.T, rows)

    _, eigenvectors = torch.linalg.eigh(covariance)
    basis = eigenvectors[:, -decoder.in_features :]
    with torch.no_grad():
        decoder.weight.copy_(basis)


def train(config: dict[str, Any]) -> None:
    """Build the model and run the training loop.

    Args:
        config: Full configuration dictionary.
    """
    # Create validation DataLoader and determine the number of time points
    val_loader, NTR = generate_val_loader(config)

    # Initialize the model
    model = FCModel(NTR, config["model"], config["data"])

    # Load Julia data generation functions (used for online training data).
    # Import juliacall only here, because the import starts Julia.
    from juliacall import Main as jl

    jl.include("data/data_generation.jl")

    initial_train_loader = None
    basis_initialization_seconds = 0.0
    basis_mode = config["model"]["basis_initialization"]
    if basis_mode == "joint_pca":
        if model.derivative_mode != "derived":
            raise ValueError(
                "joint_pca basis initialization requires derivative_mode='derived'"
            )
        started = time.perf_counter()
        initial_train_loader = generate_train_loader(config)
        initialize_joint_pca_basis(model, initial_train_loader)
        basis_initialization_seconds = time.perf_counter() - started
        print(
            "  Initialized trainable joint PCA basis in "
            f"{basis_initialization_seconds:.2f} s"
        )
    elif basis_mode != "learned":
        raise ValueError(
            "model.basis_initialization must be 'learned' or 'joint_pca', got "
            f"{basis_mode!r}"
        )

    # Print the number of trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Create the trainer
    trainer = Trainer(model, val_loader, config)

    # Train the model
    trainer.train(initial_train_loader)
