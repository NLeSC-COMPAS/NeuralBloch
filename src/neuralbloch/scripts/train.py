import time
import torch
from neuralbloch.config_schema import Config
from neuralbloch.data import DATA_GENERATION_JL
from neuralbloch.data.data_loading import TensorBatchLoader
from neuralbloch.data.data_loading import generate_train_loader
from neuralbloch.data.data_loading import generate_val_loader
from neuralbloch.model import FCModel
from neuralbloch.trainer import Trainer
from neuralbloch.utils import get_device


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


def train(config: Config) -> None:
    """Build the model and run the training loop.

    Args:
        config: The validated configuration.
    """
    device = get_device(config.experiment.device)
    # Create validation DataLoader and determine the number of time points
    val_loader, NTR = generate_val_loader(
        config.experiment.dataset_for_validation_path,
        config.training.validation_batch_size,
        device,
    )

    # Initialize the model
    model = FCModel(NTR, config.model, config.data.parameter_ranges)

    # Load Julia data generation functions (used for online training data).
    # Import juliacall only here, because the import starts Julia.
    from juliacall import Main as jl

    jl.include(DATA_GENERATION_JL)

    # Julia reads a plain dict with string paths
    julia_config = config.model_dump(mode="json")

    initial_train_loader = None
    basis_initialization_seconds = 0.0
    if model.basis_initialization == "joint_pca":
        started = time.perf_counter()
        initial_train_loader = generate_train_loader(
            julia_config, config.training.train_batch_size, device
        )
        initialize_joint_pca_basis(model, initial_train_loader)
        basis_initialization_seconds = time.perf_counter() - started
        print(
            "  Initialized trainable joint PCA basis in "
            f"{basis_initialization_seconds:.2f} s"
        )

    # Print the number of trainable parameters
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {n_params:,}")

    # Create the trainer
    trainer = Trainer(
        model,
        val_loader,
        config.training,
        config.experiment,
        config.data,
        config.model,
        julia_config,
    )

    # Train the model
    trainer.train(initial_train_loader)
