from pathlib import Path
from typing import Annotated
from typing import Any
from cyclopts import App
from cyclopts import Parameter
from neuralbloch.config import load_config
from neuralbloch.config_schema import Config
from neuralbloch.data import DATA_GENERATION_JL


app = App(name="neuralbloch", help="Train and export the NeuralBloch surrogate.")


def main() -> None:
    """The entry point of the ``neuralbloch`` command."""
    app()


# TODO: re-use from `types`
ConfigDir = Annotated[Path, Parameter(help="Directory of the YAML config files.")]


def _start(config_dir: Path, *, seed_python: bool = True) -> tuple[Any, Config]:
    """Start Julia, load the config and seed the RNGs.

    Args:
        config_dir: Directory containing the YAML configuration files.
        seed_python: Also seed Python's RNGs with ``experiment.seed``.

    Returns:
        The Julia ``Main`` module and the validated ``Config``.
    """
    # Before Julia starts, so that a bad config fails fast
    config = load_config(config_dir)

    # juliacall must be imported before torch, and the import starts Julia.
    # So keep this import and the task imports inside the commands.
    from juliacall import Main as jl  # ruff: noqa: PLC0415

    # Online samples are drawn by Julia, so Python's RNG seed alone does not
    # make training runs reproducible.
    julia_seed = config.experiment.seed
    jl.seval(f"begin using Random; Random.seed!({julia_seed}); true end")

    if seed_python:
        from neuralbloch.utils import set_seed

        set_seed(config.experiment.seed)

    return jl, config


@app.command
def generate(*, config_dir: ConfigDir = Path("config")) -> None:
    """Generate validation and test datasets using Bloch simulations."""
    jl, config = _start(config_dir, seed_python=False)
    jl.include(DATA_GENERATION_JL)
    # Julia reads a plain dict with string paths
    jl.save_dataset(config.model_dump(mode="json"))

    from neuralbloch.data.split_dataset import split_dataset

    split_dataset(
        dataset_unsplit_path=config.experiment.dataset_unsplit_path,
        dataset_for_validation_path=config.experiment.dataset_for_validation_path,
        dataset_for_testing_path=config.experiment.dataset_for_testing_path,
        seed=config.experiment.seed,
    )


@app.command
def train(*, config_dir: ConfigDir = Path("config")) -> None:
    """Train the surrogate model."""
    jl, config = _start(config_dir)
    jl.include(DATA_GENERATION_JL)
    jl.seval("begin using BlochSimulators; CUDALibs(); true end")

    from neuralbloch.scripts.train import train

    train(config)


@app.command
def test_accuracy(*, config_dir: ConfigDir = Path("config")) -> None:
    """Evaluate prediction accuracy on the test dataset."""
    _, config = _start(config_dir)

    from neuralbloch.scripts.test_accuracy import test_accuracy

    test_accuracy(
        pt_file_path=config.experiment.pt_file_path,
        dataset_for_testing_path=config.experiment.dataset_for_testing_path,
        device=config.experiment.device,
    )


@app.command
def test_inference(*, config_dir: ConfigDir = Path("config")) -> None:
    """Benchmark surrogate inference speed."""
    _, config = _start(config_dir)

    from neuralbloch.scripts.test_inference import test_inference

    test_inference(
        pt_file_path=config.experiment.pt_file_path,
        device=config.experiment.device,
        n_runs=config.testing.n_runs,
        infer_batch_size=config.testing.infer_batch_size,
        batch_size=config.testing.batch_size,
    )


@app.command
def export_onnx(*, config_dir: ConfigDir = Path("config")) -> None:
    """Export the trained model to ONNX, one model for each target."""
    _, config = _start(config_dir)

    from neuralbloch.scripts.export_onnx import export_onnx

    export = config.export
    export_onnx(
        targets=export.targets,
        fp16=export.fp16,
        opset_version=export.opset_version,
        n_verify_samples=export.n_verify_samples,
        pt_file_path=config.experiment.pt_file_path,
    )


@app.command
def export_bundle(*, config_dir: ConfigDir = Path("config")) -> None:
    """Write the model bundle that NeuralBloch.jl reads, and its release file."""
    _, config = _start(config_dir)

    from neuralbloch.scripts.export_bundle import export_bundle

    export_bundle(config.export, config.experiment.pt_file_path)
