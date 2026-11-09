"""``neuralbloch`` command line interface.

Runs one task of the pipeline with the configuration in ``--config-dir``
(default ``config``)::

    uv run neuralbloch generate
    uv run neuralbloch train
    uv run neuralbloch export-bundle --config-dir path/to/config

The imports of juliacall and the task modules are inside the commands on
purpose. juliacall must be imported before torch, and its import starts Julia,
so it waits until the arguments are parsed.
"""

from pathlib import Path
from typing import Annotated
from typing import Any
from cyclopts import App
from cyclopts import Parameter
from neuralbloch.config import load_config
from neuralbloch.data import DATA_GENERATION_JL


app = App(name="neuralbloch", help="Train and export the NeuralBloch surrogate.")

# TODO: re-use from `types`
ConfigDir = Annotated[Path, Parameter(help="Directory of the YAML config files.")]


def _start(config_dir: Path, *, seed_python: bool = True) -> tuple[Any, dict[str, Any]]:
    """Start Julia, load the config and seed the RNGs.

    Args:
        config_dir: Directory containing the YAML configuration files.
        seed_python: Also seed Python's RNGs with ``experiment.SEED``.

    Returns:
        The Julia ``Main`` module and the loaded configuration.
    """
    from juliacall import Main as jl  # ruff: noqa: PLC0415

    config = load_config(config_dir)
    # Online samples are drawn by Julia, so Python's RNG seed alone does not
    # make training runs reproducible.
    julia_seed = int(config["experiment"]["SEED"])
    jl.seval(f"begin using Random; Random.seed!({julia_seed}); true end")

    if seed_python:
        from neuralbloch.utils import set_seed

        set_seed(config["experiment"]["SEED"])

    return jl, config


@app.command
def generate(*, config_dir: ConfigDir = Path("config")) -> None:
    """Generate validation and test datasets using Bloch simulations."""
    jl, config = _start(config_dir, seed_python=False)
    jl.include(DATA_GENERATION_JL)
    jl.save_dataset(config)

    from neuralbloch.data.split_dataset import split_dataset

    split_dataset(config["experiment"])


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

    test_accuracy(config)


@app.command
def test_inference(*, config_dir: ConfigDir = Path("config")) -> None:
    """Benchmark surrogate inference speed."""
    _, config = _start(config_dir)

    from neuralbloch.scripts.test_inference import test_inference

    test_inference(config)


@app.command
def export_onnx(*, config_dir: ConfigDir = Path("config")) -> None:
    """Export the trained model to ONNX, one model for each target."""
    _, config = _start(config_dir)

    from neuralbloch.scripts.export_onnx import export_onnx

    export_onnx(config)


@app.command
def export_bundle(*, config_dir: ConfigDir = Path("config")) -> None:
    """Write the model bundle that NeuralBloch.jl reads, and its release file."""
    _, config = _start(config_dir)

    from neuralbloch.scripts.export_bundle import export_bundle

    export_bundle(config)


def main() -> None:
    """The entry point of the ``neuralbloch`` command."""
    app()


if __name__ == "__main__":
    main()
