"""``neuralbloch`` command line interface.

Runs one task of the pipeline with the configuration in ``--config-dir``
(default ``config``)::

    uv run neuralbloch generate
    uv run neuralbloch train
    uv run neuralbloch export_bundle --config-dir path/to/config

Some imports are inside ``main()`` on purpose. juliacall must be imported
before torch, and its import starts Julia, so it waits until the arguments are
parsed. The task modules are imported only when their task runs.
"""

import argparse
from neuralbloch.config import load_config
from neuralbloch.data import DATA_GENERATION_JL


def main() -> None:
    """The entry point of the ``neuralbloch`` command."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser(prog="neuralbloch")
    parser.add_argument(
        "task",
        choices=[
            "generate",
            "train",
            "test_accuracy",
            "test_inference",
            "export_onnx",
            "export_bundle",
        ],
    )
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    # Before Julia starts, so that a bad config fails fast
    config = load_config(args.config_dir)

    # juliacall must be imported before torch. The import starts Julia, so it
    # waits until the arguments are parsed.
    from juliacall import Main as jl  # ruff: noqa: PLC0415

    # Online samples are drawn by Julia, so Python's RNG seed alone does not
    # make training runs reproducible.
    julia_seed = int(config["experiment"]["seed"])
    jl.seval(f"begin using Random; Random.seed!({julia_seed}); true end")

    # Generate a new dataset using Julia and split it into
    # training and validation sets
    if args.task == "generate":
        jl.include(DATA_GENERATION_JL)
        jl.save_dataset(config)

        from neuralbloch.data.split_dataset import split_dataset

        split_dataset(config["experiment"])

        return

    if args.task == "train":
        jl.include(DATA_GENERATION_JL)
        jl.seval("begin using BlochSimulators; CUDALibs(); true end")

    from neuralbloch.utils import set_seed

    set_seed(config["experiment"]["seed"])

    # Train the surrogate model
    if args.task == "train":
        from neuralbloch.scripts.train import train

        train(config)

    # Evaluate prediction accuracy on the test dataset
    elif args.task == "test_accuracy":
        from neuralbloch.scripts.test_accuracy import test_accuracy

        test_accuracy(config)

    # Benchmark surrogate inference time
    elif args.task == "test_inference":
        from neuralbloch.scripts.test_inference import test_inference

        test_inference(config)

    # Export the trained model to ONNX
    elif args.task == "export_onnx":
        from neuralbloch.scripts.export_onnx import export_onnx

        export_onnx(config)

    # Write the model bundle that NeuralBloch.jl reads
    elif args.task == "export_bundle":
        from neuralbloch.scripts.export_bundle import export_bundle

        export_bundle(config)


if __name__ == "__main__":
    main()
