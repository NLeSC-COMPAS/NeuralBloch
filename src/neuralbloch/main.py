"""The command-line entry point of NeuralBloch (the `neuralbloch` command)."""

import argparse
from neuralbloch.config import load_config
from neuralbloch.julia import data_generation
from neuralbloch.julia import julia
from neuralbloch.julia import setup_julia


JULIA_TASKS = {"generate", "train", "export_bundle"}


def main() -> None:
    """The entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "task",
        choices=[
            "setup_julia",
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

    if args.task == "setup_julia":
        setup_julia()
        return

    # Load configuration files
    config = load_config(args.config_dir)

    # Start Julia only for the tasks that use it, and before the imports below,
    # because juliacall must be imported before torch
    if args.task in JULIA_TASKS:
        jl = julia()
        # Online samples are drawn by Julia, so Python's RNG seed alone does not
        # make training runs reproducible.
        julia_seed = int(config["experiment"]["SEED"])
        jl.seval(f"begin using Random; Random.seed!({julia_seed}); true end")

    # NOTE: Since we use lazy-load of Julia these need to be here
    from neuralbloch.data.split_dataset import split_dataset
    from neuralbloch.scripts.export_bundle import export_bundle
    from neuralbloch.scripts.export_onnx import export_onnx
    from neuralbloch.scripts.test_accuracy import test_accuracy
    from neuralbloch.scripts.test_inference import test_inference
    from neuralbloch.scripts.train import train
    from neuralbloch.utils import set_seed

    # Generate a new dataset using Julia and split it into
    # training and validation sets
    if args.task == "generate":
        data_generation().save_dataset(config)

        split_dataset(config["experiment"])

        return

    if args.task == "train":
        data_generation().seval("begin using BlochSimulators; CUDALibs(); true end")

    set_seed(config["experiment"]["SEED"])

    # Train the surrogate model
    if args.task == "train":
        train(config)

    # Evaluate prediction accuracy on the test dataset
    elif args.task == "test_accuracy":
        test_accuracy(config)

    # Benchmark surrogate inference time
    elif args.task == "test_inference":
        test_inference(config)

    # Export the trained model to ONNX
    elif args.task == "export_onnx":
        export_onnx(config)

    # Write the model bundle that NeuralBloch.jl reads
    elif args.task == "export_bundle":
        export_bundle(config)


if __name__ == "__main__":
    main()
