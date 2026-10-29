import argparse
from juliacall import Main as jl
from src.config import load_config


def main() -> None:
    """The entry point."""
    # Parse command-line arguments
    parser = argparse.ArgumentParser()
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

    # Load configuration files and set random seed
    config = load_config(args.config_dir)
    # Online samples are drawn by Julia, so Python's RNG seed alone does not
    # make training runs reproducible.
    julia_seed = int(config["experiment"]["SEED"])
    jl.seval(f"begin using Random; Random.seed!({julia_seed}); true end")

    # Generate a new dataset using Julia and split it into
    # training and validation sets
    if args.task == "generate":
        jl.include("data/data_generation.jl")
        jl.save_dataset(config)

        from data.split_dataset import split_dataset

        split_dataset(config["experiment"])

        return

    if args.task == "train":
        jl.include("data/data_generation.jl")
        jl.seval("begin using BlochSimulators; CUDALibs(); true end")

    from src.utils import set_seed

    set_seed(config["experiment"]["SEED"])

    # Train the surrogate model
    if args.task == "train":
        from scripts.train import train

        train(config)

    # Evaluate prediction accuracy on the test dataset
    elif args.task == "test_accuracy":
        from scripts.test_accuracy import test_accuracy

        test_accuracy(config)

    # Benchmark surrogate inference time
    elif args.task == "test_inference":
        from scripts.test_inference import test_inference

        test_inference(config)

    # Export the trained model to ONNX
    elif args.task == "export_onnx":
        from scripts.export_onnx import export_onnx

        export_onnx(config)

    # Write the model bundle that NeuralBloch.jl reads
    elif args.task == "export_bundle":
        from scripts.export_bundle import export_bundle

        export_bundle(config)


if __name__ == "__main__":
    main()
