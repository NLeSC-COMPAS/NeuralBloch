import argparse
from src.config import load_config
from src.utils import set_seed


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
            "build_engine",
        ],
    )
    parser.add_argument("--config-dir", default="config")
    args = parser.parse_args()

    # Load configuration files and set random seed
    config = load_config(args.config_dir)
    set_seed(config["experiment"]["SEED"])

    # Generate a new dataset using Julia and split it into
    # training and validation sets
    if args.task == "generate":
        from juliacall import Main as jl

        jl.include("data/data_generation.jl")
        jl.save_dataset(config)

        from data.split_dataset import split_dataset

        split_dataset(config["experiment"])

    # Train the surrogate model
    elif args.task == "train":
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

    # Build a TensorRT engine from the exported ONNX model
    elif args.task == "build_engine":
        from scripts.build_engine import build_engine

        build_engine(config)


if __name__ == "__main__":
    main()
