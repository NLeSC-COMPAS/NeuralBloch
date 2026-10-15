from pathlib import Path
from typing import Any
import tensorrt as trt
import torch
from scripts.export_onnx import TARGETS
from scripts.export_onnx import TargetModel
from scripts.export_onnx import check_outputs
from scripts.export_onnx import target_paths
from scripts.export_onnx import training_domain
from scripts.export_onnx import verification_tolerance
from src.model import FCModel
from src.utils import load_model
from src.utils import sample_parameters


def build_engine(config: dict[str, Any]) -> None:
    """Build a TensorRT engine per target from the exported ONNX models.

    Each target's ONNX model is parsed into a TensorRT network and compiled
    into a serialized engine with a dynamic batch size profile, at the path
    ``export.TARGETS`` names. Each engine is then verified against the
    PyTorch model.

    Args:
        config: Full configuration dictionary.

    Raises:
        RuntimeError: If no GPU is available, an ONNX model cannot be
            parsed, or an engine build fails.
        FileNotFoundError: If an ONNX model has not been exported yet.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Building a TensorRT engine requires a CUDA GPU")

    export_config = config["export"]
    paths = target_paths(export_config)

    for onnx_path, _ in paths.values():
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run `uv run run.py export_onnx` first."
            )

    logger = trt.Logger(trt.Logger.WARNING)

    # PyTorch float32 reference, and parameters from its training domain
    model, checkpoint = load_model(config, "cuda")
    x = sample_parameters(
        training_domain(config, checkpoint), export_config["N_VERIFY_SAMPLES"]
    ).to("cuda")

    precision = "FP16" if export_config["FP16"] else "FP32"

    for target, (onnx_path, engine_path) in paths.items():
        build_target_engine(
            onnx_path, engine_path, export_config, config["model"]["input_dim"], logger
        )
        print(f"\nBuilt TensorRT engine ({precision}) for {target} at {engine_path}")

        verify_engine(model, target, engine_path, x, logger, config)


def build_target_engine(
    onnx_path: Path,
    engine_path: Path,
    export_config: dict[str, Any],
    input_dim: int,
    logger: trt.Logger,
) -> None:
    """Build the TensorRT engine of one ONNX model and write it to a file.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Where to write the serialized engine.
        export_config: The ``export`` configuration (batch size profile).
        input_dim: The number of input parameters per sample.
        logger: TensorRT logger.

    Raises:
        RuntimeError: If the ONNX model cannot be parsed or the build fails.
    """
    # Parse the ONNX model into a TensorRT network
    builder = trt.Builder(logger)
    network = builder.create_network()
    parser = trt.OnnxParser(network, logger)

    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"Failed to parse {onnx_path}:\n{errors}")

    builder_config = builder.create_builder_config()

    # Dynamic batch size profile: the engine accepts any batch size between
    # MIN_BATCH_SIZE and MAX_BATCH_SIZE, with kernels tuned for OPT_BATCH_SIZE
    profile = builder.create_optimization_profile()
    profile.set_shape(
        "x",
        (export_config["MIN_BATCH_SIZE"], input_dim),
        (export_config["OPT_BATCH_SIZE"], input_dim),
        (export_config["MAX_BATCH_SIZE"], input_dim),
    )
    builder_config.add_optimization_profile(profile)

    # Compile and serialize the engine. TensorRT networks are strongly typed,
    # so the engine precision (FP32 or FP16) is inherited from the ONNX model.
    serialized_engine = builder.build_serialized_network(network, builder_config)

    if serialized_engine is None:
        raise RuntimeError(f"TensorRT engine build failed for {onnx_path}")

    engine_path.write_bytes(serialized_engine)


def verify_engine(
    model: FCModel,
    target: str,
    engine_path: Path,
    x: torch.Tensor,
    logger: trt.Logger,
    config: dict[str, Any],
) -> None:
    """Compare one target's TensorRT engine against the PyTorch model.

    The engine must have exactly the input ``x`` and the target's outputs.
    It runs directly on PyTorch CUDA tensors, so no separate CUDA bindings
    are required.

    Args:
        model: The trained PyTorch model, on the GPU.
        target: The target the engine was built for.
        engine_path: Path to the serialized TensorRT engine.
        x: Physical parameters ``[T1, T2, B1]`` on the GPU.
        logger: TensorRT logger.
        config: Full configuration dictionary.

    Raises:
        RuntimeError: If the engine's tensors are not the target's, or its
            predictions deviate from PyTorch.
    """
    with torch.no_grad():
        reference = TargetModel(model, target)(x)

    # Load the serialized engine
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    outputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
    if inputs != ["x"] or sorted(outputs) != sorted(TARGETS[target].outputs):
        raise RuntimeError(
            f"The engine at {engine_path} has inputs {inputs} and outputs "
            f"{outputs}; target {target} has input x and outputs "
            f"{list(TARGETS[target].outputs)}"
        )

    # Bind the input and allocate output buffers as PyTorch CUDA tensors
    context.set_input_shape("x", tuple(x.shape))
    context.set_tensor_address("x", x.data_ptr())

    buffers: dict[str, torch.Tensor] = {}

    for name in outputs:
        shape = tuple(context.get_tensor_shape(name))
        buffers[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
        context.set_tensor_address(name, buffers[name].data_ptr())

    # Run inference on a dedicated CUDA stream
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    check_outputs(
        f"TensorRT vs PyTorch float32, target {target}",
        model,
        target,
        reference,
        [buffers[name] for name in TARGETS[target].outputs],
        tolerance=verification_tolerance(config),
    )
