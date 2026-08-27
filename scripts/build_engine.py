from pathlib import Path
from typing import Any
import numpy as np
import tensorrt as trt
import torch
from scripts.test_accuracy import nrmse
from src.utils import load_model
from src.utils import sample_parameters


def build_engine(config: dict[str, Any]) -> None:
    """Build a TensorRT engine from the exported ONNX model.

    The ONNX model is parsed into a TensorRT network and compiled into a
    serialized engine with a dynamic batch size profile. The engine is then
    verified against the PyTorch model.

    Args:
        config: Full configuration dictionary.

    Raises:
        RuntimeError: If no GPU is available, the ONNX model cannot be
            parsed, or the engine build fails.
        FileNotFoundError: If the ONNX model has not been exported yet.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Building a TensorRT engine requires a CUDA GPU")

    export_config = config["export"]

    onnx_path = Path(export_config["ONNX_FILE_PATH"])
    engine_path = Path(export_config["ENGINE_FILE_PATH"])

    if not onnx_path.exists():
        raise FileNotFoundError(
            f"{onnx_path} not found. Run `uv run run.py export_onnx` first."
        )

    logger = trt.Logger(trt.Logger.WARNING)

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
    input_dim = config["model"]["input_dim"]

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
        raise RuntimeError("TensorRT engine build failed")

    engine_path.write_bytes(serialized_engine)

    precision = "FP16" if export_config["FP16"] else "FP32"
    print(f"\nBuilt TensorRT engine ({precision}) at {engine_path}")

    verify_engine(config, engine_path, logger)


def verify_engine(
    config: dict[str, Any],
    engine_path: Path,
    logger: trt.Logger,
) -> None:
    """Compare TensorRT engine predictions against the PyTorch model.

    The engine runs directly on PyTorch CUDA tensors, so no separate
    CUDA bindings are required.

    Args:
        config: Full configuration dictionary.
        engine_path: Path to the serialized TensorRT engine.
        logger: TensorRT logger.

    Raises:
        RuntimeError: If the engine predictions deviate from PyTorch.
    """
    n_samples = config["export"]["N_VERIFY_SAMPLES"]

    # PyTorch float32 reference predictions
    model, _ = load_model(config, "cuda")

    x = sample_parameters(config, n_samples).to("cuda")

    with torch.no_grad():
        reference = model(x)

    # Load the serialized engine
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    context = engine.create_execution_context()

    # Bind the input and allocate output buffers as PyTorch CUDA tensors
    context.set_input_shape("x", tuple(x.shape))
    context.set_tensor_address("x", x.data_ptr())

    outputs: dict[str, torch.Tensor] = {}

    for i in range(engine.num_io_tensors):
        name = engine.get_tensor_name(i)

        if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT:
            shape = tuple(context.get_tensor_shape(name))
            outputs[name] = torch.empty(shape, dtype=torch.float32, device="cuda")
            context.set_tensor_address(name, outputs[name].data_ptr())

    # Run inference on a dedicated CUDA stream
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    context.execute_async_v3(stream.cuda_stream)
    stream.synchronize()

    # FP16 engines trade a small amount of accuracy for speed
    tolerance = 1.0 if config["export"]["FP16"] else 0.1

    print(f"\n=== TensorRT vs PyTorch float32 ({n_samples} samples) ===")

    expected_shape = (n_samples, model.NTR, 2)

    for name, ref in zip(["signal", "dT1", "dT2"], reference, strict=True):
        if tuple(ref.shape) != expected_shape:
            raise RuntimeError(
                f"PyTorch {name} output has shape {tuple(ref.shape)}, "
                f"expected {expected_shape}"
            )

        pred_tensor = outputs[name]
        if tuple(pred_tensor.shape) != expected_shape:
            raise RuntimeError(
                f"TensorRT {name} output has shape {tuple(pred_tensor.shape)}, "
                f"expected {expected_shape}"
            )

        ref = ref.cpu().numpy()
        pred = pred_tensor.cpu().numpy()
        max_diff = np.abs(pred - ref).max()
        error = nrmse(pred, ref).mean()

        print(f"  {name:6s} | max abs diff: {max_diff:.2e} | NRMSE: {error:.5f}%")

        if error > tolerance:
            raise RuntimeError(
                f"TensorRT predictions for {name} deviate from PyTorch "
                f"(NRMSE {error:.5f}% > {tolerance}%)"
            )
