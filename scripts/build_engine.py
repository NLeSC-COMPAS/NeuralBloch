from pathlib import Path
from typing import Any
import tensorrt as trt
import torch
from scripts.export_onnx import TARGETS
from scripts.export_onnx import check_outputs
from scripts.export_onnx import onnx_precision
from scripts.export_onnx import require_exported
from scripts.export_onnx import staged
from scripts.export_onnx import target_paths
from scripts.export_onnx import training_domain
from src.model import FCModel
from src.utils import load_model
from src.utils import sample_parameters


def build_engine(config: dict[str, Any]) -> None:
    """Build a TensorRT engine for each target from the exported ONNX models.

    For each target, the function parses the ONNX model into a TensorRT network
    and compiles it into a serialized engine with a dynamic batch size profile.
    Then it compares each engine with the PyTorch model. Only an engine that
    passes this check replaces the file that ``export.TARGETS`` gives
    (``staged``).

    Args:
        config: Full configuration dictionary.

    Raises:
        RuntimeError: If no GPU is available, an ONNX model cannot be
            parsed, an engine build fails, or an engine fails its check.
        FileNotFoundError: If an ONNX model has not been exported yet.
        ValueError: If the batch sizes are not in order (``check_batch_profile``).
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Building a TensorRT engine requires a CUDA GPU")

    export_config = config["export"]
    check_batch_profile(export_config)
    paths = target_paths(export_config)
    require_exported(paths)

    logger = trt.Logger(trt.Logger.WARNING)

    # Use one timing cache for all builds. The networks of the smaller targets are
    # parts of the largest network, so TensorRT does not time their layers again.
    # Keep the builder and the configuration that make the cache until the
    # builds are done, so that the cache does not live longer than its creator
    cache_builder = trt.Builder(logger)
    cache_config = cache_builder.create_builder_config()
    timing_cache = cache_config.create_timing_cache(b"")

    # Load the PyTorch float32 reference and sample parameters from its training domain
    model, checkpoint = load_model(config, "cuda")
    x = sample_parameters(
        training_domain(config, checkpoint), export_config["N_VERIFY_SAMPLES"]
    ).to("cuda")

    # TensorRT networks are strongly typed, so each engine has the precision
    # of its ONNX model
    precision = onnx_precision([onnx_path for onnx_path, _ in paths.values()])

    for target, (onnx_path, engine_path) in paths.items():
        with staged(engine_path) as staged_path:
            build_target_engine(
                onnx_path, staged_path, export_config, logger, timing_cache
            )
            print(
                f"\nBuilt TensorRT engine ({precision}) for {target} at {engine_path}"
            )

            verify_engine(model, target, staged_path, x, logger, precision)


def check_batch_profile(export_config: dict[str, Any]) -> None:
    """Raise an error if the batch sizes of the engine profile are not in order.

    An engine accepts the batch sizes from MIN_BATCH_SIZE to MAX_BATCH_SIZE,
    and TensorRT tunes it for OPT_BATCH_SIZE. The check of the engines runs
    N_VERIFY_SAMPLES in one batch, so this number must also be in the range.

    Raises:
        ValueError: If 1 <= MIN_BATCH_SIZE <= OPT_BATCH_SIZE <= MAX_BATCH_SIZE
            and MIN_BATCH_SIZE <= N_VERIFY_SAMPLES <= MAX_BATCH_SIZE are not
            true.
    """
    low, high = export_config["MIN_BATCH_SIZE"], export_config["MAX_BATCH_SIZE"]
    for key in ("OPT_BATCH_SIZE", "N_VERIFY_SAMPLES"):
        if not 1 <= low <= export_config[key] <= high:
            raise ValueError(
                f"export.{key} is {export_config[key]}, but it must be in the "
                f"batch size range of the engines, MIN_BATCH_SIZE {low} to "
                f"MAX_BATCH_SIZE {high}"
            )


def build_target_engine(
    onnx_path: Path,
    engine_path: Path,
    export_config: dict[str, Any],
    logger: trt.Logger,
    timing_cache: trt.ITimingCache,
) -> None:
    """Build the TensorRT engine of one ONNX model and write it to a file.

    Args:
        onnx_path: Path to the ONNX model.
        engine_path: Where to write the serialized engine.
        export_config: The ``export`` configuration (batch size profile).
        logger: TensorRT logger.
        timing_cache: The layer timings that all builds share.

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

    if network.num_inputs != 1:
        raise RuntimeError(
            f"{onnx_path} has {network.num_inputs} inputs; the model has one, x"
        )

    builder_config = builder.create_builder_config()
    if not builder_config.set_timing_cache(timing_cache, ignore_mismatch=False):
        raise RuntimeError("TensorRT cannot use the shared timing cache")

    # Dynamic batch size profile: the engine accepts any batch size between
    # MIN_BATCH_SIZE and MAX_BATCH_SIZE, with kernels tuned for OPT_BATCH_SIZE
    x = network.get_input(0)
    sample = tuple(x.shape)[1:]
    profile = builder.create_optimization_profile()
    profile.set_shape(
        x.name,
        (export_config["MIN_BATCH_SIZE"], *sample),
        (export_config["OPT_BATCH_SIZE"], *sample),
        (export_config["MAX_BATCH_SIZE"], *sample),
    )
    if builder_config.add_optimization_profile(profile) < 0:
        raise RuntimeError(
            f"TensorRT refuses the batch size profile of {onnx_path}: "
            f"{(export_config['MIN_BATCH_SIZE'], *sample)}, "
            f"{(export_config['OPT_BATCH_SIZE'], *sample)}, "
            f"{(export_config['MAX_BATCH_SIZE'], *sample)}"
        )

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
    precision: str,
) -> None:
    """Compare the TensorRT engine of one target with the PyTorch model.

    The engine must have exactly the input ``x`` and the outputs of the target.
    The engine runs directly on PyTorch CUDA tensors, so it does not need
    separate CUDA bindings.

    Args:
        model: The trained PyTorch model, on the GPU.
        target: The target the engine was built for.
        engine_path: Path to the serialized TensorRT engine.
        x: Physical parameters ``[T1, T2, B1]`` on the GPU.
        logger: TensorRT logger.
        precision: The precision of the engine, ``"FP32"`` or ``"FP16"``.

    Raises:
        RuntimeError: If the engine tensors are not the target tensors, or its
            predictions fail ``check_outputs``.
    """
    # Load the serialized engine
    runtime = trt.Runtime(logger)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"TensorRT cannot load the engine at {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError(
            f"TensorRT cannot make an execution context for {engine_path}"
        )

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
    if not context.set_input_shape("x", tuple(x.shape)):
        raise RuntimeError(
            f"The engine at {engine_path} does not accept input shape {tuple(x.shape)}"
        )
    bind(context, "x", x)

    # Fill the outputs with NaN. If the engine does not write them,
    # check_outputs then refuses them. Memory from the PyTorch cache can
    # otherwise hold correct values from an earlier check.
    buffers: dict[str, torch.Tensor] = {}

    for name in outputs:
        shape = tuple(context.get_tensor_shape(name))
        buffers[name] = torch.full(
            shape, float("nan"), dtype=torch.float32, device="cuda"
        )
        bind(context, name, buffers[name])

    # Run inference on a dedicated CUDA stream
    stream = torch.cuda.Stream()
    torch.cuda.synchronize()

    if not context.execute_async_v3(stream.cuda_stream):
        raise RuntimeError(f"The engine at {engine_path} failed to run")
    stream.synchronize()

    check_outputs(
        f"TensorRT vs PyTorch float32, target {target}",
        model,
        target,
        x,
        [buffers[name].cpu().numpy() for name in TARGETS[target].outputs],
        precision,
    )


def bind(context: trt.IExecutionContext, name: str, tensor: torch.Tensor) -> None:
    """Give the engine tensor ``name`` the memory of a PyTorch CUDA tensor.

    Raises:
        RuntimeError: If TensorRT refuses the address.
    """
    if not context.set_tensor_address(name, tensor.data_ptr()):
        raise RuntimeError(f"TensorRT refuses the address of engine tensor {name}")
