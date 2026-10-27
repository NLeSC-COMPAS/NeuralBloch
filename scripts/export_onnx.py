import contextlib
import copy
import os
import tempfile
from collections.abc import Iterator
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import numpy as np
import numpy.typing as npt
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn
from scripts.test_accuracy import nrmse
from src.model import FCModel
from src.utils import file_sha256
from src.utils import load_model
from src.utils import sample_parameters


@dataclass(frozen=True)
class Target:
    """A method of ``FCModel`` that has its own exported model.

    Attributes:
        method: The ``FCModel`` method that the target runs.
        outputs: The names of its outputs, in the order that the method returns
            them. The ONNX model and the TensorRT engine use these names for
            their outputs.
        bundle_file: The name of its ONNX model in the model bundle.
    """

    method: str
    outputs: tuple[str, ...]
    bundle_file: str


# The data that a caller can get from the surrogate. The keys are the names that
# NeuralBloch.jl uses for these data. Each target has its own ONNX model and its
# own TensorRT engine. An engine calculates all outputs of its model on each
# call. Thus, a separate model for each target lets a caller do only the work
# that it needs.
TARGETS: dict[str, Target] = {
    "signal_and_derivs": Target("forward", ("signal", "dT1", "dT2"), "model.onnx"),
    "signal": Target("forward_signal", ("signal",), "model_signal.onnx"),
    "coeffs": Target("forward_coeffs", ("coeffs",), "model_coeffs.onnx"),
}


class TargetModel(nn.Module):
    """``FCModel`` with the method of one target as its ``forward``.

    ``torch.onnx.export`` traces only ``forward``. Thus, the export uses this
    wrapper for each target, and the three methods of ``FCModel`` do not change.
    The outputs are always a tuple, in the output order of the target.

    With ``fp16``, the model runs in half precision behind an FP32 interface.
    The wrapper casts the inputs to FP16 and the outputs back to FP32. TensorRT
    engines get their precision from the ONNX model. Thus, an export of this
    wrapper gives an FP16 engine, and the input and output dtypes do not change.
    In this mode, the wrapper holds an FP16 copy of the model and does not
    change ``model``.
    """

    def __init__(self, model: FCModel, target: str, fp16: bool = False) -> None:
        super().__init__()
        self.method = TARGETS[target].method
        self.fp16 = fp16
        self.model = copy.deepcopy(model).half() if fp16 else model
        # A new module starts in training mode. Use the mode of the model, so
        # that a wrapped evaluation model also exports as an evaluation model
        self.train(model.training)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the target's method on physical parameters ``[T1, T2, B1]``."""
        outputs = getattr(self.model, self.method)(x.half() if self.fp16 else x)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return tuple(output.float() for output in outputs) if self.fp16 else outputs


def export_onnx(config: dict[str, Any]) -> None:
    """Export the trained surrogate model to ONNX, one model per target.

    The function loads the trained checkpoint. It exports each target
    (``TARGETS``) to the ONNX model that ``export.TARGETS`` gives for it, with a
    dynamic batch dimension, and records the sha256 of the checkpoint in the
    model (``stamp_checkpoint``). Then it uses ONNX Runtime to compare each ONNX
    model with the PyTorch model. A model replaces the old file only if it
    passes this check (``staged``).

    Args:
        config: Full configuration dictionary.
    """
    export_config = config["export"]
    paths = target_paths(export_config)

    # The exported ONNX model is device independent, so export on CPU
    model, checkpoint = load_model(config, "cpu")

    fp16 = export_config["FP16"]
    precision = "FP16" if fp16 else "FP32"

    x = sample_parameters(
        training_domain(config, checkpoint), export_config["N_VERIFY_SAMPLES"]
    )

    checkpoint_sha256 = file_sha256(config["experiment"]["PT_FILE_PATH"])

    for target, (onnx_path, _) in paths.items():
        with staged(onnx_path) as staged_path:
            # A batch size > 1 is sufficient to trace the model
            export_target(
                model,
                target,
                staged_path,
                example=x[:2],
                opset=export_config["OPSET_VERSION"],
                fp16=fp16,
            )
            stamp_checkpoint(staged_path, checkpoint_sha256)
            print(
                f"\nExported target {target} of the model from epoch "
                f"{checkpoint['epoch']} to {onnx_path} ({precision})"
            )
            verify_onnx(model, target, staged_path, x, precision)


@contextlib.contextmanager
def staged(path: Path) -> Iterator[Path]:
    """Give a temporary file that replaces ``path`` only if the block succeeds.

    The temporary file is in the directory of ``path`` and has its suffix. Write
    and check the new file in the block. If the block raises an error, the
    function deletes the temporary file, and ``path`` does not change. Thus, a
    model that fails its check does not replace a good model, and a caller does
    not find a model that failed its check at ``path``.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(
        prefix=f".{path.name}-", suffix=path.suffix, dir=path.parent
    )
    os.close(fd)
    temporary = Path(name)
    try:
        yield temporary
        # mkstemp makes the file readable only by its owner
        temporary.chmod(0o644)
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# The key of the ONNX metadata that gives the sha256 of the checkpoint file that
# the model was exported from. build_engine and export_bundle use it to find ONNX
# models from a different checkpoint.
CHECKPOINT_KEY = "neuralbloch.checkpoint_sha256"


def stamp_checkpoint(onnx_path: Path, checkpoint_sha256: str) -> None:
    """Record the sha256 of the source checkpoint in the metadata of an ONNX model."""
    onnx_model = onnx.load(str(onnx_path))
    props = [p for p in onnx_model.metadata_props if p.key != CHECKPOINT_KEY]
    del onnx_model.metadata_props[:]
    onnx_model.metadata_props.extend(props)
    onnx_model.metadata_props.add(key=CHECKPOINT_KEY, value=checkpoint_sha256)
    onnx.save(onnx_model, str(onnx_path))


def onnx_checkpoint(onnx_path: Path) -> str | None:
    """Return the checkpoint sha256 that ``stamp_checkpoint`` recorded, or ``None``."""
    onnx_model = onnx.load(str(onnx_path), load_external_data=False)
    for prop in onnx_model.metadata_props:
        if prop.key == CHECKPOINT_KEY:
            return prop.value
    return None


def require_checkpoint(
    onnx_path: Path, checkpoint_sha256: str, checkpoint_path: str | Path
) -> None:
    """Raise an error if an ONNX model does not come from a checkpoint.

    Args:
        onnx_path: The ONNX model.
        checkpoint_sha256: The sha256 of the checkpoint file.
        checkpoint_path: The path of the checkpoint file, for the error message.

    Raises:
        ValueError: If the model does not record its checkpoint, or records a
            different checkpoint.
    """
    recorded = onnx_checkpoint(onnx_path)
    if recorded is None:
        raise ValueError(
            f"{onnx_path} does not record its checkpoint. Export it again with "
            "`uv run run.py export_onnx`"
        )
    if recorded != checkpoint_sha256:
        raise ValueError(
            f"{onnx_path} comes from a different checkpoint than {checkpoint_path}. "
            "Export it again with `uv run run.py export_onnx`"
        )


def target_paths(export_config: dict[str, Any]) -> dict[str, tuple[Path, Path]]:
    """Return the ONNX and engine paths of each target from ``export.TARGETS``.

    Args:
        export_config: The ``export`` configuration.

    Returns:
        ``(onnx_path, engine_path)`` per target, in ``TARGETS`` order.

    Raises:
        ValueError: If ``export.TARGETS`` does not name all targets and only
            them, or two of the paths are the same file.
    """
    configured = export_config.get("TARGETS") or {}
    if set(configured) != set(TARGETS):
        raise ValueError(
            f"export.TARGETS must name the targets {', '.join(TARGETS)}; "
            f"it names {', '.join(configured) or 'none'}"
        )
    paths = {
        target: (
            Path(configured[target]["ONNX_FILE_PATH"]),
            Path(configured[target]["ENGINE_FILE_PATH"]),
        )
        for target in TARGETS
    }
    # Each export writes its file, so two exports to one file keep only the last
    seen: dict[Path, str] = {}
    for target, target_files in paths.items():
        for key, path in zip(("ONNX_FILE_PATH", "ENGINE_FILE_PATH"), target_files):
            name = f"{target}.{key}"
            if (other := seen.setdefault(path.resolve(), name)) != name:
                raise ValueError(
                    f"export.TARGETS {name} and {other} are the same file, {path}"
                )
    return paths


def require_exported(paths: dict[str, tuple[Path, Path]]) -> None:
    """Raise an error if an ONNX model from ``export_onnx`` is missing.

    Args:
        paths: The targets' paths (``target_paths``).

    Raises:
        FileNotFoundError: If a target's ONNX model is missing.
    """
    for onnx_path, _ in paths.values():
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run `uv run run.py export_onnx` first."
            )


def training_domain(
    config: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Return the configuration from which to sample the verification parameters.

    Checkpoints contain the data configuration of their training. Thus, the
    verification samples the training domain of the model. Old checkpoints do
    not contain this configuration. For these checkpoints, the function uses
    ``config/data.yaml``.

    Args:
        config: Full configuration dictionary.
        checkpoint: The checkpoint the model was loaded from.

    Returns:
        A configuration ``sample_parameters`` accepts.
    """
    return {"data": checkpoint.get("data_config", config["data"])}


def export_target(
    model: FCModel,
    target: str,
    onnx_path: Path,
    *,
    example: torch.Tensor,
    opset: int,
    fp16: bool = False,
) -> None:
    """Export one target of ``model`` to the ONNX model at ``onnx_path``.

    The model has one input, ``x``: Float32 ``(batch, 3)`` physical parameters
    ``[T1, T2, B1]``, with a dynamic batch dimension. Its outputs are the
    outputs of the target, in Float32, with the names in ``TARGETS``. The file
    contains the weights, so it is self-contained.

    Args:
        model: The trained model.
        target: A key of ``TARGETS``.
        onnx_path: Where to write the ONNX model.
        example: Input used to trace the model (a batch size > 1).
        opset: The ONNX opset version.
        fp16: If ``True``, the model runs in FP16 behind an FP32 interface.
    """
    torch.onnx.export(
        TargetModel(model, target, fp16=fp16),
        (example,),
        str(onnx_path),
        input_names=["x"],
        output_names=list(TARGETS[target].outputs),
        dynamic_shapes={"x": {0: "batch"}},
        opset_version=opset,
        dynamo=True,
        # Store the weights inside the ONNX file itself, so the exported
        # model is a single self-contained file
        external_data=False,
    )


def onnx_precision(paths: list[Path]) -> str:
    """Return the precision of the ONNX model weights, ``"FP16"`` or ``"FP32"``.

    Raises:
        ValueError: If the models do not all have the same precision.
    """
    precisions = set()
    for path in paths:
        graph = onnx.load(str(path)).graph
        half = any(w.data_type == onnx.TensorProto.FLOAT16 for w in graph.initializer)
        precisions.add("FP16" if half else "FP32")
    if len(precisions) != 1:
        raise ValueError(
            "The ONNX models mix FP16 and FP32; export them again together"
        )
    return precisions.pop()


def verify_onnx(
    model: FCModel,
    target: str,
    onnx_path: Path,
    x: torch.Tensor,
    precision: str,
) -> None:
    """Compare ONNX Runtime predictions of one target against PyTorch.

    Args:
        model: The trained PyTorch model (on CPU).
        target: The target the ONNX model was exported for.
        onnx_path: Path to the exported ONNX model.
        x: The physical parameters ``[T1, T2, B1]`` for the predictions.
        precision: ``"FP32"`` or ``"FP16"``, the precision of the ONNX model.

    Raises:
        RuntimeError: If the predictions fail ``check_outputs``.
    """
    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    predictions = [
        np.asarray(prediction, dtype=np.float32)
        for prediction in session.run(list(TARGETS[target].outputs), {"x": x.numpy()})
    ]

    check_outputs(
        f"ONNX Runtime vs PyTorch float32, target {target}",
        model,
        target,
        x,
        predictions,
        precision,
    )


def check_outputs(
    title: str,
    model: FCModel,
    target: str,
    x: torch.Tensor,
    predictions: Sequence[npt.NDArray[np.float32]],
    precision: str,
) -> None:
    """Check the predictions of an exported model for one target against PyTorch.

    Each output must have the same shape as the PyTorch output, and all its
    values must be finite. Its mean NRMSE against the PyTorch output must be
    0.1 % or less. For an FP16 model, the limit is 1 %, because FP16 is faster
    but a little less accurate.

    Args:
        title: The name of the comparison, for the printed report.
        model: The PyTorch model, on the device of ``x``.
        target: The target of the predictions.
        x: The physical parameters for the predictions.
        predictions: The outputs of the exported model, in the target order.
        precision: The precision of the exported model, ``"FP32"`` or ``"FP16"``.

    Raises:
        RuntimeError: If an output has a shape different from the PyTorch
            output, has a value that is not finite, or its error is more than
            the tolerance.
    """
    tolerance = 1.0 if precision == "FP16" else 0.1

    with torch.no_grad():
        reference = [output.cpu().numpy() for output in TargetModel(model, target)(x)]

    print(f"\n=== {title} ({len(x)} samples) ===")

    for name, ref, pred in zip(
        TARGETS[target].outputs, reference, predictions, strict=True
    ):
        if pred.shape != ref.shape:
            raise RuntimeError(
                f"Exported {name} output has shape {pred.shape}, PyTorch's {ref.shape}"
            )

        # NaN is not more than the tolerance, so the error test does not
        # find it
        not_finite = np.count_nonzero(~np.isfinite(pred))
        if not_finite:
            raise RuntimeError(
                f"Exported {name} output has {not_finite} values that are not finite"
            )

        max_diff = np.abs(pred - ref).max()
        error = nrmse(pred, ref).mean()

        print(f"  {name:12s} | max abs diff: {max_diff:.2e} | NRMSE: {error:.5f}%")

        if not error <= tolerance:
            raise RuntimeError(
                f"Exported predictions for {name} deviate from PyTorch "
                f"(NRMSE {error:.5f}% > {tolerance}%)"
            )
