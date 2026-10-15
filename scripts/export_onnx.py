import copy
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
from src.utils import load_model
from src.utils import sample_parameters


@dataclass(frozen=True)
class Target:
    """One method of ``FCModel`` that is exported on its own.

    Attributes:
        method: The ``FCModel`` method the target runs.
        outputs: The names of its outputs, in the order the method returns
            them. They name the outputs of the ONNX model and the TensorRT
            engine.
    """

    method: str
    outputs: tuple[str, ...]


# What a caller can ask the surrogate for, keyed by the names NeuralBloch.jl
# uses for them. Each target is exported to its own ONNX model and built into
# its own TensorRT engine: an engine computes every output of its model on
# each call, so a separate model per target is what lets a caller skip the
# work it does not need.
TARGETS: dict[str, Target] = {
    "signal_and_derivatives": Target("forward", ("signal", "dT1", "dT2")),
    "signal": Target("forward_signal", ("signal",)),
    "coefficients": Target("forward_coefficients", ("coefficients",)),
}


class TargetModel(nn.Module):
    """``FCModel`` with one target's method as its ``forward``.

    ``torch.onnx.export`` traces ``forward`` only, so each target is exported
    through this wrapper, and ``FCModel`` keeps its three methods as they are.
    The outputs are always a tuple, in the target's output order.

    With ``fp16``, the model runs in half precision behind an FP32 interface:
    inputs are cast to FP16 and outputs back to FP32. TensorRT engines inherit
    their precision from the ONNX model, so exporting this wrapper yields an
    FP16 engine with unchanged input and output dtypes. The wrapper then holds
    an FP16 copy of the model and leaves ``model`` itself untouched.
    """

    def __init__(self, model: FCModel, target: str, fp16: bool = False) -> None:
        super().__init__()
        self.method = TARGETS[target].method
        self.fp16 = fp16
        self.model = copy.deepcopy(model).half() if fp16 else model
        # A new module starts in training mode; follow the model's own mode,
        # so that wrapping an evaluation model exports one
        self.train(model.training)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        """Run the target's method on physical parameters ``[T1, T2, B1]``."""
        outputs = getattr(self.model, self.method)(x.half() if self.fp16 else x)
        if isinstance(outputs, torch.Tensor):
            outputs = (outputs,)
        return tuple(output.float() for output in outputs) if self.fp16 else outputs


def rank(model: FCModel) -> int:
    """The number of signal basis coefficients per sample.

    This is the width of the ``coefficients`` output and the rank of the
    signal decoder ``U_signal``, read from the decoder itself: a rank-reduced
    checkpoint (``scripts/orthogonalize_basis.py``) has fewer than
    ``config/model.yaml`` names.
    """
    decoder = model.U_signal[0]
    assert isinstance(decoder, nn.Linear)
    return decoder.in_features


def export_onnx(config: dict[str, Any]) -> None:
    """Export the trained surrogate model to ONNX, one model per target.

    The trained checkpoint is loaded, and each target (``TARGETS``) is
    exported to the ONNX model that ``export.TARGETS`` names for it, with a
    dynamic batch dimension, and verified against the PyTorch model using
    ONNX Runtime.

    Args:
        config: Full configuration dictionary.
    """
    export_config = config["export"]
    paths = target_paths(export_config)

    # The exported ONNX model is device independent, so export on CPU
    model, checkpoint = load_model(config, "cpu")

    fp16 = export_config["FP16"]

    x = sample_parameters(
        training_domain(config, checkpoint), export_config["N_VERIFY_SAMPLES"]
    )

    for target, (onnx_path, _) in paths.items():
        # Any batch size > 1 traces the model
        export_target(
            model,
            target,
            onnx_path,
            example=x[:2],
            opset=export_config["OPSET_VERSION"],
            fp16=fp16,
        )
        print(
            f"\nExported target {target} of the model from epoch "
            f"{checkpoint['epoch']} to {onnx_path} ({'FP16' if fp16 else 'FP32'})"
        )
        verify_onnx(model, target, onnx_path, x)


def target_paths(export_config: dict[str, Any]) -> dict[str, tuple[Path, Path]]:
    """The ONNX and engine paths of every target, from ``export.TARGETS``.

    Args:
        export_config: The ``export`` configuration.

    Returns:
        ``(onnx_path, engine_path)`` per target, in ``TARGETS`` order.

    Raises:
        ValueError: If ``export.TARGETS`` does not name exactly the targets.
    """
    configured = export_config.get("TARGETS") or {}
    if set(configured) != set(TARGETS):
        raise ValueError(
            f"export.TARGETS must name the targets {', '.join(TARGETS)}; "
            f"it names {', '.join(configured) or 'none'}"
        )
    return {
        target: (
            Path(configured[target]["ONNX_FILE_PATH"]),
            Path(configured[target]["ENGINE_FILE_PATH"]),
        )
        for target in TARGETS
    }


def require_exported(paths: dict[str, tuple[Path, Path]]) -> None:
    """Refuse to go on before ``export_onnx`` has written every ONNX model.

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
    """The configuration to sample verification parameters from.

    Checkpoints record the data configuration they were trained with, so
    verification samples the model's own training domain. Checkpoints from
    before that was recorded fall back to ``config/data.yaml``.

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

    The model takes one input, ``x``: Float32 ``(batch, 3)`` physical
    parameters ``[T1, T2, B1]``, with a dynamic batch dimension. Its outputs
    are the target's, Float32 and named as in ``TARGETS``. The weights are
    stored inside the file, so it is self-contained.

    Args:
        model: The trained model.
        target: A key of ``TARGETS``.
        onnx_path: Where to write the ONNX model.
        example: Input used to trace the model (a batch size > 1).
        opset: The ONNX opset version.
        fp16: Run the model in FP16 behind an FP32 interface.
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
    """``"FP16"`` or ``"FP32"``: the precision the ONNX models' weights have.

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
) -> None:
    """Compare ONNX Runtime predictions of one target against PyTorch.

    Args:
        model: The trained PyTorch model (on CPU).
        target: The target the ONNX model was exported for.
        onnx_path: Path to the exported ONNX model.
        x: Physical parameters ``[T1, T2, B1]`` to predict for.

    Raises:
        RuntimeError: If an output has another shape than PyTorch's or
            deviates from it.
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
        onnx_precision([onnx_path]),
    )


def check_outputs(
    title: str,
    model: FCModel,
    target: str,
    x: torch.Tensor,
    predictions: Sequence[npt.NDArray[np.float32]],
    precision: str,
) -> None:
    """Check an exported model's predictions of one target against PyTorch's.

    Every output must have the shape of PyTorch's, and a mean NRMSE against it
    of at most 0.1 %, or 1 % for an FP16 model, which trades a small amount of
    accuracy for speed.

    Args:
        title: What is compared, for the printed report.
        model: The PyTorch model, on the device of ``x``.
        target: The target the predictions are of.
        x: The physical parameters they were predicted for.
        predictions: The exported model's outputs, in the target's order.
        precision: ``"FP32"`` or ``"FP16"``, the exported model's precision.

    Raises:
        RuntimeError: If an output has another shape than PyTorch's or
            deviates from it.
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

        max_diff = np.abs(pred - ref).max()
        error = nrmse(pred, ref).mean()

        print(f"  {name:12s} | max abs diff: {max_diff:.2e} | NRMSE: {error:.5f}%")

        if error > tolerance:
            raise RuntimeError(
                f"Exported predictions for {name} deviate from PyTorch "
                f"(NRMSE {error:.5f}% > {tolerance}%)"
            )
