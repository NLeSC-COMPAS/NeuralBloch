import copy
from pathlib import Path
from typing import Any
import numpy as np
import onnxruntime as ort
import torch
import torch.nn as nn
from scripts.test_accuracy import nrmse
from src.model import FCModel
from src.utils import load_model
from src.utils import sample_parameters


class HalfPrecisionModel(nn.Module):
    """Wrapper that runs the model in FP16 behind an FP32 interface.

    Inputs are cast to FP16, the wrapped model runs entirely in half
    precision, and outputs are cast back to FP32. TensorRT engines inherit
    their precision from the ONNX model, so exporting this wrapper yields
    an FP16 engine with unchanged input and output dtypes.
    """

    def __init__(self, model: FCModel) -> None:
        super().__init__()
        self.model = model.half()

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run the FP16 model on FP32 inputs and return FP32 outputs."""
        m, dm_dT1, dm_dT2 = self.model(x.half())
        return m.float(), dm_dT1.float(), dm_dT2.float()


def export_onnx(config: dict[str, Any]) -> None:
    """Export the trained surrogate model to ONNX.

    The trained checkpoint is loaded, exported to an ONNX model with a
    dynamic batch dimension, and verified against the PyTorch model using
    ONNX Runtime.

    Args:
        config: Full configuration dictionary.
    """
    onnx_path = Path(config["export"]["ONNX_FILE_PATH"])

    # The exported ONNX model is device independent, so export on CPU
    model, checkpoint = load_model(config, "cpu")

    fp16 = config["export"]["FP16"]

    export_model: nn.Module = (
        HalfPrecisionModel(copy.deepcopy(model)) if fp16 else model
    )

    # Example input used to trace the model (any batch size > 1 works)
    x = sample_parameters(config, 2)

    torch.onnx.export(
        export_model,
        (x,),
        str(onnx_path),
        input_names=["x"],
        output_names=["signal", "dT1", "dT2"],
        dynamic_shapes={"x": {0: "batch"}},
        opset_version=config["export"]["OPSET_VERSION"],
        dynamo=True,
        # Store the weights inside the ONNX file itself, so the exported
        # model is a single self-contained file
        external_data=False,
    )

    precision = "FP16" if fp16 else "FP32"
    print(
        f"\nExported model from epoch {checkpoint['epoch']} to {onnx_path} ({precision})"
    )

    verify_onnx(model, onnx_path, config)


def verify_onnx(
    model: FCModel,
    onnx_path: Path,
    config: dict[str, Any],
) -> None:
    """Compare ONNX Runtime predictions against the PyTorch model.

    Args:
        model: The trained PyTorch model (on CPU).
        onnx_path: Path to the exported ONNX model.
        config: Full configuration dictionary.

    Raises:
        RuntimeError: If an output has the wrong shape or deviates from PyTorch.
    """
    n_samples = config["export"]["N_VERIFY_SAMPLES"]

    x = sample_parameters(config, n_samples)

    with torch.no_grad():
        reference = model(x)

    session = ort.InferenceSession(
        str(onnx_path),
        providers=["CPUExecutionProvider"],
    )

    predictions = session.run(None, {"x": x.numpy()})

    # FP16 models trade a small amount of accuracy for speed
    tolerance = 1.0 if config["export"]["FP16"] else 0.1

    print(f"\n=== ONNX Runtime vs PyTorch float32 ({n_samples} samples) ===")

    expected_shape = (n_samples, model.NTR, 2)

    for name, ref, pred in zip(
        ["Signal", "dT1", "dT2"], reference, predictions, strict=True
    ):
        if tuple(ref.shape) != expected_shape:
            raise RuntimeError(
                f"PyTorch {name} output has shape {tuple(ref.shape)}, "
                f"expected {expected_shape}"
            )

        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != expected_shape:
            raise RuntimeError(
                f"ONNX {name} output has shape {pred.shape}, expected {expected_shape}"
            )

        ref = ref.numpy()
        max_diff = np.abs(pred - ref).max()
        error = nrmse(pred, ref).mean()

        print(f"  {name:6s} | max abs diff: {max_diff:.2e} | NRMSE: {error:.5f}%")

        if error > tolerance:
            raise RuntimeError(
                f"ONNX predictions for {name} deviate from PyTorch "
                f"(NRMSE {error:.5f}% > {tolerance}%)"
            )
