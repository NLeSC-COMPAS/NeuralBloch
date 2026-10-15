import copy
from dataclasses import dataclass
import torch
import torch.nn as nn
from src.model import FCModel


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
        if target not in TARGETS:
            raise ValueError(
                f"Unknown target {target!r}; the targets are {', '.join(TARGETS)}"
            )
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


def output_shape(model: FCModel, output: str, n: int) -> tuple[int, ...]:
    """The shape of output ``output`` for a batch of ``n`` samples.

    The signal and its derivatives are ``(n, NTR, 2)``, with real and
    imaginary parts paired on the last axis; the coefficients are
    ``(n, rank)``.
    """
    if output == "coefficients":
        return (n, rank(model))
    return (n, model.NTR, 2)
