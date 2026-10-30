from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import numpy as np
import onnx
import pytest
import torch
from onnx import helper
from onnx import numpy_helper
from scripts.export_onnx import TARGETS
from scripts.export_onnx import TargetModel
from scripts.export_onnx import check_outputs
from scripts.export_onnx import onnx_precision
from scripts.export_onnx import staged
from scripts.export_onnx import target_paths
from src.model import FCModel


NTR = 5

X = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9], [2.0, 0.3, 1.1]])

# Each output of each target
TARGET_OUTPUTS = [
    (name, output) for name, target in TARGETS.items() for output in target.outputs
]


def sizes(config: Mapping[str, Any]) -> dict[str, tuple[int, ...]]:
    """Return the shape of each output after the batch dimension.

    NeuralBloch.jl uses these shapes to check the engines.
    """
    return {
        "signal": (NTR, 2),
        "dT1": (NTR, 2),
        "dT2": (NTR, 2),
        "coeffs": (config["NB_signal"],),
    }


@pytest.fixture(params=["direct", "derived"])
def config(request: pytest.FixtureRequest) -> dict[str, Any]:
    """Each of the two model configurations of conftest.py."""
    return request.getfixturevalue(f"{request.param}_config")


def test_targets_are_fcmodel_methods() -> None:
    assert list(TARGETS) == ["signal_and_derivs", "signal", "coeffs"]
    for target in TARGETS.values():
        assert callable(getattr(FCModel, target.method))


@pytest.mark.parametrize("name", list(TARGETS))
def test_target_model_is_its_method_bitwise(
    config: dict[str, Any], make_model: Callable[..., FCModel], name: str
) -> None:
    torch.manual_seed(0)
    model = make_model(NTR, config).eval()
    methods = {
        "signal_and_derivs": lambda: model(X),
        "signal": lambda: (model.forward_signal(X),),
        "coeffs": lambda: (model.forward_coeffs(X),),
    }
    with torch.no_grad():
        expected = methods[name]()
        outputs = TargetModel(model, name)(X)
    assert isinstance(outputs, tuple)
    assert len(outputs) == len(TARGETS[name].outputs)
    for output, reference in zip(outputs, expected, strict=True):
        assert torch.equal(output, reference)


@pytest.mark.parametrize(("name", "output_name"), TARGET_OUTPUTS)
def test_output_shapes(
    direct_config: dict[str, Any],
    make_model: Callable[..., FCModel],
    name: str,
    output_name: str,
) -> None:
    model = make_model(NTR, direct_config).eval()
    assert model.rank == direct_config["NB_signal"]
    with torch.no_grad():
        outputs = TargetModel(model, name)(X)
    output = dict(zip(TARGETS[name].outputs, outputs, strict=True))[output_name]
    assert tuple(output.shape) == (len(X), *sizes(direct_config)[output_name])


def test_fp16_runs_in_half_behind_an_fp32_interface(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    torch.manual_seed(0)
    model = make_model(NTR, direct_config).eval()
    half = TargetModel(model, "signal_and_derivs", fp16=True)
    assert all(p.dtype == torch.float16 for p in half.parameters())
    # The wrapper converts a copy, so the model stays FP32
    assert all(p.dtype == torch.float32 for p in model.parameters())
    with torch.no_grad():
        outputs = half(X)
        expected = model(X)
    for output, reference in zip(outputs, expected, strict=True):
        assert output.dtype == torch.float32
        torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


def test_every_target_needs_its_paths() -> None:
    targets = {name: {"ONNX_FILE_PATH": f"{name}.onnx"} for name in TARGETS}
    paths = target_paths({"TARGETS": targets})
    assert list(paths) == list(TARGETS)
    assert paths["signal"] == Path("signal.onnx")

    del targets["coeffs"]
    with pytest.raises(ValueError, match="it names signal_and_derivs"):
        target_paths({"TARGETS": targets})
    with pytest.raises(ValueError, match="it names none"):
        target_paths({"ONNX_FILE_PATH": "outputs/model.onnx"})


def test_each_path_is_a_different_file() -> None:
    targets = {name: {"ONNX_FILE_PATH": f"{name}.onnx"} for name in TARGETS}
    # For example, a copied entry that keeps the ONNX file of another target
    targets["coeffs"]["ONNX_FILE_PATH"] = "./signal_and_derivs.onnx"
    with pytest.raises(
        ValueError,
        match="coeffs.ONNX_FILE_PATH and signal_and_derivs.ONNX_FILE_PATH "
        "are the same file",
    ):
        target_paths({"TARGETS": targets})


def test_outputs_need_the_pytorch_shape_and_finite_values(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> None:
    torch.manual_seed(0)
    model = make_model(NTR, direct_config).eval()
    with torch.no_grad():
        (signal,) = TargetModel(model, "signal")(X)
    signal = signal.numpy()

    check_outputs("same", model, "signal", X, [signal.copy()], "FP32")
    with pytest.raises(RuntimeError, match="shape"):
        check_outputs("shape", model, "signal", X, [signal[:2]], "FP32")
    # One NaN gives a NaN mean error, and NaN is not more than the tolerance
    nan = signal.copy()
    nan[0, 0, 0] = np.nan
    with pytest.raises(RuntimeError, match="1 values that are not finite"):
        check_outputs("nan", model, "signal", X, [nan], "FP16")


def test_a_file_replaces_the_old_file_only_after_its_check(tmp_path: Path) -> None:
    path = tmp_path / "model.onnx"
    path.write_bytes(b"old")
    # The error must come from inside staged, so that staged deletes the new file
    with pytest.raises(RuntimeError, match="failed check"):  # noqa: PT012
        with staged(path) as new:
            assert new.parent == path.parent
            assert new.suffix == ".onnx"
            new.write_bytes(b"rejected")
            raise RuntimeError("failed check")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]

    with staged(path) as new:
        new.write_bytes(b"new")
    assert path.read_bytes() == b"new"
    assert path.stat().st_mode & 0o777 == 0o644
    assert list(tmp_path.iterdir()) == [path]


def test_precision_is_read_from_the_weights(tmp_path: Path) -> None:
    def model_with(weights: type, path: Path) -> Path:
        w = numpy_helper.from_array(np.zeros(1, weights), "w")
        onnx.save(helper.make_model(helper.make_graph([], "g", [], [], [w])), path)
        return path

    fp32 = model_with(np.float32, tmp_path / "fp32.onnx")
    fp16 = model_with(np.float16, tmp_path / "fp16.onnx")
    assert onnx_precision([fp32, fp32]) == "FP32"
    assert onnx_precision([fp16]) == "FP16"
    with pytest.raises(ValueError, match="mix FP16 and FP32"):
        onnx_precision([fp32, fp16])
