import shutil
from collections.abc import Callable
from collections.abc import Mapping
from pathlib import Path
from typing import Any
import onnx
import pytest
import torch
from scripts.export_onnx import TARGETS
from scripts.export_onnx import export_target
from scripts.export_onnx import onnx_checkpoint
from scripts.export_onnx import onnx_precision
from scripts.export_onnx import require_checkpoint
from scripts.export_onnx import stamp_checkpoint
from scripts.export_onnx import verify_onnx
from src.model import FCModel


NTR = 5

X = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9], [2.0, 0.3, 1.1]])


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


def assert_tensor(
    value: onnx.ValueInfoProto,
    name: str,
    sizes: tuple[int, ...],
) -> str:
    """Check a graph input or output and return the name of its batch dimension."""
    assert value.name == name
    tensor = value.type.tensor_type
    assert tensor.elem_type == onnx.TensorProto.FLOAT, name
    dims = tensor.shape.dim
    assert len(dims) == 1 + len(sizes), name
    # The batch dimension is symbolic: the model takes any batch size
    assert dims[0].dim_param, name
    assert tuple(d.dim_value for d in dims[1:]) == sizes, name
    return dims[0].dim_param


@pytest.mark.parametrize("name", list(TARGETS))
def test_each_target_exports_exactly_its_outputs(
    direct_config: dict[str, Any], onnx_models: dict[str, Path], name: str
) -> None:
    onnx_model = onnx.load(str(onnx_models[name]))
    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph

    assert len(graph.input) == 1
    batch = assert_tensor(graph.input[0], "x", (3,))
    assert [o.name for o in graph.output] == list(TARGETS[name].outputs)
    for output in graph.output:
        dim = assert_tensor(output, output.name, sizes(direct_config)[output.name])
        assert dim == batch


def test_fp16_export_keeps_fp32_inputs_and_outputs(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel], tmp_path: Path
) -> None:
    model = make_model(NTR, direct_config).eval()
    path = tmp_path / "signal.onnx"
    export_target(model, "signal", path, example=X[:2], opset=18, fp16=True)
    graph = onnx.load(str(path)).graph
    assert_tensor(graph.input[0], "x", (3,))
    assert_tensor(graph.output[0], "signal", sizes(direct_config)["signal"])
    assert onnx_precision([path]) == "FP16"


@pytest.mark.parametrize("name", list(TARGETS))
def test_an_exported_model_gives_the_pytorch_values(
    exported_model: FCModel, onnx_models: dict[str, Path], name: str
) -> None:
    # The export traced the model with batch 2. Verify with batch 3, so
    # that the batch dimension is dynamic in the values too
    verify_onnx(exported_model, name, onnx_models[name], X, "FP32")


def test_a_model_from_other_weights_deviates(
    direct_config: dict[str, Any],
    make_model: Callable[..., FCModel],
    onnx_models: dict[str, Path],
) -> None:
    torch.manual_seed(1)
    other = make_model(NTR, direct_config).eval()
    with pytest.raises(RuntimeError, match="deviate"):
        verify_onnx(other, "signal", onnx_models["signal"], X, "FP32")


def test_a_model_records_its_checkpoint(
    onnx_models: dict[str, Path], tmp_path: Path
) -> None:
    path = Path(shutil.copy(onnx_models["signal"], tmp_path))
    assert onnx_checkpoint(path) is None
    with pytest.raises(ValueError, match="does not record its checkpoint"):
        require_checkpoint(path, "a" * 64, "model.pt")

    stamp_checkpoint(path, "a" * 64)
    stamp_checkpoint(path, "b" * 64)
    assert onnx_checkpoint(path) == "b" * 64
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    assert len(onnx_model.metadata_props) == 1
    require_checkpoint(path, "b" * 64, "model.pt")
    with pytest.raises(ValueError, match="from a different checkpoint"):
        require_checkpoint(path, "a" * 64, "model.pt")
