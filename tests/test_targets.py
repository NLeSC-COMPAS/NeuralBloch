from collections.abc import Callable
from pathlib import Path
import numpy as np
import onnx
import pytest
import torch
from onnx import helper
from onnx import numpy_helper
from scripts.export_onnx import TARGETS
from scripts.export_onnx import TargetModel
from scripts.export_onnx import check_outputs
from scripts.export_onnx import export_target
from scripts.export_onnx import onnx_checkpoint
from scripts.export_onnx import onnx_precision
from scripts.export_onnx import require_checkpoint
from scripts.export_onnx import staged
from scripts.export_onnx import stamp_checkpoint
from scripts.export_onnx import target_paths
from scripts.export_onnx import verify_onnx
from src.model import FCModel


NTR = 5

DIRECT_CONFIG = {
    "derivative_mode": "direct",
    "activation": "relu",
    "normalize_inputs": False,
    "NB_signal": 3,
    "NB_dT1": 2,
    "NB_dT2": 2,
    "hidden_size_signal": 4,
    "hidden_size_dT1": 4,
    "hidden_size_dT2": 4,
    "num_layers_signal": 2,
    "num_layers_dT1": 2,
    "num_layers_dT2": 2,
}

DERIVED_CONFIG = {
    "derivative_mode": "derived",
    "activation": "gelu",
    "normalize_inputs": False,
    "NB_signal": 3,
    "hidden_size_signal": 4,
    "num_layers_signal": 3,
}

X = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9], [2.0, 0.3, 1.1]])

# The shape of each output after the batch dimension. NeuralBloch.jl uses these
# shapes to check the engines.
SIZES = {
    "signal": (NTR, 2),
    "dT1": (NTR, 2),
    "dT2": (NTR, 2),
    "coeffs": (DIRECT_CONFIG["NB_signal"],),
}

# Each output of each target
TARGET_OUTPUTS = [
    (name, output) for name, target in TARGETS.items() for output in target.outputs
]

# A batch profile in the correct order
BATCH_SIZES = {
    "MIN_BATCH_SIZE": 10,
    "OPT_BATCH_SIZE": 20,
    "MAX_BATCH_SIZE": 30,
    "N_VERIFY_SAMPLES": 30,
}


def test_targets_are_fcmodel_methods() -> None:
    assert list(TARGETS) == ["signal_and_derivs", "signal", "coeffs"]
    for target in TARGETS.values():
        assert callable(getattr(FCModel, target.method))


@pytest.mark.parametrize("name", list(TARGETS))
@pytest.mark.parametrize(
    "config", [DIRECT_CONFIG, DERIVED_CONFIG], ids=["direct", "derived"]
)
def test_target_model_is_its_method_bitwise(
    config: dict[str, object], name: str
) -> None:
    torch.manual_seed(0)
    model = FCModel(NTR, config).eval()
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
def test_output_shapes(name: str, output_name: str) -> None:
    model = FCModel(NTR, DIRECT_CONFIG).eval()
    assert model.rank == DIRECT_CONFIG["NB_signal"]
    with torch.no_grad():
        outputs = TargetModel(model, name)(X)
    output = dict(zip(TARGETS[name].outputs, outputs, strict=True))[output_name]
    assert tuple(output.shape) == (len(X), *SIZES[output_name])


def test_fp16_runs_in_half_behind_an_fp32_interface() -> None:
    torch.manual_seed(0)
    model = FCModel(NTR, DIRECT_CONFIG).eval()
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
    targets = {
        name: {"ONNX_FILE_PATH": f"{name}.onnx", "ENGINE_FILE_PATH": f"{name}.e"}
        for name in TARGETS
    }
    paths = target_paths({"TARGETS": targets})
    assert list(paths) == list(TARGETS)
    assert paths["signal"] == (Path("signal.onnx"), Path("signal.e"))

    del targets["coeffs"]
    with pytest.raises(ValueError, match="it names signal_and_derivs"):
        target_paths({"TARGETS": targets})
    with pytest.raises(ValueError, match="it names none"):
        target_paths({"ONNX_FILE_PATH": "outputs/model.onnx"})


def test_each_path_is_a_different_file() -> None:
    targets = {
        name: {"ONNX_FILE_PATH": f"{name}.onnx", "ENGINE_FILE_PATH": f"{name}.e"}
        for name in TARGETS
    }
    # For example, a copied entry that keeps the ONNX file of another target
    targets["coeffs"]["ONNX_FILE_PATH"] = "./signal_and_derivs.onnx"
    with pytest.raises(
        ValueError,
        match="coeffs.ONNX_FILE_PATH and signal_and_derivs.ONNX_FILE_PATH "
        "are the same file",
    ):
        target_paths({"TARGETS": targets})
    targets["coeffs"]["ONNX_FILE_PATH"] = "signal.e"
    with pytest.raises(ValueError, match="and signal.ENGINE_FILE_PATH"):
        target_paths({"TARGETS": targets})


@pytest.fixture
def check_batch_profile() -> Callable[[dict[str, int]], None]:
    """Return check_batch_profile, or skip the test if TensorRT is not available."""
    pytest.importorskip("tensorrt", reason="needs TensorRT")
    from scripts.build_engine import check_batch_profile

    return check_batch_profile


def test_batch_sizes_in_order_are_accepted(
    check_batch_profile: Callable[[dict[str, int]], None],
) -> None:
    check_batch_profile(BATCH_SIZES)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("OPT_BATCH_SIZE", 31),
        ("OPT_BATCH_SIZE", 9),
        ("N_VERIFY_SAMPLES", 40),
        ("N_VERIFY_SAMPLES", 5),
        ("MIN_BATCH_SIZE", 0),
    ],
)
def test_batch_sizes_out_of_order_are_refused(
    check_batch_profile: Callable[[dict[str, int]], None], key: str, value: int
) -> None:
    with pytest.raises(ValueError, match="batch size range"):
        check_batch_profile({**BATCH_SIZES, key: value})


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
def test_each_target_exports_exactly_its_outputs(name: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = FCModel(NTR, DIRECT_CONFIG).eval()
    path = tmp_path / f"{name}.onnx"
    export_target(model, name, path, example=X[:2], opset=18)
    onnx_model = onnx.load(str(path))
    onnx.checker.check_model(onnx_model)
    graph = onnx_model.graph

    assert len(graph.input) == 1
    batch = assert_tensor(graph.input[0], "x", (3,))
    assert [o.name for o in graph.output] == list(TARGETS[name].outputs)
    for output in graph.output:
        dim = assert_tensor(output, output.name, SIZES[output.name])
        assert dim == batch


def test_fp16_export_keeps_fp32_inputs_and_outputs(tmp_path: Path) -> None:
    model = FCModel(NTR, DIRECT_CONFIG).eval()
    path = tmp_path / "signal.onnx"
    export_target(model, "signal", path, example=X[:2], opset=18, fp16=True)
    graph = onnx.load(str(path)).graph
    assert_tensor(graph.input[0], "x", (3,))
    assert_tensor(graph.output[0], "signal", SIZES["signal"])
    assert onnx_precision([path]) == "FP16"


@pytest.mark.parametrize("name", list(TARGETS))
def test_an_exported_model_gives_the_pytorch_values(name: str, tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = FCModel(NTR, DIRECT_CONFIG).eval()
    path = tmp_path / f"{name}.onnx"
    # Trace with batch 2 and verify with batch 3, so that the
    # batch dimension is dynamic in the values too
    export_target(model, name, path, example=X[:2], opset=18)
    verify_onnx(model, name, path, X, "FP32")


def test_a_model_from_other_weights_deviates(tmp_path: Path) -> None:
    torch.manual_seed(0)
    exported = FCModel(NTR, DIRECT_CONFIG).eval()
    torch.manual_seed(1)
    other = FCModel(NTR, DIRECT_CONFIG).eval()
    path = tmp_path / "signal.onnx"
    export_target(exported, "signal", path, example=X[:2], opset=18)
    with pytest.raises(RuntimeError, match="deviate"):
        verify_onnx(other, "signal", path, X, "FP32")


def test_outputs_need_the_pytorch_shape_and_finite_values() -> None:
    torch.manual_seed(0)
    model = FCModel(NTR, DIRECT_CONFIG).eval()
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


def test_a_model_records_its_checkpoint(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = FCModel(NTR, DIRECT_CONFIG).eval()
    path = tmp_path / "signal.onnx"
    export_target(model, "signal", path, example=X[:2], opset=18)
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
