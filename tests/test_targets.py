import tempfile
import unittest
from pathlib import Path
import onnx
import torch
from scripts.export_onnx import TARGETS
from scripts.export_onnx import TargetModel
from scripts.export_onnx import export_target
from scripts.export_onnx import output_shape
from scripts.export_onnx import rank
from scripts.export_onnx import target_paths
from src.model import FCModel


NTR = 5

SEPARATE_CONFIG = {
    "input_dim": 3,
    "derivative_mode": "separate",
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

FORWARD_CONFIG = {
    "input_dim": 3,
    "derivative_mode": "forward",
    "activation": "gelu",
    "NB_signal": 3,
    "hidden_size_signal": 4,
    "num_layers_signal": 3,
}

X = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9], [2.0, 0.3, 1.1]])

# ONNX's TensorProto element type for Float32
FLOAT = onnx.TensorProto.FLOAT


def as_tuple(
    outputs: torch.Tensor | tuple[torch.Tensor, ...],
) -> tuple[torch.Tensor, ...]:
    return (outputs,) if isinstance(outputs, torch.Tensor) else outputs


class TargetModelTest(unittest.TestCase):
    def test_targets_are_fcmodel_methods(self) -> None:
        self.assertEqual(
            list(TARGETS), ["signal_and_derivatives", "signal", "coefficients"]
        )
        for target in TARGETS.values():
            self.assertTrue(callable(getattr(FCModel, target.method)))

    def test_target_model_is_its_method_bitwise(self) -> None:
        torch.manual_seed(0)
        for config in (SEPARATE_CONFIG, FORWARD_CONFIG):
            model = FCModel(NTR, config).eval()
            for name, target in TARGETS.items():
                with self.subTest(mode=config["derivative_mode"], target=name):
                    with torch.no_grad():
                        outputs = TargetModel(model, name)(X)
                        expected = as_tuple(getattr(model, target.method)(X))
                    self.assertIsInstance(outputs, tuple)
                    self.assertEqual(len(outputs), len(target.outputs))
                    for output, reference in zip(outputs, expected, strict=True):
                        self.assertTrue(torch.equal(output, reference))

    def test_output_shapes(self) -> None:
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
        self.assertEqual(rank(model), SEPARATE_CONFIG["NB_signal"])
        for name, target in TARGETS.items():
            with torch.no_grad():
                outputs = TargetModel(model, name)(X)
            for output_name, output in zip(target.outputs, outputs, strict=True):
                with self.subTest(target=name, output=output_name):
                    self.assertEqual(
                        tuple(output.shape), output_shape(model, output_name, len(X))
                    )
        self.assertEqual(output_shape(model, "coefficients", 7), (7, 3))
        self.assertEqual(output_shape(model, "dT2", 7), (7, NTR, 2))

    def test_fp16_runs_in_half_behind_an_fp32_interface(self) -> None:
        torch.manual_seed(0)
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
        half = TargetModel(model, "signal_and_derivatives", fp16=True)
        self.assertTrue(all(p.dtype == torch.float16 for p in half.parameters()))
        # The wrapper converts a copy; the model itself stays FP32
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        with torch.no_grad():
            outputs = half(X)
            expected = model(X)
        for output, reference in zip(outputs, expected, strict=True):
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)

    def test_unknown_target_is_refused(self) -> None:
        model = FCModel(NTR, SEPARATE_CONFIG)
        with self.assertRaisesRegex(ValueError, "Unknown target 'derivatives'"):
            TargetModel(model, "derivatives")


class TargetPathsTest(unittest.TestCase):
    def test_every_target_needs_its_paths(self) -> None:
        targets = {
            name: {"ONNX_FILE_PATH": f"{name}.onnx", "ENGINE_FILE_PATH": f"{name}.e"}
            for name in TARGETS
        }
        paths = target_paths({"TARGETS": targets})
        self.assertEqual(list(paths), list(TARGETS))
        self.assertEqual(paths["signal"], (Path("signal.onnx"), Path("signal.e")))

        del targets["coefficients"]
        with self.assertRaisesRegex(ValueError, "it names signal_and_derivatives"):
            target_paths({"TARGETS": targets})
        with self.assertRaisesRegex(ValueError, "it names none"):
            target_paths({"ONNX_FILE_PATH": "outputs/model.onnx"})


class ExportTest(unittest.TestCase):
    def assert_tensor(
        self,
        value: onnx.ValueInfoProto,
        name: str,
        sizes: tuple[int, ...],
    ) -> str:
        """Check one graph input or output; return its batch dimension's name."""
        self.assertEqual(value.name, name)
        tensor = value.type.tensor_type
        self.assertEqual(tensor.elem_type, FLOAT, name)
        dims = tensor.shape.dim
        self.assertEqual(len(dims), 1 + len(sizes), name)
        # The batch dimension is symbolic: the model takes any batch size
        self.assertTrue(dims[0].dim_param, name)
        self.assertEqual(tuple(d.dim_value for d in dims[1:]), sizes, name)
        return dims[0].dim_param

    def test_each_target_exports_exactly_its_outputs(self) -> None:
        torch.manual_seed(0)
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
        sizes = {"coefficients": (rank(model),)}
        with tempfile.TemporaryDirectory() as tmp:
            for name, target in TARGETS.items():
                with self.subTest(target=name):
                    path = Path(tmp) / f"{name}.onnx"
                    export_target(model, name, path, example=X[:2], opset=18)
                    onnx_model = onnx.load(str(path))
                    onnx.checker.check_model(onnx_model)
                    graph = onnx_model.graph

                    self.assertEqual(len(graph.input), 1)
                    batch = self.assert_tensor(graph.input[0], "x", (3,))
                    self.assertEqual(
                        [o.name for o in graph.output], list(target.outputs)
                    )
                    for output in graph.output:
                        dim = self.assert_tensor(
                            output, output.name, sizes.get(output.name, (NTR, 2))
                        )
                        self.assertEqual(dim, batch)

    def test_fp16_export_keeps_fp32_inputs_and_outputs(self) -> None:
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signal.onnx"
            export_target(model, "signal", path, example=X[:2], opset=18, fp16=True)
            graph = onnx.load(str(path)).graph
            self.assert_tensor(graph.input[0], "x", (3,))
            self.assert_tensor(graph.output[0], "signal", (NTR, 2))
            weights = {init.data_type for init in graph.initializer}
            self.assertIn(onnx.TensorProto.FLOAT16, weights)
