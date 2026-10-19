import tempfile
import unittest
from pathlib import Path
import numpy as np
import onnx
import torch
from onnx import helper
from onnx import numpy_helper
from scripts.export_onnx import TARGETS
from scripts.export_onnx import TargetModel
from scripts.export_onnx import export_target
from scripts.export_onnx import onnx_precision
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

# The shape of each output after the batch dimension. NeuralBloch.jl uses these
# shapes to check the engines.
SIZES = {
    "signal": (NTR, 2),
    "dT1": (NTR, 2),
    "dT2": (NTR, 2),
    "coefficients": (SEPARATE_CONFIG["NB_signal"],),
}


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
            with torch.no_grad():
                methods = {
                    "signal_and_derivatives": model(X),
                    "signal": (model.forward_signal(X),),
                    "coefficients": (model.forward_coefficients(X),),
                }
            for name, expected in methods.items():
                with self.subTest(mode=config["derivative_mode"], target=name):
                    with torch.no_grad():
                        outputs = TargetModel(model, name)(X)
                    self.assertIsInstance(outputs, tuple)
                    self.assertEqual(len(outputs), len(TARGETS[name].outputs))
                    for output, reference in zip(outputs, expected, strict=True):
                        self.assertTrue(torch.equal(output, reference))

    def test_output_shapes(self) -> None:
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
        self.assertEqual(model.rank, SEPARATE_CONFIG["NB_signal"])
        for name, target in TARGETS.items():
            with torch.no_grad():
                outputs = TargetModel(model, name)(X)
            for output_name, output in zip(target.outputs, outputs, strict=True):
                with self.subTest(target=name, output=output_name):
                    self.assertEqual(tuple(output.shape), (len(X), *SIZES[output_name]))

    def test_fp16_runs_in_half_behind_an_fp32_interface(self) -> None:
        torch.manual_seed(0)
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
        half = TargetModel(model, "signal_and_derivatives", fp16=True)
        self.assertTrue(all(p.dtype == torch.float16 for p in half.parameters()))
        # The wrapper converts a copy, so the model stays FP32
        self.assertTrue(all(p.dtype == torch.float32 for p in model.parameters()))
        with torch.no_grad():
            outputs = half(X)
            expected = model(X)
        for output, reference in zip(outputs, expected, strict=True):
            self.assertEqual(output.dtype, torch.float32)
            torch.testing.assert_close(output, reference, rtol=2e-2, atol=2e-2)


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

    def test_each_path_is_a_different_file(self) -> None:
        targets = {
            name: {"ONNX_FILE_PATH": f"{name}.onnx", "ENGINE_FILE_PATH": f"{name}.e"}
            for name in TARGETS
        }
        # For example, a copied entry that keeps the ONNX file of another target
        targets["coefficients"]["ONNX_FILE_PATH"] = "./signal_and_derivatives.onnx"
        with self.assertRaisesRegex(
            ValueError,
            "coefficients.ONNX_FILE_PATH and signal_and_derivatives.ONNX_FILE_PATH "
            "are the same file",
        ):
            target_paths({"TARGETS": targets})
        targets["coefficients"]["ONNX_FILE_PATH"] = "signal.e"
        with self.assertRaisesRegex(ValueError, "and signal.ENGINE_FILE_PATH"):
            target_paths({"TARGETS": targets})


class ExportTest(unittest.TestCase):
    def assert_tensor(
        self,
        value: onnx.ValueInfoProto,
        name: str,
        sizes: tuple[int, ...],
    ) -> str:
        """Check a graph input or output and return the name of its batch dimension."""
        self.assertEqual(value.name, name)
        tensor = value.type.tensor_type
        self.assertEqual(tensor.elem_type, onnx.TensorProto.FLOAT, name)
        dims = tensor.shape.dim
        self.assertEqual(len(dims), 1 + len(sizes), name)
        # The batch dimension is symbolic: the model takes any batch size
        self.assertTrue(dims[0].dim_param, name)
        self.assertEqual(tuple(d.dim_value for d in dims[1:]), sizes, name)
        return dims[0].dim_param

    def test_each_target_exports_exactly_its_outputs(self) -> None:
        torch.manual_seed(0)
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
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
                            output, output.name, SIZES[output.name]
                        )
                        self.assertEqual(dim, batch)

    def test_fp16_export_keeps_fp32_inputs_and_outputs(self) -> None:
        model = FCModel(NTR, SEPARATE_CONFIG).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "signal.onnx"
            export_target(model, "signal", path, example=X[:2], opset=18, fp16=True)
            graph = onnx.load(str(path)).graph
            self.assert_tensor(graph.input[0], "x", (3,))
            self.assert_tensor(graph.output[0], "signal", SIZES["signal"])
            self.assertEqual(onnx_precision([path]), "FP16")


class PrecisionTest(unittest.TestCase):
    def test_precision_is_read_from_the_weights(self) -> None:
        def model_with(weights: type, path: Path) -> Path:
            w = numpy_helper.from_array(np.zeros(1, weights), "w")
            onnx.save(helper.make_model(helper.make_graph([], "g", [], [], [w])), path)
            return path

        with tempfile.TemporaryDirectory() as tmp:
            fp32 = model_with(np.float32, Path(tmp) / "fp32.onnx")
            fp16 = model_with(np.float16, Path(tmp) / "fp16.onnx")
            self.assertEqual(onnx_precision([fp32, fp32]), "FP32")
            self.assertEqual(onnx_precision([fp16]), "FP16")
            with self.assertRaisesRegex(ValueError, "mix FP16 and FP32"):
                onnx_precision([fp32, fp16])
