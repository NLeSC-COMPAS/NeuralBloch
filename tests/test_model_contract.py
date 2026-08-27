import math
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from data.data_loading import load_data_from_h5_file
from data.data_loading import make_loader
from scripts.test_accuracy import nrmse
from src.loss import nmae_loss
from src.model import FCModel
from src.utils import complex_to_real_imag
from src.utils import sample_parameters


MODEL_CONFIG = {
    "input_dim": 3,
    "NB_signal": 2,
    "NB_dT1": 2,
    "NB_dT2": 2,
    "hidden_size_signal": 4,
    "hidden_size_dT1": 4,
    "hidden_size_dT2": 4,
    "num_layers_signal": 2,
    "num_layers_dT1": 2,
    "num_layers_dT2": 2,
}

SAMPLING_CONFIG = {
    "data": {
        "T1_lower": 0.1,
        "T1_upper": 5.0,
        "T2_lower": 0.01,
        "T2_upper": 2.0,
        "B1_lower": 0.8,
        "B1_upper": 1.2,
    }
}


class ModelContractTest(unittest.TestCase):
    def test_model_logs_physical_parameters_and_returns_pairs(self) -> None:
        model = FCModel(5, MODEL_CONFIG).eval()
        captured_inputs: list[torch.Tensor] = []

        def capture_input(
            _module: torch.nn.Module,
            inputs: tuple[torch.Tensor, ...],
        ) -> None:
            captured_inputs.append(inputs[0].detach().clone())

        handle = model.subnet_signal.register_forward_pre_hook(capture_input)
        x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])

        try:
            with torch.no_grad():
                outputs = model(x)
        finally:
            handle.remove()

        expected_input = torch.cat((torch.log(x[..., :2]), x[..., 2:]), dim=-1)
        self.assertEqual(len(captured_inputs), 1)
        torch.testing.assert_close(captured_inputs[0], expected_input)

        self.assertEqual(model.NTR, 5)
        for decoder in (model.U_signal, model.U_dT1, model.U_dT2):
            self.assertIsInstance(decoder, torch.nn.Sequential)
            self.assertIsInstance(decoder[-1], torch.nn.Unflatten)

        for output in outputs:
            self.assertEqual(output.shape, (2, 5, 2))
            complex_output = torch.view_as_complex(output.contiguous())
            self.assertEqual(complex_output.shape, (2, 5))

    def test_model_output_contract_supports_multiple_batch_sizes(self) -> None:
        model = FCModel(5, MODEL_CONFIG).eval()

        for batch_size in (1, 3, 8):
            x = torch.tensor([[0.8, 0.07, 1.0]]).repeat(batch_size, 1)
            with torch.no_grad():
                outputs = model(x)
            for output in outputs:
                self.assertEqual(output.shape, (batch_size, 5, 2))

    def test_complex_conversion_pairs_real_and_imaginary_values(self) -> None:
        values = np.array(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
            dtype=np.complex64,
        )

        paired = complex_to_real_imag(values)

        self.assertEqual(paired.dtype, np.float32)
        self.assertEqual(paired.shape, (2, 2, 2))
        np.testing.assert_array_equal(paired[..., 0], values.real)
        np.testing.assert_array_equal(paired[..., 1], values.imag)
        restored = torch.view_as_complex(torch.from_numpy(paired).contiguous())
        np.testing.assert_array_equal(restored.numpy(), values)

    def test_h5_loader_preserves_physical_parameters_and_target_shape(self) -> None:
        T1 = np.array([0.8, 1.2], dtype=np.float32)
        T2 = np.array([0.07, 0.1], dtype=np.float32)
        B1 = np.array([0.9, 1.1], dtype=np.float32)
        signal = np.array(
            [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
            dtype=np.complex64,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "data.h5"
            with h5py.File(path, "w") as file:
                file.create_dataset("T1", data=T1)
                file.create_dataset("T2", data=T2)
                file.create_dataset("B1", data=B1)
                file.create_dataset("signal", data=signal)
                file.create_dataset("deriv_T1", data=signal)
                file.create_dataset("deriv_T2", data=signal)

            params, loaded_signal, dT1, dT2, n_samples, NTR = load_data_from_h5_file(
                str(path)
            )

        np.testing.assert_array_equal(params, np.stack((T1, T2, B1), axis=1))
        self.assertEqual(n_samples, 2)
        self.assertEqual(NTR, 2)

        paired = complex_to_real_imag(loaded_signal)
        loader = make_loader(
            params,
            paired,
            complex_to_real_imag(dT1),
            complex_to_real_imag(dT2),
            batchsize=2,
            device="cpu",
            shuffle=False,
        )
        x, sig, dT1_batch, dT2_batch = next(iter(loader))
        self.assertEqual(x.shape, (2, 3))
        self.assertEqual(sig.shape, (2, 2, 2))
        self.assertEqual(dT1_batch.shape, (2, 2, 2))
        self.assertEqual(dT2_batch.shape, (2, 2, 2))

    def test_parameter_sampling_is_physical_and_log_uniform(self) -> None:
        torch.manual_seed(42)
        samples = sample_parameters(SAMPLING_CONFIG, 20_000)

        self.assertEqual(samples.shape, (20_000, 3))
        for column, lower, upper in (
            (0, 0.1, 5.0),
            (1, 0.01, 2.0),
            (2, 0.8, 1.2),
        ):
            self.assertTrue(torch.all(samples[:, column] >= lower))
            self.assertTrue(torch.all(samples[:, column] <= upper))

        expected_log_T1_mean = (math.log(0.1) + math.log(5.0)) / 2
        expected_log_T2_mean = (math.log(0.01) + math.log(2.0)) / 2
        self.assertAlmostEqual(
            torch.log(samples[:, 0]).mean().item(),
            expected_log_T1_mean,
            delta=0.03,
        )
        self.assertAlmostEqual(
            torch.log(samples[:, 1]).mean().item(),
            expected_log_T2_mean,
            delta=0.03,
        )


class MetricReductionTest(unittest.TestCase):
    def test_metrics_reduce_over_all_non_batch_dimensions(self) -> None:
        true = np.array(
            [
                [[1.0, 2.0], [3.0, 4.0]],
                [[2.0, 1.0], [4.0, 3.0]],
            ],
            dtype=np.float32,
        )
        pred = true + np.array(
            [
                [[0.5, -0.5], [1.0, -1.0]],
                [[-0.25, 0.25], [-0.5, 0.5]],
            ],
            dtype=np.float32,
        )

        flat_true = true.reshape(2, -1)
        flat_pred = pred.reshape(2, -1)
        np.testing.assert_allclose(nrmse(pred, true), nrmse(flat_pred, flat_true))

        torch_true = torch.from_numpy(true)
        torch_pred = torch.from_numpy(pred)
        flat_loss = nmae_loss(
            torch_pred.reshape(2, -1),
            torch_true.reshape(2, -1),
        )
        torch.testing.assert_close(nmae_loss(torch_pred, torch_true), flat_loss)

    def test_metrics_handle_perfect_predictions_and_zero_targets(self) -> None:
        target = torch.zeros((2, 3, 2))
        self.assertEqual(nmae_loss(target, target).item(), 0.0)
        self.assertTrue(torch.isfinite(nmae_loss(torch.ones_like(target), target)))

        target_np = np.zeros((2, 3, 2), dtype=np.float32)
        self.assertTrue(np.all(nrmse(target_np, target_np) == 0.0))
        self.assertTrue(np.all(np.isfinite(nrmse(np.ones_like(target_np), target_np))))


if __name__ == "__main__":
    unittest.main()
