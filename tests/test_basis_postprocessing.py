import copy
import tempfile
import unittest
from pathlib import Path
import h5py
import numpy as np
import torch
from scripts.orthogonalize_basis import load_checkpoint
from scripts.orthogonalize_basis import postprocess_basis
from src.basis import orthogonalize_shared_basis
from src.basis import representative_coeffs
from src.basis import truncate_shared_basis
from src.basis import worst_relative_output_difference
from src.model import FCModel


MODEL_CONFIG = {
    "activation": "gelu",
    "derivative_mode": "derived",
    "NB_signal": 4,
    "hidden_size_signal": 8,
    "num_layers_signal": 3,
}
DATA_CONFIG = {
    "T1_lower": 0.1,
    "T1_upper": 5.0,
    "T2_lower": 0.01,
    "T2_upper": 2.0,
    "B1_lower": 0.8,
    "B1_upper": 1.2,
}


class BasisPostprocessingTest(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(42)
        self.model = FCModel(5, MODEL_CONFIG, DATA_CONFIG).eval()
        self.fit_params = torch.tensor(
            [[0.4, 0.04, 0.8], [0.8, 0.07, 1.0], [1.2, 0.1, 0.9], [2.0, 0.2, 1.1]]
        )
        self.held_out = torch.tensor([[0.55, 0.05, 1.15], [3.0, 0.3, 0.85]])

    def test_ordering_preserves_full_rank_and_truncation_uses_first_columns(
        self,
    ) -> None:
        ordered = copy.deepcopy(self.model)
        orthogonalize_shared_basis(ordered, self.fit_params)

        self.assertLess(
            worst_relative_output_difference(self.model, ordered, self.held_out),
            1e-5,
        )
        basis = ordered.U_signal[0].weight.detach()
        torch.testing.assert_close(basis.T @ basis, torch.eye(4), rtol=1e-5, atol=1e-5)

        smaller = truncate_shared_basis(ordered, MODEL_CONFIG, DATA_CONFIG, 2)
        self.assertEqual(smaller.U_signal[0].weight.shape, (10, 2))
        with torch.no_grad():
            coefficients = representative_coeffs(ordered, self.held_out)
            expected = tuple(
                (value[:, :2] @ basis[:, :2].T).unflatten(-1, (5, 2))
                for value in coefficients
            )
            actual = smaller(self.held_out)
        for predicted, target in zip(actual, expected, strict=True):
            torch.testing.assert_close(predicted, target, rtol=1e-5, atol=1e-6)

    def test_command_writes_new_checkpoints_and_keeps_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            checkpoint_path = root / "trained.pt"
            dataset_path = root / "validation.h5"
            output_path = root / "ordered.pt"
            checkpoint = {
                "NTR": 5,
                "epoch": 10,
                "model_config": MODEL_CONFIG,
                "data_config": DATA_CONFIG,
                "model_state": self.model.state_dict(),
            }
            torch.save(checkpoint, checkpoint_path)
            original_bytes = checkpoint_path.read_bytes()
            params = self.fit_params.numpy()
            with h5py.File(dataset_path, "w") as dataset:
                for index, key in enumerate(("T1", "T2", "B1")):
                    dataset[key] = params[:, index]
                zeros = np.zeros((len(params), 5), dtype=np.complex64)
                for key in ("signal", "deriv_T1", "deriv_T2"):
                    dataset[key] = zeros

            written = postprocess_basis(
                checkpoint_path, dataset_path, output_path, [2], "cpu"
            )
            self.assertEqual(written, [output_path, root / "ordered_r2.pt"])
            self.assertEqual(checkpoint_path.read_bytes(), original_bytes)
            ordered, _ = load_checkpoint(output_path, "cpu")
            smaller, saved = load_checkpoint(root / "ordered_r2.pt", "cpu")
            self.assertLess(
                worst_relative_output_difference(self.model, ordered, self.held_out),
                1e-5,
            )
            self.assertEqual(smaller.U_signal[0].in_features, 2)
            self.assertEqual(saved["model_config"]["NB_signal"], 2)

            with self.assertRaises(ValueError):
                postprocess_basis(
                    checkpoint_path, dataset_path, checkpoint_path, [], "cpu"
                )
