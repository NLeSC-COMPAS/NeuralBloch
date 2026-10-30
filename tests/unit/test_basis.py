import copy
from typing import Any
import torch
from src.basis import orthogonalize_shared_basis
from src.basis import representative_coeffs
from src.basis import truncate_shared_basis
from src.basis import worst_relative_output_difference
from src.model import FCModel


FIT_PARAMS = torch.tensor(
    [[0.4, 0.04, 0.8], [0.8, 0.07, 1.0], [1.2, 0.1, 0.9], [2.0, 0.2, 1.1]]
)
HELD_OUT = torch.tensor([[0.55, 0.05, 1.15], [3.0, 0.3, 0.85]])


def test_ordering_preserves_full_rank_and_truncation_uses_first_columns(
    rank4_config: dict[str, Any], data_config: dict[str, Any], rank4_model: FCModel
) -> None:
    ordered = copy.deepcopy(rank4_model)
    orthogonalize_shared_basis(ordered, FIT_PARAMS)

    assert worst_relative_output_difference(rank4_model, ordered, HELD_OUT) < 1e-5
    basis = ordered.U_signal[0].weight.detach()
    torch.testing.assert_close(basis.T @ basis, torch.eye(4), rtol=1e-5, atol=1e-5)

    smaller = truncate_shared_basis(ordered, rank4_config, data_config, 2)
    assert smaller.U_signal[0].weight.shape == (10, 2)
    with torch.no_grad():
        coefficients = representative_coeffs(ordered, HELD_OUT)
        expected = tuple(
            (value[:, :2] @ basis[:, :2].T).unflatten(-1, (5, 2))
            for value in coefficients
        )
        actual = smaller(HELD_OUT)
    for predicted, target in zip(actual, expected, strict=True):
        torch.testing.assert_close(predicted, target, rtol=1e-5, atol=1e-6)
