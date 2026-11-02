import numpy as np
import torch
from src.loss import nmae_loss
from src.metrics import nrmse


def test_metrics_reduce_over_all_non_batch_dimensions() -> None:
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


def test_metrics_handle_perfect_predictions_and_zero_targets() -> None:
    target = torch.zeros((2, 3, 2))
    assert nmae_loss(target, target).item() == 0.0
    assert torch.isfinite(nmae_loss(torch.ones_like(target), target))

    target_np = np.zeros((2, 3, 2), dtype=np.float32)
    assert np.all(nrmse(target_np, target_np) == 0.0)
    assert np.all(np.isfinite(nrmse(np.ones_like(target_np), target_np)))
