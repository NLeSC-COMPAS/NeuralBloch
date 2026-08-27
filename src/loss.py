import torch
import torch.nn as nn


# def nrmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     rms_target = target.pow(2).mean().sqrt().clamp(min=1e-8)
#     return ((pred - target).pow(2).mean().sqrt()) / rms_target


def nmae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the mean absolute error normalized by the target magnitude.

    Args:
        pred: Predicted values, shape ``(B, ...)``.
        target: Ground-truth values with the same shape as ``pred``.

    Returns:
        The batch-averaged normalized MAE as a scalar tensor.
    """
    reduce_dims = tuple(range(1, pred.ndim))
    mae_error = (pred - target).abs().mean(dim=reduce_dims)
    mae_target = target.abs().mean(dim=reduce_dims).clamp(min=1e-8)
    return (mae_error / mae_target).mean()


class Loss(nn.Module):
    """Equal-weight normalized MAE on signal and both derivatives."""

    def forward(
        self,
        pred_signal: torch.Tensor,
        pred_dT1: torch.Tensor,
        pred_dT2: torch.Tensor,
        gt_signal: torch.Tensor,
        gt_dT1: torch.Tensor,
        gt_dT2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute the equal-weight loss over the signal and its derivatives.

        Args:
            pred_signal: Predicted signal.
            pred_dT1: Predicted derivative with respect to T1.
            pred_dT2: Predicted derivative with respect to T2.
            gt_signal: Ground-truth signal.
            gt_dT1: Ground-truth derivative with respect to T1.
            gt_dT2: Ground-truth derivative with respect to T2.

        Returns:
            A tuple of the mean total loss followed by the signal, T1, and T2
            loss terms.
        """
        loss = nmae_loss
        loss_signal = loss(pred_signal, gt_signal)
        loss_dT1 = loss(pred_dT1, gt_dT1)
        loss_dT2 = loss(pred_dT2, gt_dT2)
        total = (loss_signal + loss_dT1 + loss_dT2) / 3.0
        return total, loss_signal, loss_dT1, loss_dT2
