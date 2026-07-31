import torch
import torch.nn as nn


# def nrmse_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
#     rms_target = target.pow(2).mean().sqrt().clamp(min=1e-8)
#     return ((pred - target).pow(2).mean().sqrt()) / rms_target


def nmae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mae_error = (pred - target).abs().mean(dim=1)  # (B,)
    mae_target = target.abs().mean(dim=1).clamp(min=1e-8)  # (B,)
    return (mae_error / mae_target).mean()


class Loss(nn.Module):
    """Equal-weight L2 loss on signal and all three derivatives."""

    def forward(
        self,
        pred_signal,
        pred_dT1,
        pred_dT2,
        gt_signal,
        gt_dT1,
        gt_dT2,
    ) -> torch.Tensor:

        loss = nmae_loss
        loss_signal = loss(pred_signal, gt_signal)
        loss_dT1 = loss(pred_dT1, gt_dT1)
        loss_dT2 = loss(pred_dT2, gt_dT2)
        total = (loss_signal + loss_dT1 + loss_dT2) / 3.0
        return total, loss_signal, loss_dT1, loss_dT2
