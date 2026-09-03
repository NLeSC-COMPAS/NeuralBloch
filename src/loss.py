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


def relative_l2_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Compute the relative L2 error averaged over samples.

    Args:
        pred: Predicted values, shape ``(B, ...)``.
        target: Ground-truth values with the same shape as ``pred``.

    Returns:
        The batch-averaged relative L2 error as a scalar tensor.
    """
    reduce_dims = tuple(range(1, pred.ndim))
    error = torch.linalg.vector_norm(pred - target, dim=reduce_dims)
    target_norm = torch.linalg.vector_norm(target, dim=reduce_dims).clamp(min=1e-8)
    return (error / target_norm).mean()


def relative_l2_squared_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Compute mean squared relative L2 error over samples."""
    reduce_dims = tuple(range(1, pred.ndim))
    error_energy = (pred - target).square().sum(dim=reduce_dims)
    target_energy = target.square().sum(dim=reduce_dims).clamp(min=1e-8)
    return (error_energy / target_energy).mean()


def hybrid_relative_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Blend relative L1 and L2 errors with equal weight."""
    return 0.5 * (nmae_loss(pred, target) + relative_l2_loss(pred, target))


class Loss(nn.Module):
    """Equal-weight loss on the signal and both derivatives."""

    def __init__(self, loss_function: str = "nmae") -> None:
        """Initialize the loss selected by name.

        Args:
            loss_function: Name of the normalized loss function.
        """
        super().__init__()

        if loss_function == "nmae":
            self.loss_function = nmae_loss
        elif loss_function == "relative_l2":
            self.loss_function = relative_l2_loss
        elif loss_function == "relative_l2_squared":
            self.loss_function = relative_l2_squared_loss
        elif loss_function == "hybrid_relative":
            self.loss_function = hybrid_relative_loss
        else:
            raise ValueError(
                "loss_function must be 'nmae', 'relative_l2', "
                "'relative_l2_squared', or 'hybrid_relative', got "
                f"{loss_function!r}"
            )

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
        loss_signal = self.loss_function(pred_signal, gt_signal)
        loss_dT1 = self.loss_function(pred_dT1, gt_dT1)
        loss_dT2 = self.loss_function(pred_dT2, gt_dT2)
        total = (loss_signal + loss_dT1 + loss_dT2) / 3.0
        return total, loss_signal, loss_dT1, loss_dT2
