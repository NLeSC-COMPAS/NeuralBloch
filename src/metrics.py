import numpy as np
import numpy.typing as npt


def nrmse(
    pred: npt.NDArray[np.float32],
    true: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute the per-sample normalized RMSE as a percentage.

    Args:
        pred: Predicted values, shape ``(N, ...)``.
        true: Ground-truth values with the same shape as ``pred``.

    Returns:
        An array of per-sample NRMSE values in percent.
    """
    reduce_axes = tuple(range(1, pred.ndim))
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=reduce_axes))
    scale = np.sqrt(np.mean(true**2, axis=reduce_axes)) + 1e-12
    return 100.0 * rmse / scale
