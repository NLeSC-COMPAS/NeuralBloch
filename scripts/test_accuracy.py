from typing import Any
import numpy as np
import numpy.typing as npt
import torch
from data.data_loading import load_data_from_h5_file
from src.utils import get_device
from src.utils import load_model
from src.utils import split_real_imag


def test_accuracy(config: dict[str, Any]) -> None:
    """Evaluate the trained model on the test dataset and print NRMSE scores.

    Args:
        config: Full configuration dictionary.
    """
    device = get_device(config)
    model, _ = load_model(config, device)
    params, signal, dT1, dT2, N, _ = load_data_from_h5_file(
        config["experiment"]["DATASET_FOR_TESTING_PATH"]
    )

    x = torch.tensor(params, dtype=torch.float32, device=device)

    infer_batch_size = 200

    p_sig_list, p_dT1_list, p_dT2_list = [], [], []

    with torch.no_grad():
        for i in range(0, N, infer_batch_size):
            x_batch = x[i : i + infer_batch_size]
            ps, pd1, pd2 = model(x_batch)
            p_sig_list.append(ps.cpu())
            p_dT1_list.append(pd1.cpu())
            p_dT2_list.append(pd2.cpu())

    p_sig = torch.cat(p_sig_list).numpy()
    p_dT1 = torch.cat(p_dT1_list).numpy()
    p_dT2 = torch.cat(p_dT2_list).numpy()

    sig_true = split_real_imag(signal)
    dT1_true = split_real_imag(dT1)
    dT2_true = split_real_imag(dT2)

    nrmse_sig = nrmse(p_sig, sig_true)
    nrmse_dT1 = nrmse(p_dT1, dT1_true)
    nrmse_dT2 = nrmse(p_dT2, dT2_true)

    print(f"\n=== Test results ({N} samples) ===")
    print(f"  Signal NRMSE : {nrmse_sig.mean():.3f}% ± {nrmse_sig.std():.3f}%")
    print(f"  dT1    NRMSE : {nrmse_dT1.mean():.3f}% ± {nrmse_dT1.std():.3f}%")
    print(f"  dT2    NRMSE : {nrmse_dT2.mean():.3f}% ± {nrmse_dT2.std():.3f}%")


def nrmse(
    pred: npt.NDArray[np.float32],
    true: npt.NDArray[np.float32],
) -> npt.NDArray[np.float32]:
    """Compute the per-sample normalized RMSE as a percentage.

    Args:
        pred: Predicted values, shape ``(N, M)``.
        true: Ground-truth values with the same shape as ``pred``.

    Returns:
        An array of per-sample NRMSE values in percent.
    """
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    scale = np.sqrt(np.mean(true**2, axis=1)) + 1e-12
    return 100.0 * rmse / scale
