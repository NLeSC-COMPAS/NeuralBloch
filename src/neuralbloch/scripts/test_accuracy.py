from pathlib import Path
import torch
from neuralbloch.data.data_loading import load_data_from_h5_file
from neuralbloch.metrics import nrmse
from neuralbloch.utils import complex_to_real_imag
from neuralbloch.utils import get_device
from neuralbloch.utils import load_model


def test_accuracy(
    pt_file_path: Path,
    dataset_for_testing_path: Path,
    device: str,
) -> None:
    """Evaluate the trained model on the test dataset and print NRMSE scores.

    Args:
        pt_file_path: The checkpoint file (``experiment.pt_file_path``).
        dataset_for_testing_path: The test dataset
            (``experiment.dataset_for_testing_path``).
        device: The wanted device (``experiment.device``). The CPU is used
            if CUDA is not available.
    """
    device = get_device(device)
    model, _ = load_model(pt_file_path, device)
    params, signal, dT1, dT2, N, _ = load_data_from_h5_file(dataset_for_testing_path)

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

    sig_true = complex_to_real_imag(signal)
    dT1_true = complex_to_real_imag(dT1)
    dT2_true = complex_to_real_imag(dT2)

    nrmse_sig = nrmse(p_sig, sig_true)
    nrmse_dT1 = nrmse(p_dT1, dT1_true)
    nrmse_dT2 = nrmse(p_dT2, dT2_true)

    print(f"\n=== Test results ({N} samples) ===")
    print(f"  Signal NRMSE : {nrmse_sig.mean():.3f}% ± {nrmse_sig.std():.3f}%")
    print(f"  dT1    NRMSE : {nrmse_dT1.mean():.3f}% ± {nrmse_dT1.std():.3f}%")
    print(f"  dT2    NRMSE : {nrmse_dT2.mean():.3f}% ± {nrmse_dT2.std():.3f}%")
