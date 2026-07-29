from pathlib import Path

import numpy as np
import onnxruntime as ort

from data.data_loading import load_data_from_h5_file
from src.utils import split_real_imag


def test_accuracy_onnx(config):
    session = ort.InferenceSession(
        str(Path("outputs/model.onnx")),
        providers=["CPUExecutionProvider"],
    )
    input_name = session.get_inputs()[0].name

    params, signal, dT1, dT2, N, _ = load_data_from_h5_file(
        config["experiment"]["DATASET_FOR_TESTING_PATH"]
    )

    infer_batch_size = 200

    p_sig_list, p_dT1_list, p_dT2_list = [], [], []

    for i in range(0, N, infer_batch_size):
        x_batch = params[i : i + infer_batch_size].astype(np.float32, copy=False)
        ps, pd1, pd2 = session.run(None, {input_name: x_batch})
        p_sig_list.append(ps)
        p_dT1_list.append(pd1)
        p_dT2_list.append(pd2)

    p_sig = np.concatenate(p_sig_list, axis=0)
    p_dT1 = np.concatenate(p_dT1_list, axis=0)
    p_dT2 = np.concatenate(p_dT2_list, axis=0)

    sig_true = split_real_imag(signal)
    dT1_true = split_real_imag(dT1)
    dT2_true = split_real_imag(dT2)

    nrmse_sig = nrmse(p_sig, sig_true)
    nrmse_dT1 = nrmse(p_dT1, dT1_true)
    nrmse_dT2 = nrmse(p_dT2, dT2_true)

    print(f"\n=== Test results with ONNX Runtime ({N} samples) ===")
    print(f"  Signal NRMSE : {nrmse_sig.mean():.3f}% ± {nrmse_sig.std():.3f}%")
    print(f"  dT1    NRMSE : {nrmse_dT1.mean():.3f}% ± {nrmse_dT1.std():.3f}%")
    print(f"  dT2    NRMSE : {nrmse_dT2.mean():.3f}% ± {nrmse_dT2.std():.3f}%")


def nrmse(pred, true):
    rmse = np.sqrt(np.mean((pred - true) ** 2, axis=1))
    scale = np.sqrt(np.mean(true ** 2, axis=1)) + 1e-12
    return 100.0 * rmse / scale