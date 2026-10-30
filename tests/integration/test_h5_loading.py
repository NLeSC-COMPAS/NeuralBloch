from collections.abc import Callable
from pathlib import Path
import numpy as np
from data.data_loading import load_data_from_h5_file
from data.data_loading import make_loader
from src.utils import complex_to_real_imag


def test_h5_loader_preserves_physical_parameters_and_target_shape(
    write_dataset: Callable[..., None], tmp_path: Path
) -> None:
    T1 = np.array([0.8, 1.2], dtype=np.float32)
    T2 = np.array([0.07, 0.1], dtype=np.float32)
    B1 = np.array([0.9, 1.1], dtype=np.float32)
    signal = np.array(
        [[1 + 2j, 3 + 4j], [5 + 6j, 7 + 8j]],
        dtype=np.complex64,
    )

    path = tmp_path / "data.h5"
    write_dataset(path, np.stack((T1, T2, B1), axis=1), signal, signal, signal)

    params, loaded_signal, dT1, dT2, n_samples, NTR = load_data_from_h5_file(str(path))

    np.testing.assert_array_equal(params, np.stack((T1, T2, B1), axis=1))
    assert n_samples == 2
    assert NTR == 2

    paired = complex_to_real_imag(loaded_signal)
    loader = make_loader(
        params,
        paired,
        complex_to_real_imag(dT1),
        complex_to_real_imag(dT2),
        batchsize=2,
        device="cpu",
        shuffle=False,
    )
    x, sig, dT1_batch, dT2_batch = next(iter(loader))
    assert x.shape == (2, 3)
    assert sig.shape == (2, 2, 2)
    assert dT1_batch.shape == (2, 2, 2)
    assert dT2_batch.shape == (2, 2, 2)
