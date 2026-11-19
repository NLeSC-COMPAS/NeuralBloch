from pathlib import Path
import h5py
import numpy as np


def split_dataset(
    dataset_unsplit_path: Path,
    dataset_for_validation_path: Path,
    dataset_for_testing_path: Path,
    seed: int,
) -> None:
    """Split an unsplit dataset into validation and test datasets.

    Args:
        dataset_unsplit_path: The dataset to split.
        dataset_for_validation_path: Where to write the validation dataset.
        dataset_for_testing_path: Where to write the test dataset.
        seed: Seed of the random split.
    """
    with h5py.File(dataset_unsplit_path, "r") as f:
        N = f["T1"].shape[0]

        # Randomly select samples for validation and testing reproducibly.
        rng = np.random.default_rng(seed)
        idx = rng.permutation(N)

        n_test = max(1, int(N * 0.15))

        va_idx = idx[n_test:]
        te_idx = idx[:n_test]

        print(f"Total: {N}  →  validation: {len(va_idx)}  test: {len(te_idx)}")

        # Sort indices for efficient HDF5 reading
        va_idx_sorted = np.sort(va_idx)
        te_idx_sorted = np.sort(te_idx)

        keys = list(f.keys())

        # Create separate validation and test datasets
        with h5py.File(dataset_for_validation_path, "w") as f_va:
            with h5py.File(dataset_for_testing_path, "w") as f_te:
                for key, value in f.attrs.items():
                    f_va.attrs[key] = value
                    f_te.attrs[key] = value

                for key in keys:
                    data = f[key][:]
                    # Handle parameter arrays (1D) and signal arrays (2D)
                    if data.ndim == 1:
                        f_va.create_dataset(key, data=data[va_idx_sorted])
                        f_te.create_dataset(key, data=data[te_idx_sorted])
                    else:
                        f_va.create_dataset(key, data=data[va_idx_sorted, :])
                        f_te.create_dataset(key, data=data[te_idx_sorted, :])

    print(f"Saved validation dataset → {dataset_for_validation_path}")
    print(f"Saved test dataset  → {dataset_for_testing_path}")
