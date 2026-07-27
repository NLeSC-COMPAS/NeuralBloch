import h5py
import numpy as np

def split_dataset(config):
    """Split an unsplit dataset into validation and test datasets."""

    with h5py.File(config["DATASET_UNSPLIT_PATH"], "r") as f:

        N = f["T1"].shape[0]

        # Randomly select samples for validation and testing
        idx     = np.random.permutation(N)

        n_test  = max(1, int(N * 0.15))

        va_idx  = idx[n_test:]
        te_idx  = idx[:n_test]

        print(f"Total: {N}  →  validation: {len(va_idx)}  test: {len(te_idx)}")

        # Sort indices for efficient HDF5 reading
        va_idx_sorted = np.sort(va_idx)
        te_idx_sorted = np.sort(te_idx)

        keys = list(f.keys())

        # Create separate validation and test datasets
        with h5py.File(config["DATASET_FOR_VALIDATION_PATH"], "w") as f_va:
            with h5py.File(config["DATASET_FOR_TESTING_PATH"], "w") as f_te:
                for key in keys:
                    data = f[key][:]
                    # Handle parameter arrays (1D) and signal arrays (2D)
                    if data.ndim == 1:
                        f_va.create_dataset(key, data=data[va_idx_sorted])
                        f_te.create_dataset(key, data=data[te_idx_sorted])
                    else:
                        f_va.create_dataset(key, data=data[va_idx_sorted, :])
                        f_te.create_dataset(key, data=data[te_idx_sorted, :])

    print(f"Saved validation dataset → {config["DATASET_FOR_VALIDATION_PATH"]}")
    print(f"Saved test dataset  → {config["DATASET_FOR_TESTING_PATH"]}")