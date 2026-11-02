"""The smoke test of the Julia data generation (data/data_generation.jl).

The Julia code reads the configuration keys by name, and no other test runs
it. Run these tests after a change of the configuration keys. A test that
simulates takes about a minute, because the first simulation compiles the CUDA
kernels.
"""

import subprocess
from collections.abc import Callable
from pathlib import Path
import h5py
import numpy as np
import pytest
import torch
import yaml
from data.data_loading import generate_val_loader
from data.data_loading import load_data_from_h5_file
from src.config import load_config


pytestmark = pytest.mark.julia

# Make the training data as scripts/train.py does after run.py, with the seed
# of run.py, and save the tensors, the batch size and the shuffle of the loader
# in the file of the second argument
TRAIN_LOADER = """\
import sys
from juliacall import Main as jl  # before torch, as in run.py
from src.config import load_config
config = load_config(sys.argv[1])
seed = config["experiment"]["SEED"]
jl.seval(f"begin using Random; Random.seed!({seed}); true end")
jl.include("data/data_generation.jl")
import torch
from data.data_loading import generate_train_loader
loader = generate_train_loader(config)
torch.save(
    {
        "tensors": [tensor.cpu() for tensor in loader.tensors],
        "batch_size": loader.batch_size,
        "shuffle": loader.shuffle,
    },
    sys.argv[2],
)
"""


def test_the_julia_training_data_have_the_layout_of_the_model(
    julia_config_dir: Path,
    run_python: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
    pytestconfig: pytest.Config,
) -> None:
    config = load_config(julia_config_dir)
    output = tmp_path / "train_loader.pt"

    run_python("-c", TRAIN_LOADER, str(julia_config_dir), str(output))

    saved = torch.load(output)
    params, *targets = saved["tensors"]
    data = config["data"]
    NTR = len(np.loadtxt(pytestconfig.rootpath / data["RF_ANGLE_FILE"]))
    training = config["training"]
    n_sampled = training["ONLINE_BATCH_SIZE"] * training["REGEN_EVERY"]
    # generate_data keeps the samples with T1 > T2. NTR is not a possible
    # number of samples, so a target with the axes of the Julia arrays,
    # (NTR, N), fails
    assert 0 < len(params) <= n_sampled < NTR
    for target in targets:
        assert target.shape == (len(params), NTR, 2)
    # Each column is in the range of its parameter: the order is T1, T2, B1
    for column, name in enumerate(("T1", "T2", "B1")):
        values = params[:, column]
        assert data[f"{name}_lower"] * (1 - 1e-6) <= values.min()
        assert values.max() <= data[f"{name}_upper"] * (1 + 1e-6)
    assert (params[:, 0] > params[:, 1]).all()
    assert saved["batch_size"] == training["TRAIN_BATCH_SIZE"]
    assert saved["shuffle"]


def test_ranges_without_a_sample_with_t1_above_t2_give_an_error_that_names_them(
    julia_config_dir: Path,
    run_python: Callable[..., subprocess.CompletedProcess[str]],
    tmp_path: Path,
) -> None:
    # generate_data keeps only the samples with T1 > T2. Without a sample, the
    # simulation on the GPU fails with a CUDA error that does not give the
    # cause, so generate_data must raise an error before it
    path = julia_config_dir / "data.yaml"
    data = yaml.safe_load(path.read_text())
    data.update(T1_lower=0.1, T1_upper=0.2, T2_lower=0.3, T2_upper=2.0)
    path.write_text(yaml.safe_dump(data))

    result = run_python(
        "-c",
        TRAIN_LOADER,
        str(julia_config_dir),
        str(tmp_path / "train_loader.pt"),
        succeeds=False,
    )

    assert (
        "None of the 100 samples has T1 > T2: check the ranges data.T1_lower to "
        "data.T1_upper (0.1 to 0.2) and data.T2_lower to data.T2_upper (0.3 to "
        "2.0)"
    ) in result.stderr


def test_the_generate_task_writes_the_validation_and_test_datasets(
    julia_config_dir: Path,
    run_python: Callable[..., subprocess.CompletedProcess[str]],
    pytestconfig: pytest.Config,
) -> None:
    config = load_config(julia_config_dir)

    run_python("run.py", "generate", "--config-dir", str(julia_config_dir))

    experiment, data = config["experiment"], config["data"]
    params, signal, dT1, dT2, N, NTR = load_data_from_h5_file(
        experiment["DATASET_UNSPLIT_PATH"]
    )
    # The grid of N_SAMPLES, without the points with T1 <= T2
    side = round(data["N_SAMPLES"] ** (1 / 3))
    T1 = np.geomspace(data["T1_lower"], data["T1_upper"], side)
    T2 = np.geomspace(data["T2_lower"], data["T2_upper"], side)
    assert N == side * (T1[:, None] > T2[None, :]).sum()
    assert NTR == len(np.loadtxt(pytestconfig.rootpath / data["RF_ANGLE_FILE"]))
    for target in (signal, dT1, dT2):
        assert target.shape == (N, NTR)
        assert np.iscomplexobj(target)
    assert (params[:, 0] > params[:, 1]).all()

    # Each sample is in one of the two datasets of split_dataset
    validation = load_data_from_h5_file(experiment["DATASET_FOR_VALIDATION_PATH"])
    test = load_data_from_h5_file(experiment["DATASET_FOR_TESTING_PATH"])
    assert test[4] == max(1, int(N * 0.15))
    assert sorted(map(tuple, np.concatenate([validation[0], test[0]]))) == sorted(
        map(tuple, params)
    )
    # HDF5.jl writes the attributes as bytes
    with h5py.File(experiment["DATASET_FOR_VALIDATION_PATH"]) as file:
        assert file.attrs["derivative_method"].decode() == data["DERIVATIVE_METHOD"]

    # The validation loader gives the targets in the layout of the training data
    loader, _ = generate_val_loader(config)
    for target in loader.tensors[1:]:
        assert target.shape == (validation[4], NTR, 2)
