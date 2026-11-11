"""The smoke test of the Julia data generation (data_generation.jl).

The Julia code reads the configuration keys by name, and no other test runs
it. Run these tests after a change of the configuration keys. A test that
simulates takes about a minute, because the first simulation compiles the CUDA
kernels.
"""

import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
import h5py
import numpy as np
import pytest
import yaml
from helpers import has_gpu
from neuralbloch.config import load_config
from neuralbloch.data import DATA_GENERATION_JL
from neuralbloch.data.data_loading import generate_train_loader
from neuralbloch.data.data_loading import generate_val_loader
from neuralbloch.data.data_loading import load_data_from_h5_file


@pytest.fixture(scope="session")
def julia_data_generation() -> None:
    """Include data_generation.jl in Julia Main, once for all tests."""
    # Not at the top: the import starts Julia, and pytest imports this file
    # also on a machine without a GPU
    from juliacall import Main

    Main.include(DATA_GENERATION_JL)


@pytest.fixture
def julia_config_dir(tmp_path: Path, pytestconfig: pytest.Config) -> Path:
    """A copy of config/ for a small data generation, in tmp_path.

    The Julia code reads the configuration keys by name, so the copy keeps all
    keys and values of config/. Only the sizes are smaller, and the datasets
    and the outputs are in tmp_path. The RF tables stay the tables of the
    repository, with paths relative to the repository root.
    """
    config = load_config(pytestconfig.rootpath / "config")
    config["experiment"].update(
        DATASET_UNSPLIT_PATH=str(tmp_path / "dataset_unsplit.h5"),
        DATASET_FOR_VALIDATION_PATH=str(tmp_path / "dataset_for_validation.h5"),
        DATASET_FOR_TESTING_PATH=str(tmp_path / "dataset_for_testing.h5"),
        PT_FILE_PATH=str(tmp_path / "model.pt"),
        LOG_FILE_PATH=str(tmp_path / "log"),
    )
    # A grid of 5 values of T1, T2 and B1
    config["data"]["N_SAMPLES"] = 125
    # 100 samples of the online training data
    config["training"].update(ONLINE_BATCH_SIZE=50, REGEN_EVERY=2)

    directory = tmp_path / "config"
    directory.mkdir()
    for name, section in config.items():
        (directory / f"{name}.yaml").write_text(yaml.safe_dump(section))
    return directory


@pytest.fixture
def run_python(
    pytestconfig: pytest.Config,
) -> Callable[..., subprocess.CompletedProcess[str]]:
    """Return a function that runs Python in the repository root.

    The function takes the arguments of the Python command (for example -m
    and a module) and fails the test with the output of a failed run. With
    succeeds=False, it fails the test with the output of a run that succeeds.
    A run that takes more than 30 minutes fails: the first run in a new
    virtual environment installs the Julia packages (a few minutes), but a
    run that hangs, for example on the GPU, must not stop the tests.
    """

    def run(*args: str, succeeds: bool = True) -> subprocess.CompletedProcess[str]:
        result = subprocess.run(
            [sys.executable, *args],
            cwd=pytestconfig.rootpath,
            capture_output=True,
            text=True,
            timeout=1800,
        )
        output = result.stdout[-2000:] + result.stderr[-5000:]
        assert (result.returncode == 0) == succeeds, output
        return result

    return run


@has_gpu
@pytest.mark.usefixtures("julia_data_generation")
def test_the_julia_training_data_have_the_layout_of_the_model(
    julia_config_dir: Path,
    pytestconfig: pytest.Config,
) -> None:
    config = load_config(julia_config_dir)

    loader = generate_train_loader(config)

    params, *targets = (tensor.cpu() for tensor in loader.tensors)
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
    assert loader.batch_size == training["TRAIN_BATCH_SIZE"]
    assert loader.shuffle


@has_gpu
@pytest.mark.usefixtures("julia_data_generation")
def test_ranges_without_a_sample_with_t1_above_t2_give_an_error_that_names_them(
    julia_config_dir: Path,
) -> None:
    from juliacall import JuliaError  # noqa: PLC0415

    # generate_data keeps only the samples with T1 > T2. Without a sample, the
    # simulation on the GPU fails with a CUDA error that does not give the
    # cause, so generate_data must raise an error before it
    config = load_config(julia_config_dir)
    config["data"].update(T1_lower=0.1, T1_upper=0.2, T2_lower=0.3, T2_upper=2.0)

    with pytest.raises(JuliaError) as error:
        generate_train_loader(config)

    assert (
        "None of the 100 samples has T1 > T2: check the ranges data.T1_lower to "
        "data.T1_upper (0.1 to 0.2) and data.T2_lower to data.T2_upper (0.3 to "
        "2.0)"
    ) in str(error.value)


@has_gpu
def test_the_generate_task_writes_the_validation_and_test_datasets(
    julia_config_dir: Path,
    run_python: Callable[..., subprocess.CompletedProcess[str]],
    pytestconfig: pytest.Config,
) -> None:
    config = load_config(julia_config_dir)

    run_python(
        "-m", "neuralbloch.cli", "generate", "--config-dir", str(julia_config_dir)
    )

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
