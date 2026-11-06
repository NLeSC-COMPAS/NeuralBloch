"""Fixtures that all tests share.

Make the configurations, the models, the checkpoints and the datasets of the
tests only with these fixtures, not with dicts, FCModel(...) and torch.save in
each test. Then a change of the configuration, of the FCModel signature or of
the checkpoint format changes only this file of the tests.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from types import ModuleType
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock
from unittest.mock import create_autospec
import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch
from neuralbloch.data.data_loading import TensorBatchLoader
from neuralbloch.model import FCModel
from neuralbloch.scripts import export_bundle
from neuralbloch.scripts.export_onnx import TARGETS
from neuralbloch.scripts.export_onnx import export_target
from neuralbloch.utils import RF_TABLE_KEYS
from neuralbloch.utils import rf_sha256
from neuralbloch.utils import sample_parameters


@pytest.fixture(autouse=True)
def no_julia(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an import of juliacall fail in each test without the marker julia.

    The import starts Julia, and installs it at the first use. A test that
    needs Julia must have the marker; another test can replace juliacall with
    a mock (julia_batch_mock).
    """
    if request.node.get_closest_marker("julia") is None:
        monkeypatch.setitem(sys.modules, "juliacall", None)


@pytest.fixture
def data_config(tmp_path: Path) -> dict[str, Any]:
    """The data configuration of config/data.yaml, with small RF tables.

    The RF tables are in tmp_path and have 5 values, so the models of this
    configuration have NTR = 5.
    """
    rf_tables = {}
    for key in RF_TABLE_KEYS:
        path = tmp_path / f"{key}.txt"
        path.write_text("".join(f"{i}.5\n" for i in range(5)))
        rf_tables[key] = str(path)
    return {
        "n_samples": 40000,
        "derivative_method": "forward_sensitivity",
        **rf_tables,
        "TR_s": 0.00558,
        "TE_s": 0.002745,
        "TI_s": 0.01084,
        "TW_s": 0.0,
        "max_state": 64,
        "repetitions": 5,
        "inversion_prepulse": True,
        "wait_spoiling": True,
        "py_undersampling_factor": 1,
        "delta_k_spoil_rad_per_m": 0.0,
        "parameter_ranges": {
            "T1_s": {"lower": 0.1, "upper": 6.0},
            "T2_s": {"lower": 0.01, "upper": 2.0},
            "B1": {"lower": 0.5, "upper": 1.5},
        },
    }


@pytest.fixture
def config(
    tmp_path: Path,
    data_config: dict[str, Any],
    direct_config: dict[str, Any],
) -> dict[str, Any]:
    """The configuration of a short training on CPU.

    It has the sections experiment, training, data (data_config) and model
    (direct_config), with the keys that Trainer reads and the batch sizes of
    the loaders. It does not have the keys of the Julia data generation. The
    outputs are in tmp_path. A test can change the values of this dict: the
    dicts of the other fixtures do not change.
    """
    return {
        "experiment": {
            "device": "cpu",
            "pt_file_path": str(tmp_path / "model.pt"),
            "log_dir": str(tmp_path / "log"),
        },
        "training": {
            "n_epochs": 2,
            "train_batch_size": 4,
            "learning_rate": 1e-3,
            "validation_batch_size": 8,
            "validate_every_n_epochs": 1,
            "compile_train_step": False,
            "LR_SCHEDULER": "cosine",
            "loss_function": "nmae",
            "regen_every_n_epochs": 100,
        },
        "data": dict(data_config),
        "model": dict(direct_config),
    }


@pytest.fixture
def synthetic_loader(data_config: dict[str, Any]) -> Callable[..., TensorBatchLoader]:
    """Return a function that makes a TensorBatchLoader of random data.

    The inputs are parameters in the ranges of data_config, and the signal and
    the two derivatives are random (N, 5, 2) tensors, as for NTR = 5. The data
    come from the global torch generator: seed it to get the same data.
    """

    def make(
        n_samples: int,
        batch_size: int,
        *,
        shuffle: bool = False,
    ) -> TensorBatchLoader:
        params = sample_parameters({"data": data_config}, n_samples)
        targets = (torch.randn(n_samples, 5, 2) for _ in range(3))
        return TensorBatchLoader((params, *targets), batch_size, shuffle=shuffle)

    return make


@pytest.fixture
def write_checkpoint() -> Callable[..., None]:
    """Return a function that saves a checkpoint as Trainer.train() saves it."""

    def write(
        path: Path,
        model: FCModel,
        model_config: dict[str, Any],
        data_config: dict[str, Any],
        epoch: int = 1,
    ) -> None:
        checkpoint = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "NTR": model.NTR,
            "model_config": model_config,
            "data_config": data_config,
            "rf_sha256": rf_sha256(data_config),
        }
        torch.save(checkpoint, path)

    return write


@pytest.fixture
def write_dataset() -> Callable[..., None]:
    """Return a function that writes an HDF5 dataset as save_dataset writes it.

    The function takes the parameters (N, 3), in the order T1, T2, B1, and the
    complex signal and derivatives (N, NTR).
    """

    def write(
        path: Path,
        params: npt.ArrayLike,
        signal: npt.ArrayLike,
        dT1: npt.ArrayLike,
        dT2: npt.ArrayLike,
    ) -> None:
        params = np.asarray(params, dtype=np.float32)
        with h5py.File(path, "w") as file:
            for index, key in enumerate(("T1", "T2", "B1")):
                file.create_dataset(key, data=params[:, index])
            for key, values in (
                ("signal", signal),
                ("deriv_T1", dT1),
                ("deriv_T2", dT2),
            ):
                file.create_dataset(key, data=np.asarray(values, dtype=np.complex64))

    return write


@pytest.fixture
def julia_batch_mock(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Replace juliacall with a mock; return the batch of its generate_data.

    The mock module in sys.modules has a Main with two Mocks: generate_data
    gives this batch, and include does nothing. A test can check their calls.
    The batch has 6 samples with T1 > T2 and NTR = 5, with the targets in the
    layout of the Julia arrays, (NTR, N). A test can change the batch.
    """
    rng = np.random.default_rng(0)
    batch = SimpleNamespace(
        T1=rng.uniform(1.0, 2.0, 6),
        T2=rng.uniform(0.05, 0.5, 6),
        B1=rng.uniform(0.5, 1.5, 6),
        **{
            key: rng.standard_normal((5, 6)) + 1j * rng.standard_normal((5, 6))
            for key in ("signal", "dM_dT1", "dM_dT2")
        },
    )
    juliacall = ModuleType("juliacall")
    juliacall.__dict__["Main"] = SimpleNamespace(
        include=Mock(return_value=None),
        generate_data=Mock(return_value=batch),
    )
    monkeypatch.setitem(sys.modules, "juliacall", juliacall)
    return batch


@pytest.fixture
def export_config(tmp_path: Path) -> dict[str, Any]:
    """The export configuration of config/export.yaml, for the models of NTR 5.

    The ONNX models and the bundle are in tmp_path. The family is of rank 3,
    the rank of direct_config.
    """
    return {
        "targets": {
            target: {"onnx_file_path": str(tmp_path / "onnx" / spec.bundle_file)}
            for target, spec in TARGETS.items()
        },
        "opset_version": 18,
        "fp16": False,
        "n_verify_samples": 8,
        "bundle_dir": str(tmp_path / "out" / "bundle"),
        "model_family": "fisp5-rank3",
        "model_version": "0.1.0",
        "offline": False,
    }


@pytest.fixture
def git_and_julia_in_export_mock(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replace the steps of export_bundle that run git and Julia with mocks.

    The mocks have the signatures of the real functions, so a call with other
    arguments fails. They give the commit "a" * 40, a release that does not
    exist, and BlochSimulators 0.10.0.
    """
    for name, value in (
        ("clean_commit", "a" * 40),
        ("require_unreleased", None),
        ("blochsimulators_facts", ("0.10.0", None)),
    ):
        mock = create_autospec(getattr(export_bundle, name), return_value=value)
        monkeypatch.setattr(export_bundle, name, mock)


@pytest.fixture(scope="session")
def direct_config() -> dict[str, Any]:
    """A small model that has its own networks for dT1 and dT2.

    onnx_models needs it, so all tests share this dict: do not change it.
    """
    return {
        "derivative_mode": "direct",
        "activation": "relu",
        "normalize_inputs": False,
        "NB_signal": 3,
        "NB_dT1": 2,
        "NB_dT2": 2,
        "hidden_size_signal": 4,
        "hidden_size_dT1": 4,
        "hidden_size_dT2": 4,
        "num_layers_signal": 2,
        "num_layers_dT1": 2,
        "num_layers_dT2": 2,
    }


@pytest.fixture
def derived_config() -> dict[str, Any]:
    """A small model that derives dT1 and dT2 from its signal network."""
    return {
        "derivative_mode": "derived",
        "activation": "gelu",
        "normalize_inputs": False,
        "NB_signal": 3,
        "hidden_size_signal": 4,
        "num_layers_signal": 3,
    }


@pytest.fixture(scope="session")
def make_model() -> Callable[..., FCModel]:
    """Return a function that makes an FCModel from a model configuration."""

    def make(
        NTR: int,
        config: dict[str, Any],
        data_config: dict[str, Any] | None = None,
    ) -> FCModel:
        return FCModel(NTR, config, data_config)

    return make


@pytest.fixture
def rank4_config(derived_config: dict[str, Any]) -> dict[str, Any]:
    """A derived model of rank 4, for the tests of the shared basis."""
    return {**derived_config, "NB_signal": 4, "hidden_size_signal": 8}


@pytest.fixture
def rank4_model(
    rank4_config: dict[str, Any],
    data_config: dict[str, Any],
    make_model: Callable[..., FCModel],
) -> FCModel:
    """The model of rank4_config, with the weights of seed 42."""
    torch.manual_seed(42)
    return make_model(5, rank4_config, data_config).eval()


@pytest.fixture(scope="session")
def exported_model(
    direct_config: dict[str, Any], make_model: Callable[..., FCModel]
) -> FCModel:
    """The direct model with the weights of seed 0 that onnx_models exports.

    All tests share this model: do not change it.
    """
    torch.manual_seed(0)
    return make_model(5, direct_config).eval()


@pytest.fixture(scope="session")
def onnx_models(
    exported_model: FCModel, tmp_path_factory: pytest.TempPathFactory
) -> dict[str, Path]:
    """The ONNX model of each target of exported_model.

    An export takes about a second, so all tests share these files. Copy a
    file before a test changes it.
    """
    directory = tmp_path_factory.mktemp("onnx")
    # A batch of 2 is enough to trace the model
    example = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])
    paths = {target: directory / spec.bundle_file for target, spec in TARGETS.items()}
    for target, path in paths.items():
        export_target(exported_model, target, path, example=example, opset=18)
    return paths
