"""Fixtures that all tests share.

Make the configurations, the models, the checkpoints and the datasets of the
tests only with these fixtures, not with dicts, FCModel(...) and torch.save in
each test. Then a change of the configuration, of the FCModel signature or of
the checkpoint format changes only this file of the tests.
"""

import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any
import h5py
import numpy as np
import numpy.typing as npt
import pytest
import torch
from src.model import FCModel
from src.utils import RF_TABLE_KEYS
from src.utils import rf_sha256


@pytest.fixture(autouse=True)
def no_julia(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Make an import of juliacall fail in each test without the marker julia.

    The import starts Julia, and installs it at the first use. A test that
    needs Julia must have the marker; another test can put a fake juliacall
    module in sys.modules.
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
        "N_SAMPLES": 40000,
        "DERIVATIVE_METHOD": "forward_sensitivity",
        **rf_tables,
        "TR_S": 0.00558,
        "TE_S": 0.002745,
        "TI_S": 0.01084,
        "TW_S": 0.0,
        "MAX_STATE": 64,
        "REPETITIONS": 5,
        "INVERSION_PREPULSE": True,
        "WAIT_SPOILING": True,
        "PY_UNDERSAMPLING_FACTOR": 1,
        "DELTA_K_SPOIL_RAD_PER_M": 0.0,
        "T1_lower": 0.1,
        "T1_upper": 6.0,
        "T2_lower": 0.01,
        "T2_upper": 2.0,
        "B1_lower": 0.5,
        "B1_upper": 1.5,
    }


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
def direct_config() -> dict[str, Any]:
    """A small model that has its own networks for dT1 and dT2."""
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


@pytest.fixture
def make_model() -> Callable[..., FCModel]:
    """Return a function that makes an FCModel from a model configuration."""

    def make(
        NTR: int,
        config: dict[str, Any],
        data_config: dict[str, Any] | None = None,
    ) -> FCModel:
        return FCModel(NTR, config, data_config)

    return make
