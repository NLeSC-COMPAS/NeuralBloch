"""Fixtures that all tests share.

Make the model configurations and the models of the tests only with these
fixtures, not with dicts and FCModel(...) in each test. Then a change of the
configuration or of the FCModel signature changes only this file.
"""

from collections.abc import Callable
from typing import Any
import pytest
from src.model import FCModel


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
