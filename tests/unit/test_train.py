from typing import Any
import pytest
from neuralbloch.scripts import train as train_script


@pytest.fixture(autouse=True)
def no_validation_dataset(monkeypatch: pytest.MonkeyPatch) -> None:
    # FCModel checks basis_initialization before train() uses the loader
    monkeypatch.setattr(train_script, "generate_val_loader", lambda config: (None, 5))


def test_a_direct_model_refuses_basis_initialization(config: dict[str, Any]) -> None:
    config["model"]["derivative"]["basis_initialization"] = "joint_pca"
    with pytest.raises(
        ValueError, match=r"mode 'direct' needs the keys \['dT1', 'dT2'\]"
    ):
        train_script.train(config)


def test_an_unknown_basis_initialization_is_refused(
    config: dict[str, Any], derived_config: dict[str, Any]
) -> None:
    derived_config["derivative"]["basis_initialization"] = "pca"
    config["model"] = derived_config
    with pytest.raises(ValueError, match="must be 'learned' or 'joint_pca'"):
        train_script.train(config)
