"""The config schema (types.py): the config in config/ passes, bad values fail."""

import math
from pathlib import Path
from typing import Any
import pytest
from pydantic import ValidationError
from neuralbloch.config import load_config
from neuralbloch.types import Config


CONFIG_DIR = Path(__file__).parents[2] / "config"


# =====================================================================================#
# Helpers: set, remove or read one value by its dotted key path, like "data.TR_s"
# =====================================================================================#
def set_at(config: dict[str, Any], key_path: str, value: object) -> None:
    """Set the value at "a.b.c" of a config dict: config["a"]["b"]["c"] = value."""
    *parents, last = key_path.split(".")
    node = config
    for key in parents:
        node = node[key]
    node[last] = value


def remove_at(config: dict[str, Any], key_path: str) -> None:
    """Remove the key at "a.b.c" of a config dict: del config["a"]["b"]["c"]."""
    *parents, last = key_path.split(".")
    node = config
    for key in parents:
        node = node[key]
    del node[last]


def item_at(config: dict[str, Any], key_path: str) -> object:
    """Return the value at "a.b.c" of a config dict: config["a"]["b"]["c"]."""
    value: Any = config
    for key in key_path.split("."):
        value = value[key]
    return value


def attribute_at(config: Config, key_path: str) -> object:
    """Return the value at "a.b.c" of a validated config: config.a.b.c."""
    value: object = config
    for key in key_path.split("."):
        value = getattr(value, key)
    return value


# =====================================================================================#
# Fixtures
# =====================================================================================#
@pytest.fixture
def config_dict() -> dict[str, Any]:
    return load_config(CONFIG_DIR)


@pytest.fixture
def plateau() -> dict[str, Any]:
    return {"name": "plateau", "patience_epochs": 100, "factor": 0.5, "min_lr": 1e-5}


@pytest.fixture
def derived() -> dict[str, Any]:
    return {"mode": "derived", "basis_initialization": "joint_pca"}


# =====================================================================================#


def test_config_dir_is_valid(config_dict: dict[str, Any]) -> None:
    """Test that the YAML files in config/ follow the schema."""
    Config.model_validate(config_dict)


@pytest.mark.parametrize(
    "key_path",
    [
        "data.rf_angle_file",
        "data.rf_phase_file",
        "experiment.dataset_unsplit_path",
        "experiment.dataset_for_validation_path",
        "experiment.dataset_for_testing_path",
        "experiment.pt_file_path",
        "experiment.log_dir",
        "export.targets.signal_and_derivs.onnx_file_path",
        "export.targets.signal.onnx_file_path",
        "export.targets.coeffs.onnx_file_path",
        "export.bundle_dir",
    ],
)
def test_paths_become_path_objects(config_dict: dict[str, Any], key_path: str) -> None:
    yaml_value = item_at(config_dict, key_path)
    assert isinstance(yaml_value, str)
    config = Config.model_validate(config_dict)
    assert attribute_at(config, key_path) == Path(yaml_value)


# The float keys of config/
@pytest.mark.parametrize(
    ("key_path", "value"),
    [
        ("data.parameter_ranges.T1_s.lower", 1),
        ("data.parameter_ranges.T1_s.upper", 6),
        ("data.parameter_ranges.T2_s.lower", 1),
        ("data.parameter_ranges.T2_s.upper", 2),
        ("data.parameter_ranges.B1.lower", 1),
        ("data.parameter_ranges.B1.upper", 2),
        ("data.TR_s", 1),
        ("data.TE_s", 1),
        ("data.TI_s", 0),
        ("data.TW_s", 0),
        ("data.delta_k_spoil_rad_per_m", 0),
        ("training.learning_rate", 1),
    ],
)
def test_int_is_accepted_for_a_float(
    config_dict: dict[str, Any], key_path: str, value: int
) -> None:
    set_at(config_dict, key_path, value)
    validated = attribute_at(Config.model_validate(config_dict), key_path)
    assert validated == value
    assert isinstance(validated, float)


@pytest.mark.parametrize(
    "key_path",
    [
        "training.n_epochs",
        "data.parameter_ranges.T1_s.lower",
        "model.derivative.dT1",
        "export.targets.coeffs",
    ],
)
def test_missing_keys_are_refused(config_dict: dict[str, Any], key_path: str) -> None:
    remove_at(config_dict, key_path)
    with pytest.raises(ValidationError, match="Field required"):
        Config.model_validate(config_dict)


@pytest.mark.parametrize(
    ("key_path", "value", "message"),
    [
        # The rules of Section, which all sections follow
        ("training.unknown_key", 1, "Extra inputs are not permitted"),
        # No conversion of strings and bools
        ("training.n_epochs", "5000", "Input should be a valid integer"),
        ("training.n_epochs", True, "Input should be a valid integer"),
        ("export.fp16", "false", "Input should be a valid boolean"),
        ("training.learning_rate", math.inf, "finite number"),
        ("training.learning_rate", math.nan, "finite number"),
        # data.yaml
        ("data.n_samples", 0, "greater than 0"),
        ("data.TR_s", 0.0, "greater than 0"),
        ("data.TI_s", -0.1, "greater than or equal to 0"),
        ("data.derivative_method", "central_difference", "forward_sensitivity"),
        ("data.parameter_ranges.B1", {"lower": 1.5, "upper": 0.5}, "must be below"),
        ("data.parameter_ranges.B1", {"lower": 1.0, "upper": 1.0}, "must be below"),
        # T1 and T2 are sampled with log
        ("data.parameter_ranges.T2_s.lower", 0.0, "greater than 0"),
        # model.yaml
        ("model.activation", "sigmoid", "relu"),
        ("model.signal.num_layers", 1, "greater than or equal to 2"),
        ("model.signal.rank", 0, "greater than 0"),
        ("model.derivative.mode", "mixed", "does not match any of the expected tags"),
        # training.yaml
        ("training.loss_function", "mse", "nmae"),
        ("training.lr_scheduler", {"name": "step"}, "expected tags"),
        ("training.lr_scheduler", {"name": "cosine", "factor": 0.5}, "Extra inputs"),
        # testing.yaml
        ("testing.n_runs", 0, "greater than 0"),
        # export.yaml
        ("export.opset_version", 0, "greater than 0"),
        # YAML reads an unquoted 1.0 as a number
        ("export.model_version", 1.0, "Input should be a valid string"),
    ],
)
def test_bad_values_are_refused(
    config_dict: dict[str, Any], key_path: str, value: object, message: str
) -> None:
    set_at(config_dict, key_path, value)
    with pytest.raises(ValidationError, match=message):
        Config.model_validate(config_dict)


def test_derived_mode_with_a_smooth_activation_is_valid(
    config_dict: dict[str, Any], derived: dict[str, Any]
) -> None:
    config_dict["model"]["derivative"] = derived
    config_dict["model"]["activation"] = "silu"
    config = Config.model_validate(config_dict)
    assert config.model.derivative.mode == "derived"


def test_derived_mode_refuses_relu(
    config_dict: dict[str, Any], derived: dict[str, Any]
) -> None:
    config_dict["model"]["derivative"] = derived
    config_dict["model"]["activation"] = "relu"
    with pytest.raises(ValidationError, match="activation must be smooth"):
        Config.model_validate(config_dict)


def test_derived_mode_refuses_the_keys_of_direct_mode(
    config_dict: dict[str, Any], derived: dict[str, Any]
) -> None:
    # The direct block keeps its dT1 and dT2
    config_dict["model"]["derivative"].update(derived)
    config_dict["model"]["activation"] = "silu"
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        Config.model_validate(config_dict)


def test_plateau_scheduler_is_valid(
    config_dict: dict[str, Any], plateau: dict[str, Any]
) -> None:
    config_dict["training"]["lr_scheduler"] = plateau
    config = Config.model_validate(config_dict)
    assert config.training.lr_scheduler.name == "plateau"


@pytest.mark.parametrize(
    ("key", "value", "message"),
    [
        ("factor", 1.0, "less than 1"),
        ("factor", 0.0, "greater than 0"),
        ("patience_epochs", 0, "greater than 0"),
        ("min_lr", -1e-5, "greater than or equal to 0"),
    ],
)
def test_plateau_scheduler_bad_values(
    config_dict: dict[str, Any],
    plateau: dict[str, Any],
    key: str,
    value: object,
    message: str,
) -> None:
    plateau[key] = value
    config_dict["training"]["lr_scheduler"] = plateau
    with pytest.raises(ValidationError, match=message):
        Config.model_validate(config_dict)


@pytest.mark.parametrize(
    "path",
    [
        "outputs/model.onnx",
        "./outputs/model.onnx",
        "outputs/../outputs/model.onnx",
    ],
)
def test_two_targets_cannot_write_one_file(
    config_dict: dict[str, Any], path: str
) -> None:
    set_at(config_dict, "export.targets.coeffs.onnx_file_path", path)
    with pytest.raises(ValidationError, match="coeffs and signal_and_derivs are the"):
        Config.model_validate(config_dict)


@pytest.mark.parametrize(
    "family", ["fisp600", "fisp600-rank16-rank8", "FISP600-rank16", "fisp600_rank16"]
)
def test_bad_model_family_is_refused(config_dict: dict[str, Any], family: str) -> None:
    config_dict["export"]["model_family"] = family
    with pytest.raises(ValidationError, match="must be lowercase words"):
        Config.model_validate(config_dict)


@pytest.mark.parametrize("version", ["0.1", "v0.1.0", "0.1.0+build.1"])
def test_bad_model_version_is_refused(
    config_dict: dict[str, Any], version: str
) -> None:
    config_dict["export"]["model_version"] = version
    with pytest.raises(ValidationError, match="must be a semantic version"):
        Config.model_validate(config_dict)
