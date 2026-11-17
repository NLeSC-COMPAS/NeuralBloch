"""load_config: it reads the YAML files of a config dir and validates them."""

import shutil
from pathlib import Path
import pytest
import yaml
from pydantic import ValidationError
from neuralbloch.config import load_config
from neuralbloch.config_schema import Config


CONFIG_DIR = Path(__file__).parents[2] / "config"


@pytest.fixture
def config_dir(tmp_path: Path) -> Path:
    """Copy config/ to tmp_path, so that a test can change it.

    Returns:
        The path of the copy.
    """
    return Path(shutil.copytree(CONFIG_DIR, tmp_path / "config"))


def test_load_config_returns_a_validated_config() -> None:
    config = load_config(CONFIG_DIR)
    assert isinstance(config, Config)
    assert config.data.model_dump(mode="json") == yaml.safe_load(
        (CONFIG_DIR / "data.yaml").read_text()
    )


def test_load_config_refuses_a_bad_value(config_dir: Path) -> None:
    data_yaml = config_dir / "data.yaml"
    data = yaml.safe_load(data_yaml.read_text())
    data["TR_s"] = 0.0
    data_yaml.write_text(yaml.safe_dump(data))
    with pytest.raises(ValidationError, match=r"data\.TR_s\n.*greater than 0"):
        load_config(config_dir)


def test_load_config_refuses_an_empty_file(config_dir: Path) -> None:
    (config_dir / "testing.yaml").write_text("")
    with pytest.raises(ValidationError, match=r"testing\n.*valid dictionary"):
        load_config(config_dir)
