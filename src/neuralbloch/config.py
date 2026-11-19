from pathlib import Path
from typing import Any
import yaml
from neuralbloch.config_schema import Config


def load_config(config_dir: str | Path) -> Config:
    """Load, merge and validate all YAML configuration files.

    A configuration that does not follow the schema in ``config_schema.py`` raises the
    ``ValidationError`` of pydantic, which lists each wrong key (``data.TR_s``).

    Args:
        config_dir: Directory containing the YAML configuration files.

    Returns:
        The validated ``Config``, with one attribute for each section (``data``,
        ``experiment``, ``model``, ``training``, ``testing``, ``export``).
    """
    config_dir = Path(config_dir)

    raw: dict[str, Any] = {}

    # One file for each section of the schema
    for name in Config.model_fields:
        config_file = config_dir / f"{name}.yaml"
        with config_file.open() as f:
            raw[name] = yaml.safe_load(f)

    return Config.model_validate(raw)
