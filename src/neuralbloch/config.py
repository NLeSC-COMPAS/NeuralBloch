from pathlib import Path
from typing import Any
import yaml
from neuralbloch.types import Config


def load_config(config_dir: str | Path) -> dict[str, Any]:
    """Load, merge and validate all YAML configuration files.

    A configuration that does not follow the schema in ``types.py`` raises the
    ``ValidationError`` of pydantic, which lists each wrong key (``data.TR_s``).

    Args:
        config_dir: Directory containing the YAML configuration files.

    Returns:
        A dictionary mapping each section of ``Config`` (``data``,
        ``experiment``, ``model``, ``training``, ``testing``, ``export``) to
        the parsed contents of its file.
    """
    config_dir = Path(config_dir)

    config: dict[str, Any] = {}

    # One file for each section of the schema
    for name in Config.model_fields:
        config_file = config_dir / f"{name}.yaml"
        with config_file.open() as f:
            config[name] = yaml.safe_load(f)

    # The callers still read the dict, so the validated model is not returned
    Config.model_validate(config)

    return config
