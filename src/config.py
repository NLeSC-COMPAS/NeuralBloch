from pathlib import Path
from typing import Any
import yaml


def load_config(config_dir: str | Path) -> dict[str, Any]:
    """Load and merge all YAML configuration files.

    Args:
        config_dir: Directory containing the YAML configuration files.

    Returns:
        A dictionary mapping each config name (``data``, ``experiment``,
        ``model``, ``training``, ``testing``) to its parsed contents.
    """
    config_dir = Path(config_dir)

    config: dict[str, Any] = {}

    for name in ["data", "experiment", "model", "training", "testing"]:
        with open(config_dir / f"{name}.yaml") as f:
            config[name] = yaml.safe_load(f)

    return config
