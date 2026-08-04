from pathlib import Path
import yaml


def load_config(config_dir):
    """Load and merge all YAML configuration files.

    Args:
        config_dir: Directory containing the YAML configuration files.

    Returns:
        A dictionary mapping each config name (``data``, ``experiment``,
        ``model``, ``training``, ``testing``) to its parsed contents.
    """
    config_dir = Path(config_dir)

    config = {}

    for name in ["data", "experiment", "model", "training", "testing"]:
        with open(config_dir / f"{name}.yaml") as f:
            config[name] = yaml.safe_load(f)

    return config
