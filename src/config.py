from pathlib import Path
import yaml


def load_config(config_dir):
    config_dir = Path(config_dir)

    config = {}

    for name in ["data", "experiment", "model", "training", "testing"]:
        with open(config_dir / f"{name}.yaml") as f:
            config[name] = yaml.safe_load(f)

    return config
