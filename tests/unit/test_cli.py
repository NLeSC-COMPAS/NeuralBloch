"""The commands of neuralbloch.cli make the directories of their output files."""

import sys
from collections.abc import Callable
from pathlib import Path
import pytest
import yaml
from neuralbloch import cli


@pytest.mark.parametrize(
    ("command", "keys"),
    [
        pytest.param(
            cli.generate,
            (
                "dataset_unsplit_path",
                "dataset_for_validation_path",
                "dataset_for_testing_path",
            ),
            id="generate",
        ),
        pytest.param(cli.train, ("pt_file_path",), id="train"),
    ],
)
def test_a_command_makes_the_output_directories_before_julia_starts(
    command: Callable[..., None],
    keys: tuple[str, ...],
    config_dir: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Put each output file in a new directory
    experiment_yaml = config_dir / "experiment.yaml"
    experiment = yaml.safe_load(experiment_yaml.read_text())
    outputs = {key: tmp_path / "new" / key / "file" for key in keys}
    experiment.update({key: str(path) for key, path in outputs.items()})
    experiment_yaml.write_text(yaml.safe_dump(experiment))
    # The import of juliacall fails. Thus, the command stops before Julia starts
    monkeypatch.setitem(sys.modules, "juliacall", None)

    with pytest.raises(ImportError):
        command(config_dir=config_dir)

    assert all(path.parent.is_dir() for path in outputs.values())
