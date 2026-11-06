"""The style of the files in config/ (issue #20) that yamllint does not check."""

import re
from collections.abc import Iterator
from pathlib import Path
import pytest
import yaml


# Physics symbols keep their case in key names
PHYSICS_SYMBOLS = {"T1", "T2", "B1", "TR", "TE", "TI", "TW", "dT1", "dT2"}
MAX_KEY_LEVELS = 3
CONFIG_FILES = sorted((Path(__file__).parents[2] / "config").glob("*.yaml"))

# A value of a YAML file, as yaml.safe_load gives it
type YamlValue = (
    str | int | float | bool | None | list[YamlValue] | dict[str, YamlValue]
)


def items(
    node: YamlValue, path: tuple[str, ...] = ()
) -> Iterator[tuple[tuple[str, ...], YamlValue]]:
    """Yield the key path and the value of each leaf of a YAML tree."""
    if isinstance(node, dict):
        for key, value in node.items():
            yield from items(value, (*path, key))
    else:
        yield path, node


def is_number(text: str) -> bool:
    try:
        float(text)
    except ValueError:
        return False
    return True


@pytest.fixture(params=CONFIG_FILES, ids=lambda path: path.name)
def config_file(request: pytest.FixtureRequest) -> Path:
    return request.param


def test_all_config_files_are_found() -> None:
    assert [path.stem for path in CONFIG_FILES] == [
        "data",
        "experiment",
        "export",
        "model",
        "testing",
        "training",
    ]


def test_keys_are_lower_snake_case_or_physics_symbols(config_file: Path) -> None:
    for path, _ in items(yaml.safe_load(config_file.read_text())):
        for key in path:
            for word in key.split("_"):
                assert word in PHYSICS_SYMBOLS or re.fullmatch(r"[a-z0-9]+", word), (
                    f"{'.'.join(path)}: {key!r}"
                )


def test_keys_have_at_most_three_levels(config_file: Path) -> None:
    for path, _ in items(yaml.safe_load(config_file.read_text())):
        assert len(path) <= MAX_KEY_LEVELS, ".".join(path)


def test_values_are_not_null_or_numbers_in_strings(config_file: Path) -> None:
    # PyYAML reads 1e-3 as a string: write 1.0e-3
    for path, value in items(yaml.safe_load(config_file.read_text())):
        assert value is not None, ".".join(path)
        assert not (isinstance(value, str) and is_number(value)), ".".join(path)


def test_there_are_no_section_headers(config_file: Path) -> None:
    for number, line in enumerate(config_file.read_text().splitlines(), 1):
        assert not line.startswith("# ---"), f"{config_file.name}:{number}"
