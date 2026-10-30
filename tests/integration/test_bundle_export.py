import shutil
import tarfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import Mock
import pytest
import torch
from scripts.export_bundle import REPO_ROOT
from scripts.export_bundle import check_bundle
from scripts.export_bundle import export_bundle
from scripts.export_bundle import file_sha256
from scripts.export_bundle import git
from scripts.export_bundle import rf_table_length
from scripts.export_bundle import tarball_name
from scripts.export_bundle import write_tarball
from scripts.export_onnx import stamp_checkpoint
from src.model import FCModel


NTR = 5


@dataclass
class ExportSetup:
    """A small checkpoint, its ONNX models and the configuration of the export."""

    checkpoint: Path
    onnx: dict[str, Path]
    bundle: Path
    config: dict[str, Any]
    tarball: Path


@pytest.fixture
def export(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    direct_config: dict[str, Any],
    data_config: dict[str, Any],
    exported_model: FCModel,
    onnx_models: dict[str, Path],
    write_checkpoint: Callable[..., None],
) -> ExportSetup:
    """export_bundle, from a small checkpoint to the bundle and its release file."""
    source = tmp_path / "src"
    source.mkdir()
    checkpoint = source / "model.pt"
    # A copy, because the tests stamp and change the ONNX models
    onnx = {t: Path(shutil.copy(path, source)) for t, path in onnx_models.items()}
    bundle = tmp_path / "out" / "bundle"
    config = {
        "experiment": {"PT_FILE_PATH": str(checkpoint)},
        "model": {},
        "data": {},
        "export": {
            "TARGETS": {
                target: {"ONNX_FILE_PATH": str(path)} for target, path in onnx.items()
            },
            "N_VERIFY_SAMPLES": 8,
            "BUNDLE_DIR": str(bundle),
            "MODEL_FAMILY": "fisp5-rank3",
            "MODEL_VERSION": "0.1.0",
            "OFFLINE": False,
        },
    }
    tarball = bundle.with_name(tarball_name("fisp5-rank3", "0.1.0"))
    # git and Julia are not part of this test
    for name, value in (
        ("clean_commit", "a" * 40),
        ("require_unreleased", None),
        ("blochsimulators_facts", ("0.10.0", None)),
    ):
        monkeypatch.setattr(f"scripts.export_bundle.{name}", Mock(return_value=value))
    write_checkpoint(checkpoint, exported_model, direct_config, data_config, epoch=7)
    setup = ExportSetup(checkpoint, onnx, bundle, config, tarball)
    stamp_onnx_models(setup)
    return setup


def stamp_onnx_models(setup: ExportSetup) -> None:
    """Record the sha256 of the checkpoint in the ONNX model of each target."""
    for path in setup.onnx.values():
        stamp_checkpoint(path, file_sha256(setup.checkpoint))


def drop_from_checkpoint(setup: ExportSetup, key: str) -> None:
    """Delete ``key`` from the checkpoint, and stamp the ONNX models again."""
    checkpoint = torch.load(setup.checkpoint)
    del checkpoint[key]
    torch.save(checkpoint, setup.checkpoint)
    stamp_onnx_models(setup)


def test_the_bundle_and_its_release_file_describe_the_checkpoint(
    export: ExportSetup, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    export_bundle(export.config)
    card = check_bundle(export.bundle)
    model = card["model"]
    assert (model["family"], model["version"], model["ntr"], model["rank"]) == (
        "fisp5-rank3",
        "0.1.0",
        NTR,
        3,
    )
    for target, path in export.onnx.items():
        assert model[target]["sha256"] == file_sha256(path)
    assert card["provenance"]["checkpoint_epoch"] == 7
    assert card["provenance"]["precision"] == "FP32"
    printed = capsys.readouterr().out
    assert "gh release create" in printed
    assert "WARNING" not in printed

    # The release file holds the same bundle, and the bundle in it gives
    # the same release file again
    extracted = tmp_path / "extracted"
    with tarfile.open(export.tarball) as tar:
        tar.extractall(extracted, filter="data")
    assert check_bundle(extracted) == card
    again = write_tarball(extracted, tmp_path / "again.tar.gz")
    assert again == file_sha256(export.tarball)


def test_the_warnings_come_again_after_the_release_command(
    export: ExportSetup, capsys: pytest.CaptureFixture[str]
) -> None:
    # A checkpoint from before the checkpoints recorded the RF tables
    drop_from_checkpoint(export, "rf_sha256")
    export_bundle(export.config)
    printed = capsys.readouterr().out
    command = printed.index("gh release create")
    assert "WARNING: the checkpoint does not record" in printed[:command]
    assert "WARNING: the checkpoint does not record" in printed[command:]


def assert_refused(setup: ExportSetup, error: str) -> None:
    """Check that export_bundle raises ``error`` and writes nothing."""
    with pytest.raises(ValueError, match=error):
        export_bundle(setup.config)
    assert not setup.bundle.parent.exists()


def test_onnx_models_of_another_checkpoint_are_refused(export: ExportSetup) -> None:
    stamp_checkpoint(export.onnx["signal"], "0" * 64)
    assert_refused(export, "from a different checkpoint")


def test_a_checkpoint_without_its_configuration_is_refused(
    export: ExportSetup,
) -> None:
    # The model of config/ can have a different architecture, so the
    # export must not make the model before it checks the checkpoint
    drop_from_checkpoint(export, "model_config")
    assert_refused(export, "does not store model_config")


def test_a_family_with_another_rank_is_refused(export: ExportSetup) -> None:
    export.config["export"]["MODEL_FAMILY"] = "fisp5-rank16"
    assert_refused(export, "states rank 16")


def test_rf_tables_changed_after_the_training_are_refused(
    export: ExportSetup, data_config: dict[str, Any]
) -> None:
    path = Path(data_config["RF_PHASE_FILE"])
    path.write_text(path.read_text().replace("0.5", "9.5"))
    assert_refused(export, "is not the RF table of the training")


def test_git_runs_in_this_repository(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    assert Path(git("rev-parse", "--show-toplevel")) == REPO_ROOT
    with pytest.raises(RuntimeError, match="is not a git command"):
        git("no-such-command")


def test_the_repository_rf_tables_have_600_values() -> None:
    for name in ("rf_angle_list.txt", "rf_phase_list.txt"):
        assert rf_table_length(REPO_ROOT / "data" / "rf_data" / name) == 600
