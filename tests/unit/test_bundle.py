import os
import subprocess
import tarfile
import tomllib
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock
import pytest
from scripts.export_bundle import CARD
from scripts.export_bundle import RF_FILES
from scripts.export_bundle import SCHEMA_VERSION
from scripts.export_bundle import CardInputs
from scripts.export_bundle import card_text
from scripts.export_bundle import check_bundle
from scripts.export_bundle import check_family
from scripts.export_bundle import check_rf_provenance
from scripts.export_bundle import check_undersampling
from scripts.export_bundle import check_version
from scripts.export_bundle import config_sha256
from scripts.export_bundle import derivative_method
from scripts.export_bundle import family_rank
from scripts.export_bundle import file_sha256
from scripts.export_bundle import release_command
from scripts.export_bundle import release_tag
from scripts.export_bundle import require_configs
from scripts.export_bundle import require_unreleased
from scripts.export_bundle import rf_table_length
from scripts.export_bundle import tarball_name
from scripts.export_bundle import write_bundle
from scripts.export_bundle import write_tarball
from scripts.export_onnx import TARGETS
from src.utils import RF_TABLE_KEYS
from src.utils import rf_sha256


DATA_CONFIG = {
    "N_SAMPLES": 40000,
    "DERIVATIVE_METHOD": "forward_sensitivity",
    "RF_ANGLE_FILE": "data/rf_data/rf_angle_list.txt",
    "RF_PHASE_FILE": "data/rf_data/rf_phase_list.txt",
    "TR_S": 0.00558,
    "TE_S": 0.002745,
    "TI_S": 0.01084,
    "TW_S": 0.0,
    "MAX_STATE": 64,
    "REPETITIONS": 5,
    "INVERSION_PREPULSE": True,
    "WAIT_SPOILING": True,
    "PY_UNDERSAMPLING_FACTOR": 1,
    "DELTA_K_SPOIL_RAD_PER_M": 0.0,
    "T1_lower": 0.1,
    "T1_upper": 6,
    "T2_lower": 0.01,
    "T2_upper": 2.0,
    "B1_lower": 0.5,
    "B1_upper": 1.5,
}

INPUTS = CardInputs(
    family="fisp600-rank16",
    version="0.1.0-beta.1",
    ntr=600,
    rank=16,
    onnx_sha256={target: f"{i}" * 64 for i, target in enumerate(TARGETS)},
    data_config=DATA_CONFIG,
    finite_difference_steps=None,
    upstream="NeuralBloch@" + "a" * 40,
    checkpoint_epoch=4970,
    config_sha256="c" * 64,
    blochsimulators="0.10.0",
    precision="FP32",
)


def test_card_states_every_field() -> None:
    card = tomllib.loads(card_text(INPUTS))
    # The first field is the version of the card schema, 1.0
    assert next(iter(card)) == "schema_version"
    assert card["schema_version"] == SCHEMA_VERSION
    assert SCHEMA_VERSION == "1.0"
    model = card["model"]
    assert (model["family"], model["version"], model["ntr"], model["rank"]) == (
        "fisp600-rank16",
        "0.1.0-beta.1",
        600,
        16,
    )
    for i, (target, spec) in enumerate(TARGETS.items()):
        assert model[target] == {"onnx": spec.bundle_file, "sha256": f"{i}" * 64}
    assert card["domain"] == {
        "T1": [0.1, 6.0],
        "T2": [0.01, 2.0],
        "B1": [0.5, 1.5],
        "constraint": "T2 < T1",
    }
    assert card["derivatives"] == {"method": "forward sensitivity"}
    assert card["sequence"] == {
        "kind": "FISP3D",
        "TR": 0.00558,
        "TE": 0.002745,
        "TI": 0.01084,
        "TW": 0.0,
        "max_state": 64,
        "repetitions": 5,
        "inversion_prepulse": True,
        "wait_spoiling": True,
        "undersampling_factor": 1,
        "delta_k_spoil": 0.0,
        "rf_angle": "rf_data/rf_angle_list.txt",
        "rf_phase": "rf_data/rf_phase_list.txt",
    }
    assert card["provenance"] == {
        "upstream": "NeuralBloch@" + "a" * 40,
        "checkpoint_epoch": 4970,
        "config_sha256": "c" * 64,
        "blochsimulators": "0.10.0",
        "precision": "FP32",
    }


def test_integers_floats_and_booleans_keep_their_types() -> None:
    card = tomllib.loads(card_text(INPUTS))
    # The card writes an integer bound (T1_upper: 6) with a decimal point
    assert isinstance(card["domain"]["T1"][1], float)
    assert isinstance(card["sequence"]["max_state"], int)
    assert isinstance(card["sequence"]["TW"], float)
    assert card["sequence"]["wait_spoiling"] is True


def test_every_field_names_its_source() -> None:
    for line in card_text(INPUTS).splitlines():
        if " = " in line and not line.startswith(("onnx", "sha256", "kind")):
            assert "  # " in line, line


def test_finite_differences_state_their_steps() -> None:
    inputs = replace(
        INPUTS,
        data_config={**DATA_CONFIG, "DERIVATIVE_METHOD": "finite_difference"},
        finite_difference_steps=(0.02, 0.0025),
    )
    card = tomllib.loads(card_text(inputs))
    assert card["derivatives"] == {
        "method": "forward difference",
        "dT1_step": 0.02,
        "dT2_step": 0.0025,
    }


def test_steps_belong_to_finite_differences_only() -> None:
    with pytest.raises(ValueError, match="Finite-difference steps"):
        card_text(replace(INPUTS, finite_difference_steps=(0.02, 0.0025)))
    fd = {**DATA_CONFIG, "DERIVATIVE_METHOD": "finite_difference"}
    with pytest.raises(ValueError, match="Finite-difference steps"):
        card_text(replace(INPUTS, data_config=fd))


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"version": "0.1"}, "must be a semantic version"),
        ({"family": "fisp600-rank8"}, "states rank 8"),
        ({"onnx_sha256": {"signal": "0" * 64}}, "exactly the targets"),
        ({"precision": "fp16"}, 'must be "FP32" or "FP16"'),
    ],
    ids=["version", "family", "onnx_sha256", "precision"],
)
def test_card_inputs_agree(change: dict[str, object], words: str) -> None:
    with pytest.raises(ValueError, match=words):
        replace(INPUTS, **change)


def test_the_derivative_method_is_required() -> None:
    # Old configurations without DERIVATIVE_METHOD used finite differences,
    # but data_generation.jl now defaults to forward sensitivity
    config = {k: v for k, v in DATA_CONFIG.items() if k != "DERIVATIVE_METHOD"}
    with pytest.raises(ValueError, match="does not give DERIVATIVE_METHOD"):
        derivative_method(config)
    with pytest.raises(ValueError, match="Unknown DERIVATIVE_METHOD 'adjoint'"):
        derivative_method({"DERIVATIVE_METHOD": "adjoint"})


def test_a_sequence_without_undersampling_is_bundled() -> None:
    check_undersampling(DATA_CONFIG)


@pytest.mark.parametrize(
    ("factor", "message"),
    [(2, "requires 1"), (0, "requires 1"), (True, "integer")],
    ids=["2", "0", "True"],
)
def test_undersampling_factors_other_than_1_are_refused(
    factor: object, message: str
) -> None:
    data = {**DATA_CONFIG, "PY_UNDERSAMPLING_FACTOR": factor}
    with pytest.raises(ValueError, match=message):
        check_undersampling(data)


@pytest.mark.parametrize(
    ("change", "words"),
    [
        ({"MAX_STATE": 64.0}, "MAX_STATE must be an integer"),
        ({"WAIT_SPOILING": "yes"}, "WAIT_SPOILING must be true or false"),
        ({"TR_S": "5.58 ms"}, "TR_S must be a number"),
        ({"T1_lower": 6.0}, "T1_lower = 6.0 must be below T1_upper"),
        ({"TE_S": float("nan")}, "must be finite"),
    ],
    ids=["MAX_STATE", "WAIT_SPOILING", "TR_S", "T1_lower", "TE_S"],
)
def test_malformed_configuration_is_refused(
    change: dict[str, object], words: str
) -> None:
    inputs = replace(INPUTS, data_config={**DATA_CONFIG, **change})
    with pytest.raises(ValueError, match=words):
        card_text(inputs)


def test_family_names_state_their_rank() -> None:
    assert family_rank("fisp600-rank16") == 16
    assert family_rank("fisp600b-rank16") == 16
    assert family_rank("fisp600-rank8-narrow") == 8


@pytest.mark.parametrize(
    "name",
    [
        "fisp600-r16",
        "fisp600",
        "Fisp600-rank16",
        "fisp600-rank16-rank8",
        "fisp600-rank016",
        "fisp600_rank16",
        "fisp600--rank16",
        # A YAML block scalar, which $ would accept
        pytest.param("fisp600-rank16\n", id="final-newline"),
        pytest.param("", id="empty"),
    ],
)
def test_malformed_family_names_are_refused(name: str) -> None:
    with pytest.raises(ValueError, match="rank<N>"):
        family_rank(name)


def test_family_must_state_the_checkpoints_rank() -> None:
    check_family("fisp600-rank16", 16)
    with pytest.raises(ValueError, match="states rank 16.*has rank 12"):
        check_family("fisp600-rank16", 12)


@pytest.mark.parametrize(
    "version", ["0.1.0-beta.1", "1.0.0", "0.1.0-rc.1", "2.10.3-beta.x-y.1"]
)
def test_version_is_a_semantic_version(version: str) -> None:
    assert check_version(version) == version


@pytest.mark.parametrize(
    "version",
    [
        "1",
        "0.1",
        "v0.1.0",
        "01.0.0",
        "0.1.0-beta.01",  # a numeric identifier with a leading zero
        "0.1.0-",
        "0.1.0+build.1",  # the tag would contain build metadata
        pytest.param("0.1.0\n", id="final-newline"),
        # A fullwidth digit, which \d would accept
        pytest.param("０.1.0", id="fullwidth-digit"),
    ],
)
def test_versions_that_are_not_semantic_are_refused(version: str) -> None:
    with pytest.raises(ValueError, match="must be a semantic version"):
        check_version(version)


# YAML reads an unquoted 1 or 1.0 as a number
@pytest.mark.parametrize("version", [1, 1.0, True])
def test_unquoted_yaml_versions_are_refused(version: object) -> None:
    with pytest.raises(ValueError, match="in quotes"):
        check_version(version)


def test_checkpoint_must_store_its_configuration() -> None:
    require_configs({"model_config": {}, "data_config": {}}, "model.pt")
    with pytest.raises(ValueError, match="model.pt does not store data_config"):
        require_configs({"model_config": {}}, "model.pt")
    with pytest.raises(ValueError, match="model_config and data_config"):
        require_configs({"NTR": 600, "epoch": 4190}, "model.pt")


def test_config_hash_ignores_key_order() -> None:
    a = config_sha256({"x": 1, "y": 2}, {"T1_lower": 0.1, "T1_upper": 6.0})
    b = config_sha256({"y": 2, "x": 1}, {"T1_upper": 6.0, "T1_lower": 0.1})
    assert a == b
    assert len(a) == 64
    assert a != config_sha256({"x": 1, "y": 3}, {})


@pytest.fixture
def files(tmp_path: Path) -> dict[str, Path]:
    """Write the source files of a bundle with 3 readouts in tmp_path/src."""
    (tmp_path / "src").mkdir()
    sources: dict[str, Path] = {}
    for target, spec in TARGETS.items():
        path = tmp_path / "src" / f"{target}.onnx"
        path.write_bytes(target.encode())
        sources[spec.bundle_file] = path
    for name, _, _ in RF_FILES.values():
        path = tmp_path / "src" / Path(name).name
        path.write_text("".join(f"{i}.5\n" for i in range(3)))
        sources[name] = path
    return sources


def card(files: dict[str, Path], ntr: int = 3) -> str:
    sha = {t: file_sha256(files[s.bundle_file]) for t, s in TARGETS.items()}
    return card_text(replace(INPUTS, ntr=ntr, onnx_sha256=sha))


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    """Return tmp_path/bundle. The directory does not exist yet."""
    return tmp_path / "bundle"


def test_written_bundle_checks_out(bundle: Path, files: dict[str, Path]) -> None:
    write_bundle(bundle, card(files), files)
    assert check_bundle(bundle)["model"]["ntr"] == 3
    assert sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")) == sorted(
        [CARD, "rf_data", *files]
    )
    assert (bundle / "model.onnx").read_bytes() == b"signal_and_derivs"


def test_a_new_bundle_replaces_the_old_one_whole(
    tmp_path: Path, bundle: Path, files: dict[str, Path]
) -> None:
    write_bundle(bundle, card(files), files)
    (bundle / "stale.onnx").write_bytes(b"left over")
    write_bundle(bundle, card(files), files)
    assert not (bundle / "stale.onnx").exists()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle", "src"]


def test_a_failed_write_keeps_the_old_bundle(
    tmp_path: Path, bundle: Path, files: dict[str, Path]
) -> None:
    write_bundle(bundle, card(files), files)
    before = (bundle / CARD).read_text()
    with pytest.raises(FileNotFoundError, match="No such file or directory"):
        write_bundle(bundle, card(files), {**files, "x": tmp_path / "missing"})
    # Also, a bundle that fails its check does not replace the old bundle
    with pytest.raises(ValueError, match="does not have ntr = 4 values"):
        write_bundle(bundle, card(files, ntr=4), files)
    assert (bundle / CARD).read_text() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle", "src"]


def test_only_a_bundle_is_replaced(
    tmp_path: Path, bundle: Path, files: dict[str, Path]
) -> None:
    # A directory without a card, for example outputs/, is not deleted
    bundle.mkdir()
    (bundle / "checkpoint.pt").write_bytes(b"weights")
    with pytest.raises(ValueError, match="is not a bundle"):
        write_bundle(bundle, card(files), files)
    assert (bundle / "checkpoint.pt").read_bytes() == b"weights"
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle", "src"]
    # The sources of a bundle cannot be inside the bundle that it replaces
    (bundle / CARD).write_text("")
    inside = {**files, "model.onnx": bundle / "checkpoint.pt"}
    with pytest.raises(ValueError, match="is inside"):
        write_bundle(bundle, card(files), inside)
    assert (bundle / "checkpoint.pt").exists()
    # A bundle is replaced
    write_bundle(bundle, card(files), files)
    assert not (bundle / "checkpoint.pt").exists()


def test_a_failed_swap_restores_the_old_bundle(
    tmp_path: Path, bundle: Path, files: dict[str, Path]
) -> None:
    write_bundle(bundle, card(files), files)
    before = (bundle / CARD).read_bytes()
    rename = Path.rename
    calls = []

    def fail_second(path: Path, target: Path) -> Path:
        # The first rename moves the old bundle away, the second moves the
        # new bundle in
        calls.append(path)
        if len(calls) == 2:
            raise OSError("disk full")
        return rename(path, target)

    with pytest.MonkeyPatch.context() as m:
        m.setattr(Path, "rename", fail_second)
        with pytest.raises(OSError, match="disk full"):
            write_bundle(bundle, card(files), files)
    assert (bundle / CARD).read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle", "src"]


@pytest.mark.parametrize(
    "make",
    [
        lambda old: (old.mkdir(), (old / CARD).write_text("")),
        lambda old: old.write_text(""),
        lambda old: old.symlink_to(old.parent / "nowhere"),
    ],
    ids=["directory", "file", "link"],
)
def test_a_left_over_old_bundle_is_deleted(
    tmp_path: Path, bundle: Path, files: dict[str, Path], make: Callable[[Path], object]
) -> None:
    write_bundle(bundle, card(files), files)
    make(tmp_path / ".bundle-old")
    write_bundle(bundle, card(files), files)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle", "src"]


def test_a_link_is_not_replaced(
    tmp_path: Path, bundle: Path, files: dict[str, Path]
) -> None:
    # The new bundle would replace the link, and the bundle that the link
    # points to would not change
    target = tmp_path / "scratch"
    write_bundle(target, card(files), files)
    bundle.symlink_to(target)
    with pytest.raises(ValueError, match="is a symbolic link"):
        write_bundle(bundle, card(files), files)
    assert bundle.is_symlink()
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle", "scratch", "src"]


def test_check_finds_damaged_bundles(bundle: Path, files: dict[str, Path]) -> None:
    write_bundle(bundle, card(files), files)
    (bundle / "model_signal.onnx").write_bytes(b"changed")
    with pytest.raises(ValueError, match="model_signal.onnx does not have"):
        check_bundle(bundle)
    (bundle / "model.onnx").unlink()
    with pytest.raises(ValueError, match="model.onnx, which is missing"):
        check_bundle(bundle)
    write_bundle(bundle, card(files), files)
    (bundle / "rf_data" / "rf_phase_list.txt").unlink()
    with pytest.raises(ValueError, match="rf_phase_list.txt, which is missing"):
        check_bundle(bundle)
    card_file = bundle / CARD
    card_file.write_text(card_file.read_text().replace('"1.0"', '"2.0"'))
    with pytest.raises(ValueError, match="has schema_version '2.0', not '1.0'"):
        check_bundle(bundle)


SMALL_BUNDLE_FILES = {
    "model_card.toml": b'schema_version = "1.0"\n',
    "model.onnx": b"\x00onnx",
    "rf_data/rf_angle_list.txt": b"1.5\n2.5\n",
}
TARBALL_NAME = tarball_name("fisp600-rank16", "0.1.0-beta.1")


@pytest.fixture
def small_bundle(bundle: Path) -> Path:
    """Write a small bundle in tmp_path/bundle."""
    (bundle / "rf_data").mkdir(parents=True)
    for name, data in SMALL_BUNDLE_FILES.items():
        (bundle / name).write_bytes(data)
    return bundle


@pytest.fixture
def tarball(tmp_path: Path) -> Path:
    """Return tmp_path/TARBALL_NAME. The file does not exist yet."""
    return tmp_path / TARBALL_NAME


def test_the_bundle_sits_at_the_root(small_bundle: Path, tarball: Path) -> None:
    write_tarball(small_bundle, tarball)
    # Other users can read the release file, as the bundle
    assert tarball.stat().st_mode & 0o777 == 0o644
    with tarfile.open(tarball) as tar:
        assert tar.getnames() == [
            "model.onnx",
            "model_card.toml",
            "rf_data",
            "rf_data/rf_angle_list.txt",
        ]
        for name, data in SMALL_BUNDLE_FILES.items():
            member = tar.extractfile(name)
            assert member is not None
            assert member.read() == data


def test_the_same_files_give_the_same_bytes(small_bundle: Path, tarball: Path) -> None:
    sha256 = write_tarball(small_bundle, tarball)
    first = tarball.read_bytes()
    assert sha256 == file_sha256(tarball)
    # The times and modes of the files have no effect on the tarball
    os.utime(small_bundle / "model.onnx", (0, 1e9))
    (small_bundle / "model.onnx").chmod(0o600)
    assert write_tarball(small_bundle, tarball) == sha256
    assert tarball.read_bytes() == first
    with tarfile.open(tarball) as tar:
        for member in tar.getmembers():
            assert member.mtime == 0, member.name
            assert (member.uid, member.gid) == (0, 0), member.name
            assert (member.uname, member.gname) == ("", ""), member.name
            assert member.mode == (0o755 if member.isdir() else 0o644), member.name
    # gzip header: no file name (flag bit 3) and mtime 0
    assert first[3] & 0x08 == 0
    assert first[4:8] == b"\0\0\0\0"


def test_a_failed_write_leaves_no_partial_tarball(
    tmp_path: Path, small_bundle: Path, tarball: Path
) -> None:
    write_tarball(small_bundle, tarball)
    before = tarball.read_bytes()
    with pytest.MonkeyPatch.context() as m:
        m.setattr(tarfile.TarFile, "addfile", Mock(side_effect=OSError("full")))
        with pytest.raises(OSError, match="full"):
            write_tarball(small_bundle, tarball)
    assert tarball.read_bytes() == before
    assert sorted(p.name for p in tmp_path.iterdir()) == ["bundle", TARBALL_NAME]
    with pytest.raises(FileNotFoundError, match="not a bundle directory"):
        write_tarball(tmp_path / "missing", tmp_path / "other.tar.gz")
    assert not (tmp_path / "other.tar.gz").exists()


def test_release_names() -> None:
    assert (
        release_tag("fisp600-rank16", "0.1.0-beta.1")
        == "model-fisp600-rank16-v0.1.0-beta.1"
    )
    assert TARBALL_NAME == "neuralbloch-model-fisp600-rank16-v0.1.0-beta.1.tar.gz"


def test_the_release_command() -> None:
    card = tomllib.loads(card_text(INPUTS))
    command = release_command(card, Path("outputs/model.tar.gz"), "f" * 64)
    assert command[:8] == [
        "gh",
        "release",
        "create",
        "model-fisp600-rank16-v0.1.0-beta.1",
        "outputs/model.tar.gz",
        "--target",
        "a" * 40,
        "--prerelease",
    ]
    assert command[8:10] == ["--title", "Model fisp600-rank16 v0.1.0-beta.1"]
    notes = command[command.index("--notes") + 1]
    assert "T1 0.1 to 6 s, T2 0.01 to 2 s, B1 0.5 to 1.5 (T2 < T1)" in notes
    assert "f" * 64 in notes
    # A version without a pre-release part is also not marked as latest
    card["model"]["version"] = "0.1.0"
    command = release_command(card, Path("outputs/model.tar.gz"), "f" * 64)
    assert "--latest=false" in command
    assert "--prerelease" not in command


TAG = "model-fisp600-rank16-v0.1.0-beta.1"
DENIED = "git@github.com: Permission denied (publickey)."


def run_git(local: str, remote: str | Exception, offline: bool = False) -> str | None:
    """Run require_unreleased with git output; return its warning."""

    def run(args: list[str], **kwargs: object) -> subprocess.CompletedProcess:
        if args[1] == "ls-remote":
            if isinstance(remote, Exception):
                raise remote
            return subprocess.CompletedProcess(args, 0, remote, "")
        return subprocess.CompletedProcess(args, 0, local, "")

    with pytest.MonkeyPatch.context() as m:
        m.setattr("scripts.export_bundle.subprocess.run", run)
        return require_unreleased(TAG, offline)


def test_a_new_version_is_not_released() -> None:
    assert run_git("", "") is None
    assert run_git("", "", offline=True) is None


@pytest.mark.parametrize(
    ("local", "remote"),
    [(TAG, ""), ("", f"abc\trefs/tags/{TAG}")],
    ids=["local-tag", "remote-tag"],
)
def test_a_released_version_is_refused(local: str, remote: str) -> None:
    with pytest.raises(RuntimeError, match="Increase"):
        run_git(local, remote)


@pytest.mark.parametrize(
    ("failure", "reason"),
    [
        (subprocess.CalledProcessError(128, "git", stderr=DENIED), DENIED),
        (subprocess.TimeoutExpired("git", 60), "timed out"),
    ],
    ids=["CalledProcessError", "TimeoutExpired"],
)
def test_without_origin_the_export_stops_unless_offline(
    failure: Exception, reason: str
) -> None:
    with pytest.raises(RuntimeError, match="export.OFFLINE") as raised:
        run_git("", failure)
    assert reason in str(raised.value)
    warning = run_git("", failure, offline=True)
    assert "cannot get the tags of origin" in str(warning)
    assert reason in str(warning)


def test_offline_does_not_permit_a_version_released_here() -> None:
    error = subprocess.CalledProcessError(128, "git", stderr=DENIED)
    with pytest.raises(RuntimeError, match="Increase"):
        run_git(TAG, error, offline=True)


def test_rf_tables_count_numbers(tmp_path: Path) -> None:
    path = tmp_path / "rf.txt"
    path.write_text("1.0\n\n2.5\n3\n")
    assert rf_table_length(path) == 3
    path.write_text("1.0\nangle\n")
    with pytest.raises(ValueError, match="rf.txt:2: not a finite number"):
        rf_table_length(path)


# NeuralBloch.jl does not read 1_000 or Arabic-Indic digits, and the angles must
# be finite
@pytest.mark.parametrize("value", ["1_000", "١٢", "nan", "inf", "-Infinity"])
def test_rf_tables_accept_only_finite_ascii_numbers(tmp_path: Path, value: str) -> None:
    path = tmp_path / "rf.txt"
    path.write_text(f"1.0\n{value}\n")
    with pytest.raises(ValueError, match="rf.txt:2: not a finite"):
        rf_table_length(path)


def test_rf_files_are_the_rf_tables_of_the_training() -> None:
    assert sorted(data_key for _, data_key, _ in RF_FILES.values()) == sorted(
        RF_TABLE_KEYS
    )


def test_the_rf_tables_must_be_the_tables_of_the_training(tmp_path: Path) -> None:
    data = dict(DATA_CONFIG)
    for key in RF_TABLE_KEYS:
        data[key] = tmp_path / f"{key}.txt"
        data[key].write_text("1.0\n2.0\n")
    checkpoint = {"rf_sha256": rf_sha256(data)}
    assert check_rf_provenance(checkpoint, data) is None
    data["RF_PHASE_FILE"].write_text("1.0\n3.0\n")
    with pytest.raises(
        ValueError, match="RF_PHASE_FILE.txt \\(RF_PHASE_FILE\\) is not the RF table"
    ):
        check_rf_provenance(checkpoint, data)
    # An old checkpoint does not record the tables: only a warning
    warning = check_rf_provenance({}, data)
    assert "the checkpoint does not record" in str(warning)
