import os
import tarfile
import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
from unittest import mock
from scripts.export_bundle import CARD
from scripts.export_bundle import RF_FILES
from scripts.export_bundle import SCHEMA_VERSION
from scripts.export_bundle import CardInputs
from scripts.export_bundle import card_text
from scripts.export_bundle import check_bundle
from scripts.export_bundle import check_family
from scripts.export_bundle import check_undersampling
from scripts.export_bundle import check_version
from scripts.export_bundle import config_sha256
from scripts.export_bundle import derivative_method
from scripts.export_bundle import family_rank
from scripts.export_bundle import file_sha256
from scripts.export_bundle import release_command
from scripts.export_bundle import release_tag
from scripts.export_bundle import require_configs
from scripts.export_bundle import rf_table_length
from scripts.export_bundle import tarball_name
from scripts.export_bundle import write_bundle
from scripts.export_bundle import write_tarball
from scripts.export_onnx import TARGETS


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


class CardTest(unittest.TestCase):
    def test_card_states_every_field(self) -> None:
        card = tomllib.loads(card_text(INPUTS))
        # The first field is the version of the card schema, 1.0
        self.assertEqual(next(iter(card)), "schema_version")
        self.assertEqual(card["schema_version"], SCHEMA_VERSION)
        self.assertEqual(SCHEMA_VERSION, "1.0")
        model = card["model"]
        self.assertEqual(
            (model["family"], model["version"], model["ntr"], model["rank"]),
            ("fisp600-rank16", "0.1.0-beta.1", 600, 16),
        )
        for i, (target, spec) in enumerate(TARGETS.items()):
            self.assertEqual(
                model[target], {"onnx": spec.bundle_file, "sha256": f"{i}" * 64}
            )
        self.assertEqual(
            card["domain"],
            {
                "T1": [0.1, 6.0],
                "T2": [0.01, 2.0],
                "B1": [0.5, 1.5],
                "constraint": "T2 < T1",
            },
        )
        self.assertEqual(card["derivatives"], {"method": "forward sensitivity"})
        self.assertEqual(
            card["sequence"],
            {
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
            },
        )
        self.assertEqual(
            card["provenance"],
            {
                "upstream": "NeuralBloch@" + "a" * 40,
                "checkpoint_epoch": 4970,
                "config_sha256": "c" * 64,
                "blochsimulators": "0.10.0",
                "precision": "FP32",
            },
        )

    def test_integers_floats_and_booleans_keep_their_types(self) -> None:
        card = tomllib.loads(card_text(INPUTS))
        # The card writes an integer bound (T1_upper: 6) with a decimal point
        self.assertIsInstance(card["domain"]["T1"][1], float)
        self.assertIsInstance(card["sequence"]["max_state"], int)
        self.assertIsInstance(card["sequence"]["TW"], float)
        self.assertIs(card["sequence"]["wait_spoiling"], True)

    def test_every_field_names_its_source(self) -> None:
        for line in card_text(INPUTS).splitlines():
            if " = " in line and not line.startswith(("onnx", "sha256", "kind")):
                with self.subTest(line=line):
                    self.assertIn("  # ", line)

    def test_finite_differences_state_their_steps(self) -> None:
        inputs = replace(
            INPUTS,
            data_config={**DATA_CONFIG, "DERIVATIVE_METHOD": "finite_difference"},
            finite_difference_steps=(0.02, 0.0025),
        )
        card = tomllib.loads(card_text(inputs))
        self.assertEqual(
            card["derivatives"],
            {"method": "forward difference", "dT1_step": 0.02, "dT2_step": 0.0025},
        )

    def test_steps_belong_to_finite_differences_only(self) -> None:
        with self.assertRaisesRegex(ValueError, "Finite-difference steps"):
            card_text(replace(INPUTS, finite_difference_steps=(0.02, 0.0025)))
        fd = {**DATA_CONFIG, "DERIVATIVE_METHOD": "finite_difference"}
        with self.assertRaisesRegex(ValueError, "Finite-difference steps"):
            card_text(replace(INPUTS, data_config=fd))

    def test_derivative_method_defaults_like_data_generation(self) -> None:
        config = {k: v for k, v in DATA_CONFIG.items() if k != "DERIVATIVE_METHOD"}
        self.assertEqual(derivative_method(config), "forward sensitivity")
        with self.assertRaisesRegex(ValueError, "Unknown DERIVATIVE_METHOD 'adjoint'"):
            derivative_method({"DERIVATIVE_METHOD": "adjoint"})

    def test_only_a_sequence_without_undersampling_is_bundled(self) -> None:
        check_undersampling(DATA_CONFIG)
        for factor, message in (
            (2, "requires 1"),
            (0, "requires 1"),
            (True, "integer"),
        ):
            with self.subTest(factor=factor):
                data = {**DATA_CONFIG, "PY_UNDERSAMPLING_FACTOR": factor}
                with self.assertRaisesRegex(ValueError, message):
                    check_undersampling(data)

    def test_malformed_configuration_is_refused(self) -> None:
        cases = [
            ({"MAX_STATE": 64.0}, "MAX_STATE must be an integer"),
            ({"WAIT_SPOILING": "yes"}, "WAIT_SPOILING must be true or false"),
            ({"TR_S": "5.58 ms"}, "TR_S must be a number"),
            ({"T1_lower": 6.0}, "T1_lower = 6.0 must be below T1_upper"),
            ({"TE_S": float("nan")}, "must be finite"),
        ]
        for change, words in cases:
            with self.subTest(change=change):
                inputs = replace(INPUTS, data_config={**DATA_CONFIG, **change})
                with self.assertRaisesRegex(ValueError, words):
                    card_text(inputs)


class FamilyTest(unittest.TestCase):
    def test_family_names_state_their_rank(self) -> None:
        self.assertEqual(family_rank("fisp600-rank16"), 16)
        self.assertEqual(family_rank("fisp600b-rank16"), 16)
        self.assertEqual(family_rank("fisp600-rank8-narrow"), 8)
        for name in (
            "fisp600-r16",
            "fisp600",
            "Fisp600-rank16",
            "fisp600-rank16-rank8",
            "fisp600-rank016",
            "fisp600_rank16",
            "fisp600--rank16",
            "",
        ):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "rank<N>"):
                family_rank(name)

    def test_family_must_state_the_checkpoints_rank(self) -> None:
        check_family("fisp600-rank16", 16)
        with self.assertRaisesRegex(ValueError, "states rank 16.*has rank 12"):
            check_family("fisp600-rank16", 12)

    def test_version_is_a_semantic_version(self) -> None:
        for version in ("0.1.0-beta.1", "1.0.0", "0.1.0-rc.1", "2.10.3-beta.x-y.1"):
            with self.subTest(version=version):
                self.assertEqual(check_version(version), version)
        refused = (
            "1",
            "0.1",
            "v0.1.0",
            "01.0.0",
            "0.1.0-beta.01",  # a numeric identifier with a leading zero
            "0.1.0-",
            "0.1.0+build.1",  # the tag would contain build metadata
            "0.1.0\n",
            "０.1.0",  # a fullwidth digit, which \d would accept
        )
        for version in refused:
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "must be a semantic version"):
                    check_version(version)
        # YAML reads an unquoted 1 or 1.0 as a number
        for version in (1, 1.0, True):
            with self.subTest(version=version):
                with self.assertRaisesRegex(ValueError, "in quotes"):
                    check_version(version)


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_must_store_its_configuration(self) -> None:
        require_configs({"model_config": {}, "data_config": {}}, "model.pt")
        with self.assertRaisesRegex(ValueError, "model.pt does not store data_config"):
            require_configs({"model_config": {}}, "model.pt")
        with self.assertRaisesRegex(ValueError, "model_config and data_config"):
            require_configs({"NTR": 600, "epoch": 4190}, "model.pt")

    def test_config_hash_ignores_key_order(self) -> None:
        a = config_sha256({"x": 1, "y": 2}, {"T1_lower": 0.1, "T1_upper": 6.0})
        b = config_sha256({"y": 2, "x": 1}, {"T1_upper": 6.0, "T1_lower": 0.1})
        self.assertEqual(a, b)
        self.assertEqual(len(a), 64)
        self.assertNotEqual(a, config_sha256({"x": 1, "y": 3}, {}))


class BundleTest(unittest.TestCase):
    def setUp(self) -> None:
        # Make source files for a bundle with 3 readouts, and set the bundle path
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        (self.root / "src").mkdir()
        self.files: dict[str, str | Path] = {}
        for target, spec in TARGETS.items():
            path = self.root / "src" / f"{target}.onnx"
            path.write_bytes(target.encode())
            self.files[spec.bundle_file] = path
        for name, _, _ in RF_FILES.values():
            path = self.root / "src" / Path(name).name
            path.write_text("".join(f"{i}.5\n" for i in range(3)))
            self.files[name] = path
        self.bundle = self.root / "bundle"

    def card(self, ntr: int = 3) -> str:
        sha = {t: file_sha256(self.files[s.bundle_file]) for t, s in TARGETS.items()}
        return card_text(replace(INPUTS, ntr=ntr, onnx_sha256=sha))

    def test_written_bundle_checks_out(self) -> None:
        write_bundle(self.bundle, self.card(), self.files)
        self.assertEqual(check_bundle(self.bundle)["model"]["ntr"], 3)
        self.assertEqual(
            sorted(str(p.relative_to(self.bundle)) for p in self.bundle.rglob("*")),
            sorted([CARD, "rf_data", *self.files]),
        )
        self.assertEqual(
            (self.bundle / "model.onnx").read_bytes(), b"signal_and_derivatives"
        )

    def test_a_new_bundle_replaces_the_old_one_whole(self) -> None:
        write_bundle(self.bundle, self.card(), self.files)
        (self.bundle / "stale.onnx").write_bytes(b"left over")
        write_bundle(self.bundle, self.card(), self.files)
        self.assertFalse((self.bundle / "stale.onnx").exists())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["bundle", "src"])

    def test_a_failed_write_keeps_the_old_bundle(self) -> None:
        write_bundle(self.bundle, self.card(), self.files)
        before = (self.bundle / CARD).read_text()
        with self.assertRaises(FileNotFoundError):
            write_bundle(
                self.bundle, self.card(), {**self.files, "x": self.root / "missing"}
            )
        # Also, a bundle that fails its check does not replace the old bundle
        with self.assertRaisesRegex(ValueError, "does not have ntr = 4 values"):
            write_bundle(self.bundle, self.card(ntr=4), self.files)
        self.assertEqual((self.bundle / CARD).read_text(), before)
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["bundle", "src"])

    def test_only_a_bundle_is_replaced(self) -> None:
        # A directory without a card, for example outputs/, is not deleted
        self.bundle.mkdir()
        (self.bundle / "checkpoint.pt").write_bytes(b"weights")
        with self.assertRaisesRegex(ValueError, "is not a bundle"):
            write_bundle(self.bundle, self.card(), self.files)
        self.assertEqual((self.bundle / "checkpoint.pt").read_bytes(), b"weights")
        self.assertEqual(sorted(p.name for p in self.root.iterdir()), ["bundle", "src"])
        # The sources of a bundle cannot be inside the bundle that it replaces
        (self.bundle / CARD).write_text("")
        inside = {**self.files, "model.onnx": self.bundle / "checkpoint.pt"}
        with self.assertRaisesRegex(ValueError, "is inside"):
            write_bundle(self.bundle, self.card(), inside)
        self.assertTrue((self.bundle / "checkpoint.pt").exists())
        # A bundle is replaced
        write_bundle(self.bundle, self.card(), self.files)
        self.assertFalse((self.bundle / "checkpoint.pt").exists())

    def test_check_finds_damaged_bundles(self) -> None:
        write_bundle(self.bundle, self.card(), self.files)
        (self.bundle / "model_signal.onnx").write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "model_signal.onnx does not have"):
            check_bundle(self.bundle)
        (self.bundle / "model.onnx").unlink()
        with self.assertRaisesRegex(ValueError, "model.onnx, which is missing"):
            check_bundle(self.bundle)
        card = self.bundle / CARD
        card.write_text(card.read_text().replace('"1.0"', '"2.0"'))
        with self.assertRaisesRegex(ValueError, "has schema_version '2.0', not '1.0'"):
            check_bundle(self.bundle)


class TarballTest(unittest.TestCase):
    def setUp(self) -> None:
        # Make a small bundle and set the path of its release file
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.bundle = self.root / "bundle"
        (self.bundle / "rf_data").mkdir(parents=True)
        self.files = {
            "model_card.toml": b'schema_version = "1.0"\n',
            "model.onnx": b"\x00onnx",
            "rf_data/rf_angle_list.txt": b"1.5\n2.5\n",
        }
        for name, data in self.files.items():
            (self.bundle / name).write_bytes(data)
        self.tarball = self.root / tarball_name("fisp600-rank16", "0.1.0-beta.1")

    def test_the_bundle_sits_at_the_root(self) -> None:
        write_tarball(self.bundle, self.tarball)
        with tarfile.open(self.tarball) as tar:
            self.assertEqual(
                tar.getnames(),
                [
                    "model.onnx",
                    "model_card.toml",
                    "rf_data",
                    "rf_data/rf_angle_list.txt",
                ],
            )
            for name, data in self.files.items():
                member = tar.extractfile(name)
                assert member is not None
                self.assertEqual(member.read(), data)

    def test_the_same_files_give_the_same_bytes(self) -> None:
        sha256 = write_tarball(self.bundle, self.tarball)
        first = self.tarball.read_bytes()
        self.assertEqual(sha256, file_sha256(self.tarball))
        # The times and modes of the files have no effect on the tarball
        os.utime(self.bundle / "model.onnx", (0, 1e9))
        (self.bundle / "model.onnx").chmod(0o600)
        self.assertEqual(write_tarball(self.bundle, self.tarball), sha256)
        self.assertEqual(self.tarball.read_bytes(), first)
        with tarfile.open(self.tarball) as tar:
            for member in tar.getmembers():
                with self.subTest(member=member.name):
                    self.assertEqual(member.mtime, 0)
                    self.assertEqual((member.uid, member.gid), (0, 0))
                    self.assertEqual((member.uname, member.gname), ("", ""))
                    self.assertEqual(member.mode, 0o755 if member.isdir() else 0o644)
        # gzip header: no file name (flag bit 3) and mtime 0
        self.assertEqual(first[3] & 0x08, 0)
        self.assertEqual(first[4:8], b"\0\0\0\0")

    def test_a_failed_write_leaves_no_partial_tarball(self) -> None:
        write_tarball(self.bundle, self.tarball)
        before = self.tarball.read_bytes()
        with mock.patch.object(tarfile.TarFile, "addfile", side_effect=OSError("full")):
            with self.assertRaisesRegex(OSError, "full"):
                write_tarball(self.bundle, self.tarball)
        self.assertEqual(self.tarball.read_bytes(), before)
        self.assertEqual(
            sorted(p.name for p in self.root.iterdir()), ["bundle", self.tarball.name]
        )
        with self.assertRaisesRegex(FileNotFoundError, "not a bundle directory"):
            write_tarball(self.root / "missing", self.root / "other.tar.gz")
        self.assertFalse((self.root / "other.tar.gz").exists())

    def test_release_names(self) -> None:
        self.assertEqual(
            release_tag("fisp600-rank16", "0.1.0-beta.1"),
            "model-fisp600-rank16-v0.1.0-beta.1",
        )
        self.assertEqual(
            self.tarball.name, "neuralbloch-model-fisp600-rank16-v0.1.0-beta.1.tar.gz"
        )

    def test_the_release_command(self) -> None:
        card = tomllib.loads(card_text(INPUTS))
        command = release_command(card, Path("outputs/model.tar.gz"), "f" * 64)
        self.assertEqual(
            command[:8],
            [
                "gh",
                "release",
                "create",
                "model-fisp600-rank16-v0.1.0-beta.1",
                "outputs/model.tar.gz",
                "--target",
                "a" * 40,
                "--prerelease",
            ],
        )
        self.assertEqual(
            command[8:10], ["--title", "Model fisp600-rank16 v0.1.0-beta.1"]
        )
        notes = command[command.index("--notes") + 1]
        self.assertIn("T1 0.1 to 6 s, T2 0.01 to 2 s, B1 0.5 to 1.5 (T2 < T1)", notes)
        self.assertIn("f" * 64, notes)
        # A version without a pre-release part is also not marked as latest
        card["model"]["version"] = "0.1.0"
        command = release_command(card, Path("outputs/model.tar.gz"), "f" * 64)
        self.assertIn("--latest=false", command)
        self.assertNotIn("--prerelease", command)


class RfTableTest(unittest.TestCase):
    def test_rf_tables_count_numbers(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "rf.txt"
            path.write_text("1.0\n\n2.5\n3\n")
            self.assertEqual(rf_table_length(path), 3)
            path.write_text("1.0\nangle\n")
            with self.assertRaisesRegex(ValueError, "rf.txt:2: not a number"):
                rf_table_length(path)

    def test_the_repository_rf_tables_have_600_values(self) -> None:
        for name in ("rf_angle_list.txt", "rf_phase_list.txt"):
            self.assertEqual(rf_table_length(Path("data/rf_data") / name), 600)
