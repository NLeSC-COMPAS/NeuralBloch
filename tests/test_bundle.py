import tempfile
import tomllib
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from scripts.export_bundle import FORMAT
from scripts.export_bundle import ONNX_FILES
from scripts.export_bundle import CardInputs
from scripts.export_bundle import card_text
from scripts.export_bundle import check_bundle
from scripts.export_bundle import check_family
from scripts.export_bundle import check_version
from scripts.export_bundle import config_sha256
from scripts.export_bundle import derivative_method
from scripts.export_bundle import family_rank
from scripts.export_bundle import file_sha256
from scripts.export_bundle import onnx_precision
from scripts.export_bundle import require_configs
from scripts.export_bundle import rf_table_length
from scripts.export_bundle import write_bundle
from scripts.export_onnx import export_target
from src.model import FCModel


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

MODEL_CONFIG = {
    "input_dim": 3,
    "NB_signal": 2,
    "NB_dT1": 2,
    "NB_dT2": 2,
    "hidden_size_signal": 4,
    "hidden_size_dT1": 4,
    "hidden_size_dT2": 4,
    "num_layers_signal": 2,
    "num_layers_dT1": 2,
    "num_layers_dT2": 2,
}

INPUTS = CardInputs(
    family="fisp600-rank16",
    version=1,
    ntr=600,
    rank=16,
    onnx_sha256={target: f"{i}" * 64 for i, target in enumerate(ONNX_FILES)},
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
        self.assertEqual(card["format"], FORMAT)
        model = card["model"]
        self.assertEqual(
            (model["family"], model["version"], model["ntr"], model["rank"]),
            ("fisp600-rank16", 1, 600, 16),
        )
        for i, (target, onnx) in enumerate(ONNX_FILES.items()):
            self.assertEqual(model[target], {"onnx": onnx, "sha256": f"{i}" * 64})
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
        # An integer bound (T1_upper: 6) is written as a number with a point
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

    def test_version_is_a_positive_integer(self) -> None:
        self.assertEqual(check_version(2), 2)
        for version in (0, -1, 1.0, "1", True):
            with self.subTest(version=version), self.assertRaises(ValueError):
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

    def test_onnx_precision(self) -> None:
        model = FCModel(5, MODEL_CONFIG).eval()
        x = torch.tensor([[0.8, 0.07, 1.0], [1.2, 0.1, 0.9]])
        with tempfile.TemporaryDirectory() as tmp:
            fp32, fp16 = Path(tmp) / "fp32.onnx", Path(tmp) / "fp16.onnx"
            export_target(model, "coefficients", fp32, example=x, opset=18)
            export_target(model, "coefficients", fp16, example=x, opset=18, fp16=True)
            self.assertEqual(onnx_precision([fp32, fp32]), "FP32")
            self.assertEqual(onnx_precision([fp16]), "FP16")
            with self.assertRaisesRegex(ValueError, "mix FP16 and FP32"):
                onnx_precision([fp32, fp16])


class BundleTest(unittest.TestCase):
    def make_sources(self, root: Path, ntr: int = 3) -> dict[str, str | Path]:
        files: dict[str, str | Path] = {}
        for target, name in ONNX_FILES.items():
            path = root / "src" / f"{target}.onnx"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(target.encode())
            files[name] = path
        for name in ("rf_data/rf_angle_list.txt", "rf_data/rf_phase_list.txt"):
            path = root / "src" / Path(name).name
            path.write_text("".join(f"{i}.5\n" for i in range(ntr)))
            files[name] = path
        return files

    def card_for(self, files: dict[str, str | Path], ntr: int = 3) -> str:
        sha = {t: file_sha256(files[name]) for t, name in ONNX_FILES.items()}
        return card_text(replace(INPUTS, ntr=ntr, onnx_sha256=sha))

    def test_written_bundle_checks_out(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.make_sources(root)
            bundle = root / "out" / "bundle"
            write_bundle(bundle, self.card_for(files), files)
            card = check_bundle(bundle)
            self.assertEqual(card["model"]["ntr"], 3)
            self.assertEqual(
                sorted(str(p.relative_to(bundle)) for p in bundle.rglob("*")),
                sorted(["card.toml", "rf_data", *files]),
            )
            self.assertEqual(
                (bundle / "model.onnx").read_bytes(), b"signal_and_derivatives"
            )

    def test_a_new_bundle_replaces_the_old_one_whole(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.make_sources(root)
            bundle = root / "bundle"
            write_bundle(bundle, self.card_for(files), files)
            (bundle / "stale.onnx").write_bytes(b"left over")
            write_bundle(bundle, self.card_for(files), files)
            self.assertFalse((bundle / "stale.onnx").exists())
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["bundle", "src"])

    def test_a_failed_write_keeps_the_old_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.make_sources(root)
            bundle = root / "bundle"
            write_bundle(bundle, self.card_for(files), files)
            before = (bundle / "card.toml").read_text()
            with self.assertRaises(FileNotFoundError):
                write_bundle(bundle, "format = 2\n", {**files, "x": root / "missing"})
            self.assertEqual((bundle / "card.toml").read_text(), before)
            self.assertEqual(sorted(p.name for p in root.iterdir()), ["bundle", "src"])

    def test_check_finds_damaged_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            files = self.make_sources(root)
            bundle = root / "bundle"
            write_bundle(bundle, self.card_for(files), files)

            (bundle / "model_signal.onnx").write_bytes(b"changed")
            with self.assertRaisesRegex(ValueError, "model_signal.onnx does not have"):
                check_bundle(bundle)

            write_bundle(bundle, self.card_for(files, ntr=4), files)
            with self.assertRaisesRegex(ValueError, "does not have ntr = 4 values"):
                check_bundle(bundle)

            (bundle / "model.onnx").unlink()
            with self.assertRaisesRegex(ValueError, "model.onnx, which is missing"):
                check_bundle(bundle)

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
