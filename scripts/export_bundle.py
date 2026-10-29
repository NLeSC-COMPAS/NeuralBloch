import gzip
import hashlib
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import tarfile
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from scripts.export_onnx import TARGETS
from scripts.export_onnx import onnx_precision
from scripts.export_onnx import require_checkpoint
from scripts.export_onnx import require_exported
from scripts.export_onnx import target_paths
from scripts.export_onnx import training_domain
from scripts.export_onnx import verify_onnx
from src.utils import RF_TABLE_KEYS
from src.utils import file_sha256
from src.utils import load_checkpoint
from src.utils import model_from_checkpoint
from src.utils import rf_sha256
from src.utils import sample_parameters


# The model bundle that NeuralBloch.jl reads contains the ONNX model of each
# target, the RF tables of the training sequence, and model_card.toml, which
# describes them. This script generates the card. Do not write it by hand. Each
# field comes from the checkpoint, from the export, or from a constant below that
# gives its source.

# The version of the card schema, "MAJOR.MINOR". src/model.jl in NeuralBloch.jl
# reads it. Increase MINOR for a new optional field that readers can ignore.
# NeuralBloch.jl reads all MINOR versions of its MAJOR. Increase MAJOR for all
# other changes: a field gets a new name, is removed, gets a new type or a new
# meaning, or becomes required, or the card gets a new name. Change NeuralBloch.jl
# at the same time, because it does not accept a different MAJOR.
SCHEMA_VERSION = "1.0"

CARD = "model_card.toml"

# The RF tables of the bundle, by card key: (file in the bundle, key of the source
# file in the data configuration, contents of the table).
RF_FILES = {
    "rf_angle": ("rf_data/rf_angle_list.txt", "RF_ANGLE_FILE", "flip angles, degrees"),
    "rf_phase": ("rf_data/rf_phase_list.txt", "RF_PHASE_FILE", "phases, degrees"),
}

# Each DERIVATIVE_METHOD name in data/data_generation.jl -> the name in the card.
DERIVATIVE_METHODS = {
    "forward_sensitivity": "forward sensitivity",
    "finite_difference": "forward difference",
}

# data/data_generation.jl keeps only samples with T1 > T2. This is a physical
# requirement, so it is a constant here and not a setting.
CONSTRAINT = "T2 < T1"


# The training domain: (card key, lower key, upper key, how it was sampled).
DOMAIN_FIELDS = (
    ("T1", "T1_lower", "T1_upper", "seconds, sampled log-uniformly"),
    ("T2", "T2_lower", "T2_upper", "seconds, sampled log-uniformly"),
    ("B1", "B1_lower", "B1_upper", "relative to nominal, sampled uniformly"),
)

# The name of a family has the form <sequence><ntr>-rank<rank>[-<qualifier>], for
# example fisp600-rank16 or fisp600-rank16-narrow. A qualifier makes the names of
# two families different. The checks are only these: the name is lowercase words
# with hyphens between them, and exactly one of these words is rank<N>, with the
# rank of the model.
# The checks use fullmatch, because $ also matches before a final newline.
FAMILY_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
RANK_WORD = re.compile(r"rank([1-9][0-9]*)")

# The version of a family is a semantic version (semver.org): MAJOR.MINOR.PATCH,
# with an optional pre-release part such as -beta.1. PATCH counts new exports of
# the same weights. MINOR counts new weights or a wider domain. MAJOR stays 0
# until the family is stable. Build metadata (+...) is not permitted, because the
# version is part of the release tag and the download URL. The patterns use
# [0-9] and not \d, because \d also matches other Unicode digits.
PRERELEASE_PART = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    rf"(?:-{PRERELEASE_PART}(?:\.{PRERELEASE_PART})*)?"
)


@dataclass(frozen=True)
class CardInputs:
    """The data that ``export_bundle`` collects for the card.

    Attributes:
        family: The model family, ``<sequence><ntr>-rank<rank>[-qualifier]``.
        version: The family's semantic version (``SEMVER``), e.g.
            ``0.1.0-beta.1``.
        ntr: The number of readouts per voxel.
        rank: The number of signal basis coefficients per voxel.
        onnx_sha256: The sha256 of each target's ONNX model.
        data_config: The checkpoint's data configuration.
        finite_difference_steps: The T1 and T2 steps of BlochSimulators if the
            model was trained on finite-difference derivatives. Otherwise ``None``.
        upstream: The NeuralBloch commit that exported the bundle.
        checkpoint_epoch: The checkpoint's epoch.
        config_sha256: The hash of the configurations in the checkpoint.
        blochsimulators: The BlochSimulators version of the export
            environment.
        precision: The precision of the ONNX models, ``"FP32"`` or ``"FP16"``.
    """

    family: str
    version: str
    ntr: int
    rank: int
    onnx_sha256: Mapping[str, str]
    data_config: Mapping[str, Any]
    finite_difference_steps: tuple[float, float] | None
    upstream: str
    checkpoint_epoch: int
    config_sha256: str
    blochsimulators: str
    precision: str

    def __post_init__(self) -> None:
        """Raise an error if the fields do not agree.

        Raises:
            ValueError: If the version or the family is not valid, the family
                gives a different rank, the finite-difference steps do not
                match the derivative method, ``onnx_sha256`` does not have
                exactly the targets, or the precision is unknown.
        """
        check_version(self.version)
        check_family(self.family, self.rank)
        method = derivative_method(self.data_config)
        if (method == "forward difference") != (
            self.finite_difference_steps is not None
        ):
            raise ValueError(
                "Finite-difference steps are necessary for a model trained on "
                "finite-difference derivatives, and not permitted for other "
                f"models. This model used {method}"
            )
        if set(self.onnx_sha256) != set(TARGETS):
            raise ValueError(
                f"The card needs the sha256 of exactly the targets {list(TARGETS)}, "
                f"not {list(self.onnx_sha256)}"
            )
        if self.precision not in ("FP32", "FP16"):
            raise ValueError(
                f'The precision must be "FP32" or "FP16", not {self.precision!r}'
            )


def require_configs(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    """Raise an error if a checkpoint does not contain its own configuration.

    The card describes the model in the checkpoint. Thus, all fields must come
    from the training configuration of the model. Older checkpoints do not
    contain this configuration. This check gives a clearer error than the
    ``KeyError`` of ``model_from_checkpoint``.

    Args:
        checkpoint: The loaded checkpoint.
        path: The path of the checkpoint file, for the error message.

    Raises:
        ValueError: If ``model_config`` or ``data_config`` is missing.
    """
    missing = [key for key in ("model_config", "data_config") if key not in checkpoint]
    if missing:
        raise ValueError(
            f"{path} does not store {' and '.join(missing)}, so the model card "
            "cannot come from it. Train the model again, or add its training "
            "configuration to the checkpoint yourself "
            "(checkpoint['model_config'] = ..., checkpoint['data_config'] = ..., "
            "torch.save(checkpoint, path))."
        )


def family_rank(family: str) -> int:
    """Return the rank that a family name gives.

    Args:
        family: A family name, e.g. ``fisp600-rank16``.

    Returns:
        The number after ``rank``.

    Raises:
        ValueError: If the name does not follow the pattern.
    """
    ranks = [
        match.group(1)
        for word in family.split("-")
        if (match := RANK_WORD.fullmatch(word)) is not None
    ]
    if not FAMILY_PATTERN.fullmatch(family) or len(ranks) != 1:
        raise ValueError(
            f"Model family {family!r} must be lowercase words with hyphens between "
            "them, and exactly one word must be rank<N>. The form is "
            "<sequence><ntr>-rank<rank>, for example fisp600-rank16, with an "
            "optional qualifier (fisp600-rank16-narrow)"
        )
    return int(ranks[0])


def check_family(family: str, model_rank: int) -> None:
    """Raise an error if the rank in a family name is not the rank of the model.

    Raises:
        ValueError: If the name is not valid or gives a different rank.
    """
    stated = family_rank(family)
    if stated != model_rank:
        raise ValueError(
            f"Model family {family!r} states rank {stated}, but the checkpoint's "
            f"signal decoder has rank {model_rank}: set export.MODEL_FAMILY to a "
            f"rank{model_rank} family"
        )


def check_version(version: object) -> str:
    """Return the family version if it is a semantic version (``SEMVER``).

    Raises:
        ValueError: If the version is not a semantic version.
    """
    if not isinstance(version, str):
        raise ValueError(
            "export.MODEL_VERSION must be a semantic version in quotes, such as "
            f'"0.1.0-beta.1", not {version!r}'
        )
    if not SEMVER.fullmatch(version):
        raise ValueError(
            f"export.MODEL_VERSION {version!r} must be a semantic version "
            "MAJOR.MINOR.PATCH, optionally with a pre-release such as -beta.1, "
            "and without build metadata"
        )
    return version


def derivative_method(data_config: Mapping[str, Any]) -> str:
    """Return the method that calculated the training derivatives, as the card names it.

    The data configuration must give the method. data_generation.jl uses
    forward sensitivity if DERIVATIVE_METHOD is missing, but before commit
    088ba08 it used only finite differences, and its configurations did not
    give DERIVATIVE_METHOD. Thus, a missing method does not tell which method
    calculated the derivatives.

    Raises:
        ValueError: If the data configuration does not give the method, or
            names an unknown method.
    """
    if "DERIVATIVE_METHOD" not in data_config:
        raise ValueError(
            "The checkpoint's data_config does not give DERIVATIVE_METHOD, so the "
            "card cannot state the derivative method. Checkpoints from before "
            "commit 088ba08 were trained with finite_difference. Add the method "
            "of the training to the checkpoint yourself "
            "(checkpoint['data_config']['DERIVATIVE_METHOD'] = ..., "
            "torch.save(checkpoint, path))."
        )
    method = data_config["DERIVATIVE_METHOD"]
    if method not in DERIVATIVE_METHODS:
        raise ValueError(
            f"Unknown DERIVATIVE_METHOD {method!r}; "
            f"known: {', '.join(DERIVATIVE_METHODS)}"
        )
    return DERIVATIVE_METHODS[method]


def check_rf_provenance(
    checkpoint: Mapping[str, Any], data_config: Mapping[str, Any]
) -> str | None:
    """Raise an error if the RF tables changed after the training.

    The trainer records the sha256 of the RF tables in the checkpoint
    (``rf_sha256``). Older checkpoints do not record them. For these, the
    function cannot check the tables, and it returns a warning.

    Args:
        checkpoint: The loaded checkpoint.
        data_config: The checkpoint's data configuration.

    Returns:
        The warning if the checkpoint does not record its RF tables, else
        ``None``.

    Raises:
        ValueError: If an RF table does not have the sha256 that the checkpoint
            records.
    """
    recorded = checkpoint.get("rf_sha256")
    if recorded is None:
        return (
            "the checkpoint does not record the sha256 of its RF tables, so the "
            "export cannot check that the RF tables are the tables of the "
            "training: " + ", ".join(str(data_config[k]) for k in RF_TABLE_KEYS)
        )
    current = rf_sha256(data_config)
    for key in RF_TABLE_KEYS:
        if recorded.get(key) != current[key]:
            raise ValueError(
                f"{data_config[key]} ({key}) is not the RF table of the training: "
                "its sha256 is not the sha256 that the checkpoint records"
            )
    return None


def check_undersampling(data_config: Mapping[str, Any]) -> None:
    """Raise an error if the training sequence is undersampled.

    With PY_UNDERSAMPLING_FACTOR R, BlochSimulators gives R readouts for each RF
    pulse, so NTR = R * (length of the RF tables). Schema 1.0 and NeuralBloch.jl
    require one RF table value for each readout, thus R = 1.

    Raises:
        ValueError: If PY_UNDERSAMPLING_FACTOR is not the integer 1.
    """
    factor = integer(data_config["PY_UNDERSAMPLING_FACTOR"], "PY_UNDERSAMPLING_FACTOR")
    if factor != 1:
        raise ValueError(
            f"PY_UNDERSAMPLING_FACTOR is {factor!r}. A bundle requires 1, because "
            f"the RF tables of schema {SCHEMA_VERSION} have one value for each "
            "readout, and BlochSimulators gives PY_UNDERSAMPLING_FACTOR readouts "
            "for each RF pulse"
        )


def config_sha256(
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> str:
    """Return the sha256 of the checkpoint configurations. Key order has no effect."""
    canonical = json.dumps(
        {"model_config": model_config, "data_config": data_config},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def rf_table_length(path: str | Path) -> int:
    """Return the number of values in an RF table (one number per non-empty line).

    NeuralBloch.jl reads each line with ``tryparse(Float64, line)``. Python
    ``float`` also accepts ``_`` between digits and digits that are not ASCII
    (for example ``١``), so the function refuses ``_`` and characters that are
    not ASCII. It also refuses values that are not finite.

    Raises:
        ValueError: If a line is not a finite number.
    """
    count = 0
    with open(path, encoding="utf-8") as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                value = float(line)
            except ValueError:
                value = math.nan
            if "_" in line or not line.isascii() or not math.isfinite(value):
                raise ValueError(f"{path}:{number}: not a finite number: {line!r}")
            count += 1
    return count


def toml_value(value: object) -> str:
    """Return the TOML literal of a string, boolean, integer, float or list of them."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"A card value must be finite, not {value}")
        return repr(value)
    if isinstance(value, str):
        # A JSON string is a valid TOML basic string
        return json.dumps(value)
    if isinstance(value, list | tuple):
        return "[" + ", ".join(toml_value(v) for v in value) + "]"
    raise TypeError(f"No TOML literal for {value!r}")


def number(value: object, name: str) -> float:
    """Return a card number: a float, or an integer as a float.

    Raises:
        ValueError: If ``value`` is not a number.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{name} must be a number, not {value!r}")


def integer(value: object, name: str) -> int:
    """Return a card integer.

    Raises:
        ValueError: If ``value`` is not an integer.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be an integer, not {value!r}")


def boolean(value: object, name: str) -> bool:
    """Return a card boolean.

    Raises:
        ValueError: If ``value`` is not true or false.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be true or false, not {value!r}")


# The sequence that data/data_generation.jl simulates. It is FISP3D(...), made
# from these fields of the data configuration: (card key, data key, type, unit).
SEQUENCE_KIND = "FISP3D"
SEQUENCE_FIELDS = (
    ("TR", "TR_S", number, "seconds"),
    ("TE", "TE_S", number, "seconds"),
    ("TI", "TI_S", number, "seconds"),
    ("TW", "TW_S", number, "seconds"),
    ("max_state", "MAX_STATE", integer, ""),
    ("repetitions", "REPETITIONS", integer, ""),
    ("inversion_prepulse", "INVERSION_PREPULSE", boolean, ""),
    ("wait_spoiling", "WAIT_SPOILING", boolean, ""),
    ("undersampling_factor", "PY_UNDERSAMPLING_FACTOR", integer, ""),
    ("delta_k_spoil", "DELTA_K_SPOIL_RAD_PER_M", number, "rad/m"),
)


def card_text(inputs: CardInputs) -> str:
    """Return the text of model_card.toml with the source of each field.

    The card uses schema ``SCHEMA_VERSION``.

    ``CardInputs`` checks that its fields agree. This function checks the
    values of the data configuration while it writes them.

    Raises:
        ValueError: If a value has the wrong type, or a domain range is empty.
    """
    data = inputs.data_config
    method = derivative_method(data)

    lines: list[str] = []

    def add(line: str, comment: str) -> None:
        lines.append(f"{line:<36}  # {comment}" if comment else line)

    def field(key: str, value: object, comment: str = "") -> None:
        add(f"{key} = {toml_value(value)}", comment)

    def table(name: str, comment: str = "") -> None:
        lines.append("")
        add(f"[{name}]", comment)

    lines += [
        "# NeuralBloch model card. The export_bundle task of NeuralBloch generates",
        "# this file (uv run run.py export_bundle). Do not edit it by hand. Units",
        "# are physical: seconds for T1, T2 and times, B1 relative to nominal,",
        "# derivatives per second.",
        "",
    ]
    field(
        "schema_version", SCHEMA_VERSION, "of this card: a newer MINOR only adds fields"
    )

    table("model")
    field("family", inputs.family, "export.MODEL_FAMILY: <sequence><ntr>-rank<rank>")
    field("version", inputs.version, "export.MODEL_VERSION: semantic version")
    field("ntr", inputs.ntr, "readouts per voxel: the checkpoint's NTR")
    field("rank", inputs.rank, "basis coefficients per voxel: U_signal's rank")

    for target, spec in TARGETS.items():
        table(f"model.{target}", f"FCModel.{spec.method}: {', '.join(spec.outputs)}")
        field("onnx", spec.bundle_file)
        field("sha256", inputs.onnx_sha256[target])

    table("domain", "where the network was trained: data_config")
    for key, lower, upper, comment in DOMAIN_FIELDS:
        low, high = number(data[lower], lower), number(data[upper], upper)
        if not low < high:
            raise ValueError(f"{lower} = {low} must be below {upper} = {high}")
        field(key, [low, high], f"{lower}, {upper}: {comment}")
    field("constraint", CONSTRAINT, "data/data_generation.jl keeps only T1 > T2")

    table("derivatives", "how the training targets were computed")
    field("method", method, "data_config DERIVATIVE_METHOD")
    if inputs.finite_difference_steps is not None:
        dT1, dT2 = inputs.finite_difference_steps
        field("dT1_step", dT1, "BlochSimulators.DEFAULT_ΔT₁")
        field("dT2_step", dT2, "BlochSimulators.DEFAULT_ΔT₂")

    table("sequence", "FISP3D(...) in data/data_generation.jl, from data_config")
    field("kind", SEQUENCE_KIND)
    for key, data_key, convert, unit in SEQUENCE_FIELDS:
        field(
            key,
            convert(data[data_key], data_key),
            ", ".join(filter(None, (data_key, unit))),
        )
    for key, (name, data_key, holds) in RF_FILES.items():
        field(key, name, f"{data_key}: {holds}")

    table("provenance")
    field("upstream", inputs.upstream, "the NeuralBloch commit that exported it")
    field("checkpoint_epoch", inputs.checkpoint_epoch, "the checkpoint's epoch")
    field("config_sha256", inputs.config_sha256, "of model_config and data_config")
    field("blochsimulators", inputs.blochsimulators, "in the export environment")
    field("precision", inputs.precision, "of the ONNX models")

    return "\n".join(lines) + "\n"


def write_bundle(
    bundle_dir: str | Path,
    card: str,
    files: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Write a bundle: ``card`` as model_card.toml and ``files`` copied into it.

    The function first makes the bundle in a new directory next to
    ``bundle_dir`` and checks it (``check_bundle``). Then the new directory
    replaces ``bundle_dir``. Thus, ``bundle_dir`` never contains a mix of two
    bundles, or a bundle that fails the check.

    The function deletes the old ``bundle_dir``. Thus, it replaces only a
    directory that contains model_card.toml, and it does not accept sources
    inside ``bundle_dir``. It also does not accept a symbolic link as
    ``bundle_dir``, because the new directory would replace the link and not
    the bundle that the link points to.

    Args:
        bundle_dir: The directory of the bundle. If it exists, it must be a
            bundle, and it is replaced.
        card: The text of model_card.toml.
        files: The other files of the bundle: path in the bundle -> source.

    Returns:
        The parsed card, from ``check_bundle``.

    Raises:
        ValueError: If ``bundle_dir`` is a symbolic link, exists and is not a
            bundle, a source is inside ``bundle_dir``, or the assembled bundle
            fails ``check_bundle``.
        KeyError: If ``card`` does not have a table or key that
            ``check_bundle`` reads.
    """
    bundle_dir = Path(bundle_dir)
    if bundle_dir.is_symlink():
        raise ValueError(
            f"{bundle_dir} is a symbolic link, so it is not replaced. Set "
            f"export.BUNDLE_DIR to the directory itself, {bundle_dir.resolve()}"
        )
    if bundle_dir.exists() and not (bundle_dir / CARD).is_file():
        raise ValueError(
            f"{bundle_dir} is not a bundle (it has no {CARD}), so it is not "
            "replaced. Set export.BUNDLE_DIR to a new directory or to a bundle"
        )
    for source in files.values():
        if Path(source).resolve().is_relative_to(bundle_dir.resolve()):
            raise ValueError(
                f"{source} is inside {bundle_dir}, which the new bundle replaces. "
                "Keep the sources of the bundle outside export.BUNDLE_DIR"
            )
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    new = Path(tempfile.mkdtemp(prefix=f".{bundle_dir.name}-", dir=bundle_dir.parent))
    old = bundle_dir.with_name(f".{bundle_dir.name}-old")
    moved = False
    try:
        for name, source in files.items():
            target = new / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        # The card can contain characters that are not ASCII (ΔT₁)
        (new / CARD).write_text(card, encoding="utf-8")
        new.chmod(0o755)
        parsed = check_bundle(new)
        if bundle_dir.exists():
            # An earlier write can leave the old bundle
            remove(old)
            bundle_dir.rename(old)
            moved = True
        new.rename(bundle_dir)
    except BaseException:
        shutil.rmtree(new, ignore_errors=True)
        if moved and not bundle_dir.exists():
            old.rename(bundle_dir)
        raise
    if moved:
        shutil.rmtree(old)
    return parsed


def remove(path: Path) -> None:
    """Delete a file, a symbolic link or a directory tree, if it exists."""
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)


def check_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Read the card of a bundle again and check the files that it names.

    Args:
        bundle_dir: The bundle's directory.

    Returns:
        The parsed card.

    Raises:
        ValueError: If the card has a different schema version, a file that it
            names is missing, the hash of an ONNX model is not the hash in the
            card, or an RF table does not have ``ntr`` values.
        KeyError: If the card does not have a table or key that the function
            reads.
    """
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / CARD, "rb") as f:
        card = tomllib.load(f)
    if card.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(
            f"{bundle_dir / CARD} has schema_version "
            f"{card.get('schema_version')!r}, not {SCHEMA_VERSION!r}"
        )
    for target in TARGETS:
        entry = card["model"][target]
        path = bundle_dir / entry["onnx"]
        if not path.is_file():
            raise ValueError(
                f"{bundle_dir / CARD} names {entry['onnx']}, which is missing"
            )
        if file_sha256(path) != entry["sha256"]:
            raise ValueError(f"{path} does not have the sha256 the card states")
    for key in RF_FILES:
        path = bundle_dir / card["sequence"][key]
        if not path.is_file():
            raise ValueError(
                f"{bundle_dir / CARD} names {card['sequence'][key]}, which is missing"
            )
        if rf_table_length(path) != card["model"]["ntr"]:
            raise ValueError(
                f"{path} does not have ntr = {card['model']['ntr']} values"
            )
    return card


# Each family version is published as a GitHub release of this repository. The
# release has one file, the tarball of the bundle. dev/bind_bundle.jl in
# NeuralBloch.jl makes the download URL from the same two names.


def release_tag(family: str, version: str) -> str:
    """Return the release tag of a family version.

    For example ``model-fisp600-rank16-v0.1.0-beta.1``.
    """
    return f"model-{family}-v{version}"


def tarball_name(family: str, version: str) -> str:
    """Return the name of the file in the release of a family version.

    For example ``neuralbloch-model-fisp600-rank16-v0.1.0-beta.1.tar.gz``.
    """
    return f"neuralbloch-{release_tag(family, version)}.tar.gz"


def write_tarball(bundle_dir: str | Path, tarball: str | Path) -> str:
    """Pack a bundle into a reproducible tarball, the release file.

    The tarball has the bundle files at its root. With the same zlib, the same
    files always give the same bytes. Thus, you can calculate the sha256 that
    NeuralBloch.jl pins for the release again from the bundle. (A different
    zlib can compress the same files to different bytes.) For this, the tarball
    has: members
    in sorted order, USTAR headers with mtime 0, owner 0:0, mode 0644 for files
    and 0755 for directories, and a gzip header with mtime 0 and no file name.
    The function first writes a temporary file next to the tarball. Then this
    file replaces the tarball. Thus, a failed write does not leave a partial
    tarball.

    Args:
        bundle_dir: The directory of the bundle.
        tarball: The path of the tarball. If it exists, it is replaced.

    Returns:
        The sha256 of the tarball, in hexadecimal.

    Raises:
        FileNotFoundError: If ``bundle_dir`` is not a directory.
    """
    bundle_dir, tarball = Path(bundle_dir), Path(tarball)
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"{bundle_dir} is not a bundle directory")
    paths = sorted(
        bundle_dir.rglob("*"), key=lambda path: path.relative_to(bundle_dir).as_posix()
    )
    tarball.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{tarball.name}-", dir=tarball.parent)
    try:
        with (
            open(fd, "wb") as raw,
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed,
            tarfile.open(
                fileobj=compressed, mode="w", format=tarfile.USTAR_FORMAT
            ) as tar,
        ):
            for path in paths:
                info = tarfile.TarInfo(path.relative_to(bundle_dir).as_posix())
                info.mtime, info.uid, info.gid = 0, 0, 0
                info.uname, info.gname = "", ""
                if path.is_dir():
                    info.type, info.mode = tarfile.DIRTYPE, 0o755
                    tar.addfile(info)
                else:
                    info.size, info.mode = path.stat().st_size, 0o644
                    with open(path, "rb") as f:
                        tar.addfile(info, f)
        # mkstemp makes the file readable only by its owner
        os.chmod(temporary, 0o644)
        Path(temporary).replace(tarball)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
    return file_sha256(tarball)


def release_command(card: Mapping[str, Any], tarball: Path, sha256: str) -> list[str]:
    """Return the ``gh`` command that publishes a bundle tarball as its release.

    The command tags the release at the commit that the card names. It
    publishes a version with a pre-release part (``-beta.1``) as a GitHub
    pre-release. It does not mark other versions as latest, so the latest
    release stays a code release.

    Args:
        card: The parsed card of the bundle.
        tarball: The tarball that ``write_tarball`` wrote.
        sha256: The sha256 of the tarball.

    Returns:
        The command, as a list with one argument in each item.
    """
    model, domain = card["model"], card["domain"]
    family, version = model["family"], model["version"]
    provenance = card["provenance"]
    ranges = ", ".join(
        f"{name} {domain[name][0]:g} to {domain[name][1]:g}{unit}"
        for name, unit in (("T1", " s"), ("T2", " s"), ("B1", ""))
    )
    notes = (
        f"Model {family} {version} for NeuralBloch.jl: {CARD}, an ONNX model "
        "per target and the RF tables of its sequence. "
        f"{card['sequence']['kind']}, {model['ntr']} readouts, rank {model['rank']}; "
        f"trained on {ranges} ({domain['constraint']}) with "
        f"{card['derivatives']['method']} derivatives; checkpoint epoch "
        f"{provenance['checkpoint_epoch']}, {provenance['precision']}. "
        f"sha256 of {tarball.name}: {sha256}"
    )
    # The version has no build metadata, so a hyphen in it starts the
    # pre-release part
    latest = "--prerelease" if "-" in version else "--latest=false"
    return [
        "gh",
        "release",
        "create",
        release_tag(family, version),
        str(tarball),
        "--target",
        provenance["upstream"].removeprefix("NeuralBloch@"),
        latest,
        "--title",
        f"Model {family} v{version}",
        "--notes",
        notes,
    ]


def export_bundle(config: dict[str, Any]) -> None:
    """Write the model bundle that NeuralBloch.jl reads, from the exported targets.

    The bundle (``export.BUNDLE_DIR``) contains the ONNX model of each target,
    the RF tables of the training sequence and ``model_card.toml``. The card
    gives the family and version of the model, its shapes, the training domain,
    the derivative method, the sequence and the provenance (see ``card_text``).
    The function also writes the release file of the bundle next to the bundle
    (``write_tarball``). It prints the command that publishes this file, but it
    does not publish anything.

    The card must describe the checkpoint correctly. Thus, the export raises an
    error for: a checkpoint without its training configuration, a family name
    with a rank that is not the rank of the checkpoint, ONNX models that come
    from a different checkpoint or do not give the same results as the
    checkpoint, RF tables that changed after the training
    (``check_rf_provenance``), and uncommitted changes in this repository. (The
    card gives the commit of this repository.) It also raises an error if the
    release of the family version exists (``require_unreleased``).

    Args:
        config: Full configuration dictionary.
    """
    export_config = config["export"]
    checkpoint_path = Path(config["experiment"]["PT_FILE_PATH"])
    # Check the configurations before the model is made from them
    checkpoint = load_checkpoint(config, "cpu")
    require_configs(checkpoint, checkpoint_path)
    model = model_from_checkpoint(checkpoint, "cpu")

    family = export_config["MODEL_FAMILY"]
    version = check_version(export_config["MODEL_VERSION"])
    check_family(family, model.rank)

    commit = clean_commit()
    upstream = f"NeuralBloch@{commit}"

    # The export continues after a warning. Thus, give each warning when it
    # occurs, and again at the end, where the user reads the result.
    warnings: list[str] = []

    def warn(warning: str | None) -> None:
        if warning is not None:
            print(f"\nWARNING: {warning}")
            warnings.append(warning)

    warn(require_unreleased(release_tag(family, version), export_config["OFFLINE"]))

    data_config = checkpoint["data_config"]
    method = derivative_method(data_config)
    check_undersampling(data_config)
    # check_bundle checks this again, but only after the slow steps below
    for key, (_, data_key, _) in RF_FILES.items():
        values = rf_table_length(data_config[data_key])
        if values != model.NTR:
            raise ValueError(
                f"{data_config[data_key]} ({key}) has {values} values; "
                f"the checkpoint has NTR = {model.NTR}"
            )
    warn(check_rf_provenance(checkpoint, data_config))

    # The ONNX models must come from this checkpoint. Check the checkpoint that
    # export_onnx recorded in each model, then compare each model with it.
    onnx_paths = target_paths(export_config)
    require_exported(onnx_paths)
    checkpoint_sha256 = file_sha256(checkpoint_path)
    for onnx_path in onnx_paths.values():
        require_checkpoint(onnx_path, checkpoint_sha256, checkpoint_path)
    precision = onnx_precision(list(onnx_paths.values()))
    x = sample_parameters(
        training_domain(checkpoint), export_config["N_VERIFY_SAMPLES"]
    )
    for target, onnx_path in onnx_paths.items():
        verify_onnx(model, target, onnx_path, x, precision)

    blochsimulators, steps = blochsimulators_facts(method == "forward difference")

    card = card_text(
        CardInputs(
            family=family,
            version=version,
            ntr=model.NTR,
            rank=model.rank,
            onnx_sha256={t: file_sha256(path) for t, path in onnx_paths.items()},
            data_config=data_config,
            finite_difference_steps=steps,
            upstream=upstream,
            checkpoint_epoch=checkpoint["epoch"],
            config_sha256=config_sha256(checkpoint["model_config"], data_config),
            blochsimulators=blochsimulators,
            precision=precision,
        )
    )

    files: dict[str, str | Path] = {
        TARGETS[target].bundle_file: path for target, path in onnx_paths.items()
    }
    files |= {name: data_config[data_key] for name, data_key, _ in RF_FILES.values()}

    bundle_dir = Path(export_config["BUNDLE_DIR"])
    parsed = write_bundle(bundle_dir, card, files)

    # Write the release file next to the bundle
    tarball = bundle_dir.with_name(tarball_name(family, version))
    sha256 = write_tarball(bundle_dir, tarball)

    print(
        f"\nWrote the model bundle {family}-v{version} to {bundle_dir}: "
        f"NTR {model.NTR}, rank {model.rank}, {method} derivatives, {precision}, "
        f"from {checkpoint_path} (epoch {checkpoint['epoch']}) and {upstream}"
    )
    print(
        f"\nWrote its release file {tarball} ({tarball.stat().st_size:,} bytes), "
        f"sha256 {sha256}. Nothing was published. To publish it, push "
        f"{commit[:7]}, bind the file in "
        "NeuralBloch.jl (julia --project dev/bind_bundle.jl FILE) and run its "
        "tests, then create the release:\n\n"
        + shlex.join(release_command(parsed, tarball, sha256))
    )
    for warning in warnings:
        print(f"\nWARNING: {warning}. Make sure of this before you publish.")


def clean_commit() -> str:
    """Return the commit of this repository if it has no uncommitted changes.

    Returns:
        The full hash of ``HEAD``.

    Raises:
        RuntimeError: If ``git status`` gives a change (a changed or staged
            file, or a file that is not tracked and not ignored), or git fails.
    """
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "The model card names the NeuralBloch commit that exported it, so "
            "commit or remove these changes first:\n" + status
        )
    return git("rev-parse", "HEAD")


def require_unreleased(tag: str, offline: bool) -> str | None:
    """Raise an error if the release tag of a family version exists.

    Users can pin the sha256 of a published release file, so a published
    version must not get a new file. The function looks for the tag in this
    repository and on ``origin``.

    Args:
        tag: The release tag (``release_tag``).
        offline: What to do if the function cannot get the tags of ``origin``,
            for example without a network. If ``True``, return a warning. If
            ``False``, raise an error.

    Returns:
        The warning if ``offline`` and the tags of ``origin`` are not
        available, else ``None``.

    Raises:
        RuntimeError: If the tag exists, or the tags of ``origin`` are not
            available and ``offline`` is ``False``.
    """
    exists = RuntimeError(
        f"The release {tag} exists, and a published version does not change. "
        "Increase export.MODEL_VERSION"
    )
    if git("tag", "--list", tag):
        raise exists
    # Do not wait for a password or a passphrase
    env = {**os.environ, "GIT_TERMINAL_PROMPT": "0"}
    env.setdefault("GIT_SSH_COMMAND", "ssh -o BatchMode=yes")
    try:
        remote = subprocess.run(
            ["git", "ls-remote", "--tags", "origin", f"refs/tags/{tag}"],
            check=True,
            capture_output=True,
            text=True,
            cwd=REPO_ROOT,
            env=env,
            timeout=60,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as error:
        reason = (getattr(error, "stderr", None) or str(error)).strip()
        unknown = (
            f"cannot get the tags of origin ({reason}), so the export does not "
            f"know if the release {tag} exists"
        )
        if not offline:
            raise RuntimeError(
                f"The export {unknown}. Correct the access to origin, or set "
                "export.OFFLINE to true to continue without this check"
            ) from error
        return f"{unknown}. Do not publish a version that exists"
    if remote:
        raise exists
    return None


# The card names the commit of this repository. Thus, git runs in this
# repository and not in the current directory, which can be a different
# repository.
REPO_ROOT = Path(__file__).resolve().parents[1]


def git(*args: str) -> str:
    """Return the stripped output of a git command in this repository.

    Raises:
        RuntimeError: If git fails.
    """
    try:
        result = subprocess.run(
            ["git", *args], check=True, capture_output=True, text=True, cwd=REPO_ROOT
        )
    except subprocess.CalledProcessError as error:
        raise RuntimeError(
            f"git {' '.join(args)} failed in {REPO_ROOT}: {error.stderr.strip()}"
        ) from error
    return result.stdout.strip()


def blochsimulators_facts(
    finite_difference: bool,
) -> tuple[str, tuple[float, float] | None]:
    """Return the BlochSimulators version and its T1 and T2 forward-difference steps.

    The training data come from simulations in the Julia environment of this
    repository (juliapkg.json), so the function reads the data from this
    environment. It reads the version from the Project.toml file of the package
    and does not load the package. (When Julia loads the package, the package
    also initializes CUDA, in addition to the CUDA of PyTorch.) The function
    loads the package only when it needs the steps, and reads them from it.

    Args:
        finite_difference: Whether the model was trained on finite-difference
            derivatives.

    Returns:
        The version, and ``(dT1_step, dT2_step)`` if ``finite_difference``,
        else ``None``.
    """
    from juliacall import Main as jl

    jl.seval("import TOML")
    version = str(
        jl.seval(
            "TOML.parsefile(joinpath(dirname(dirname(Base.locate_package("
            'Base.identify_package("BlochSimulators")))), "Project.toml"))["version"]'
        )
    )
    steps = None
    if finite_difference:
        jl.seval("using BlochSimulators")
        steps = (
            float(jl.seval("BlochSimulators.DEFAULT_ΔT₁")),
            float(jl.seval("BlochSimulators.DEFAULT_ΔT₂")),
        )
    return version, steps
