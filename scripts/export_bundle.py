import hashlib
import json
import math
import re
import shutil
import subprocess
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
from scripts.export_onnx import TARGETS
from scripts.export_onnx import onnx_precision
from scripts.export_onnx import rank
from scripts.export_onnx import require_exported
from scripts.export_onnx import target_paths
from scripts.export_onnx import training_domain
from scripts.export_onnx import verify_onnx
from src.utils import load_model
from src.utils import sample_parameters


# The model bundle NeuralBloch.jl reads: the ONNX model of every target, the
# RF tables of the training sequence, and card.toml, which describes them.
# The card is generated here, never written by hand: every field comes from
# the checkpoint, from the export, or from a constant below that names its
# source.

# The card format NeuralBloch.jl reads. Change it with every change of the
# card's fields, together with NeuralBloch.jl's src/model.jl.
FORMAT = 3

CARD = "card.toml"

# The bundle's file for each target (TARGETS in scripts/export_onnx.py).
ONNX_FILES = {
    "signal_and_derivatives": "model.onnx",
    "signal": "model_signal.onnx",
    "coefficients": "model_coefficients.onnx",
}

# The bundle's RF tables, by card key: (file in the bundle, the data
# configuration's key for the source file, what the table holds).
RF_FILES = {
    "rf_angle": ("rf_data/rf_angle_list.txt", "RF_ANGLE_FILE", "flip angles, degrees"),
    "rf_phase": ("rf_data/rf_phase_list.txt", "RF_PHASE_FILE", "phases, degrees"),
}

# DERIVATIVE_METHOD as data/data_generation.jl names it -> the card's name,
# with the same default as data_generation.jl.
DERIVATIVE_METHODS = {
    "forward_sensitivity": "forward sensitivity",
    "finite_difference": "forward difference",
}
DEFAULT_DERIVATIVE_METHOD = "forward_sensitivity"

# data/data_generation.jl keeps only samples with T1 > T2: a physical
# requirement, so it is a constant here rather than a setting.
CONSTRAINT = "T2 < T1"


# The training domain: (card key, lower key, upper key, how it was sampled).
DOMAIN_FIELDS = (
    ("T1", "T1_lower", "T1_upper", "seconds, sampled log-uniformly"),
    ("T2", "T2_lower", "T2_upper", "seconds, sampled log-uniformly"),
    ("B1", "B1_lower", "B1_upper", "relative to nominal, sampled uniformly"),
)

# A family is named <sequence><ntr>-rank<rank>, optionally with a qualifier,
# e.g. fisp600-rank16 or fisp600-rank16-narrow: lowercase words joined by
# hyphens, exactly one of them rank<N>.
FAMILY_PATTERN = re.compile(r"^[a-z0-9]+(-[a-z0-9]+)*$")
RANK_WORD = re.compile(r"^rank([1-9][0-9]*)$")

# A family's version is a semantic version (semver.org): MAJOR.MINOR.PATCH,
# optionally with a pre-release such as -beta.1. PATCH counts re-exports of
# the same weights, MINOR new weights or a wider domain, and MAJOR stays 0
# until the family is stable. Build metadata (+...) is not allowed: the
# version is part of the release tag and the download URL. Written with
# [0-9], since \d also matches other Unicode digits.
PRERELEASE_PART = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    rf"(?:-{PRERELEASE_PART}(?:\.{PRERELEASE_PART})*)?"
)


@dataclass(frozen=True)
class CardInputs:
    """Everything the card states, gathered by ``export_bundle``.

    Attributes:
        family: The model family, ``<sequence><ntr>-rank<rank>[-qualifier]``.
        version: The family's semantic version (``SEMVER``), e.g.
            ``0.1.0-beta.1``.
        ntr: Readouts per voxel.
        rank: Signal basis coefficients per voxel.
        onnx_sha256: The sha256 of each target's ONNX model.
        data_config: The checkpoint's data configuration.
        finite_difference_steps: BlochSimulators' T1 and T2 steps, for a
            model trained on finite-difference derivatives; else ``None``.
        upstream: The NeuralBloch commit that exported the bundle.
        checkpoint_epoch: The checkpoint's epoch.
        config_sha256: The hash of the checkpoint's configurations.
        blochsimulators: The BlochSimulators version of the export
            environment.
        precision: ``"FP32"`` or ``"FP16"``, of the ONNX models.
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


def require_configs(checkpoint: Mapping[str, Any], path: str | Path) -> None:
    """Refuse a checkpoint that does not record its own configuration.

    The card describes the model in the checkpoint, so every field must come
    from the configuration it was trained with. Older checkpoints did not
    store it, and ``load_model`` then falls back to ``config/``, which may
    describe another model.

    Args:
        checkpoint: The loaded checkpoint.
        path: Where it was loaded from, for the message.

    Raises:
        ValueError: If ``model_config`` or ``data_config`` is missing.
    """
    missing = [key for key in ("model_config", "data_config") if key not in checkpoint]
    if missing:
        raise ValueError(
            f"{path} does not store {' and '.join(missing)}, so the model card "
            "cannot be taken from it: checkpoints written before the training "
            "configuration was stored in them would otherwise be described by "
            "config/, which may belong to another model. Retrain, or add the "
            "configuration the checkpoint was trained with to it deliberately "
            "(checkpoint['model_config'] = ..., checkpoint['data_config'] = ..., "
            "torch.save(checkpoint, path))."
        )


def family_rank(family: str) -> int:
    """The rank a family name states.

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
        if (match := RANK_WORD.match(word)) is not None
    ]
    if not FAMILY_PATTERN.match(family) or len(ranks) != 1:
        raise ValueError(
            f"Model family {family!r} must be lowercase words joined by hyphens, "
            "exactly one of them rank<N>: <sequence><ntr>-rank<rank>, e.g. "
            "fisp600-rank16, optionally with a qualifier (fisp600-rank16-narrow)"
        )
    return int(ranks[0])


def check_family(family: str, model_rank: int) -> None:
    """Refuse a family name whose rank is not the model's.

    Raises:
        ValueError: If the name is malformed or states another rank.
    """
    stated = family_rank(family)
    if stated != model_rank:
        raise ValueError(
            f"Model family {family!r} states rank {stated}, but the checkpoint's "
            f"signal decoder has rank {model_rank}: set export.MODEL_FAMILY to a "
            f"rank{model_rank} family"
        )


def check_version(version: object) -> str:
    """The family version, which must be a semantic version (``SEMVER``).

    Raises:
        ValueError: If it is not.
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
    """How the training targets' derivatives were computed, as the card says it.

    Raises:
        ValueError: If the data configuration names an unknown method.
    """
    method = data_config.get("DERIVATIVE_METHOD", DEFAULT_DERIVATIVE_METHOD)
    if method not in DERIVATIVE_METHODS:
        raise ValueError(
            f"Unknown DERIVATIVE_METHOD {method!r}; "
            f"known: {', '.join(DERIVATIVE_METHODS)}"
        )
    return DERIVATIVE_METHODS[method]


def config_sha256(
    model_config: Mapping[str, Any],
    data_config: Mapping[str, Any],
) -> str:
    """The sha256 of a checkpoint's configurations, independent of key order."""
    canonical = json.dumps(
        {"model_config": model_config, "data_config": data_config},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(canonical.encode()).hexdigest()


def file_sha256(path: str | Path) -> str:
    """The sha256 of a file's contents, in hexadecimal."""
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def rf_table_length(path: str | Path) -> int:
    """The number of values in an RF table: one number per non-empty line.

    Raises:
        ValueError: If a line is not a number.
    """
    count = 0
    with open(path) as f:
        for number, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                float(line)
            except ValueError:
                raise ValueError(f"{path}:{number}: not a number: {line!r}") from None
            count += 1
    return count


def toml_value(value: object) -> str:
    """A TOML literal for a string, boolean, integer, float or list of them."""
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
    """A card number: a float, or an integer written as one.

    Raises:
        ValueError: If ``value`` is not a number.
    """
    if isinstance(value, int | float) and not isinstance(value, bool):
        return float(value)
    raise ValueError(f"{name} must be a number, not {value!r}")


def integer(value: object, name: str) -> int:
    """A card integer.

    Raises:
        ValueError: If ``value`` is not an integer.
    """
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be an integer, not {value!r}")


def boolean(value: object, name: str) -> bool:
    """A card boolean.

    Raises:
        ValueError: If ``value`` is not true or false.
    """
    if isinstance(value, bool):
        return value
    raise ValueError(f"{name} must be true or false, not {value!r}")


# The sequence data/data_generation.jl simulates: FISP3D(...) built from these
# fields of the data configuration: (card key, data key, type, unit).
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
    """The text of card.toml, format ``FORMAT``, with each field's source.

    Raises:
        ValueError: If a value has the wrong type, a domain range is empty,
            or the finite-difference steps do not match the method.
    """
    data = inputs.data_config
    method = derivative_method(data)
    if (method == "forward difference") != (inputs.finite_difference_steps is not None):
        raise ValueError(
            "Finite-difference steps belong to, and only to, a model trained on "
            f"finite-difference derivatives; this one used {method}"
        )

    lines: list[str] = []

    def field(key: str, value: object, comment: str = "") -> None:
        line = f"{key} = {toml_value(value)}"
        lines.append(f"{line:<36}  # {comment}" if comment else line)

    def table(name: str, comment: str = "") -> None:
        lines.append("")
        header = f"[{name}]"
        lines.append(f"{header:<36}  # {comment}" if comment else header)

    lines += [
        "# NeuralBloch model card. Generated by NeuralBloch's export_bundle",
        "# (uv run run.py export_bundle): do not edit by hand. Units are physical:",
        "# seconds for T1, T2 and times, B1 relative to nominal, derivatives per",
        "# second.",
        "",
    ]
    field("format", FORMAT, "the card format NeuralBloch.jl reads")

    table("model")
    field("family", inputs.family, "export.MODEL_FAMILY: <sequence><ntr>-rank<rank>")
    field("version", inputs.version, "export.MODEL_VERSION: semantic version")
    field("ntr", inputs.ntr, "readouts per voxel: the checkpoint's NTR")
    field("rank", inputs.rank, "basis coefficients per voxel: U_signal's rank")

    for target, spec in TARGETS.items():
        table(f"model.{target}", f"FCModel.{spec.method}: {', '.join(spec.outputs)}")
        field("onnx", ONNX_FILES[target])
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
        field("dT1_step", number(dT1, "dT1_step"), "BlochSimulators.DEFAULT_ΔT₁")
        field("dT2_step", number(dT2, "dT2_step"), "BlochSimulators.DEFAULT_ΔT₂")

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
) -> None:
    """Write a bundle: ``card`` as card.toml and ``files`` copied into it.

    The bundle is assembled in a new directory next to ``bundle_dir``,
    checked (``check_bundle``), and then takes its place, so ``bundle_dir``
    never holds a mix of two bundles, nor one that fails the check.

    Args:
        bundle_dir: The bundle's directory; an existing one is replaced.
        card: The text of card.toml.
        files: The bundle's other files: path in the bundle -> source.

    Raises:
        ValueError: If the assembled bundle fails ``check_bundle``.
    """
    bundle_dir = Path(bundle_dir)
    bundle_dir.parent.mkdir(parents=True, exist_ok=True)
    new = Path(tempfile.mkdtemp(prefix=f".{bundle_dir.name}-", dir=bundle_dir.parent))
    old = bundle_dir.with_name(f".{bundle_dir.name}-old")
    moved = False
    try:
        for name, source in files.items():
            target = new / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        (new / CARD).write_text(card)
        new.chmod(0o755)
        check_bundle(new)
        if bundle_dir.exists():
            shutil.rmtree(old, ignore_errors=True)
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


def check_bundle(bundle_dir: str | Path) -> dict[str, Any]:
    """Read a bundle's card back and check the files it names.

    Args:
        bundle_dir: The bundle's directory.

    Returns:
        The parsed card.

    Raises:
        ValueError: If the card has another format, a file it names is
            missing, an ONNX model's hash differs from the card's, or an RF
            table does not have ``ntr`` values.
    """
    bundle_dir = Path(bundle_dir)
    with open(bundle_dir / CARD, "rb") as f:
        card = tomllib.load(f)
    if card.get("format") != FORMAT:
        raise ValueError(f"{bundle_dir / CARD} has format {card.get('format')!r}")
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
        if rf_table_length(path) != card["model"]["ntr"]:
            raise ValueError(
                f"{path} does not have ntr = {card['model']['ntr']} values"
            )
    return card


def export_bundle(config: dict[str, Any]) -> None:
    """Write the model bundle NeuralBloch.jl reads, from the exported targets.

    The bundle (``export.BUNDLE_DIR``) holds the ONNX model of every target,
    the RF tables of the training sequence and ``card.toml``, which states
    the model's family and version, its shapes, training domain, derivative
    method, sequence and provenance (see ``card_text``).

    The card must describe the checkpoint truthfully, so the export refuses
    a checkpoint without its training configuration, a family name whose
    rank is not the checkpoint's, ONNX models that do not reproduce the
    checkpoint, and uncommitted changes in this repository, whose commit the
    card names.

    Args:
        config: Full configuration dictionary.
    """
    export_config = config["export"]
    checkpoint_path = Path(config["experiment"]["PT_FILE_PATH"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    require_configs(checkpoint, checkpoint_path)

    model, _ = load_model(config, "cpu")
    model_rank = rank(model)
    family = export_config["MODEL_FAMILY"]
    version = check_version(export_config["MODEL_VERSION"])
    check_family(family, model_rank)

    upstream = f"NeuralBloch@{clean_commit()}"

    data_config = checkpoint["data_config"]
    method = derivative_method(data_config)
    for key, (_, data_key, _) in RF_FILES.items():
        values = rf_table_length(data_config[data_key])
        if values != model.NTR:
            raise ValueError(
                f"{data_config[data_key]} ({key}) has {values} values; "
                f"the checkpoint has NTR = {model.NTR}"
            )

    # The ONNX models must be this checkpoint's: each is checked against it
    paths = target_paths(export_config)
    require_exported(paths)
    onnx_paths = {target: onnx_path for target, (onnx_path, _) in paths.items()}
    precision = onnx_precision(list(onnx_paths.values()))
    x = sample_parameters(
        training_domain(config, checkpoint), export_config["N_VERIFY_SAMPLES"]
    )
    for target, onnx_path in onnx_paths.items():
        verify_onnx(model, target, onnx_path, x)

    blochsimulators, steps = blochsimulators_facts(method)

    card = card_text(
        CardInputs(
            family=family,
            version=version,
            ntr=model.NTR,
            rank=model_rank,
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
        ONNX_FILES[target]: path for target, path in onnx_paths.items()
    }
    files |= {name: data_config[data_key] for name, data_key, _ in RF_FILES.values()}

    bundle_dir = Path(export_config["BUNDLE_DIR"])
    write_bundle(bundle_dir, card, files)

    print(
        f"\nWrote the model bundle {family}-v{version} to {bundle_dir}: "
        f"NTR {model.NTR}, rank {model_rank}, {method} derivatives, {precision}, "
        f"from {checkpoint_path} (epoch {checkpoint['epoch']}) and {upstream}"
    )


def clean_commit() -> str:
    """The commit of this repository, which must have no uncommitted changes.

    Returns:
        The full hash of ``HEAD``.

    Raises:
        RuntimeError: If tracked or untracked files differ from ``HEAD``.
    """
    status = git("status", "--porcelain")
    if status:
        raise RuntimeError(
            "The model card names the NeuralBloch commit that exported it, so "
            "commit or remove these changes first:\n" + status
        )
    return git("rev-parse", "HEAD")


def git(*args: str) -> str:
    """The output of a git command in this repository, stripped."""
    return subprocess.run(
        ["git", *args], check=True, capture_output=True, text=True
    ).stdout.strip()


def blochsimulators_facts(method: str) -> tuple[str, tuple[float, float] | None]:
    """BlochSimulators' version, and its T1 and T2 steps for forward differences.

    The training data are simulated in this repository's Julia environment
    (juliapkg.json), so it is the one asked. The version is read from the
    package's Project.toml, without loading the package, which would also
    initialize CUDA next to PyTorch's; the steps, when needed, from the
    package itself.

    Args:
        method: The card's derivative method.

    Returns:
        The version, and ``(dT1_step, dT2_step)`` for ``"forward
        difference"`` or ``None`` otherwise.
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
    if method == "forward difference":
        jl.seval("using BlochSimulators")
        steps = (
            float(jl.seval("BlochSimulators.DEFAULT_ΔT₁")),
            float(jl.seval("BlochSimulators.DEFAULT_ΔT₂")),
        )
    return version, steps
