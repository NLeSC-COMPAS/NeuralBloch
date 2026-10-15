import subprocess
from pathlib import Path
from typing import Any
import onnx
import torch
from scripts.export_onnx import target_paths
from scripts.export_onnx import verify_onnx
from src.bundle import ONNX_FILES
from src.bundle import RF_FILES
from src.bundle import CardInputs
from src.bundle import card_text
from src.bundle import check_bundle
from src.bundle import check_family
from src.bundle import check_version
from src.bundle import config_sha256
from src.bundle import derivative_method
from src.bundle import file_sha256
from src.bundle import require_configs
from src.bundle import rf_table_length
from src.bundle import write_bundle
from src.targets import rank
from src.utils import load_model


def export_bundle(config: dict[str, Any]) -> None:
    """Write the model bundle NeuralBloch.jl reads, from the exported targets.

    The bundle (``export.BUNDLE_DIR``) holds the ONNX model of every target,
    the RF tables of the training sequence and ``card.toml``, which states
    the model's family and version, its shapes, training domain, derivative
    method, sequence and provenance (``src/bundle.py``).

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
    family = export_config["MODEL_FAMILY"]
    version = check_version(export_config["MODEL_VERSION"])
    check_family(family, rank(model))

    upstream = f"NeuralBloch@{clean_commit()}"

    data_config = checkpoint["data_config"]
    for key, (_, data_key) in RF_FILES.items():
        values = rf_table_length(data_config[data_key])
        if values != model.NTR:
            raise ValueError(
                f"{data_config[data_key]} ({key}) has {values} values; "
                f"the checkpoint has NTR = {model.NTR}"
            )

    # The ONNX models must be this checkpoint's: each is checked against it
    paths = target_paths(export_config)
    for target, (onnx_path, _) in paths.items():
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"{onnx_path} not found. Run `uv run run.py export_onnx` first."
            )
        verify_onnx(model, target, onnx_path, config, checkpoint)
    precision = onnx_precision([onnx_path for onnx_path, _ in paths.values()])

    blochsimulators, steps = blochsimulators_facts(derivative_method(data_config))

    card = card_text(
        CardInputs(
            family=family,
            version=version,
            ntr=model.NTR,
            rank=rank(model),
            onnx_sha256={
                target: file_sha256(onnx_path)
                for target, (onnx_path, _) in paths.items()
            },
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
        ONNX_FILES[target]: onnx_path for target, (onnx_path, _) in paths.items()
    }
    files |= {name: data_config[data_key] for name, data_key in RF_FILES.values()}

    bundle_dir = Path(export_config["BUNDLE_DIR"])
    write_bundle(bundle_dir, card, files)
    check_bundle(bundle_dir)

    print(
        f"\nWrote the model bundle {family}-v{version} to {bundle_dir}: "
        f"NTR {model.NTR}, rank {rank(model)}, {derivative_method(data_config)} "
        f"derivatives, {precision}, from {checkpoint_path} (epoch "
        f"{checkpoint['epoch']}) and {upstream}"
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


def onnx_precision(paths: list[Path]) -> str:
    """``"FP16"`` or ``"FP32"``: the precision the ONNX models' weights have.

    Raises:
        ValueError: If the models do not all have the same precision.
    """
    precisions = set()
    for path in paths:
        graph = onnx.load(str(path)).graph
        half = any(w.data_type == onnx.TensorProto.FLOAT16 for w in graph.initializer)
        precisions.add("FP16" if half else "FP32")
    if len(precisions) != 1:
        raise ValueError(
            "The ONNX models mix FP16 and FP32; export them again together"
        )
    return precisions.pop()


def blochsimulators_facts(method: str) -> tuple[str, tuple[float, float] | None]:
    """BlochSimulators' version, and its T1 and T2 steps for forward differences.

    The training data are simulated in this repository's Julia environment
    (juliapkg.json), so it is the one asked.

    Args:
        method: The card's derivative method.

    Returns:
        The version, and ``(dT1_step, dT2_step)`` for ``"forward
        difference"`` or ``None`` otherwise.
    """
    from juliacall import Main as jl

    jl.seval("using BlochSimulators")
    version = str(jl.seval("string(pkgversion(BlochSimulators))"))
    steps = None
    if method == "forward difference":
        steps = (
            float(jl.seval("BlochSimulators.DEFAULT_ΔT₁")),
            float(jl.seval("BlochSimulators.DEFAULT_ΔT₂")),
        )
    return version, steps
