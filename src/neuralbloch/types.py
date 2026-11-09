"""The schema of the config: one Pydantic model for each config file."""

import re
from pathlib import Path
from typing import Annotated
from typing import Literal
from typing import Self
from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveFloat
from pydantic import PositiveInt
from pydantic import field_validator
from pydantic import model_validator


# =============================================================================
# Model family and version names (export.model_family, export.model_version)
# =============================================================================


# The name of a family has the form <sequence><ntr>-rank<rank>[-<qualifier>], for
# ex:`fisp600-rank16`, `fisp600-rank16-narrow`.
FAMILY_PATTERN = re.compile(r"[a-z0-9]+(-[a-z0-9]+)*")
RANK_WORD = re.compile(r"rank[1-9][0-9]*")

# The version of a family is semantic with an optional pre-release part such
# as -beta.1.
PRERELEASE_PART = r"(?:0|[1-9][0-9]*|[0-9]*[A-Za-z-][0-9A-Za-z-]*)"
SEMVER = re.compile(
    r"(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)"
    rf"(?:-{PRERELEASE_PART}(?:\.{PRERELEASE_PART})*)?"
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
    rank_words = [word for word in family.split("-") if RANK_WORD.fullmatch(word)]
    if not FAMILY_PATTERN.fullmatch(family) or len(rank_words) != 1:
        raise ValueError(
            f"Model family {family!r} must be lowercase words with hyphens between "
            "them, and exactly one word must be rank<N>. The form is "
            "<sequence><ntr>-rank<rank>, for example fisp600-rank16, with an "
            "optional qualifier (fisp600-rank16-narrow)"
        )
    return int(rank_words[0].removeprefix("rank"))


def check_family(family: str, model_rank: int) -> None:
    """Raise an error if the rank in a family name is not the rank of the model.

    Raises:
        ValueError: If the name is not valid or gives a different rank.
    """
    stated = family_rank(family)
    if stated != model_rank:
        raise ValueError(
            f"Model family {family!r} states rank {stated}, but the checkpoint's "
            f"signal decoder has rank {model_rank}: set export.model_family to a "
            f"rank{model_rank} family"
        )


def check_version(version: object) -> str:
    """Return the family version if it is a semantic version (``SEMVER``).

    Raises:
        TypeError: If the version is not a string.
        ValueError: If the version is not a semantic version.
    """
    if not isinstance(version, str):
        raise TypeError(
            "export.model_version must be a semantic version in quotes, such as "
            f'"0.1.0-beta.1", not {version!r}'
        )
    if not SEMVER.fullmatch(version):
        raise ValueError(
            f"export.model_version {version!r} must be a semantic version "
            "MAJOR.MINOR.PATCH, optionally with a pre-release such as -beta.1, "
            "and without build metadata"
        )
    return version


# =============================================================================
# Shared
# =============================================================================


class Section(BaseModel):
    """The rules of all sections: all keys required, no unknown keys, no conversion."""

    model_config = ConfigDict(
        extra="forbid",
        strict=True,
        allow_inf_nan=False,
        validate_assignment=True,
    )


# A path that may not exist yet. YAML gives paths as strings.
ConfigPath = Annotated[Path, Field(strict=False)]


# =============================================================================
# config/data.yaml
# =============================================================================


class ParameterRange(Section):
    """The range of one sampled tissue parameter."""

    lower: float
    upper: float

    @model_validator(mode="after")
    def lower_below_upper(self) -> Self:
        """Refuse an empty range.

        Returns:
            The validated range.

        Raises:
            ValueError: If lower is not below upper.
        """
        if not self.lower < self.upper:
            raise ValueError(f"lower = {self.lower} must be below upper = {self.upper}")
        return self


class LogRange(ParameterRange):
    """A range that is sampled and scaled with log, so it starts above 0."""

    lower: PositiveFloat


class ParameterRanges(Section):
    """``data.parameter_ranges``: the training domain."""

    T1_s: LogRange
    T2_s: LogRange
    B1: ParameterRange


class DataConfig(Section):
    """``data.yaml``: the sequence and the generated data."""

    n_samples: PositiveInt
    derivative_method: Literal["forward_sensitivity", "finite_difference"]
    rf_angle_file: ConfigPath
    rf_phase_file: ConfigPath
    TR_s: PositiveFloat
    TE_s: PositiveFloat
    TI_s: NonNegativeFloat
    TW_s: NonNegativeFloat
    max_state: PositiveInt
    repetitions: PositiveInt
    inversion_prepulse: bool
    wait_spoiling: bool
    py_undersampling_factor: PositiveInt
    delta_k_spoil_rad_per_m: float
    parameter_ranges: ParameterRanges


# =============================================================================
# config/experiment.yaml
# =============================================================================


class ExperimentConfig(Section):
    """``experiment.yaml``: the seed, the device and the file locations."""

    seed: int
    device: str
    dataset_unsplit_path: ConfigPath
    dataset_for_validation_path: ConfigPath
    dataset_for_testing_path: ConfigPath
    pt_file_path: ConfigPath
    log_dir: ConfigPath


# =============================================================================
# config/model.yaml
# =============================================================================


class Network(Section):
    """A sub-network and its decoder."""

    rank: PositiveInt
    hidden_size: PositiveInt
    # At least the input and the output layer
    num_layers: Annotated[int, Field(ge=2)]


class DirectDerivative(Section):
    """``model.derivative`` with its own networks for dT1 and dT2."""

    mode: Literal["direct"]
    dT1: Network  # ruff: ignore[mixed-case-variable-in-class-scope]
    dT2: Network  # ruff: ignore[mixed-case-variable-in-class-scope]


class DerivedDerivative(Section):
    """``model.derivative`` that derives dT1 and dT2 from the signal."""

    mode: Literal["derived"]
    basis_initialization: Literal["learned", "joint_pca"]


class ModelConfig(Section):
    """``model.yaml``: the network."""

    activation: Literal["relu", "silu", "gelu", "tanh", "softplus"]
    normalize_inputs: bool
    signal: Network
    derivative: Annotated[
        DirectDerivative | DerivedDerivative, Field(discriminator="mode")
    ]

    @model_validator(mode="after")
    def derived_mode_needs_a_smooth_activation(self) -> Self:
        """Refuse relu, which has no second derivative, in derived mode.

        Returns:
            The validated model.

        Raises:
            ValueError: If the activation is relu and the mode is derived.
        """
        if self.derivative.mode == "derived" and self.activation == "relu":
            raise ValueError("activation must be smooth when derivative.mode='derived'")
        return self


# =============================================================================
# config/training.yaml
# =============================================================================


class CosineScheduler(Section):
    """``training.lr_scheduler`` for CosineAnnealingLR."""

    name: Literal["cosine"]


class PlateauScheduler(Section):
    """``training.lr_scheduler`` for ReduceLROnPlateau."""

    name: Literal["plateau"]
    patience_epochs: PositiveInt
    factor: Annotated[float, Field(gt=0.0, lt=1.0)]
    min_lr: NonNegativeFloat


class TrainingConfig(Section):
    """``training.yaml``: the optimizer and the training loop."""

    n_epochs: PositiveInt
    train_batch_size: PositiveInt
    learning_rate: PositiveFloat
    validation_batch_size: PositiveInt
    validate_every_n_epochs: PositiveInt
    compile_train_step: bool
    lr_scheduler: Annotated[
        CosineScheduler | PlateauScheduler, Field(discriminator="name")
    ]
    loss_function: Literal[
        "nmae", "relative_l2", "relative_l2_squared", "hybrid_relative"
    ]
    online_samples_per_epoch: PositiveInt
    regen_every_n_epochs: PositiveInt


# =============================================================================
# config/testing.yaml
# =============================================================================


class TestingConfig(Section):
    """``testing.yaml``: the inference benchmark."""

    batch_size: PositiveInt
    infer_batch_size: PositiveInt
    n_runs: PositiveInt


# =============================================================================
# config/export.yaml
# =============================================================================


class Target(Section):
    """The exported ONNX model of one target."""

    onnx_file_path: ConfigPath


class Targets(Section):
    """``export.targets``: one ONNX model for each target."""

    signal_and_derivs: Target
    signal: Target
    coeffs: Target

    @model_validator(mode="after")
    def paths_are_different_files(self) -> Self:
        """Refuse two targets that write one file: only the last export would stay.

        Returns:
            The validated targets.

        Raises:
            ValueError: If two targets resolve to the same file.
        """
        seen: dict[Path, str] = {}
        for name, target in self:
            path = target.onnx_file_path.resolve()
            if path in seen:
                raise ValueError(f"{name} and {seen[path]} are the same file, {path}")
            seen[path] = name
        return self


class ExportConfig(Section):
    """``export.yaml``: the ONNX export and the model bundle."""

    targets: Targets
    opset_version: PositiveInt
    fp16: bool
    n_verify_samples: PositiveInt
    bundle_dir: ConfigPath
    model_family: str
    model_version: str
    offline: bool

    @field_validator("model_family")
    @classmethod
    def family_follows_the_pattern(cls, family: str) -> str:
        """Refuse a family name that does not follow the naming rules.

        Returns:
            The family name.
        """
        family_rank(family)
        return family

    @field_validator("model_version")
    @classmethod
    def version_is_semantic(cls, version: str) -> str:
        """Refuse a version that is not a semantic version.

        Returns:
            The version.
        """
        return check_version(version)


# =============================================================================
# All files together
# =============================================================================


class Config(Section):
    """All config sections, one for each file in the config dir."""

    data: DataConfig
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig
    testing: TestingConfig
    export: ExportConfig
