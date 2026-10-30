from collections.abc import Callable
from collections.abc import Iterator
from dataclasses import dataclass
from dataclasses import field
from itertools import pairwise
from pathlib import Path
from typing import Any
import pytest
import torch
from data.data_loading import TensorBatchLoader
from src.loss import Loss
from src.model import FCModel
from src.trainer import Trainer
from src.utils import load_checkpoint
from src.utils import load_model
from src.utils import rf_sha256


Weights = dict[str, torch.Tensor]


def copy_weights(model: FCModel) -> Weights:
    return {key: value.clone() for key, value in model.state_dict().items()}


def same_weights(a: Weights, b: Weights) -> bool:
    return a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a)


@pytest.fixture
def make_trainer(
    config: dict[str, Any],
    make_model: Callable[..., FCModel],
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> Iterator[Callable[..., Trainer]]:
    """Return a function that makes a Trainer of config, with the model of seed 0.

    The keyword arguments of the function change values of the training
    section of config. The tests change config only in this way.
    """
    trainers: list[Trainer] = []

    def make(
        val_loader: TensorBatchLoader | None = None,
        **training: object,
    ) -> Trainer:
        config["training"].update(training)
        torch.manual_seed(0)
        model = make_model(5, config["model"], config["data"])
        if val_loader is None:
            batch_size = config["training"]["VALIDATION_BATCH_SIZE"]
            val_loader = synthetic_loader(8, batch_size)
        trainer = Trainer(model, val_loader, config)
        trainers.append(trainer)
        return trainer

    yield make
    # Only train() closes the TensorBoard writer, and only at its end
    for trainer in trainers:
        trainer.writer.close()


@dataclass
class TrainingEvents:
    """What a trainer does in train()."""

    # The loader that the trainer generates at an epoch, by epoch
    generated: dict[int, TensorBatchLoader] = field(default_factory=dict)
    # The loader of each epoch, in the order of the epochs
    trained_on: list[TensorBatchLoader] = field(default_factory=list)
    # The weights of the model at each validation, by epoch: the keys are the
    # validated epochs
    weights: dict[int, Weights] = field(default_factory=dict)
    # The epoch of each checkpoint that is saved, in order
    saved: list[int] = field(default_factory=list)
    # The arguments of each scheduler step, and the learning rate after it
    scheduler_steps: list[tuple[float, ...]] = field(default_factory=list)
    learning_rates: list[float] = field(default_factory=list)


@pytest.fixture
def record_training(
    monkeypatch: pytest.MonkeyPatch,
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> Callable[..., TrainingEvents]:
    """Return a function that records what a trainer does in train().

    When the trainer generates data, it gets 10 synthetic samples, not Julia
    data. Each torch.save of the test is recorded, so write other files before
    the recording starts. With val_losses, the validations give these losses in
    this order and do not run the model: give one loss for each validation.
    """

    def record(
        trainer: Trainer,
        val_losses: list[float] | None = None,
    ) -> TrainingEvents:
        events = TrainingEvents()
        losses = None if val_losses is None else iter(val_losses)

        def epoch() -> int:
            return len(events.trained_on)

        def generate_train_loader(config: dict[str, Any]) -> TensorBatchLoader:
            batch_size = config["training"]["TRAIN_BATCH_SIZE"]
            loader = synthetic_loader(10, batch_size, shuffle=True)
            events.generated[epoch() + 1] = loader
            return loader

        train_epoch = trainer.train_epoch

        def recording_train_epoch(
            loader: TensorBatchLoader,
        ) -> tuple[float, float, float, float]:
            events.trained_on.append(loader)
            return train_epoch(loader)

        validate = trainer.validate

        def recording_validate() -> float:
            events.weights[epoch()] = copy_weights(trainer.model)
            return validate() if losses is None else next(losses)

        save = torch.save

        def recording_save(
            checkpoint: dict[str, Any],
            *args: object,
            **kwargs: object,
        ) -> None:
            events.saved.append(checkpoint["epoch"])
            save(checkpoint, *args, **kwargs)

        step = trainer.scheduler.step

        def recording_step(*args: float, **kwargs: float) -> None:
            events.scheduler_steps.append((*args, *kwargs.values()))
            step(*args, **kwargs)
            events.learning_rates.append(trainer.optimizer.param_groups[0]["lr"])

        monkeypatch.setattr("src.trainer.generate_train_loader", generate_train_loader)
        monkeypatch.setattr(trainer, "train_epoch", recording_train_epoch)
        monkeypatch.setattr(trainer, "validate", recording_validate)
        monkeypatch.setattr(torch, "save", recording_save)
        monkeypatch.setattr(trainer.scheduler, "step", recording_step)
        return events

    return record


def test_the_data_are_generated_at_the_first_epoch_and_every_regen_every_epochs(
    make_trainer: Callable[..., Trainer],
    record_training: Callable[..., TrainingEvents],
) -> None:
    trainer = make_trainer(N_EPOCHS=7, REGEN_EVERY=3)
    events = record_training(trainer)

    trainer.train()

    assert list(events.generated) == [1, 4, 7]
    loaders = list(events.generated.values())
    assert events.trained_on == [loaders[0]] * 3 + [loaders[1]] * 3 + [loaders[2]]


def test_a_first_loader_is_used_until_the_first_regeneration(
    make_trainer: Callable[..., Trainer],
    record_training: Callable[..., TrainingEvents],
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> None:
    trainer = make_trainer(N_EPOCHS=7, REGEN_EVERY=3)
    events = record_training(trainer)
    first = synthetic_loader(10, 4, shuffle=True)

    trainer.train(first)

    assert list(events.generated) == [4, 7]
    loaders = list(events.generated.values())
    assert events.trained_on == [first] * 3 + [loaders[0]] * 3 + [loaders[1]]


@pytest.mark.parametrize(
    ("n_epochs", "validate_every", "validated"),
    [
        (7, 3, [1, 3, 6, 7]),
        (6, 3, [1, 3, 6]),
        (1, 5, [1]),
        (4, 1, [1, 2, 3, 4]),
    ],
)
def test_the_first_epoch_every_validate_every_epochs_and_the_last_are_validated(
    n_epochs: int,
    validate_every: int,
    validated: list[int],
    make_trainer: Callable[..., Trainer],
    record_training: Callable[..., TrainingEvents],
) -> None:
    trainer = make_trainer(N_EPOCHS=n_epochs, VALIDATE_EVERY=validate_every)
    events = record_training(trainer)

    trainer.train()

    assert list(events.weights) == validated


def test_a_checkpoint_is_saved_only_for_a_lower_validation_loss(
    make_trainer: Callable[..., Trainer],
    record_training: Callable[..., TrainingEvents],
) -> None:
    trainer = make_trainer(N_EPOCHS=5)
    events = record_training(trainer, val_losses=[3.0, 1.0, 1.0, 0.5, 2.0])

    trainer.train()

    assert events.saved == [1, 2, 4]


def test_the_checkpoint_describes_and_restores_the_trained_model(
    config: dict[str, Any],
    make_trainer: Callable[..., Trainer],
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> None:
    trainer = make_trainer()

    trainer.train(synthetic_loader(10, 4))

    restored, checkpoint = load_model(config, "cpu")
    assert checkpoint["NTR"] == 5
    assert checkpoint["model_config"] == config["model"]
    assert checkpoint["data_config"] == config["data"]
    assert checkpoint["rf_sha256"] == rf_sha256(config["data"])
    x = synthetic_loader(6, 6).tensors[0]
    with torch.no_grad():
        for output, expected in zip(restored(x), trainer.model(x), strict=True):
            torch.testing.assert_close(output, expected)


def test_write_checkpoint_writes_the_keys_of_the_trainer(
    config: dict[str, Any],
    make_trainer: Callable[..., Trainer],
    synthetic_loader: Callable[..., TensorBatchLoader],
    write_checkpoint: Callable[..., None],
    tmp_path: Path,
) -> None:
    # The tests of the export read the checkpoints of write_checkpoint, so these
    # must have the keys of the checkpoints of the training
    trainer = make_trainer()
    trainer.train(synthetic_loader(10, 4))
    path = tmp_path / "written.pt"

    write_checkpoint(path, trainer.model, config["model"], config["data"])

    assert torch.load(path).keys() == load_checkpoint(config, "cpu").keys()


def test_the_trained_model_has_the_weights_of_the_best_epoch(
    make_trainer: Callable[..., Trainer],
    record_training: Callable[..., TrainingEvents],
) -> None:
    trainer = make_trainer(N_EPOCHS=3)
    events = record_training(trainer, val_losses=[2.0, 1.0, 3.0])

    trainer.train()

    weights = trainer.model.state_dict()
    assert same_weights(weights, events.weights[2])
    assert not same_weights(weights, events.weights[3])


def test_the_plateau_scheduler_steps_after_each_validation_with_its_loss(
    make_trainer: Callable[..., Trainer],
    record_training: Callable[..., TrainingEvents],
) -> None:
    trainer = make_trainer(N_EPOCHS=7, VALIDATE_EVERY=3, LR_SCHEDULER="plateau")
    val_losses = [4.0, 3.0, 2.0, 1.0]
    events = record_training(trainer, val_losses=val_losses)

    trainer.train()

    assert events.scheduler_steps == [(loss,) for loss in val_losses]


def test_the_cosine_scheduler_anneals_the_learning_rate_to_0_over_the_epochs(
    make_trainer: Callable[..., Trainer],
    record_training: Callable[..., TrainingEvents],
) -> None:
    trainer = make_trainer(N_EPOCHS=7, VALIDATE_EVERY=3, LR_SCHEDULER="cosine")
    events = record_training(trainer)

    trainer.train()

    rates = events.learning_rates
    assert len(rates) == 7
    assert all(later < earlier for earlier, later in pairwise(rates))
    assert rates[-1] == pytest.approx(0.0, abs=1e-12)


def test_the_epoch_loss_is_the_mean_over_the_samples(
    make_trainer: Callable[..., Trainer],
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> None:
    # Without steps, the epoch gives the loss of one model on all batches
    train_loader = synthetic_loader(10, 4)  # batches of 4, 4 and 2 samples
    val_loader = TensorBatchLoader(train_loader.tensors, 10, shuffle=False)
    trainer = make_trainer(val_loader, LEARNING_RATE=0.0)

    total, signal, dT1, dT2 = trainer.train_epoch(train_loader)

    assert total == pytest.approx(trainer.validate(), rel=1e-6)
    assert total == pytest.approx((signal + dT1 + dT2) / 3, rel=1e-6)


def test_the_validation_loss_does_not_depend_on_the_batch_size(
    make_trainer: Callable[..., Trainer],
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> None:
    tensors = synthetic_loader(10, 10).tensors

    losses = [
        make_trainer(TensorBatchLoader(tensors, batch_size, shuffle=False)).validate()
        for batch_size in (1, 3, 10)
    ]

    assert losses == pytest.approx([losses[2]] * 3, rel=1e-6)


def test_a_train_step_gives_the_losses_of_the_configured_function_before_the_step(
    make_trainer: Callable[..., Trainer],
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> None:
    # Not the default function of Loss: a trainer that ignores the
    # configuration gives other losses
    trainer = make_trainer(LOSS_FUNCTION="relative_l2")
    x, signal, dT1, dT2 = synthetic_loader(4, 4).tensors
    before = copy_weights(trainer.model)
    with torch.no_grad():
        expected = Loss("relative_l2")(*trainer.model(x), signal, dT1, dT2)

    losses = trainer._train_step(x, signal, dT1, dT2)

    torch.testing.assert_close(losses, torch.stack(expected))
    assert not same_weights(trainer.model.state_dict(), before)


def test_a_compiled_train_step_is_not_used_on_cpu(
    make_trainer: Callable[..., Trainer],
    synthetic_loader: Callable[..., TensorBatchLoader],
) -> None:
    # config/training.yaml asks for it, and without CUDA the training runs on
    # CPU. The compiled step needs CUDA graphs.
    trainer = make_trainer(COMPILE_TRAIN_STEP=True)
    before = copy_weights(trainer.model)

    trainer.train(synthetic_loader(10, 4))

    assert not same_weights(trainer.model.state_dict(), before)
