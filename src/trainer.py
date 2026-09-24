import math
import time
from collections.abc import Callable
from typing import Any
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.tensorboard import SummaryWriter
from data.data_loading import TensorBatchLoader
from data.data_loading import generate_train_loader
from src.loss import Loss
from src.model import FCModel
from src.utils import get_device


class Trainer:
    """Training loop for the surrogate model.

    Handles optimization, learning rate scheduling, validation, checkpointing,
    and TensorBoard logging.
    """

    def __init__(
        self,
        model: FCModel,
        val_loader: TensorBatchLoader,
        config: dict[str, Any],
    ) -> None:

        # Set device and move model to GPU/CPU
        self.device = get_device(config)
        self.model = model.to(self.device)

        self.config = config

        # Optimizer
        compile_train_step = config["training"].get("COMPILE_TRAIN_STEP", False)
        compile_train_step = compile_train_step and self.device.startswith("cuda")
        initial_learning_rate = float(config["training"].get("LEARNING_RATE", 1e-3))
        if compile_train_step:
            # A device-resident learning rate prevents torch.compile from
            # specializing and recompiling the Adam step after every scheduler
            # update. ``capturable`` is required for CUDA graph capture.
            learning_rate: float | torch.Tensor = torch.tensor(
                initial_learning_rate,
                device=self.device,
            )
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                capturable=True,
            )
        else:
            self.optimizer = optim.Adam(
                self.model.parameters(),
                lr=initial_learning_rate,
            )

        # Learning rate scheduler
        self.scheduler: LRScheduler

        validate_every = config["training"].get("VALIDATE_EVERY", 1)
        if validate_every <= 0:
            raise ValueError("training.VALIDATE_EVERY must be positive")

        if config["training"]["LR_SCHEDULER"] == "plateau":
            # ReduceLROnPlateau counts calls rather than epochs.  Keep its
            # original 100-epoch patience when validation is less frequent.
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                patience=math.ceil(100 / validate_every),
                factor=0.5,
                min_lr=1e-5,
            )
        elif config["training"]["LR_SCHEDULER"] == "cosine":
            self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=config["training"]["N_EPOCHS"]
            )
        else:
            raise ValueError(
                "training.LR_SCHEDULER must be 'plateau' or 'cosine', got "
                f"{config['training']['LR_SCHEDULER']!r}"
            )

        # Loss function
        self.criterion = Loss(config["training"]["LOSS_FUNCTION"])

        # Validation data remains fixed during training
        self.val_loader = val_loader

        self.train_step: Callable[
            [torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor
        ] = self._train_step
        if compile_train_step:
            # Compiling the optimizer step enables CUDA graphs and removes much
            # of the Python/kernel-launch overhead of this small network.  The
            # first full batch pays a one-time compilation cost.
            self.train_step = torch.compile(
                self._train_step,
                mode="reduce-overhead",
            )

        # TensorBoard logger
        self.writer = SummaryWriter(log_dir=config["experiment"]["LOG_FILE_PATH"])

    def train(self) -> None:
        """Main training loop.

        Periodically regenerates training data, trains the model,
        evaluates validation loss, updates the learning rate,
        and saves the best performing model.
        """
        best_val_loss = float("inf")

        pt_file_path = self.config["experiment"]["PT_FILE_PATH"]
        n_epochs = self.config["training"]["N_EPOCHS"]
        validate_every = self.config["training"].get("VALIDATE_EVERY", 1)
        training_started = time.perf_counter()
        train_loader: TensorBatchLoader | None = None

        for epoch in range(1, n_epochs + 1):
            generation_time = 0.0
            # Generate new Bloch simulation data periodically
            if epoch == 1 or (epoch - 1) % self.config["training"]["REGEN_EVERY"] == 0:
                print(f"Generating training data for epoch {epoch} ...", flush=True)
                started = time.perf_counter()
                train_loader = generate_train_loader(self.config)
                generation_time = time.perf_counter() - started
                print(
                    f"Generated {train_loader.n_samples:,} samples in "
                    f"{generation_time:.2f} s",
                    flush=True,
                )

            # Training
            started = time.perf_counter()
            if train_loader is None:
                raise RuntimeError("Training data was not generated")
            train_loss, loss_signal, loss_dT1, loss_dT2 = self.train_epoch(train_loader)
            train_time = time.perf_counter() - started

            # Validation is intentionally much less frequent than optimization.
            # Always validate the first and final epoch so a checkpoint exists
            # and the reported best model includes the end of training.
            should_validate = (
                epoch == 1 or epoch % validate_every == 0 or epoch == n_epochs
            )
            val_loss: float | None = None
            validation_time = 0.0
            if should_validate:
                started = time.perf_counter()
                val_loss = self.validate()
                validation_time = time.perf_counter() - started

            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                if val_loss is not None:
                    self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Save best model checkpoint only when validation was performed.
            if val_loss is not None and val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "epoch": epoch,
                        "NTR": self.model.NTR,
                        "model_config": dict(self.config["model"]),
                        "data_config": dict(self.config["data"]),
                    },
                    pt_file_path,
                )

            # Print training progress
            if epoch % 50 == 0 or epoch == 1 or epoch == n_epochs:
                val_text = "skipped" if val_loss is None else f"{val_loss:.4e}"
                print(
                    f"Epoch {epoch:4d}/{n_epochs} | "
                    f"train loss: {train_loss:.4e} | "
                    f"val loss: {val_text} | "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2e} | "
                    f"time: generate {generation_time:.2f}s, "
                    f"train {train_time:.2f}s, val {validation_time:.2f}s"
                )

            # Log metrics to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/signal", loss_signal, epoch)
            self.writer.add_scalar("Loss/dT1", loss_dT1, epoch)
            self.writer.add_scalar("Loss/dT2", loss_dT2, epoch)
            if val_loss is not None:
                self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)
            self.writer.add_scalar("Time/Generation", generation_time, epoch)
            self.writer.add_scalar("Time/Training", train_time, epoch)
            if should_validate:
                self.writer.add_scalar("Time/Validation", validation_time, epoch)

        elapsed = time.perf_counter() - training_started
        print(
            f"\nBest val loss: {best_val_loss:.4e}  (saved to {pt_file_path})"
            f"\nTotal training time: {elapsed:.2f} s"
        )

        # Restore best model after training
        checkpoint = torch.load(pt_file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

        print(f"Loaded best model from epoch {checkpoint['epoch']}")

        self.writer.close()

    def train_epoch(
        self,
        train_loader: TensorBatchLoader,
    ) -> tuple[float, float, float, float]:
        """Perform one epoch of training.

        Returns:
            Average training loss and individual loss components.
        """
        self.model.train()

        metric_sums = torch.zeros(4, device=self.device)
        n_samples = 0

        for x, sig, dT1, dT2 in train_loader:
            # A differently sized final batch would trigger another expensive
            # compilation and CUDA graph.  Run just that batch eagerly.
            step = (
                self.train_step
                if x.shape[0] == train_loader.batch_size
                else self._train_step
            )
            batch_metrics = step(x, sig, dT1, dT2)
            metric_sums += batch_metrics * x.shape[0]
            n_samples += x.shape[0]

        # Synchronize once per epoch, not once per mini-batch.  Calling .item()
        # inside the loop previously serialized every CUDA operation.
        metrics = (metric_sums / n_samples).tolist()
        return metrics[0], metrics[1], metrics[2], metrics[3]

    def _train_step(
        self,
        x: torch.Tensor,
        sig: torch.Tensor,
        dT1: torch.Tensor,
        dT2: torch.Tensor,
    ) -> torch.Tensor:
        """Optimize one batch and return its four detached loss metrics."""
        self.optimizer.zero_grad(set_to_none=True)

        p_sig, p_dT1, p_dT2 = self.model(x)
        losses = self.criterion(p_sig, p_dT1, p_dT2, sig, dT1, dT2)
        losses[0].backward()
        self.optimizer.step()

        return torch.stack(losses).detach()

    def validate(self) -> float:
        """Evaluate the model on the validation dataset."""
        self.model.eval()

        val_loss_sum = torch.zeros((), device=self.device)
        n_samples = 0

        with torch.inference_mode():
            for x, sig, dT1, dT2 in self.val_loader:
                p_sig, p_dT1, p_dT2 = self.model(x)

                loss, _, _, _ = self.criterion(p_sig, p_dT1, p_dT2, sig, dT1, dT2)

                val_loss_sum += loss * x.shape[0]
                n_samples += x.shape[0]

        return (val_loss_sum / n_samples).item()
