from typing import Any
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
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
        val_loader: DataLoader[tuple[torch.Tensor, ...]],
        config: dict[str, Any],
    ) -> None:

        # Set device and move model to GPU/CPU
        self.device = get_device(config)
        self.model = model.to(self.device)

        self.config = config

        # Optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        # Learning rate scheduler
        self.scheduler: LRScheduler

        if config["training"]["LR_SCHEDULER"] == "plateau":
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=100, factor=0.5, min_lr=1e-5
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
        self.criterion = Loss()

        # Validation data remains fixed during training
        self.val_loader = val_loader

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

        for epoch in range(1, self.config["training"]["N_EPOCHS"] + 1):
            # Generate new Bloch simulation data periodically
            if epoch == 1 or (epoch - 1) % self.config["training"]["REGEN_EVERY"] == 0:
                train_loader = generate_train_loader(self.config)

            # Training and validation
            train_loss, loss_signal, loss_dT1, loss_dT2 = self.train_epoch(train_loader)  # ty: ignore
            val_loss = self.validate()

            # Update learning rate
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.scheduler.step(val_loss)
            else:
                self.scheduler.step()

            # Save best model checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(
                    {
                        "model_state": self.model.state_dict(),
                        "epoch": epoch,
                        "NTR": self.model.U_signal.out_features // 2,
                    },
                    pt_file_path,
                )

            # Print training progress
            if epoch % 50 == 0 or epoch == 1:
                print(
                    f"Epoch {epoch:4d}/{self.config['training']['N_EPOCHS']} | "
                    f"train loss: {train_loss:.4e} | "
                    f"val loss: {val_loss:.4e} | "
                    f"lr: {self.optimizer.param_groups[0]['lr']:.2e}"
                )

            # Log metrics to TensorBoard
            self.writer.add_scalar("Loss/Train", train_loss, epoch)
            self.writer.add_scalar("Loss/signal", loss_signal.item(), epoch)
            self.writer.add_scalar("Loss/dT1", loss_dT1.item(), epoch)
            self.writer.add_scalar("Loss/dT2", loss_dT2.item(), epoch)
            self.writer.add_scalar("Loss/Validation", val_loss, epoch)
            self.writer.add_scalar("LR", self.optimizer.param_groups[0]["lr"], epoch)

        print(f"\nBest val loss: {best_val_loss:.4e}  (saved to {pt_file_path})")

        # Restore best model after training
        checkpoint = torch.load(pt_file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])

        print(f"Loaded best model from epoch {checkpoint['epoch']}")

        self.writer.flush()

    def train_epoch(
        self,
        train_loader: DataLoader[tuple[torch.Tensor, ...]],
    ) -> tuple[float, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform one epoch of training.

        Returns:
            Average training loss and individual loss components.
        """
        self.model.train()

        train_loss = 0.0

        for x, sig, dT1, dT2 in train_loader:
            # Move data to GPU/CPU
            x, sig, dT1, dT2 = (t.to(self.device) for t in (x, sig, dT1, dT2))

            self.optimizer.zero_grad()

            # Forward pass
            p_sig, p_dT1, p_dT2 = self.model(x)

            # Compute loss
            loss, loss_signal, loss_dT1, loss_dT2 = self.criterion(
                p_sig, p_dT1, p_dT2, sig, dT1, dT2
            )

            # Backpropagation
            loss.backward()
            self.optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train_loader)

        return train_loss, loss_signal, loss_dT1, loss_dT2  # ty: ignore

    def validate(self) -> float:
        """Evaluate the model on the validation dataset."""
        self.model.eval()

        val_loss = 0.0

        with torch.no_grad():
            for x, sig, dT1, dT2 in self.val_loader:
                x, sig, dT1, dT2 = (t.to(self.device) for t in (x, sig, dT1, dT2))

                p_sig, p_dT1, p_dT2 = self.model(x)

                loss, _, _, _ = self.criterion(p_sig, p_dT1, p_dT2, sig, dT1, dT2)

                val_loss += loss.item()

        val_loss /= len(self.val_loader)

        return val_loss
