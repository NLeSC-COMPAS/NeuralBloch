from typing import Any
import torch
import torch.nn as nn


class SubNetwork(nn.Module):
    """Fully connected network predicting a low-rank representation.

    Maps tissue parameters (T1, T2, B1) to the compressed representation of
    one output quantity.

    Architecture:
        Linear -> ReLU repeated -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
    ) -> None:
        super().__init__()

        # First layer: input parameters -> hidden representation
        layers = [nn.Linear(input_dim, hidden_size), nn.ReLU()]

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend([nn.Linear(hidden_size, hidden_size), nn.ReLU()])

        # Final layer: hidden representation -> compressed signal
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute compressed representation."""
        return self.net(x)


class FCModel(nn.Module):
    """Fully connected surrogate model.

    The model consists of three independent sub-networks:
        - Signal
        - Derivative with respect to T1
        - Derivative with respect to T2

    Each sub-network predicts a low-rank representation,
    which is decoded to the full signal length using a
    learned linear projection matrix U.

    Callers provide physical tissue parameters ``[T1, T2, B1]``. The model
    converts T1 and T2 to logarithmic coordinates before evaluating the
    sub-networks.

    Complex-valued outputs have shape ``(batch, NTR, 2)``, with real and
    imaginary values paired along the final axis.
    """

    def __init__(self, NTR: int, config: dict[str, Any]) -> None:
        super().__init__()

        input_dim = config["input_dim"]

        self.NTR = NTR

        # Learned low-rank decoding matrices
        self.U_signal = nn.Linear(config["NB_signal"], 2 * NTR, bias=False)

        self.U_dT1 = nn.Linear(config["NB_dT1"], 2 * NTR, bias=False)

        self.U_dT2 = nn.Linear(config["NB_dT2"], 2 * NTR, bias=False)

        # Sub-network for the signal
        self.subnet_signal = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_signal"],
            hidden_size=config["hidden_size_signal"],
            num_layers=config["num_layers_signal"],
        )

        # Sub-network for the T1 derivative
        self.subnet_dT1 = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_dT1"],
            hidden_size=config["hidden_size_dT1"],
            num_layers=config["num_layers_dT1"],
        )

        # Sub-network for the T2 derivative
        self.subnet_dT2 = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_dT2"],
            hidden_size=config["hidden_size_dT2"],
            num_layers=config["num_layers_dT2"],
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict signal and parameter derivatives.

        Input:
            x: Physical tissue parameters ``[T1, T2, B1]``. T1 and T2 must
                be finite and strictly positive.

        Returns:
            m: Predicted signal with shape ``(batch, NTR, 2)``.
            dm_dT1: T1 derivative with shape ``(batch, NTR, 2)``.
            dm_dT2: T2 derivative with shape ``(batch, NTR, 2)``.
        """
        # Use logarithmic T1 and T2 coordinates internally while keeping the
        # public model input in physical units.
        x = torch.cat((torch.log(x[..., :2]), x[..., 2:]), dim=-1)

        # Compute low-rank representations
        signal = self.subnet_signal(x)
        dT1 = self.subnet_dT1(x)
        dT2 = self.subnet_dT2(x)

        # Decode low-rank representations to full signal length
        m = self.U_signal(signal).unflatten(-1, (self.NTR, 2))
        dm_dT1 = self.U_dT1(dT1).unflatten(-1, (self.NTR, 2))
        dm_dT2 = self.U_dT2(dT2).unflatten(-1, (self.NTR, 2))

        return m, dm_dT1, dm_dT2
