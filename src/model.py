from typing import Any
import torch
import torch.nn as nn


def make_activation(name: str) -> nn.Module:
    """Create a hidden-layer activation by name."""
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    raise ValueError(
        f"model.activation must be 'relu', 'silu', or 'gelu', got {name!r}"
    )


class SubNetwork(nn.Module):
    """Fully connected network predicting a low-rank representation.

    Maps tissue parameters (T1, T2, B1) to the compressed representation of
    one output quantity.

    Architecture:
        Linear -> configured activation repeated -> Linear
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
        activation: str,
    ) -> None:
        super().__init__()

        # First layer: input parameters -> hidden representation
        layers = [nn.Linear(input_dim, hidden_size), make_activation(activation)]

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend(
                [nn.Linear(hidden_size, hidden_size), make_activation(activation)]
            )

        # Final layer: hidden representation -> compressed signal
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute compressed representation."""
        return self.net(x)


class InputTransform(nn.Module):
    """Map physical tissue parameters to the network's internal coordinates.

    Converts the public model input ``[T1, T2, B1]`` to the
    ``[log(T1), log(T2), B1]`` coordinates the sub-networks operate in.
    The transform has no parameters, so it adds nothing to checkpoints.
    """

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert ``[T1, T2, B1]`` to ``[log(T1), log(T2), B1]``."""
        return torch.cat((torch.log(x[..., :2]), x[..., 2:]), dim=-1)


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
        activation = config.get("activation", "relu")

        self.NTR = NTR

        # Physical [T1, T2, B1] -> internal [log(T1), log(T2), B1] coordinates
        self.input_transform = InputTransform()

        # Learned low-rank decoders with paired real-imaginary outputs
        self.U_signal = nn.Sequential(
            nn.Linear(config["NB_signal"], 2 * NTR, bias=False),
            nn.Unflatten(-1, (NTR, 2)),
        )

        self.U_dT1 = nn.Sequential(
            nn.Linear(config["NB_dT1"], 2 * NTR, bias=False),
            nn.Unflatten(-1, (NTR, 2)),
        )

        self.U_dT2 = nn.Sequential(
            nn.Linear(config["NB_dT2"], 2 * NTR, bias=False),
            nn.Unflatten(-1, (NTR, 2)),
        )

        # Sub-network for the signal
        self.subnet_signal = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_signal"],
            hidden_size=config["hidden_size_signal"],
            num_layers=config["num_layers_signal"],
            activation=activation,
        )

        # Sub-network for the T1 derivative
        self.subnet_dT1 = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_dT1"],
            hidden_size=config["hidden_size_dT1"],
            num_layers=config["num_layers_dT1"],
            activation=activation,
        )

        # Sub-network for the T2 derivative
        self.subnet_dT2 = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_dT2"],
            hidden_size=config["hidden_size_dT2"],
            num_layers=config["num_layers_dT2"],
            activation=activation,
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
        x = self.input_transform(x)

        # Compute low-rank representations
        signal = self.subnet_signal(x)
        dT1 = self.subnet_dT1(x)
        dT2 = self.subnet_dT2(x)

        # Decode low-rank representations to full signal length
        m = self.U_signal(signal)
        dm_dT1 = self.U_dT1(dT1)
        dm_dT2 = self.U_dT2(dT2)

        return m, dm_dT1, dm_dT2

    def forward_signal(self, x: torch.Tensor) -> torch.Tensor:
        """Predict the signal without evaluating the derivative sub-networks.

        The cheaper path for callers that only need the signal: it shares
        ``input_transform`` and the signal head with ``forward``, so its
        result is identical to ``forward(x)[0]``.

        Input:
            x: Physical tissue parameters ``[T1, T2, B1]``. T1 and T2 must
                be finite and strictly positive.

        Returns:
            Predicted signal with shape ``(batch, NTR, 2)``.
        """
        return self.U_signal(self.subnet_signal(self.input_transform(x)))

    def forward_coefficients(self, x: torch.Tensor) -> torch.Tensor:
        """Predict signal coefficients, stopping before the signal decoder.

        This is ``forward_signal`` without the final projection through
        ``U_signal``. A low-rank reconstruction applies that basis itself, in
        image space, so it wants the coefficients rather than the decoded
        signal: ``forward_coefficients(x) @ U_signal[0].weight.T``, unflattened
        to ``(batch, NTR, 2)``, reproduces ``forward_signal(x)`` exactly.

        In ``forward`` mode, this basis is shared by the signal and both
        derivatives. In ``separate`` mode, it is the signal basis only.

        Input:
            x: Physical tissue parameters ``[T1, T2, B1]``. T1 and T2 must
                be finite and strictly positive.

        Returns:
            Coefficients with shape ``(batch, NB_signal)``.
        """
        return self.subnet_signal(self.input_transform(x))
