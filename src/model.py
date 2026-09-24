import math
from typing import Any
import torch
import torch.nn as nn
from torch.nn import functional


def make_activation(name: str) -> nn.Module:
    """Create a hidden-layer activation by name."""
    if name == "relu":
        return nn.ReLU()
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "tanh":
        return nn.Tanh()
    if name == "softplus":
        return nn.Softplus()
    raise ValueError(
        "model.activation must be 'relu', 'silu', 'gelu', 'tanh', or "
        f"'softplus', got {name!r}"
    )


def activation_derivative(x: torch.Tensor, name: str) -> torch.Tensor:
    """Evaluate the exact first derivative of a supported activation."""
    if name == "relu":
        return (x > 0).to(x.dtype)
    if name == "silu":
        sigmoid = torch.sigmoid(x)
        return sigmoid * (1.0 + x * (1.0 - sigmoid))
    if name == "gelu":
        normal_cdf = 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))
        normal_pdf = torch.exp(-0.5 * x.square()) / math.sqrt(2.0 * math.pi)
        return normal_cdf + x * normal_pdf
    if name == "tanh":
        return 1.0 - torch.tanh(x).square()
    if name == "softplus":
        return torch.sigmoid(x)
    raise ValueError(f"Unsupported activation derivative: {name!r}")


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
        self.activation = activation

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

    def forward_with_tangents(
        self,
        x: torch.Tensor,
        tangent_T1: torch.Tensor,
        tangent_T2: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate two parameter tangents with the network value.

        This is explicit forward-mode differentiation. Keeping it in ordinary
        tensor operations makes the derivative path trainable and exportable.
        """
        value = x
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                value = layer(value)
                tangent_T1 = functional.linear(tangent_T1, layer.weight)
                tangent_T2 = functional.linear(tangent_T2, layer.weight)
            else:
                derivative = activation_derivative(value, self.activation)
                value = layer(value)
                tangent_T1 = tangent_T1 * derivative
                tangent_T2 = tangent_T2 * derivative

        return value, tangent_T1, tangent_T2


class InputTransform(nn.Module):
    """Map physical tissue parameters to the network's internal coordinates.

    Converts the public model input ``[T1, T2, B1]`` to the
    ``[log(T1), log(T2), B1]`` coordinates the sub-networks operate in.
    The transform has no parameters, so it adds nothing to checkpoints.
    """

    def __init__(
        self,
        input_dim: int,
        normalize: bool,
        data_config: dict[str, Any] | None,
    ) -> None:
        super().__init__()
        if normalize:
            if data_config is None:
                raise ValueError("data_config is required when normalizing inputs")
            lower = torch.tensor(
                [
                    math.log(data_config["T1_lower"]),
                    math.log(data_config["T2_lower"]),
                    data_config["B1_lower"],
                ]
            )
            upper = torch.tensor(
                [
                    math.log(data_config["T1_upper"]),
                    math.log(data_config["T2_upper"]),
                    data_config["B1_upper"],
                ]
            )
            offset = 0.5 * (lower + upper)
            scale = 2.0 / (upper - lower)
        else:
            offset = torch.zeros(input_dim)
            scale = torch.ones(input_dim)

        # Derived configuration follows the model's device but does not need
        # to make checkpoints larger or incompatible.
        self.offset: torch.Tensor
        self.scale: torch.Tensor
        self.register_buffer("offset", offset, persistent=False)
        self.register_buffer("scale", scale, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Convert physical parameters to normalized log-parameter space."""
        logged = torch.cat((torch.log(x[..., :2]), x[..., 2:]), dim=-1)
        return (logged - self.offset) * self.scale

    def forward_with_tangents(
        self,
        x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Transform inputs and return tangents with respect to physical T1/T2."""
        value = self(x)
        zeros = torch.zeros_like(x[..., :1])
        tangent_T1 = torch.cat((self.scale[0] / x[..., 0:1], zeros, zeros), dim=-1)
        tangent_T2 = torch.cat((zeros, self.scale[1] / x[..., 1:2], zeros), dim=-1)
        return value, tangent_T1, tangent_T2


class FCModel(nn.Module):
    """Fully connected surrogate model.

    In ``separate`` mode, three independent sub-networks predict the signal and
    its T1/T2 derivatives. In ``forward`` mode, one smooth signal network is
    differentiated by forward sensitivity and all three outputs therefore use
    exactly the same decoder basis.

    Each sub-network predicts a low-rank representation,
    which is decoded to the full signal length using a
    learned linear projection matrix U.

    Callers provide physical tissue parameters ``[T1, T2, B1]``. The model
    converts T1 and T2 to logarithmic coordinates before evaluating the
    sub-networks.

    Complex-valued outputs have shape ``(batch, NTR, 2)``, with real and
    imaginary values paired along the final axis.
    """

    def __init__(
        self,
        NTR: int,
        config: dict[str, Any],
        data_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()

        input_dim = config["input_dim"]
        activation = config.get("activation", "relu")
        self.derivative_mode = config.get("derivative_mode", "separate")

        if self.derivative_mode not in {"separate", "forward"}:
            raise ValueError(
                "model.derivative_mode must be 'separate' or 'forward', got "
                f"{self.derivative_mode!r}"
            )
        if self.derivative_mode == "forward" and activation == "relu":
            raise ValueError(
                "model.activation must be smooth when derivative_mode='forward'"
            )

        self.NTR = NTR

        # Physical [T1, T2, B1] -> internal [log(T1), log(T2), B1] coordinates
        self.input_transform = InputTransform(
            input_dim,
            config.get("normalize_inputs", False),
            data_config,
        )

        # Learned low-rank decoders with paired real-imaginary outputs
        self.U_signal = nn.Sequential(
            nn.Linear(config["NB_signal"], 2 * NTR, bias=False),
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

        self.U_dT1: nn.Sequential | None = None
        self.U_dT2: nn.Sequential | None = None
        self.subnet_dT1: SubNetwork | None = None
        self.subnet_dT2: SubNetwork | None = None

        if self.derivative_mode == "separate":
            self.U_dT1 = nn.Sequential(
                nn.Linear(config["NB_dT1"], 2 * NTR, bias=False),
                nn.Unflatten(-1, (NTR, 2)),
            )
            self.U_dT2 = nn.Sequential(
                nn.Linear(config["NB_dT2"], 2 * NTR, bias=False),
                nn.Unflatten(-1, (NTR, 2)),
            )
            self.subnet_dT1 = SubNetwork(
                input_dim=input_dim,
                output_dim=config["NB_dT1"],
                hidden_size=config["hidden_size_dT1"],
                num_layers=config["num_layers_dT1"],
                activation=activation,
            )
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
        if self.derivative_mode == "forward":
            network_input, tangent_T1, tangent_T2 = (
                self.input_transform.forward_with_tangents(x)
            )
            signal, dT1, dT2 = self.subnet_signal.forward_with_tangents(
                network_input, tangent_T1, tangent_T2
            )
            # A single rank-R decoder is shared by the signal and both exact
            # coefficient derivatives. Total deployed basis rank is therefore R.
            m = self.U_signal(signal)
            dm_dT1 = self.U_signal(dT1)
            dm_dT2 = self.U_signal(dT2)
        else:
            network_input = self.input_transform(x)
            if (
                self.subnet_dT1 is None
                or self.subnet_dT2 is None
                or self.U_dT1 is None
                or self.U_dT2 is None
            ):
                raise RuntimeError("Separate derivative heads were not initialized")
            signal = self.subnet_signal(network_input)
            dT1 = self.subnet_dT1(network_input)
            dT2 = self.subnet_dT2(network_input)
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
