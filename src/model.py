import torch
import torch.nn as nn


class SubNetwork(nn.Module):
    """
    Fully connected neural network used to predict the
    low-rank representation of one output quantity.

    Architecture:
        Linear -> ReLU repeated -> Linear

    The network maps tissue parameters (T1, T2, B1)
    to a compressed representation.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_size: int,
        num_layers: int,
    ):
        super().__init__()

        # First layer: input parameters -> hidden representation
        layers = [
            nn.Linear(input_dim, hidden_size),
            nn.ReLU()
        ]

        # Hidden layers
        for _ in range(num_layers - 2):
            layers.extend([
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU()
            ])

        # Final layer: hidden representation -> compressed signal
        layers.append(nn.Linear(hidden_size, output_dim))

        self.net = nn.Sequential(*layers)


    def forward(self, x):
        """Compute compressed representation."""
        return self.net(x)



class FCModel(nn.Module):
    """
    Fully connected surrogate model.

    The model consists of three independent sub-networks:
        - Signal
        - Derivative with respect to T1
        - Derivative with respect to T2

    Each sub-network predicts a low-rank representation,
    which is decoded to the full signal length using a
    learned linear projection matrix U.

    The output dimension is 2*NTR because complex-valued
    signals are represented using separate real and imaginary
    components.
    """

    def __init__(self, NTR, config):
        super().__init__()

        input_dim = config["input_dim"]

        # Learned low-rank decoding matrices
        self.U_signal = nn.Linear(
            config["NB_signal"],
            2 * NTR,
            bias=False
        )

        self.U_dT1 = nn.Linear(
            config["NB_dT1"],
            2 * NTR,
            bias=False
        )

        self.U_dT2 = nn.Linear(
            config["NB_dT2"],
            2 * NTR,
            bias=False
        )

        # Sub-network for the signal
        self.subnet_signal = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_signal"],
            hidden_size=config["hidden_size_signal"],
            num_layers=config["num_layers_signal"]
        )   

        # Sub-network for the T1 derivative
        self.subnet_dT1 = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_dT1"],
            hidden_size=config["hidden_size_dT1"],
            num_layers=config["num_layers_dT1"]
        )

        # Sub-network for the T2 derivative
        self.subnet_dT2 = SubNetwork(
            input_dim=input_dim,
            output_dim=config["NB_dT2"],
            hidden_size=config["hidden_size_dT2"],
            num_layers=config["num_layers_dT2"]
        )


    def forward(self, x):
        """
        Predict signal and parameter derivatives.

        Input:
            x: tissue parameters [T1, T2, B1]

        Returns:
            m       : predicted signal
            dm_dT1  : derivative with respect to T1
            dm_dT2  : derivative with respect to T2
        """

        # Compute low-rank representations
        signal = self.subnet_signal(x)
        dT1 = self.subnet_dT1(x)
        dT2 = self.subnet_dT2(x)

        # Decode low-rank representations to full signal length
        m = self.U_signal(signal)
        dm_dT1 = self.U_dT1(dT1)
        dm_dT2 = self.U_dT2(dT2)

        return m, dm_dT1, dm_dT2