import torch as th
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple


class FFN(nn.Module):
    """
    Generic Feed Forward Network class
    """

    def __init__(
        self,
        in_dim: int,
        hidden_layers: List[int],
        out_dim: int,
        activation=nn.ReLU,
        dropout: float = 0.0,
    ):
        r"""
        ## Inputs

        :param in_dim: Input dimensions
        :param hidden_layers: List of hidden layer sizes
        :param out_dim: Output dimensions
        :param activation: nn Module for activation
        :param Dropout: rate of dropout
        ```math
        e=mc^2
        ```
        """
        super(FFN, self).__init__()
        self.layers = [nn.Linear(in_dim, hidden_layers[0]), activation()]
        for i in range(1, len(hidden_layers)):
            self.layers.extend(
                [
                    nn.Linear(hidden_layers[i - 1], hidden_layers[i]),
                    activation(),
                    nn.Dropout(dropout),
                ]
            )
        self.layers.append(nn.Linear(hidden_layers[-1], out_dim))
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inp):
        r"""
        ## Inputs

        :param inp: Input vectors shape: [batch, inp_dim]

        ----
        ## Outputs

        out: [batch, out_dim]
        """
        return self.layers(inp)


class GRUEncoder(nn.Module):
    """
    Encodes Sequences using GRU
    """

    def __init__(self, in_size: int, out_dim: int, bidirectional: bool = False):
        super(GRUEncoder, self).__init__()
        self.gru = nn.GRU(
            in_size, out_dim, batch_first=True, bidirectional=bidirectional
        )

    def forward(self, batch):
        r"""
        ## Inputs

        :param batch: Input vectors shape: [batch, seq_len, in_size]

        ----
        ## Outputs

        out: [batch, seq_len, out_dim]
        """
        out_seq, _ = self.gru(batch)
        return out_seq[:, -1, :]

class WeightEncoder(nn.Module):
    """
    Uses visit data to encode weights
    """
    def __init__(self, ) -> None:
        super(WeightEncoder, self).__init__()