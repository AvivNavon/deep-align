import math
from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from rff.layers import GaussianEncoding, PositionalEncoding
from torch import nn


class Sine(nn.Module):
    def __init__(self, w0=1.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class Siren(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        w0=30.0,
        c=6.0,
        is_first=False,
        use_bias=True,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.is_first = is_first

        weight = torch.zeros(dim_out, dim_in)
        bias = torch.zeros(dim_out) if use_bias else None
        self.init_(weight, bias, c=c, w0=w0)

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias) if use_bias else None
        self.activation = Sine(w0) if activation is None else activation

    def init_(self, weight: torch.Tensor, bias: torch.Tensor, c: float, w0: float):
        dim = self.dim_in

        w_std = (1 / dim) if self.is_first else (math.sqrt(c / dim) / w0)
        weight.uniform_(-w_std, w_std)

        if bias is not None:
            # bias.uniform_(-w_std, w_std)
            bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        out = self.activation(out)
        return out


class INR(nn.Module):
    def __init__(
        self,
        in_dim: int = 2,
        n_layers: int = 3,
        up_scale: int = 4,
        out_channels: int = 1,
        pe_features: Optional[int] = None,
        fix_pe=True,
    ):
        super().__init__()
        hidden_dim = in_dim * up_scale

        if pe_features is not None:
            if fix_pe:
                self.layers = [PositionalEncoding(sigma=10, m=pe_features)]
                encoded_dim = in_dim * pe_features * 2
            else:
                self.layers = [
                    GaussianEncoding(
                        sigma=10, input_size=in_dim, encoded_size=pe_features
                    )
                ]
                encoded_dim = pe_features * 2
            self.layers.append(Siren(dim_in=encoded_dim, dim_out=hidden_dim))
        else:
            self.layers = [Siren(dim_in=in_dim, dim_out=hidden_dim)]
        for i in range(n_layers - 2):
            self.layers.append(Siren(hidden_dim, hidden_dim))
        self.layers.append(nn.Linear(hidden_dim, out_channels))
        self.seq = nn.Sequential(*self.layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.seq(x) + 0.5


class FunctionalSiren(nn.Module):
    def __init__(
        self,
        w0=30.0,
        c=6.0,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.c = c
        self.activation = Sine(w0) if activation is None else activation

    def forward(
        self, x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor
    ) -> torch.Tensor:
        out = F.linear(x, weight, bias)
        out = self.activation(out)
        return out


class FunctionalINR(nn.Module):
    def __init__(
        self, in_dim: int = 2, pe_features: Optional[int] = None, fix_pe: bool = True
    ):
        super().__init__()

        if pe_features is not None:
            if fix_pe:
                self.pe = [PositionalEncoding(sigma=10, m=pe_features)]
            else:
                self.pe = [
                    GaussianEncoding(
                        sigma=10, input_size=in_dim, encoded_size=pe_features
                    )
                ]
        else:
            self.pe = None
        self.f_siren = FunctionalSiren()

    def forward(
        self, x: torch.Tensor, weights: torch.Tensor, biases: torch.Tensor
    ) -> torch.Tensor:
        if self.pe is not None:
            x = self.pe(x)
        for w, b in zip(weights[:-1], biases[:-1]):
            x = self.f_siren(x, w.squeeze(-1).transpose(1, 0), b.squeeze(-1))
        x = F.linear(x, weights[-1].squeeze(-1).transpose(1, 0), biases[-1].squeeze(-1))
        return x + 0.5


class RBFLayer(nn.Module):
    """Transforms incoming data using a given radial basis function.
    - Input: (1, N, in_features) where N is an arbitrary batch size
    - Output: (1, N, out_features) where N is an arbitrary batch size"""

    def __init__(self, in_features, out_features):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.centres = nn.Parameter(torch.Tensor(out_features, in_features))
        self.sigmas = nn.Parameter(torch.Tensor(out_features))
        self.reset_parameters()

        self.freq = nn.Parameter(np.pi * torch.ones((1, self.out_features)))

    def reset_parameters(self):
        nn.init.uniform_(self.centres, -1, 1)
        nn.init.constant_(self.sigmas, 10)

    def forward(self, input):
        input = input[0, ...]
        size = (input.size(0), self.out_features, self.in_features)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1) * self.sigmas.unsqueeze(0)
        return self.gaussian(distances).unsqueeze(0)

    def gaussian(self, alpha):
        phi = torch.exp(-1 * alpha.pow(2))
        return phi


class FunctionalINRForModelBatch(nn.Module):
    def __init__(
        self,
        w0=30.0,
        activation=None,
    ):
        super().__init__()
        self.w0 = w0
        self.activation = Sine(w0) if activation is None else activation

    def forward(self, x, weights_and_biases):
        # x is (n_pixels, in_features)
        # weights: (bs_models, di, d{i+1}, 1)
        # biases: (bs_models, d{i+1}, 1)
        weights, biases = weights_and_biases

        # (bs_models, n_pixels, features)
        x = (weights[0].squeeze(-1).permute(0, 2, 1) @ x.permute(1, 0)).permute(
            0, 2, 1
        ) + biases[0].permute(
            0, 2, 1
        )  # (bs_models, 1, out_features)

        for i, (w, b) in enumerate(zip(weights[1:], biases[1:])):
            # (bs_models, n_pixels, in_features)
            x = self.activation(x)
            # (bs_models, n_pixels, out_features)
            x = x.bmm(w.squeeze(-1)) + b.permute(
                0, 2, 1
            )  # bias is (n_pixels, 1, out_features)
        # (bs_models, n_pixels, out_channels)
        return x + 0.5


if __name__ == "__main__":
    from experiments.utils import make_coordinates

    coords = make_coordinates((28, 28), 1).squeeze(0)
    weights = (
        torch.randn(12, 2, 32, 1),
        torch.randn(12, 32, 32, 1),
        torch.randn(12, 32, 1, 1),
    )
    biases = (torch.randn(12, 32, 1), torch.randn(12, 32, 1), torch.randn(12, 1, 1))
    model = FunctionalINRForModelBatch(w0=30.0)
    out = model(coords, (weights, biases))
    print(out.shape)
