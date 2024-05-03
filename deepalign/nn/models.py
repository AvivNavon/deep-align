import logging
from typing import Tuple

import torch
from torch import nn

from experiments.utils import count_parameters
from nn.layers import (BN, CannibalLayer, DownSampleCannibalLayer, Dropout,
                       InvariantLayer, LeakyReLU, ReLU)
from nn.layers.layers import NormalizeAndScale


class SwapAxis(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 1)


class DWSModel(nn.Module):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        input_features,
        hidden_dim,
        n_hidden=2,
        output_features=None,
        reduction="max",
        bias=True,
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        input_dim_downsample=None,
        dropout_rate=0.0,
        add_skip=False,
        add_layer_skip=False,
        init_scale=1e-4,
        init_off_diag_scale_penalty=1.0,
        bn=False,
        diagonal=False,
        hnp_setup=True,
    ):
        super().__init__()
        print(
            "the current implementation for diverse architectures assumes the "
            "initilization shapes are for networks with M>=4 layers."
        )
        assert (
            len(weight_shapes) >= 4
        ), "the current implementation for diverse architectures assumes the initilization shapes are for networks with M>=4 layers."

        if not hnp_setup:
            assert input_dim_downsample is None, "input_dim_downsample must be None for np_setup is False"

        self.hnp_setup = hnp_setup
        self.input_features = input_features
        self.input_dim_downsample = input_dim_downsample
        if output_features is None:
            output_features = hidden_dim

        self.add_skip = add_skip
        if self.add_skip:
            self.skip = nn.Linear(input_features, output_features, bias=bias)
            with torch.no_grad():
                torch.nn.init.constant_(
                    self.skip.weight, 1.0 / self.skip.weight.numel()
                )
                torch.nn.init.constant_(self.skip.bias, 0.0)

        if input_dim_downsample is None:
            layers = [
                CannibalLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                    hnp_setup=hnp_setup,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [
                        ReLU(),
                        Dropout(dropout_rate),
                        CannibalLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim
                            if i != (n_hidden - 1)
                            else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                            hnp_setup=hnp_setup,
                        ),
                    ]
                )
        else:
            layers = [
                DownSampleCannibalLayer(
                    weight_shapes=weight_shapes,
                    bias_shapes=bias_shapes,
                    in_features=input_features,
                    out_features=hidden_dim,
                    reduction=reduction,
                    bias=bias,
                    n_fc_layers=n_fc_layers,
                    num_heads=num_heads,
                    set_layer=set_layer,
                    downsample_dim=input_dim_downsample,
                    add_skip=add_layer_skip,
                    init_scale=init_scale,
                    init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                    diagonal=diagonal,
                ),
            ]
            for i in range(n_hidden):
                if bn:
                    layers.append(BN(hidden_dim, len(weight_shapes), len(bias_shapes)))

                layers.extend(
                    [
                        ReLU(),
                        Dropout(dropout_rate),
                        DownSampleCannibalLayer(
                            weight_shapes=weight_shapes,
                            bias_shapes=bias_shapes,
                            in_features=hidden_dim,
                            out_features=hidden_dim
                            if i != (n_hidden - 1)
                            else output_features,
                            reduction=reduction,
                            bias=bias,
                            n_fc_layers=n_fc_layers,
                            num_heads=num_heads if i != (n_hidden - 1) else 1,
                            set_layer=set_layer,
                            downsample_dim=input_dim_downsample,
                            add_skip=add_layer_skip,
                            init_scale=init_scale,
                            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
                            diagonal=diagonal,
                        ),
                    ]
                )
        self.layers = nn.Sequential(*layers)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        out = self.layers(x)
        if self.add_skip:
            skip_out = tuple(self.skip(w) for w in x[0]), tuple(
                self.skip(b) for b in x[1]
            )
            weight_out = tuple(ws + w for w, ws in zip(out[0], skip_out[0]))
            bias_out = tuple(bs + b for b, bs in zip(out[1], skip_out[1]))
            out = weight_out, bias_out
        return out


class DWSMatching(DWSModel):
    def __init__(
        self,
        weight_shapes: Tuple[Tuple[int, int], ...],
        bias_shapes: Tuple[
            Tuple[int,],
            ...,
        ],
        input_features,
        hidden_dim,
        n_hidden=2,
        output_features=None,
        reduction="max",
        bias=True,
        n_fc_layers=1,
        num_heads=8,
        set_layer="sab",
        input_dim_downsample=None,
        dropout_rate=0.0,
        add_skip=False,
        add_layer_skip=False,
        init_scale=1e-4,
        init_off_diag_scale_penalty=1.0,
        bn=False,
        diagonal=False,
        hnp_setup=True,
        normalize_scale=True,
    ):
        super().__init__(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=input_features,
            hidden_dim=hidden_dim,
            n_hidden=n_hidden,
            reduction=reduction,
            bias=bias,
            output_features=output_features,
            n_fc_layers=n_fc_layers,
            num_heads=num_heads,
            set_layer=set_layer,
            dropout_rate=dropout_rate,
            input_dim_downsample=input_dim_downsample,
            init_scale=init_scale,
            init_off_diag_scale_penalty=init_off_diag_scale_penalty,
            bn=bn,
            add_skip=add_skip,
            add_layer_skip=add_layer_skip,
            diagonal=diagonal,
            hnp_setup=hnp_setup,
        )

        self.normalize_scale = normalize_scale
        self.normalize_layer = NormalizeAndScale(normalize_scale)

    def forward(self, x: Tuple[Tuple[torch.tensor], Tuple[torch.tensor]]):
        x = super().forward(x)
        _, features = x
        norm_features = []
        for k in range(len(features) - 1):  # no permutations on final bias
            norm_features.append(self.normalize_layer(features[k]))
        return tuple(norm_features)


if __name__ == "__main__":
    weights = (
        torch.randn(4, 3, 16, 1),
        torch.randn(4, 16, 32, 1),
        torch.randn(4, 32, 32, 1),
        torch.randn(4, 32, 64, 1),
        torch.randn(4, 64, 64, 1),
        torch.randn(4, 64, 128, 1),
        torch.randn(4, 128, 128, 1),
        torch.randn(4, 128, 10, 1),
    )
    biases = (
        torch.randn(4, 16, 1),
        torch.randn(4, 32, 1),
        torch.randn(4, 32, 1),
        torch.randn(4, 64, 1),
        torch.randn(4, 64, 1),
        torch.randn(4, 128, 1),
        torch.randn(4, 128, 1),
        torch.randn(4, 10, 1),
    )
    in_dim = sum([w[0, :].numel() for w in weights]) + sum(
        [w[0, :].numel() for w in biases]
    )
    weight_shapes = tuple(w.shape[1:3] for w in weights)
    bias_shapes = tuple(b.shape[1:2] for b in biases)
    n_params = sum([i.numel() for i in weight_shapes + bias_shapes])

    model = DWSMatching(
        weight_shapes=weight_shapes,
        bias_shapes=bias_shapes,
        input_features=16,
        hidden_dim=64,
        n_hidden=4,
        input_dim_downsample=None,
        hnp_setup=True,
        diagonal=True
    )
    print(count_parameters(model))
