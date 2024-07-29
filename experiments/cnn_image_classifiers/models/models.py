import math

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


class LargeCNNs(nn.Module):
    def __init__(self, n_layers=3, n_filters=16, stride=1, out_dim=10):
        super(LargeCNNs, self).__init__()
        init_pow = int(math.log2(n_filters))
        l = [nn.Conv2d(3, 2 ** init_pow, 3, stride, padding="same"), nn.ReLU()]
        for i in range(n_layers):
            l.extend(
                [nn.Conv2d(2 ** min([i + init_pow, 9]), 2 ** min([i + 1 + init_pow, 9]), 3, stride, padding="same"),
                 nn.ReLU(),
                 nn.Conv2d(2 ** min([i + 1 + init_pow, 9]), 2 ** min([i + 1 + init_pow, 9]), 3, stride, padding="same"),
                 nn.ReLU(),
                 nn.MaxPool2d(2) #  if i % 2 != 0 else nn.Identity()
                 ]
            )
        self.cnn = nn.Sequential(*l)
        with torch.no_grad():
            dum_inp = torch.rand(1, 3, 32, 32)
            lat_dim = self.cnn(dum_inp).flatten().shape[0]
        self.fc = nn.Linear(lat_dim, out_dim)

        # self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(mean=0.0, std=1.0)
                if m.bias is not None:
                    m.bias.data.normal_(mean=0.0, std=1.0)
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(mean=0.0, std=1.0)
                m.bias.data.normal_(mean=0.0, std=1.0)

    def forward(self, x):
        x = self.cnn(x).flatten(start_dim=1)
        x = self.fc(x)
        return x


class BatchFunctionalCNN(nn.Module):
    def __init__(self, act="relu", n_cnn_layers=7):
        super(BatchFunctionalCNN, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten(start_dim=2)  # (bs_model, bs, ...)
        self.act = dict(tanh=nn.Tanh(), relu=nn.ReLU(), gelu=nn.GELU())[act]
        self.n_cnn_layers = n_cnn_layers

    def forward(self, x, weights_and_biases):
        # x is (bs, c, h, w)
        weights, biases = weights_and_biases
        model_bs = weights[0].shape[0]
        for i, (w, b) in enumerate(zip(weights, biases)):

            # convolutions
            # todo: this is a very hacky code - we assume that if the feature dim != 1, it is conv with (k, k) kernel
            # if w.shape[-1] != 1:  # conv with (k, k) kernel
            if i < self.n_cnn_layers:
                kernel_size = int(np.sqrt(w.shape[-1]).item())
                # w is (model_bs, di, di+1, k*k)
                # b is (model_bs, di+1, 1)
                w = w.view(model_bs, w.shape[1], w.shape[2], kernel_size, kernel_size).transpose(1, 2)
                b = b.squeeze(-1)

                if i == 0:
                    def conv(weights, biases, x=x):
                        return F.conv2d(x, weights, bias=biases, stride=1, padding="same")
                    x = torch.vmap(conv)(w, b)
                else:
                    def conv(weights, biases, x):
                        return F.conv2d(x, weights, bias=biases, stride=1, padding="same")

                    x = self.act(x)
                    x = torch.vmap(conv)(w, b, x)

                if i > 0 and (i % 2) == 0:
                    # max pool - first two conv layers
                    def max_pool(x):
                        return self.max_pool(x)

                    x = torch.vmap(max_pool)(x)

            # linear
            else:
                # flatten
                x = self.flatten(x)
                x = self.act(x)

                if w.shape[-1] != 1:
                    # reshape the linear layer (bs, conv_out, linear_out, expand) -> (bs, conv_out * expand, linear_out)
                    w = w.permute(0, 2, 1, 3).flatten(start_dim=2).permute(0, 2, 1)

                # now linear
                def linear(weight, bias, x):
                    return F.linear(x, weight, bias=bias)

                x = torch.vmap(linear)(w.squeeze(-1).transpose(1, 2), b.squeeze(-1), x)

        return x  # (bs models, bs images, n classes)


class BatchFunctionalVGG(nn.Module):
    def __init__(self, name="vgg16"):
        super(BatchFunctionalVGG, self).__init__()
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten(start_dim=2)  # (bs_model, bs, ...)
        self.act = nn.ReLU()
        self.name = name
        assert name in ["vgg16", "vgg11"]
        self.config = {
            "vgg16": [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
            "vgg11": [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
        }[name]

    def forward(self, x, weights_and_biases):
        # x is (bs, c, h, w)
        weights, biases = weights_and_biases
        model_bs = weights[0].shape[0]

        j = 0
        for c in self.config:
            if c == 'M':
                def max_pool(x):
                    return self.max_pool(x)

                x = torch.vmap(max_pool)(x)

            else:  # conv
                w, b = weights[j], biases[j]

                kernel_size = int(np.sqrt(w.shape[-1]).item())
                # w is (model_bs, di, di+1, k*k)
                # b is (model_bs, di+1, 1)
                w = w.view(model_bs, w.shape[1], w.shape[2], kernel_size, kernel_size).transpose(1, 2)
                b = b.squeeze(-1)

                if j == 0:
                    def conv(weights, biases, x=x):
                        return F.conv2d(x, weights, bias=biases, stride=1, padding="same")

                    x = torch.vmap(conv)(w, b)
                else:
                    def conv(weights, biases, x):
                        return F.conv2d(x, weights, bias=biases, stride=1, padding="same")

                    x = torch.vmap(conv)(w, b, x)

                x = self.act(x)
                j += 1

        for w, b in zip(weights[-3:], biases[-3:]):

            # flatten
            x = self.flatten(x)
            if w.shape[-1] != 1:
                # reshape the linear layer (bs, conv_out, linear_out, expand) -> (bs, conv_out * expand, linear_out)
                w = w.permute(0, 2, 1, 3).flatten(start_dim=2).permute(0, 2, 1)

            # now linear
            def linear(weight, bias, x):
                return F.linear(x, weight, bias=bias)

            x = torch.vmap(linear)(w.squeeze(-1).transpose(1, 2), b.squeeze(-1), x)

        return x  # (bs models, bs images, n classes)
