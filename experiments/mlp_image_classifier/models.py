from torch import nn
import torch.nn.functional as F


class FCNet(nn.Module):
    """Fully connected neural network with functional forward support.
    """
    def __init__(self, in_dim=28*28, hidden_dim=128, n_hidden=2):
        super().__init__()
        layers = [nn.Linear(in_dim, hidden_dim), nn.ReLU()]
        self.n_hidden = n_hidden

        for _ in range(n_hidden):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])

        layers.append(nn.Linear(hidden_dim, 10))
        self.seq = nn.Sequential(*layers)

    def forward(self, x, weights_and_biases=None, negative_slope=0.0):
        x = x.flatten(start_dim=1)
        if weights_and_biases is None:
            return self.seq(x)
        else:
            # processing a batch of images using a batch of models' weights and biases
            weights, biases = weights_and_biases

            # (bs_models, bs_images, features)
            x = (
                    (weights[0].squeeze(-1).permute(0, 2, 1) @ x.permute(1, 0)).permute(0, 2, 1) +
                    biases[0].permute(0, 2, 1)  # (bs_models, 1, out_features)
                )

            for i, (w, b) in enumerate(zip(weights[1:], biases[1:])):
                # (bs_models, bs_images, in_features)
                x = F.leaky_relu(x, negative_slope=negative_slope)  # relu(x)
                # (bs_models, bs_images, out_features)
                x = x.bmm(w.squeeze(-1)) + b.permute(0, 2, 1)  # bias is (bs, 1, out_features)
            # (bs_models, bs_images, logits)
            return x
