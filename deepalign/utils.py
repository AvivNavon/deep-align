import copy
from typing import Tuple, List, Union

import torch


def permute_weights(
    weights: Tuple[torch.Tensor], biases: Tuple[torch.Tensor], perms: List[torch.Tensor]
) -> Tuple[Tuple[torch.Tensor], Tuple[torch.Tensor]]:
    """Util function to permute a batch of weights and biases."""
    inputs_weights, inputs_bias = copy.deepcopy(weights), copy.deepcopy(
        biases
    )  # not sure needed, just to be safe
    perm_weights, perm_bias = list(copy.deepcopy(weights)), list(copy.deepcopy(biases))
    prev_perm = None
    for perm_id, (curr_perm, weight, bias) in enumerate(
        zip(perms + [None], inputs_weights, inputs_bias)
    ):
        # weights: (bs, di, d{i+1}, 1), bias: (bs, d{i+1}, 1), curr_perm: (bs, d{i+1}, d{i+1})
        if prev_perm is not None:
            perm_weights[perm_id] = torch.einsum(
                "bli,bljk->bijk", (prev_perm.transpose(1, 2), weight)
            )
            weight = perm_weights[perm_id]
        if curr_perm is not None:
            perm_weights[perm_id] = torch.einsum(
                "bimk,bmj->bijk", (weight, curr_perm.transpose(1, 2))
            )
            perm_bias[perm_id] = torch.einsum(
                "bmk,bmj->bjk", (bias, curr_perm.transpose(1, 2))
            )
        prev_perm = curr_perm
    return tuple(perm_weights), tuple(perm_bias)


def unsqueeze_like(tensor: torch.Tensor, like: torch.Tensor):
    """
    Unsqueeze last dimensions of tensor to match another tensor's number of dimensions.

    Args:
        tensor (torch.Tensor): tensor to unsqueeze
        like (torch.Tensor): tensor whose dimensions to match
    """
    n_unsqueezes = like.ndim - tensor.ndim
    if n_unsqueezes < 0:
        raise ValueError(f"tensor.ndim={tensor.ndim} > like.ndim={like.ndim}")
    elif n_unsqueezes == 0:
        return tensor
    else:
        return tensor[(...,) + (None,) * n_unsqueezes]


def avg_weights_and_biases(
    weights0, biases0, weights1, biases1, alpha: Union[torch.Tensor, float] = 0.5
):
    if isinstance(alpha, float):
        alpha = torch.tensor(alpha, dtype=weights0[0].dtype, device=weights0[0].device)

    avg_weights = tuple(
        unsqueeze_like(alpha, w0) * w0 + unsqueeze_like(1 - alpha, w1) * w1
        for w0, w1 in zip(weights0, weights1)
    )
    avg_bias = tuple(
        unsqueeze_like(alpha, b0) * b0 + unsqueeze_like(1 - alpha, b1) * b1
        for b0, b1 in zip(biases0, biases1)
    )
    return avg_weights, avg_bias


def extract_pred(features0, features1):
    """Extracts the predicted permutation matrices from the two INRs model outputs.

    """
    pred_matrices = []
    for k in range(len(features0)):
        # outer product
        f0 = features0[k]
        f1 = features1[k]
        pred = torch.einsum("bid,bjd->bij", (f0, f1))
        pred_matrices.append(pred)

    return pred_matrices
