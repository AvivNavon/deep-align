from typing import List

import numpy as np
import torch
import torchvision
from scipy.optimize import linear_sum_assignment
from scipy.sparse import coo_matrix
from torch import nn
from torchvision.utils import save_image
from tqdm import tqdm

from deepalign.utils import permute_weights, unsqueeze_like, avg_weights_and_biases
from deepalign.nn.inr_models import FunctionalINRForModelBatch
from experiments.mlp_image_classifier.models import FCNet


def matching(predicted, **kwargs):
    # Negate the probability matrix to serve as cost matrix. This function
    # yields two lists, the row and colum indices for all entries in the
    # permutation matrix we should set to 1.
    # row, col = linear_sum_assignment(-alpha.permute(0, 2, 1).detach().cpu().numpy(), **kwargs)
    perms = list(
        map(
            lambda x: linear_sum_assignment(-x, **kwargs),
            predicted.clone().detach().cpu().numpy(),
        )
    )

    # Create the permutation matrix.
    permutation_matrix = np.stack(
        [
            coo_matrix((np.ones_like(row), (row, col))).toarray()
            for row, col in perms
        ],
        axis=0,
        dtype=np.float32,
    )
    return torch.from_numpy(permutation_matrix)


# Sinkhorn differentiation from https://github.com/marvin-eisenberger/implicit-sinkhorn
class Sinkhorn(torch.autograd.Function):
    """
    An implementation of a Sinkhorn layer with our custom backward module, based on implicit differentiation
    :param c: input cost matrix, size [*,m,n], where * are arbitrarily many batch dimensions
    :param a: first input marginal, size [*,m]
    :param b: second input marginal, size [*,n]
    :param num_sink: number of Sinkhorn iterations
    :param lambd_sink: entropy regularization weight
    :return: optimized soft permutation matrix
    """

    @staticmethod
    def forward(ctx, c, a, b, num_sink, lambd_sink):
        log_p = -c / lambd_sink
        log_a = torch.log(a).unsqueeze(dim=-1)
        log_b = torch.log(b).unsqueeze(dim=-2)
        for _ in range(num_sink):
            log_p -= torch.logsumexp(log_p, dim=-2, keepdim=True) - log_b
            log_p -= torch.logsumexp(log_p, dim=-1, keepdim=True) - log_a
        p = torch.exp(log_p)

        ctx.save_for_backward(p, torch.sum(p, dim=-1), torch.sum(p, dim=-2))
        ctx.lambd_sink = lambd_sink
        return p

    @staticmethod
    def backward(ctx, grad_p):
        p, a, b = ctx.saved_tensors

        device = grad_p.device

        m, n = p.shape[-2:]
        batch_shape = list(p.shape[:-2])

        grad_p *= -1 / ctx.lambd_sink * p
        K = torch.cat(
            (
                torch.cat((torch.diag_embed(a), p), dim=-1),
                torch.cat((p.transpose(-2, -1), torch.diag_embed(b)), dim=-1),
            ),
            dim=-2,
        )[..., :-1, :-1]
        t = torch.cat(
            (grad_p.sum(dim=-1), grad_p[..., :, :-1].sum(dim=-2)), dim=-1
        ).unsqueeze(-1)
        grad_ab = torch.linalg.solve(K, t)
        grad_a = grad_ab[..., :m, :]
        grad_b = torch.cat(
            (
                grad_ab[..., m:, :],
                torch.zeros(batch_shape + [1, 1], device=device, dtype=torch.float32),
            ),
            dim=-2,
        )
        U = grad_a + grad_b.transpose(-2, -1)
        grad_p -= p * U
        grad_a = -ctx.lambd_sink * grad_a.squeeze(dim=-1)
        grad_b = -ctx.lambd_sink * grad_b.squeeze(dim=-1)
        return grad_p, grad_a, grad_b, None, None, None


class INRBatchSinkhorn:
    def __init__(self, n_iterations=1000, n_sinkhorn=20, lr=5e-2):
        self.n_iterations = n_iterations
        self.n_sinkhorn = n_sinkhorn
        self.lr = lr

    def get_loss_from_views(
            self,
            inputs_view_0,
            inputs_view_1,
            inputs,
            bi_stochastic_matrices,
            add_l2_loss=True,
            add_task_loss=True,
            device=torch.device("cpu"),
    ):
        recon_loss = torch.tensor(0.0, device=device)
        permuted_weights, permuted_biases = permute_weights(
            weights=inputs_view_1[0], biases=inputs_view_1[1], perms=bi_stochastic_matrices
        )
        if add_task_loss:
            functional_inr = FunctionalINRForModelBatch()
            # target for recon
            with torch.no_grad():
                lbl0 = functional_inr(inputs, weights_and_biases=inputs_view_0)
                lbl1 = functional_inr(inputs, weights_and_biases=inputs_view_1)

            criterion = nn.MSELoss()

            bs = permuted_weights[0].shape[0]
            alpha = torch.rand(bs).to(device)
            mixup_weights = avg_weights_and_biases(
                weights0=inputs_view_0[0],
                biases0=inputs_view_0[1],
                weights1=permuted_weights,
                biases1=permuted_biases,
                alpha=alpha,
            )
            mixup_pred = functional_inr(inputs, weights_and_biases=mixup_weights)

            recon_loss = recon_loss + criterion(
                mixup_pred,
                unsqueeze_like(alpha, lbl0) * lbl0 + unsqueeze_like(1 - alpha, lbl1) * lbl1,
            )

        # we add L1 loss
        if add_l2_loss:
            l2_loss = 0.
            for w, w_perm, b, b_perm in zip(inputs_view_0[0], permuted_weights, inputs_view_0[1], permuted_biases):
                l2_loss = l2_loss + nn.functional.mse_loss(w_perm, w) + nn.functional.mse_loss(b_perm, b)
            recon_loss = recon_loss + l2_loss
        return recon_loss

    def match_batch(
        self, inputs_view_0, inputs_view_1, permutations: List[torch.Tensor],
        inputs: torch.Tensor, add_l2_loss=True, add_task_loss=True, show_progress=False,
    ):
        device = permutations[0].device
        optimizer = torch.optim.Adam(permutations, lr=self.lr)
        progress_bar = tqdm(range(self.n_iterations)) if show_progress else range(self.n_iterations)
        for _ in progress_bar:
            optimizer.zero_grad()
            curr_bi_stochastic_matrices = []

            for perm in permutations:
                bi_stochastic_matrix = Sinkhorn.apply(
                    -perm * 1.,
                    torch.ones((perm.shape[1])).to(device),
                    torch.ones((perm.shape[2])).to(device),
                    self.n_sinkhorn,
                    1.,
                )
                curr_bi_stochastic_matrices.append(bi_stochastic_matrix)

            loss = self.get_loss_from_views(
                inputs_view_0=inputs_view_0,
                inputs_view_1=inputs_view_1,
                inputs=inputs,
                bi_stochastic_matrices=curr_bi_stochastic_matrices,
                add_l2_loss=add_l2_loss,
                add_task_loss=add_task_loss,
                device=device,
            )

            loss.backward()
            optimizer.step()

        results = {"predicted_raw": [p.clone().detach() for p in permutations]}
        bi_stochastic_matrices = []
        hard_perm = []
        for perm in permutations:
            # exact perm
            hard_perm.append(matching(perm).to(device))
            # soft perm
            bi_stochastic_matrix = Sinkhorn.apply(
                -perm * 1.,
                torch.ones((perm.shape[1])).to(device),
                torch.ones((perm.shape[2])).to(device),
                self.n_sinkhorn,
                1.,
            )
            bi_stochastic_matrices.append(bi_stochastic_matrix.clone().detach())

        results["predicted_hard"] = hard_perm
        results["predicted_soft"] = bi_stochastic_matrices
        return results


class CLFBatchSinkhorn:
    """Batch Sinkhorn algorithm for classifiers

    """
    def __init__(self, n_iterations=1000, n_sinkhorn=20, lr=5e-2, functional_model=FCNet()):
        self.n_iterations = n_iterations
        self.n_sinkhorn = n_sinkhorn
        self.lr = lr
        self.functional_model = functional_model

    def get_loss_from_views(
            self,
            inputs_view_0,
            inputs_view_1,
            batch,
            bi_stochastic_matrices,
            add_l2_loss=True,
            add_task_loss=True,
            device=torch.device("cpu"),
    ):
        recon_loss = torch.tensor(0.0, device=device)
        permuted_weights, permuted_biases = permute_weights(
            weights=inputs_view_1[0], biases=inputs_view_1[1], perms=bi_stochastic_matrices
        )

        if add_task_loss:
            functional_model = self.functional_model
            criterion = nn.CrossEntropyLoss()

            bs = permuted_weights[0].shape[0]
            alpha = torch.rand(bs).to(device)
            mixup_weights = avg_weights_and_biases(
                weights0=inputs_view_0[0],
                biases0=inputs_view_0[1],
                weights1=permuted_weights,
                biases1=permuted_biases,
                alpha=alpha,
            )
            batch = (t.to(device) for t in batch)
            inputs, targets = batch
            logits = functional_model(inputs, weights_and_biases=mixup_weights)

            recon_loss = recon_loss + criterion(
                logits.permute(1, 2, 0),  # (bs_image, classes, bs_model)
                targets.unsqueeze(1).repeat(1, logits.size(0)),  # (bs_image, bs_model)
            )

        # we add L1 loss
        if add_l2_loss:
            l2_loss = 0.
            for w, w_perm, b, b_perm in zip(inputs_view_0[0], permuted_weights, inputs_view_0[1], permuted_biases):
                l2_loss = l2_loss + nn.functional.mse_loss(w_perm, w) + nn.functional.mse_loss(b_perm, b)
            recon_loss = recon_loss + l2_loss
        return recon_loss

    def match_batch(
        self, inputs_view_0, inputs_view_1, permutations: List[torch.Tensor],
        batch: torch.Tensor, add_l2_loss=True, add_task_loss=True, show_progress=False,
    ):
        device = permutations[0].device
        optimizer = torch.optim.Adam(permutations, lr=self.lr)
        progress_bar = tqdm(range(self.n_iterations)) if show_progress else range(self.n_iterations)
        for _ in progress_bar:
            optimizer.zero_grad()
            curr_bi_stochastic_matrices = []

            for perm in permutations:
                bi_stochastic_matrix = Sinkhorn.apply(
                    -perm * 1.,
                    torch.ones((perm.shape[1])).to(device),
                    torch.ones((perm.shape[2])).to(device),
                    self.n_sinkhorn,
                    1.,
                )
                curr_bi_stochastic_matrices.append(bi_stochastic_matrix)

            loss = self.get_loss_from_views(
                inputs_view_0=inputs_view_0,
                inputs_view_1=inputs_view_1,
                batch=batch,
                bi_stochastic_matrices=curr_bi_stochastic_matrices,
                add_l2_loss=add_l2_loss,
                add_task_loss=add_task_loss,
                device=device,
            )

            loss.backward()
            optimizer.step()

        results = {"predicted_raw": [p.clone().detach() for p in permutations]}
        bi_stochastic_matrices = []
        hard_perm = []
        for perm in permutations:
            # exact perm
            hard_perm.append(matching(perm).to(device))
            # soft perm
            bi_stochastic_matrix = Sinkhorn.apply(
                -perm * 1.,
                torch.ones((perm.shape[1])).to(device),
                torch.ones((perm.shape[2])).to(device),
                self.n_sinkhorn,
                1.,
            )
            bi_stochastic_matrices.append(bi_stochastic_matrix.clone().detach())

        results["predicted_hard"] = hard_perm
        results["predicted_soft"] = bi_stochastic_matrices
        return results


class CLFBatchSinkhornUsingMultiBatch:
    """Batch Sinkhorn algorithm for classifiers

    """
    def __init__(self, n_iterations=1000, n_sinkhorn=20, lr=5e-2, functional_model=FCNet()):
        self.n_iterations = n_iterations
        self.n_sinkhorn = n_sinkhorn
        self.lr = lr
        self.functional_model = functional_model

    def get_loss_from_views(
            self,
            inputs_view_0,
            inputs_view_1,
            batch,
            bi_stochastic_matrices,
            add_l2_loss=True,
            add_task_loss=True,
            device=torch.device("cpu"),
    ):
        recon_loss = torch.tensor(0.0, device=device)
        permuted_weights, permuted_biases = permute_weights(
            weights=inputs_view_1[0], biases=inputs_view_1[1], perms=bi_stochastic_matrices
        )

        if add_task_loss:
            functional_model = self.functional_model
            criterion = nn.CrossEntropyLoss()

            bs = permuted_weights[0].shape[0]
            alpha = torch.rand(bs).to(device)
            mixup_weights = avg_weights_and_biases(
                weights0=inputs_view_0[0],
                biases0=inputs_view_0[1],
                weights1=permuted_weights,
                biases1=permuted_biases,
                alpha=alpha,
            )
            batch = (t.to(device) for t in batch)
            inputs, targets = batch
            logits = functional_model(inputs, weights_and_biases=mixup_weights)

            recon_loss = recon_loss + criterion(
                logits.permute(1, 2, 0),  # (bs_image, classes, bs_model)
                targets.unsqueeze(1).repeat(1, logits.size(0)),  # (bs_image, bs_model)
            )

        # we add L1 loss
        if add_l2_loss:
            l2_loss = 0.
            for w, w_perm, b, b_perm in zip(inputs_view_0[0], permuted_weights, inputs_view_0[1], permuted_biases):
                l2_loss = l2_loss + nn.functional.mse_loss(w_perm, w) + nn.functional.mse_loss(b_perm, b)
            recon_loss = recon_loss + l2_loss
        return recon_loss

    def match_batch(
        self, inputs_view_0, inputs_view_1, permutations: List[torch.Tensor],
        loader, add_l2_loss=True, add_task_loss=True, show_progress=False,
    ):
        device = permutations[0].device
        optimizer = torch.optim.Adam(permutations, lr=self.lr)
        progress_bar = tqdm(range(self.n_iterations)) if show_progress else range(self.n_iterations)
        for _ in progress_bar:
            optimizer.zero_grad()
            curr_bi_stochastic_matrices = []
            batch = next(iter(loader))
            batch = tuple(t.to(device) for t in batch)

            for perm in permutations:
                bi_stochastic_matrix = Sinkhorn.apply(
                    -perm * 1.,
                    torch.ones((perm.shape[1])).to(device),
                    torch.ones((perm.shape[2])).to(device),
                    self.n_sinkhorn,
                    1.,
                )
                curr_bi_stochastic_matrices.append(bi_stochastic_matrix)

            loss = self.get_loss_from_views(
                inputs_view_0=inputs_view_0,
                inputs_view_1=inputs_view_1,
                batch=batch,
                bi_stochastic_matrices=curr_bi_stochastic_matrices,
                add_l2_loss=add_l2_loss,
                add_task_loss=add_task_loss,
                device=device,
            )

            loss.backward()
            optimizer.step()

        results = {"predicted_raw": [p.clone().detach() for p in permutations]}
        bi_stochastic_matrices = []
        hard_perm = []
        for perm in permutations:
            # exact perm
            hard_perm.append(matching(perm).to(device))
            # soft perm
            bi_stochastic_matrix = Sinkhorn.apply(
                -perm * 1.,
                torch.ones((perm.shape[1])).to(device),
                torch.ones((perm.shape[2])).to(device),
                self.n_sinkhorn,
                1.,
            )
            bi_stochastic_matrices.append(bi_stochastic_matrix.clone().detach())

        results["predicted_hard"] = hard_perm
        results["predicted_soft"] = bi_stochastic_matrices
        return results


if __name__ == '__main__':
    from experiments.utils.data import MultiViewMatchingINRDataset
    from experiments.utils import get_device, make_coordinates, reshape_model_output, set_seed

    path = "/cortex/data/c-nets/inrs/artifacts/fmnist_many_views_splits.json"
    normalize = False
    batch_size = 64
    num_workers = 0
    n_iterations = 500
    device = get_device()
    set_seed(42)

    train_set = MultiViewMatchingINRDataset(
        path=path,
        split="train",
        normalize=normalize,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    batch = next(iter(train_loader))
    batch = batch.to(device)
    random_perm = [
            torch.nn.Parameter(
                torch.eye(32, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
                + torch.randn(batch_size, 32, 32, device=device) * 1e-1,
                requires_grad=True,
            )
            for _ in range(2)
        ]

    image_size = (28, 28)
    coords = make_coordinates(image_size, 1).squeeze(0).to(device)

    batch_sinkhorn = INRBatchSinkhorn(n_iterations=n_iterations)
    permutations = batch_sinkhorn.match_batch(
        inputs_view_0=(batch.weights_view_0, batch.biases_view_0),
        inputs_view_1=(batch.weights_view_1, batch.biases_view_1),
        permutations=random_perm,
        inputs=coords,
        show_progress=True,
    )

    inputs1_weights, inputs1_bias = permute_weights(
        weights=batch.weights_view_1, biases=batch.biases_view_1, perms=permutations["predicted_hard"]
    )

    functional_inr = FunctionalINRForModelBatch()
    n_steps = 16
    images = []
    for a in torch.linspace(0, 1, n_steps):
        alpha = torch.ones(batch_size).to(device) * a.item()
        mixup_weights = avg_weights_and_biases(
            weights0=batch.weights_view_0,
            biases0=batch.biases_view_0,
            weights1=inputs1_weights,
            biases1=inputs1_bias,
            alpha=alpha,
        )
        mixup_pred = functional_inr(coords, weights_and_biases=mixup_weights)
        images.append(mixup_pred)

    # avg_weights, avg_bias = avg_weights_and_biases(
    #     weights0=batch.weights_view_0,
    #     biases0=batch.biases_view_0,
    #     weights1=inputs1_weights,
    #     biases1=inputs1_bias,
    # )
    #
    # inputs = (avg_weights, avg_bias)
    # functional_inr = FunctionalINRForModelBatch()
    # outputs = functional_inr(coords, inputs)

    def process_images(images, image_size):
        recon_image = [r[:16] for r in images]  # take first 16 images for each alpha
        recon_image = [reshape_model_output(r, *image_size, r.shape[0]) for r in recon_image]  # reshape to e.g. bsx28x28x1
        recon_image = torch.cat(recon_image)  # 2*bs x 28 x 28 x 1
        return recon_image


    images = process_images(images, image_size)
    # pred_images = outputs.view(batch_size, *image_size, -1)
    save_image(
        torchvision.utils.make_grid(images, nrow=16),
        f"test_batch_sinkhorn_alignment.png"
    )
    # save_image(images.permute(0, 3, 1, 2), f"test_batch_sinkhorn_alignment.png")
