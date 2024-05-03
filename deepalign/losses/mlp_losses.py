import torch
from torch import nn
from torch.nn import functional as F

from deepalign.utils import permute_weights, avg_weights_and_biases
from deepalign.sinkhorn import Sinkhorn, matching, CLFBatchSinkhorn
from experiments.mlp_image_classifier.models import FCNet
from experiments.mlp_image_classifier.trainer import device, image_flatten_size


def get_lmc_loss_and_preds_from_permutations(
        perm, inputs_view_0, inputs_view_1, functional_model, image_batch, n_steps=16
):
    if perm is not None:
        permuted_weights, permuted_biases = permute_weights(
            weights=inputs_view_1[0], biases=inputs_view_1[1], perms=perm
        )
    else:
        permuted_weights, permuted_biases = inputs_view_1

    recon_losses = []
    predicted = []

    bs = permuted_weights[0].shape[0]

    image_batch = tuple(t.to(device) for t in image_batch)
    images, labels = image_batch

    for a in torch.linspace(0, 1, n_steps):
        alpha = torch.ones(bs).to(device) * a.item()
        mixup_weights = avg_weights_and_biases(
            weights0=inputs_view_0[0],
            biases0=inputs_view_0[1],
            weights1=permuted_weights,
            biases1=permuted_biases,
            alpha=alpha,
        )
        logits = functional_model(images, weights_and_biases=mixup_weights)
        recon_loss = F.cross_entropy(
            # todo: make sure this is the correct way to do this loss calculation
            logits.permute(1, 2, 0),  # (bs_image, classes, bs_model)
            labels.unsqueeze(1).repeat(1, logits.size(0)),  # (bs_image, bs_model)
        )
        recon_losses.append(recon_loss.item())
        predicted.append(logits)

    return recon_losses, predicted


def calc_lmc_loss(
        pred_perm, inputs_view_0, inputs_view_1, image_batch, sinkhorn_project=True,
        n_sinkhorn_iter=20, tau=1., n_steps=16,
):
    """This function calculates the reconstruction loss and the reconstruction images for a sequence of
    interpolation points.

    """
    functional_model = FCNet(in_dim=image_flatten_size)

    # soft permutation
    pred_cp = None
    if sinkhorn_project:
        pred_cp = [None] * len(pred_perm)
        for i in range(len(pred_perm)):
            pred_cp[i] = Sinkhorn.apply(
                -pred_perm[i],
                torch.ones((pred_perm[i].shape[1])).to(pred_perm[i].device),
                torch.ones((pred_perm[i].shape[2])).to(pred_perm[i].device),
                n_sinkhorn_iter,
                tau,
            )

    # exact "hard" permutation
    hard_pred_perm = []
    for mat in pred_perm:
        permutation = matching(mat)
        hard_pred_perm.append(permutation.to(mat.device))

    # soft ours
    recon_losses, recon_preds = get_lmc_loss_and_preds_from_permutations(
        pred_cp if sinkhorn_project else pred_perm, inputs_view_0, inputs_view_1,
        functional_model, image_batch=image_batch, n_steps=n_steps
    )
    # exact "hard" ours
    hard_recon_losses, hard_recon_preds = get_lmc_loss_and_preds_from_permutations(
        hard_pred_perm, inputs_view_0, inputs_view_1, functional_model, image_batch=image_batch, n_steps=n_steps
    )
    # now perm baseline
    baseline_losses, baseline_recon_preds = get_lmc_loss_and_preds_from_permutations(
        None, inputs_view_0, inputs_view_1, functional_model, image_batch=image_batch, n_steps=n_steps
    )

    results = {
        "soft": {"losses": recon_losses, "preds": recon_preds},
        "hard": {"losses": hard_recon_losses, "preds": hard_recon_preds},
        "no_alignment": {"losses": baseline_losses, "preds": baseline_recon_preds},
    }

    return results


def calc_recon_loss(
        pred_perm, inputs_view_0, inputs_view_1, image_batch, sinkhorn_project=True,
        n_sinkhorn_iter=20, tau=1., alpha=None, add_task_loss=True, add_l2_loss=True,
        eval_mode=False
):
    """Calculates the linear mode connectivity (LMC) loss between the two models after alignment.

    """
    functional_model = FCNet(in_dim=image_flatten_size)
    images, labels = image_batch

    pred_cp = None
    if sinkhorn_project:
        pred_cp = [None] * len(pred_perm)
        for i in range(len(pred_perm)):
            pred_cp[i] = Sinkhorn.apply(
                -pred_perm[i],
                torch.ones((pred_perm[i].shape[1])).to(pred_perm[i].device),
                torch.ones((pred_perm[i].shape[2])).to(pred_perm[i].device),
                n_sinkhorn_iter,
                tau,
            )

    if eval_mode:
        perms = pred_cp if sinkhorn_project else pred_perm
        hard_perms = []
        for p in perms:
            hard_perms.append(matching(p).to(p.device))
        permuted_weights, permuted_biases = permute_weights(
            weights=inputs_view_1[0], biases=inputs_view_1[1], perms=hard_perms
        )
    else:
        permuted_weights, permuted_biases = permute_weights(
            weights=inputs_view_1[0], biases=inputs_view_1[1], perms=pred_cp if sinkhorn_project else pred_perm
        )

    recon_loss = torch.tensor(0.).to(device)
    if add_task_loss:
        bs = permuted_weights[0].shape[0]
        alpha = torch.rand(bs).to(device) if alpha is None else torch.ones(bs).to(device) * alpha
        mixup_weights = avg_weights_and_biases(
            weights0=inputs_view_0[0],
            biases0=inputs_view_0[1],
            weights1=permuted_weights,
            biases1=permuted_biases,
            alpha=alpha,
        )
        logits = functional_model(images, weights_and_biases=mixup_weights)
        recon_loss = recon_loss + F.cross_entropy(
            logits.permute(1, 2, 0),  # (bs_image, classes, bs_model)
            labels.unsqueeze(1).repeat(1, logits.size(0)),  # (bs_image, bs_model)
        )

    if add_l2_loss:
        weights_loss = 0.
        for w, w_perm, b, b_perm in zip(inputs_view_0[0], permuted_weights, inputs_view_0[1], permuted_biases):
            weights_loss = weights_loss + nn.functional.mse_loss(w_perm, w) + nn.functional.mse_loss(b_perm, b)
        recon_loss = recon_loss + weights_loss

    return recon_loss


def calc_lookahead_loss(
    pred_perm, inputs_view_0, inputs_view_1, image_batch,
    n_sinkhorn_iter=20,
    n_lookahead_iter=25, lookahead_lr=5e-2,
    add_task_loss=True, add_l2_loss=True, loss_type="mse",
):
    """Lookahead loss: We take lookahead steps on the predicted permutations to achieve better permutations matrices.
    Then we compute the loss as the CE/MSE loss between the initial and final permutations matrices.

    """
    assert loss_type in ["mse", "ce"]
    batch_sinkhorn = CLFBatchSinkhorn(n_sinkhorn=n_sinkhorn_iter, n_iterations=n_lookahead_iter, lr=lookahead_lr)

    permutations = batch_sinkhorn.match_batch(
        inputs_view_0=inputs_view_0,
        inputs_view_1=inputs_view_1,
        permutations=[p.clone().detach().requires_grad_(True) for p in pred_perm],
        batch=image_batch,
        show_progress=False,
        add_task_loss=add_task_loss,
        add_l2_loss=add_l2_loss,
    )

    permutations = permutations["predicted_hard"] if loss_type == "ce" else permutations["predicted_raw"]

    loss = torch.tensor(0.0, device=device)
    criterion = nn.CrossEntropyLoss() if loss_type == "ce" else nn.MSELoss()
    for init_p, final_p in zip(pred_perm, permutations):
        gt = final_p.detach()
        if loss_type == "ce":
            gt = gt.argmax(1)
        loss = loss + criterion(init_p, gt)

    return loss


def calc_gt_perm_loss(pred_perms, gt_perms, criterion):
    """CE/MSE loss between the predicted permutations and the ground truth permutations.

    """
    assert criterion in ["mse", "ce"]
    loss = torch.tensor(0.0, device=device)
    for pred_perm, gt_perm in zip(pred_perms, gt_perms):
        if criterion == "mse":
            # convert perm indices to perm matrix
            batch_size, n = gt_perm.shape
            # create an identity matrix
            identity = torch.eye(n).repeat(batch_size, 1, 1).to(device)
            # use scatter to create permutation matrix
            perm_matrix = identity[torch.arange(batch_size).unsqueeze(1), gt_perm].permute(0, 2, 1)
            loss = loss + F.mse_loss(pred_perm, perm_matrix)

        else:
            loss = loss + F.cross_entropy(pred_perm, gt_perm)
    return loss / len(gt_perms)
