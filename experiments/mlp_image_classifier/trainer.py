import json
import logging
from argparse import ArgumentParser
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score
from tqdm import trange

from deepalign.losses.mlp_losses import calc_lmc_loss, calc_recon_loss, calc_gt_perm_loss
from deepalign.utils import extract_pred
from experiments.utils import (
    common_parser, count_parameters, get_device, set_logger, set_seed, str2bool,
)
from experiments.utils.data import MultiViewMatchingBatch, MatchingModelsDataset
from deepalign.sinkhorn import matching
from deepalign import DWSMatching
from experiments.utils.data.image_data import get_mnist_dataloaders, get_cifar10_dataloaders

set_logger()


@torch.no_grad()
def evaluate(model, loader, image_loader, add_task_loss=True, add_l2_loss=True):
    model.eval()

    perm_loss = 0.0
    recon_loss = 0.
    correct = 0.0
    total = 0.0
    predicted, gt = [], []
    recon_losses, baseline_losses, hard_recon_losses, sink_ours_losses, sink_random_losses = [], [], [], [], []
    for j, batch in enumerate(loader):
        image_batch = next(iter(image_loader))
        image_batch = tuple(t.to(device) for t in image_batch)
        batch: MultiViewMatchingBatch = batch.to(device)

        input_0 = (batch.weights_view_0, batch.biases_view_0)
        input_1 = (batch.weights_view_1, batch.biases_view_1)
        perm_input_0 = (batch.perm_weights_view_0, batch.perm_biases_view_0)

        out_0 = model(input_0)
        out_1 = model(input_1)
        perm_out_0 = model(perm_input_0)

        pred_matrices_perm_0 = extract_pred(
            out_0,
            perm_out_0,
        )

        pred_matrices = extract_pred(
            out_0,
            out_1,
        )

        # loss from GT permutations
        curr_gt_loss = calc_gt_perm_loss(
            pred_matrices_perm_0, batch.perms_view_0, criterion=args.loss, device=device
        )

        # reconstruction loss
        curr_recon_loss = calc_recon_loss(
            pred_matrices if not args.sanity else pred_matrices_perm_0,
            input_0,
            input_1 if not args.sanity else perm_input_0,
            image_batch=image_batch,
            sinkhorn_project=True,
            n_sinkhorn_iter=args.n_sink,
            add_task_loss=add_task_loss,
            add_l2_loss=add_l2_loss,
            alpha=0.5,
            eval_mode=True,
            device=device,
            image_flatten_size=image_flatten_size,
        )

        # reconstruction loss and images
        results = calc_lmc_loss(
            pred_matrices if not args.sanity else pred_matrices_perm_0,
            input_0,
            input_1 if not args.sanity else perm_input_0,
            image_batch=image_batch,
            sinkhorn_project=True,
            n_sinkhorn_iter=args.n_sink,
            device=device,
            image_flatten_size=image_flatten_size,
        )

        recon_losses.append(results["soft"]["losses"])
        hard_recon_losses.append(results["hard"]["losses"])
        baseline_losses.append(results["no_alignment"]["losses"])

        perm_loss += curr_gt_loss.item()
        recon_loss += curr_recon_loss.item()

        curr_correct = 0.
        curr_gts = []
        curr_preds = []

        for pred, gt_perm in zip(pred_matrices_perm_0, batch.perms_view_0):
            pred = matching(pred).to(device)
            curr_correct += ((pred.argmax(1)).eq(gt_perm) * 1.0).mean().item()
            curr_preds.append(pred.argmax(1).reshape(-1))
            curr_gts.append(gt_perm.reshape(-1))

        total += 1
        correct += (curr_correct / len(pred_matrices_perm_0))
        predicted.extend(curr_preds)
        gt.extend(curr_gts)

    predicted = torch.cat(predicted).int()
    gt = torch.cat(gt).int()

    avg_loss = perm_loss / total
    avg_acc = correct / total
    recon_loss = recon_loss / total

    f1 = f1_score(predicted.cpu().detach().numpy(), gt.cpu().detach().numpy(), average="macro")

    # LMC losses
    lmc_losses = dict(
        soft_alignment=np.stack(recon_losses).mean(0),  # NOTE: this is the soft alignment loss.
        no_alignment=np.stack(baseline_losses).mean(0),
        alignment=np.stack(hard_recon_losses).mean(0),
    )

    return dict(
        avg_loss=avg_loss,
        avg_acc=avg_acc,
        recon_loss=recon_loss,
        predicted=predicted,
        gt=gt,
        f1=f1,
        lmc_losses=lmc_losses,
    )


def main(
    path,
    epochs: int,
    lr: float,
    batch_size: int,
    device,
    eval_every: int,
):
    # losses
    add_l2_loss = True if args.recon_loss in ["l2", "both"] else False
    add_task_loss = True if args.recon_loss in ["lmc", "both"] else False
    logging.info(f"Using {args.recon_loss} loss (task loss: {add_task_loss}, l2 loss: {add_l2_loss})")

    # load dataset
    train_set = MatchingModelsDataset(
        path=path,
        split="train",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )
    val_set = MatchingModelsDataset(
        path=path,
        split="val",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )
    test_set = MatchingModelsDataset(
        path=path,
        split="test",
        normalize=args.normalize,
        statistics_path=args.statistics_path,
    )

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_set,
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )
    # todo: add image args to argparse
    get_loaders = dict(mnist=get_mnist_dataloaders, cifar10=get_cifar10_dataloaders)[args.data_name]
    train_image_loader, val_image_loader, test_image_loader = get_loaders(
        args.image_data_path, batch_size=args.image_batch_size
    )

    logging.info(
        f"train size {len(train_set)}, "
        f"val size {len(val_set)}, "
        f"test size {len(test_set)}"
    )

    batch = next(iter(train_loader))
    weight_shapes, bias_shapes = batch.get_weight_shapes()

    logging.info(f"weight shapes: {weight_shapes}, bias shapes: {bias_shapes}")

    model = DWSMatching(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=1,
            hidden_dim=args.dim_hidden,
            n_hidden=args.n_hidden,
            reduction=args.reduction,
            n_fc_layers=args.n_fc_layers,
            set_layer=args.set_layer,
            output_features=args.output_features
            if args.output_features is not None
            else args.dim_hidden,
            input_dim_downsample=args.input_dim_downsample,
            add_skip=args.add_skip,
            add_layer_skip=args.add_layer_skip,
            init_scale=args.init_scale,
            init_off_diag_scale_penalty=args.init_off_diag_scale,
            bn=args.add_bn,
            diagonal=args.diagonal,
            hnp_setup=args.hnp_setup,
        ).to(device)

    logging.info(f"number of parameters: {count_parameters(model)}")

    params = list(model.parameters())

    optimizer = {
        "adam": torch.optim.Adam(
            [
                dict(params=params, lr=lr),
            ],
            lr=lr,
            weight_decay=5e-4,
        ),
        "sgd": torch.optim.SGD(params, lr=lr, weight_decay=5e-4, momentum=0.9),
        "adamw": torch.optim.AdamW(
            params=params, lr=lr, amsgrad=True, weight_decay=5e-4
        ),
    }[args.optim]

    def save_model(sd):
        path = Path(args.save_path)
        artifact_path = path / args.recon_loss / f"{args.seed}"
        artifact_path.mkdir(parents=True, exist_ok=True)
        # save model
        torch.save(sd, artifact_path / f"model.pth")

        with open(artifact_path / "args.json", "w") as f:
            json.dump(vars(args), f)

        model_args = dict(
            weight_shapes=weight_shapes,
            bias_shapes=bias_shapes,
            input_features=1,
            hidden_dim=args.dim_hidden,
            n_hidden=args.n_hidden,
            reduction=args.reduction,
            n_fc_layers=args.n_fc_layers,
            set_layer=args.set_layer,
            output_features=args.output_features
            if args.output_features is not None
            else args.dim_hidden,
            input_dim_downsample=args.input_dim_downsample,
            add_skip=args.add_skip,
            add_layer_skip=args.add_layer_skip,
            init_scale=args.init_scale,
            init_off_diag_scale_penalty=args.init_off_diag_scale,
            bn=args.add_bn,
            diagonal=args.diagonal,
            hnp_setup=args.hnp_setup,
        )
        with open(artifact_path / "model_config.json", "w") as f:
            json.dump(model_args, f)

    epoch_iter = trange(epochs)
    best_test_results, best_val_results = None, None
    test_acc, test_loss = -1.0, -1.0
    best_val_recon_loss = 1e6
    best_sd = model.state_dict()
    for epoch in epoch_iter:
        for i, batch in enumerate(train_loader):
            model.train()
            optimizer.zero_grad()

            batch: MultiViewMatchingBatch = batch.to(device)
            image_batch = next(iter(train_image_loader))
            image_batch = tuple(t.to(device) for t in image_batch)

            input_0 = (batch.weights_view_0, batch.biases_view_0)
            input_1 = (batch.weights_view_1, batch.biases_view_1)
            perm_input_0 = (batch.perm_weights_view_0, batch.perm_biases_view_0)

            out_0 = model(input_0)
            out_1 = model(input_1)
            perm_out_0 = model(perm_input_0)

            pred_matrices_perm_0 = extract_pred(
                out_0,
                perm_out_0,
            )

            pred_matrices = extract_pred(
                out_0,
                out_1,
            )

            # loss from GT permutations
            gt_perm_loss = calc_gt_perm_loss(
                pred_matrices_perm_0, batch.perms_view_0, criterion=args.loss, device=device
            )

            # reconstruction loss
            recon_loss = calc_recon_loss(
                pred_matrices if not args.sanity else pred_matrices_perm_0,
                input_0,
                input_1 if not args.sanity else perm_input_0,
                image_batch=image_batch,
                sinkhorn_project=True,   # if we perms are already bi-stochastic we don't need to do anything
                n_sinkhorn_iter=args.n_sink,
                add_task_loss=add_task_loss,
                add_l2_loss=add_l2_loss,
                device=device,
                image_flatten_size=image_flatten_size,
            )

            loss = gt_perm_loss * args.supervised_loss_weight + recon_loss * args.recon_loss_weight
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if args.wandb:
                log = {
                    "train/loss": loss.item(),
                    "train/supervised_loss": gt_perm_loss.item(),
                    "train/recon_loss": recon_loss.item(),
                }
                wandb.log(log)

            epoch_iter.set_description(
                f"[{epoch} {i+1}], train loss: {loss.item():.3f}, test_loss: {test_loss:.3f}, test_acc: {test_acc:.3f}"
            )

        if (epoch + 1) % eval_every == 0:
            val_loss_dict = evaluate(
                model, val_loader, image_loader=val_image_loader,
                add_task_loss=add_task_loss, add_l2_loss=add_l2_loss,
            )
            test_loss_dict = evaluate(
                model, test_loader, image_loader=test_image_loader,
                add_task_loss=add_task_loss, add_l2_loss=add_l2_loss,
            )
            val_loss = val_loss_dict["avg_loss"]
            val_acc = val_loss_dict["avg_acc"]
            test_loss = test_loss_dict["avg_loss"]
            test_acc = test_loss_dict["avg_acc"]

            best_val_criteria = val_loss_dict["recon_loss"] <= best_val_recon_loss

            if best_val_criteria:
                best_val_recon_loss = val_loss_dict["recon_loss"]
                best_test_results = test_loss_dict
                best_val_results = val_loss_dict
                best_sd = model.state_dict()
                if args.save_model:
                    save_model(best_sd)

            if args.wandb:
                # LMC plot
                x = torch.linspace(0.0, 1.0, len(test_loss_dict["lmc_losses"]["alignment"])).numpy().tolist()
                for k, v in test_loss_dict["lmc_losses"].items():
                    plt.plot(x, v, label=k)
                plt.legend()

                log = {
                    "val/loss": val_loss,
                    "val/acc": val_acc,
                    "val/f1": val_loss_dict["f1"],
                    "val/recon_loss": val_loss_dict["recon_loss"],
                    "val/best_loss": best_val_results["avg_loss"],
                    "val/best_acc": best_val_results["avg_acc"],
                    "val/best_f1": best_val_results["f1"],
                    "val/best_recon_loss": best_val_results["recon_loss"],
                    "test/loss": test_loss,
                    "test/acc": test_acc,
                    "test/f1": test_loss_dict["f1"],
                    "test/recon_loss": test_loss_dict["recon_loss"],
                    "test/best_loss": best_test_results["avg_loss"],
                    "test/best_acc": best_test_results["avg_acc"],
                    "test/best_f1": best_test_results["f1"],
                    "test/best_recon_loss": best_test_results["recon_loss"],
                    "epoch": epoch,
                    "interpolation_loss": plt,
                }
                wandb.log(log)
                plt.close()

    if args.save_model:
        save_model(best_sd)


if __name__ == "__main__":
    parser = ArgumentParser("DEEP-ALIGN MLP matching trainer", parents=[common_parser])
    parser.set_defaults(
        data_path="",
        lr=5e-4,
        n_epochs=100,
        batch_size=32,
    )
    parser.add_argument(
        "--image-data-path",
        type=str,
        help="image data path",
    )
    parser.add_argument(
        "--image-batch-size",
        type=int,
        default=32,
        help="image batch size",
    )
    parser.add_argument(
        "--loss",
        type=str,
        choices=["ce", "mse"],
        default="ce",
        help="loss func for perm",
    )
    parser.add_argument(
        "--recon-loss",
        type=str,
        choices=["l2", "lmc", "both"],
        default="both",
        help="reconstruction loss type",
    )
    parser.add_argument(
        "--optim",
        type=str,
        default="adamw",
        choices=["adam", "sgd", "adamw"],
        help="optimizer",
    )
    parser.add_argument(
        "--data-name",
        type=str,
        default="mnist",
        choices=["mnist", "cifar10"],
        help="dataset to use",
    )
    parser.add_argument("--num-workers", type=int, default=8, help="num workers")
    parser.add_argument(
        "--reduction",
        type=str,
        default="max",
        choices=["mean", "sum", "max", "attn"],
        help="reduction strategy",
    )
    parser.add_argument(
        "--dim-hidden",
        type=int,
        default=64,
        help="dim hidden layers",
    )
    parser.add_argument(
        "--n-hidden",
        type=int,
        default=4,
        help="num hidden layers",
    )
    parser.add_argument(
        "--output-features", type=int, default=128, help="output features"
    )
    parser.add_argument(
        "--n-fc-layers",
        type=int,
        default=1,
        help="num linear layers at each ff block",
    )
    parser.add_argument(
        "--set-layer",
        type=str,
        default="sab",
        choices=["sab", "ds"],
        help="set layer",
    )
    parser.add_argument(
        "--n-heads",
        type=int,
        default=8,
        help="number of attention heads",
    )
    parser.add_argument(
        "--statistics-path",
        type=str,
        default=None,
        help="path to dataset statistics",
    )
    parser.add_argument("--eval-every", type=int, default=5, help="eval every")
    parser.add_argument(
        "--normalize", type=str2bool, default=False, help="normalize data"
    )
    parser.add_argument("--do-rate", type=float, default=0.0, help="dropout rate")
    parser.add_argument(
        "--add-skip", type=str2bool, default=False, help="add skip connection"
    )
    parser.add_argument(
        "--add-layer-skip",
        type=str2bool,
        default=False,
        help="add per layer skip connection",
    )
    parser.add_argument(
        "--add-bn", type=str2bool, default=True, help="add batch norm layers"
    )
    parser.add_argument(
        "--save-model", type=str2bool, default=False, help="save model artifacts"
    )
    parser.add_argument(
        "--diagonal", type=str2bool, default=True, help="diagonal DWSNet"
    )
    parser.add_argument(
        "--hnp-setup", type=str2bool, default=True, help="HNP vs NP setup"
    )
    parser.add_argument(
        "--sanity", type=str2bool, default=False, help="sanity check using a network and its perm"
    )
    parser.add_argument(
        "--init-scale", type=float, default=1.0, help="scale for weight initialization"
    )
    parser.add_argument(
        "--init-off-diag-scale",
        type=float,
        default=1.0,
        help="scale for weight initialization",
    )
    parser.add_argument(
        "--input-dim-downsample",
        type=int,
        default=8,
        help="input downsampling dimension",
    )
    # loss options
    parser.add_argument(
        "--recon-loss-weight",
        type=float,
        default=1.0,
        help="Reconstruction loss weight",
    )
    parser.add_argument(
        "--supervised-loss-weight",
        type=float,
        default=1.0,
        help="Reconstruction loss weight",
    )
    parser.add_argument(
        "--n-sink",
        type=int,
        default=20,
        help="Num. Sink steps",
    )
    args = parser.parse_args()

    # set seed
    set_seed(args.seed)
    # wandb
    if args.wandb:
        name = (
            f"mlp_cls_trainer_{args.data_name}_lr_{args.lr}_bs_{args.batch_size}_seed_{args.seed}"
        )
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=name,
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.config.update(args)

    device = get_device(gpus=args.gpu)

    logging.info(f"Using {args.data_name} dataset")
    image_flatten_size = dict(mnist=28 * 28, cifar10=32 * 32 * 3)[args.data_name]

    main(
        path=args.data_path,
        epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        device=device,
    )
