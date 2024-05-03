import copy
import dataclasses
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import NamedTuple, Tuple, Union, List

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Subset

from experiments.utils import make_coordinates
from experiments.utils.utils import unfold_matrices
from nn.inr_models import INR


class Batch(NamedTuple):
    weights: Tuple
    biases: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights=tuple(w.to(device) for w in self.weights),
            biases=tuple(w.to(device) for w in self.biases),
            label=self.label.to(device),
        )

    def get_weight_shapes(self):
        weight_shapes = tuple(w.shape[:2] for w in self.weights)
        bias_shapes = tuple(b.shape[:1] for b in self.biases)
        return weight_shapes, bias_shapes

    def __len__(self):
        return len(self.weights[0])


class ImageBatch(NamedTuple):
    image: torch.Tensor
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(*[t.to(device) for t in self])

    def __len__(self):
        return len(self.image)


class MultiViewMatchingBatch(NamedTuple):
    """
    Batch class for the matching task
    """

    weights_view_0: Tuple
    biases_view_0: Tuple
    weights_view_1: Tuple
    biases_view_1: Tuple
    perm_weights_view_0: Tuple
    perm_biases_view_0: Tuple
    perm_weights_view_1: Tuple
    perm_biases_view_1: Tuple
    perms_view_0: Tuple
    perms_view_1: Tuple
    label: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights_view_0=tuple(w.to(device) for w in self.weights_view_0),
            biases_view_0=tuple(w.to(device) for w in self.biases_view_0),
            weights_view_1=tuple(w.to(device) for w in self.weights_view_1),
            biases_view_1=tuple(w.to(device) for w in self.biases_view_1),
            perm_weights_view_0=tuple(w.to(device) for w in self.perm_weights_view_0),
            perm_biases_view_0=tuple(w.to(device) for w in self.perm_biases_view_0),
            perm_weights_view_1=tuple(w.to(device) for w in self.perm_weights_view_1),
            perm_biases_view_1=tuple(w.to(device) for w in self.perm_biases_view_1),
            perms_view_0=tuple(w.to(device) for w in self.perms_view_0),
            perms_view_1=tuple(w.to(device) for w in self.perms_view_1),
            label=self.label.to(device) if isinstance(self.label, torch.Tensor) else self.label,
        )

    def get_weight_shapes(self):
        # assume we have batch as first dim
        weight_shapes = tuple(w.shape[1:3] for w in self.weights_view_0)
        bias_shapes = tuple(b.shape[1:2] for b in self.biases_view_0)
        return weight_shapes, bias_shapes

    def __len__(self):
        return len(self.weights_view_0[0])


class TwoCoefSineMatchingBatch(NamedTuple):
    """
    Batch class for the matching task
    """

    weights_view_0: Tuple
    biases_view_0: Tuple
    weights_view_1: Tuple
    biases_view_1: Tuple
    perm_weights_view_0: Tuple
    perm_biases_view_0: Tuple
    perm_weights_view_1: Tuple
    perm_biases_view_1: Tuple
    perms_view_0: Tuple
    perms_view_1: Tuple
    label_view_0: Union[torch.Tensor, int]
    label_view_1: Union[torch.Tensor, int]

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            weights_view_0=tuple(w.to(device) for w in self.weights_view_0),
            biases_view_0=tuple(w.to(device) for w in self.biases_view_0),
            weights_view_1=tuple(w.to(device) for w in self.weights_view_1),
            biases_view_1=tuple(w.to(device) for w in self.biases_view_1),
            perm_weights_view_0=tuple(w.to(device) for w in self.perm_weights_view_0),
            perm_biases_view_0=tuple(w.to(device) for w in self.perm_biases_view_0),
            perm_weights_view_1=tuple(w.to(device) for w in self.perm_weights_view_1),
            perm_biases_view_1=tuple(w.to(device) for w in self.perm_biases_view_1),
            perms_view_0=tuple(w.to(device) for w in self.perms_view_0),
            perms_view_1=tuple(w.to(device) for w in self.perms_view_1),
            label_view_0=self.label_view_0.to(device),
            label_view_1=self.label_view_1.to(device),
        )

    def get_weight_shapes(self):
        # assume we have batch as first dim
        weight_shapes = tuple(w.shape[1:3] for w in self.weights_view_0)
        bias_shapes = tuple(b.shape[1:2] for b in self.biases_view_0)
        return weight_shapes, bias_shapes

    def __len__(self):
        return len(self.weights_view_0[0])


class MultiViewCNNMatchingBatch(NamedTuple):
    """
    Batch class for the matching task
    """
    # sd_weights_view_0: Tuple
    # sd_biases_view_0: Tuple
    # sd_weights_view_1: Tuple
    # sd_biases_view_1: Tuple
    weights_view_0: Tuple
    biases_view_0: Tuple
    weights_view_1: Tuple
    biases_view_1: Tuple
    perm_weights_view_0: Tuple
    perm_biases_view_0: Tuple
    # perm_weights_view_1: Tuple
    # perm_biases_view_1: Tuple
    perms_view_0: Tuple
    # perms_view_1: Tuple
    label: Union[torch.Tensor, int]
    kernel_sizes: torch.Tensor
    unpadded_weights_view_0: Tuple
    unpadded_biases_view_0: Tuple
    unpadded_weights_view_1: Tuple
    unpadded_biases_view_1: Tuple
    weight_names: List[str]
    bias_names: List[str]
    original_weight_shapes: Tuple
    original_bias_shapes: Tuple

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            # sd_weights_view_0=tuple(w.to(device) for w in self.sd_weights_view_0),
            # sd_biases_view_0=tuple(w.to(device) for w in self.sd_biases_view_0),
            # sd_weights_view_1=tuple(w.to(device) for w in self.sd_weights_view_1),
            # sd_biases_view_1=tuple(w.to(device) for w in self.sd_biases_view_1),
            weights_view_0=tuple(w.to(device) for w in self.weights_view_0),
            biases_view_0=tuple(w.to(device) for w in self.biases_view_0),
            weights_view_1=tuple(w.to(device) for w in self.weights_view_1),
            biases_view_1=tuple(w.to(device) for w in self.biases_view_1),
            perm_weights_view_0=tuple(w.to(device) for w in self.perm_weights_view_0),
            perm_biases_view_0=tuple(w.to(device) for w in self.perm_biases_view_0),
            # perm_weights_view_1=tuple(w.to(device) for w in self.perm_weights_view_1),
            # perm_biases_view_1=tuple(w.to(device) for w in self.perm_biases_view_1),
            perms_view_0=tuple(w.to(device) for w in self.perms_view_0),
            # perms_view_1=tuple(w.to(device) for w in self.perms_view_1),
            label=self.label.to(device),
            kernel_sizes=self.kernel_sizes.to(device),
            unpadded_weights_view_0=tuple(w.to(device) for w in self.unpadded_weights_view_0),
            unpadded_biases_view_0=tuple(w.to(device) for w in self.unpadded_biases_view_0),
            unpadded_weights_view_1=tuple(w.to(device) for w in self.unpadded_weights_view_1),
            unpadded_biases_view_1=tuple(w.to(device) for w in self.unpadded_biases_view_1),
            weight_names=self.weight_names,
            bias_names=self.bias_names,
            original_weight_shapes=self.original_weight_shapes,
            original_bias_shapes=self.original_bias_shapes,
        )

    def get_weight_shapes(self):
        # assume we have batch as first dim
        weight_shapes = tuple(w.shape[1:3] for w in self.weights_view_0)
        bias_shapes = tuple(b.shape[1:2] for b in self.biases_view_0)
        input_features = self.weights_view_0[0].shape[-1]
        return weight_shapes, bias_shapes, input_features

    def __len__(self):
        return len(self.weights_view_0[0])


class MultiViewCNNMatchingBatchExtended(NamedTuple):
    """
    Batch class for the matching task
    """
    sd_weights_view_0: Tuple
    sd_biases_view_0: Tuple
    sd_weights_view_1: Tuple
    sd_biases_view_1: Tuple
    weights_view_0: Tuple
    biases_view_0: Tuple
    weights_view_1: Tuple
    biases_view_1: Tuple
    perm_weights_view_0: Tuple
    perm_biases_view_0: Tuple
    perms_view_0: Tuple
    label: Union[torch.Tensor, int]
    kernel_sizes: torch.Tensor
    unpadded_weights_view_0: Tuple
    unpadded_biases_view_0: Tuple
    unpadded_weights_view_1: Tuple
    unpadded_biases_view_1: Tuple
    weight_names: List[str]
    bias_names: List[str]
    original_weight_shapes: Tuple
    original_bias_shapes: Tuple

    def _assert_same_len(self):
        assert len(set([len(t) for t in self])) == 1

    def as_dict(self):
        return self._asdict()

    def to(self, device):
        """move batch to device"""
        return self.__class__(
            sd_weights_view_0=tuple(w.to(device) for w in self.sd_weights_view_0),
            sd_biases_view_0=tuple(w.to(device) for w in self.sd_biases_view_0),
            sd_weights_view_1=tuple(w.to(device) for w in self.sd_weights_view_1),
            sd_biases_view_1=tuple(w.to(device) for w in self.sd_biases_view_1),

            weights_view_0=tuple(w.to(device) for w in self.weights_view_0),
            biases_view_0=tuple(w.to(device) for w in self.biases_view_0),
            weights_view_1=tuple(w.to(device) for w in self.weights_view_1),
            biases_view_1=tuple(w.to(device) for w in self.biases_view_1),

            perm_weights_view_0=tuple(w.to(device) for w in self.perm_weights_view_0),
            perm_biases_view_0=tuple(w.to(device) for w in self.perm_biases_view_0),

            perms_view_0=tuple(w.to(device) for w in self.perms_view_0),

            label=self.label.to(device),
            kernel_sizes=self.kernel_sizes.to(device),

            unpadded_weights_view_0=tuple(w.to(device) for w in self.unpadded_weights_view_0),
            unpadded_biases_view_0=tuple(w.to(device) for w in self.unpadded_biases_view_0),
            unpadded_weights_view_1=tuple(w.to(device) for w in self.unpadded_weights_view_1),
            unpadded_biases_view_1=tuple(w.to(device) for w in self.unpadded_biases_view_1),

            weight_names=self.weight_names,
            bias_names=self.bias_names,

            original_weight_shapes=self.original_weight_shapes,
            original_bias_shapes=self.original_bias_shapes,
        )

    def get_weight_shapes(self):
        # assume we have batch as first dim
        weight_shapes = tuple(w.shape[1:3] for w in self.weights_view_0)
        bias_shapes = tuple(b.shape[1:2] for b in self.biases_view_0)
        input_features = self.weights_view_0[0].shape[-1]
        return weight_shapes, bias_shapes, input_features

    def __len__(self):
        return len(self.weights_view_0[0])


class INRDataset(torch.utils.data.Dataset):
    """Base Dataset class for INR classification/regression"""

    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path=None,
        translation_scale=0.25,
        rotation_degree=45,
        noise_scale=1e-2,
        drop_rate=1e-2,
        resize_scale=0.2,
        pos_scale=0.0,
        quantile_dropout=0.0,
    ):
        self.split = split
        self.dataset = json.load(open(path, "r"))[self.split]

        self.augmentation = augmentation
        self.permutation = permutation
        self.normalize = normalize
        if self.normalize:
            assert statistics_path is not None
            self.stats = torch.load(statistics_path, map_location="cpu")

        self.translation_scale = translation_scale
        self.rotation_degree = rotation_degree
        self.noise_scale = noise_scale
        self.drop_rate = drop_rate
        self.resize_scale = resize_scale
        self.pos_scale = pos_scale
        self.quantile_dropout = quantile_dropout

    def __len__(self):
        return len(self.dataset)

    def _normalize(self, weights, biases):
        wm, ws = self.stats["weights"]["mean"], self.stats["weights"]["std"]
        bm, bs = self.stats["biases"]["mean"], self.stats["biases"]["std"]

        weights = tuple((w - m) / s for w, m, s in zip(weights, wm, ws))
        biases = tuple((w - m) / s for w, m, s in zip(biases, bm, bs))

        return weights, biases

    @staticmethod
    def rotation_mat(degree=30.0):
        angle = torch.empty(1).uniform_(-degree, degree)
        angle_rad = angle * (torch.pi / 180)
        rotation_matrix = torch.tensor(
            [
                [torch.cos(angle_rad), -torch.sin(angle_rad)],
                [torch.sin(angle_rad), torch.cos(angle_rad)],
            ]
        )
        return rotation_matrix

    def _augment(self, weights, biases):
        """translation and rotation

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)
        # translation
        if self.translation_scale > 0:
            translation = torch.empty(weights[0].shape[0]).uniform_(
                -self.translation_scale, self.translation_scale
            )
            order = random.sample(range(1, len(weights)), 1)[0]
            bias_res = translation
            i = 0
            for i in range(order):
                bias_res = bias_res @ weights[i]

            new_biases[i] += bias_res

        # rotation
        if self.rotation_degree > 0:
            if new_weights[0].shape[0] == 2:
                rot_mat = self.rotation_mat(self.rotation_degree)
                new_weights[0] = rot_mat @ new_weights[0]

        # noise
        new_weights = [
            w + torch.empty(w.shape).normal_(
                0, self.noise_scale * (torch.nan_to_num(w.std()) + 1e-8)
            )
            for w in new_weights
        ]
        new_biases = [
            b + torch.empty(b.shape).normal_(
                0, self.noise_scale * (torch.nan_to_num(b.std()) + 1e-8)
            )
            for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        # scale
        # todo: can also apply to deeper layers
        rand_scale = 1 + (torch.rand(1).item() - 0.5) * 2 * self.resize_scale
        new_weights[0] = new_weights[0] * rand_scale

        # positive scale
        if self.pos_scale > 0:
            for i in range(len(new_weights) - 1):
                # todo: we do a lot of duplicated stuff here
                out_dim = new_biases[i].shape[0]
                scale = torch.from_numpy(
                    np.random.uniform(
                        1 - self.pos_scale, 1 + self.pos_scale, out_dim
                    ).astype(np.float32)
                )
                inv_scale = 1.0 / scale
                new_weights[i] = new_weights[i] * scale
                new_biases[i] = new_biases[i] * scale
                new_weights[i + 1] = (new_weights[i + 1].T * inv_scale).T

        if self.quantile_dropout > 0:
            do_q = torch.empty(1).uniform_(0, self.quantile_dropout)
            q = torch.quantile(
                torch.cat([v.flatten().abs() for v in new_weights + new_biases]), q=do_q
            )
            new_weights = [torch.where(w.abs() < q, 0, w) for w in new_weights]
            new_biases = [torch.where(w.abs() < q, 0, w) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _permute(weights, biases, return_permutation=False):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        if return_permutation:
            return tuple(new_weights), tuple(new_biases), tuple(perms)
        else:
            return tuple(new_weights), tuple(new_biases)

    def __getitem__(self, item):
        path = self.dataset[item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = state_dict.pop("label")

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        return Batch(weights=weights, biases=biases, label=label)


class INRStateDataset(INRDataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.25,
        rotation_degree=45,
        noise_scale=1e-1,
        drop_rate=1e-2,
        resize_scale=0.2,
        pos_scale=0.0,
    ):
        super().__init__(
            path,
            split,
            normalize,
            augmentation,
            permutation,
            statistics_path,
            translation_scale,
            rotation_degree,
            noise_scale,
            drop_rate=drop_rate,
            resize_scale=resize_scale,
            pos_scale=pos_scale,
        )

    def __getitem__(self, item):
        path = self.dataset[item]
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = state_dict.pop("label")
        layer_names, layer_params = list(state_dict.keys()), list(state_dict.values())
        return layer_names, layer_params, label, path


class INRImageDataset(INRDataset):
    """Dataset for image reconstruction from INRs"""

    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=False,
        permutation=False,
        statistics_path="dataset/statistics.pth",
        translation_scale=0.25,
        rotation_degree=45,
        drop_rate=1e-2,
        noise_scale=1e-1,
        resize_scale=0.2,
        pos_scale=0.0,
        inr_class=INR,
        inr_kwargs={"n_layers": 3, "in_dim": 2, "up_scale": 16},
        image_size=(28, 28),
    ):
        super().__init__(
            path,
            split,
            normalize,
            augmentation,
            permutation,
            statistics_path,
            translation_scale,
            rotation_degree,
            noise_scale,
            drop_rate=drop_rate,
            resize_scale=resize_scale,
            pos_scale=pos_scale,
        )
        self.inr_class = inr_class
        self.inr_kwargs = inr_kwargs
        self.image_size = image_size

    def __getitem__(self, item):
        path = self.dataset[item]
        state_dict = torch.load(path, map_location="cpu")
        label = state_dict.pop("label")

        weight_names = [k for k in state_dict.keys() if "weight" in k]
        bias_names = [k for k in state_dict.keys() if "bias" in k]

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        if self.augmentation:
            weights, biases = self._augment(weights, biases)

        # add feature dim
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            weights, biases = self._normalize(weights, biases)

        if self.permutation:
            weights, biases = self._permute(weights, biases)

        new_state_dict = {}
        for i, k in enumerate(weight_names):
            new_state_dict[k] = weights[i].squeeze(-1).permute(1, 0)

        for i, k in enumerate(bias_names):
            new_state_dict[k] = biases[i].squeeze(-1)

        inr = self.inr_class(**self.inr_kwargs)
        inr.load_state_dict(new_state_dict)
        inr.eval()
        input = make_coordinates(self.image_size, 1)
        with torch.no_grad():
            image = inr(input)
            image = image.view(*self.image_size, -1)
            image = image.permute(2, 0, 1)
        return ImageBatch(image=image, label=label)


class MultiViewMatchingINRDataset(INRDataset):
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=True,  # NOTE: always true, it's here for compatibility
        permutation=True,  # NOTE: always true, it's here for compatibility
        statistics_path=None,
        translation_scale=0.25,
        rotation_degree=45,
        noise_scale=1e-2,
        drop_rate=5e-2,
        resize_scale=0.2,
        pos_scale=0.0,
        quantile_dropout=0.1,
        same_object_pct=1.
    ):
        super().__init__(
            path=path,
            split=split,
            normalize=normalize,
            augmentation=augmentation,
            permutation=permutation,
            statistics_path=statistics_path,
            translation_scale=translation_scale,
            rotation_degree=rotation_degree,
            noise_scale=noise_scale,
            drop_rate=drop_rate,
            resize_scale=resize_scale,
            pos_scale=pos_scale,
            quantile_dropout=quantile_dropout,
        )
        self.same_object_pct = same_object_pct
        self.model_to_views = defaultdict(list)

        # NOTE: this is specific for multi-view datasets like the one in:
        # /cortex/data/c-nets/inrs/artifacts/fmnist_many_views_splits.json
        # todo: make it more general
        for path in self.dataset:
            model_name = path.split("/")[-1]
            self.model_to_views[model_name].append(path)
        self.models = list(self.model_to_views.keys())

    def __len__(self):
        return len(self.models)

    def _prepare_data(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = state_dict.pop("label")

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        orig_weights = weights
        orig_biases = biases
        # augmentation
        weights, biases = self._augment(weights, biases)
        if self.augmentation:
            # todo: if self.augmentation is true, maybe also aug. the original weights and biases?
            pass

        # add feature dim
        orig_weights = tuple([w.unsqueeze(-1) for w in orig_weights])
        orig_biases = tuple([b.unsqueeze(-1) for b in orig_biases])
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            orig_weights, orig_biases = self._normalize(orig_weights, orig_biases)
            weights, biases = self._normalize(weights, biases)

        # if self.permutation:
        weights, biases, perms = self._permute(weights, biases, return_permutation=True)

        return dict(
            weights=orig_weights,
            biases=orig_biases,
            # augmented and permuted
            perm_weights=weights,
            perm_biases=biases,
            perms=perms,
            label=label,
        )

    def __getitem__(self, item):
        model_name = self.models[item]
        # sample two views
        if self.same_object_pct >= np.random.rand():
            if len(self.model_to_views[model_name]) >= 2:
                path0, path1 = random.sample(self.model_to_views[model_name], 2)
            else:
                print(f"Only a single view is available for {model_name}")
                # raise f"Only a single view is available for {model_name}"
                path0 = path1 = self.model_to_views[model_name][0]
        else:
            path0 = random.choice(self.model_to_views[model_name])
            model1 = random.choice(self.models)
            path1 = random.choice(self.model_to_views[model1])

        data_dict_0 = self._prepare_data(path0)
        data_dict_1 = self._prepare_data(path1)

        return MultiViewMatchingBatch(
            # orig
            weights_view_0=data_dict_0["weights"],
            biases_view_0=data_dict_0["biases"],
            weights_view_1=data_dict_1["weights"],
            biases_view_1=data_dict_1["biases"],
            # augmented and permuted
            perm_weights_view_0=data_dict_0["perm_weights"],
            perm_biases_view_0=data_dict_0["perm_biases"],
            perm_weights_view_1=data_dict_1["perm_weights"],
            perm_biases_view_1=data_dict_1["perm_biases"],
            # permutation labels
            perms_view_0=data_dict_0["perms"],
            perms_view_1=data_dict_1["perms"],
            # class label from first view
            label=data_dict_0["label"],
        )


class MatchingModelsDataset(INRDataset):
    """Dataset of models (e.g., classifiers)

    """
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=True,  # NOTE: always true, it's here for compatibility
        permutation=True,  # NOTE: always true, it's here for compatibility
        statistics_path=None,
        noise_scale=1e-2,
        drop_rate=5e-2,
        pos_scale=0.1,
        quantile_dropout=0.1,
    ):
        super().__init__(
            path=path,
            split=split,
            normalize=normalize,
            augmentation=augmentation,
            permutation=permutation,
            statistics_path=statistics_path,
            translation_scale=0.,
            rotation_degree=0,
            noise_scale=noise_scale,
            drop_rate=drop_rate,
            resize_scale=0.,
            pos_scale=pos_scale,
            quantile_dropout=quantile_dropout,
        )

    def __len__(self):
        return len(self.dataset)

    def _prepare_data(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = state_dict.pop("label")
        label = label["test_loss"]  # todo: maybe we need something more generic? Or no label?

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        orig_weights = weights
        orig_biases = biases
        # augmentation
        weights, biases = self._augment(weights, biases)
        if self.augmentation:
            # todo: if self.augmentation is true, maybe also aug. the original weights and biases?
            pass

        # add feature dim
        orig_weights = tuple([w.unsqueeze(-1) for w in orig_weights])
        orig_biases = tuple([b.unsqueeze(-1) for b in orig_biases])
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            orig_weights, orig_biases = self._normalize(orig_weights, orig_biases)
            weights, biases = self._normalize(weights, biases)

        # if self.permutation:
        weights, biases, perms = self._permute(weights, biases, return_permutation=True)

        return dict(
            weights=orig_weights,
            biases=orig_biases,
            # augmented and permuted
            perm_weights=weights,
            perm_biases=biases,
            perms=perms,
            label=label,
        )

    def __getitem__(self, item):
        path0 = self.dataset[item]
        # sample index
        path1 = random.choice(self.dataset)

        data_dict_0 = self._prepare_data(path0)
        data_dict_1 = self._prepare_data(path1)

        return MultiViewMatchingBatch(
            # orig
            weights_view_0=data_dict_0["weights"],
            biases_view_0=data_dict_0["biases"],
            weights_view_1=data_dict_1["weights"],
            biases_view_1=data_dict_1["biases"],
            # augmented and permuted
            perm_weights_view_0=data_dict_0["perm_weights"],
            perm_biases_view_0=data_dict_0["perm_biases"],
            perm_weights_view_1=data_dict_1["perm_weights"],
            perm_biases_view_1=data_dict_1["perm_biases"],
            # permutation labels
            perms_view_0=data_dict_0["perms"],
            perms_view_1=data_dict_1["perms"],
            # class label from first view
            label=data_dict_0["label"],  # todo: add label for seconds view (do this also for the INR datasets?)
        )


class MatchingCNNModelsDataset(INRDataset):
    """Dataset of CNN models (e.g., classifiers)

    """
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=True,  # NOTE: always true, it's here for compatibility
        permutation=True,  # NOTE: always true, it's here for compatibility
        statistics_path=None,
        noise_scale=1e-2,
        drop_rate=5e-2,
        pos_scale=0.1,
        quantile_dropout=0.1,
    ):
        super().__init__(
            path=path,
            split=split,
            normalize=normalize,
            augmentation=augmentation,
            permutation=permutation,
            statistics_path=statistics_path,
            translation_scale=0.,
            rotation_degree=0,
            noise_scale=noise_scale,
            drop_rate=drop_rate,
            resize_scale=0.,
            pos_scale=pos_scale,
            quantile_dropout=quantile_dropout,
        )

    def __len__(self):
        return len(self.dataset)

    def _augment(self, weights, biases):
        """translation and rotation

        :param weights:
        :param biases:
        :return:
        """
        new_weights, new_biases = list(weights), list(biases)

        # noise
        new_weights = [
            w + torch.empty(w.shape).normal_(
                0, self.noise_scale * (torch.nan_to_num(w.std()) + 1e-8)
            )
            for w in new_weights
        ]
        new_biases = [
            b + torch.empty(b.shape).normal_(
                0, self.noise_scale * (torch.nan_to_num(b.std()) + 1e-8)
            )
            for b in new_biases
        ]

        # dropout
        new_weights = [F.dropout(w, p=self.drop_rate) for w in new_weights]
        new_biases = [F.dropout(w, p=self.drop_rate) for w in new_biases]

        # positive scale
        if self.pos_scale > 0 and False:  # todo: need to somehow deal with the transition between Conv and FC layers
            for i in range(len(new_weights) - 1):
                # todo: we do a lot of duplicated stuff here
                out_dim = new_biases[i].shape[0]
                scale = torch.from_numpy(
                    np.random.uniform(
                        1 - self.pos_scale, 1 + self.pos_scale, out_dim
                    ).astype(np.float32)
                )
                inv_scale = 1.0 / scale
                new_weights[i] = (new_weights[i].permute(0, 2, 1) * scale).permute(0, 2, 1)
                new_biases[i] = (new_biases[i].permute(1, 0) * scale).permute(1, 0)
                new_weights[i + 1] = (new_weights[i + 1].T * inv_scale).T

        if self.quantile_dropout > 0:
            do_q = torch.empty(1).uniform_(0, self.quantile_dropout)
            q = torch.quantile(
                torch.cat([v.flatten().abs() for v in new_weights + new_biases]), q=do_q
            )
            new_weights = [torch.where(w.abs() < q, 0, w) for w in new_weights]
            new_biases = [torch.where(w.abs() < q, 0, w) for w in new_biases]

        return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _permute(weights, biases, return_permutation=False):
        new_weights = [None] * len(weights)
        new_biases = [None] * len(biases)
        assert len(weights) == len(biases)

        perms = []
        for i, w in enumerate(weights):
            if i != len(weights) - 1:
                perms.append(torch.randperm(w.shape[1]))

        for i, (w, b) in enumerate(zip(weights, biases)):
            if i == 0:
                new_weights[i] = w[:, perms[i], :]
                new_biases[i] = b[perms[i], :]
            elif i == len(weights) - 1:
                new_weights[i] = w[perms[-1], :, :]
                new_biases[i] = b
            else:
                # if w.shape[0] != len(perms[i - 1]):  # NOTE: patch to handle maxpool
                #     assert w.shape[0] % len(perms[i - 1]) == 0
                #     expand = w.shape[0] // len(perms[i - 1])
                #     new_perm = torch.cat([perms[i-1] + (j * len(perms[i-1])) for j in range(expand)])
                #     new_weights[i] = w[new_perm, :, :][:, perms[i], :]
                #     new_biases[i] = b[perms[i], :]
                # else:

                new_weights[i] = w[perms[i - 1], :, :][:, perms[i], :]
                new_biases[i] = b[perms[i], :]
        if return_permutation:
            return tuple(new_weights), tuple(new_biases), tuple(perms)
        else:
            return tuple(new_weights), tuple(new_biases)

    @staticmethod
    def _padding(to_pad, length):
        padding = (0, length - to_pad.size(-1))
        # Pad the tensor along the last dimension with zeros
        return torch.nn.functional.pad(to_pad, padding)

    def _prepare_data(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)

        sd_weights = [v for k, v in state_dict.items() if "weight" in k]
        sd_biases = [v for k, v in state_dict.items() if "bias" in k]

        weight_names = [w for w in state_dict.keys() if "weight" in w]
        bias_names = [b for b in state_dict.keys() if "bias" in b]

        original_weight_shapes = [list(w.shape) for n, w in state_dict.items() if "weight" in n]
        original_bias_shapes = [list(b.shape) for n, b in state_dict.items() if "bias" in n]

        # weights are of the form (out channels, in channels, k, k)
        weights = [v.transpose(1, 0) for w, v in state_dict.items() if "weight" in w]

        kernel_sizes = [v.shape[2] if len(v.shape) == 4 else -1 for v in weights]
        weights = [v.flatten(start_dim=2) if len(v.shape) == 4 else v.unsqueeze(-1) for v in weights]

        # un-padded weights
        # unpadded_weights = tuple(copy.deepcopy(weights))

        # if we have convolutions, it is possible there is a misalignment between last conv output and first
        # fully connected input. If so we need to fold the FC layer.
        for i in range(len(weights) - 1):
            if weights[i].shape[1] != weights[i + 1].shape[0]:
                weight_shapes = weights[i + 1].shape
                conv_out_shape = weights[i].shape[1]
                assert weight_shapes[0] % conv_out_shape == 0
                expand = weight_shapes[0] // conv_out_shape
                weights[i + 1] = weights[i + 1].squeeze(-1).reshape(conv_out_shape, expand, weight_shapes[1]).permute(0, 2, 1)
                break

        # todo: we move it here so the permute_weights function would work...
        unpadded_weights = tuple(copy.deepcopy(weights))

        max_features_len = max([w.shape[-1] for w in weights])
        weights = tuple([self._padding(v, max_features_len) for v in weights])

        # biases are of the form (out channels)
        biases = [v.unsqueeze(-1) for w, v in state_dict.items() if "bias" in w]
        # un-padded biases
        unpadded_biases = tuple(copy.deepcopy(biases))
        biases = tuple([self._padding(b, max_features_len) for b in biases])

        orig_weights = weights
        orig_biases = biases
        # augmentation
        weights, biases = self._augment(weights, biases)
        if self.augmentation:
            # todo: if self.augmentation is true, maybe also aug. the original weights and biases?
            pass

        if self.normalize:
            orig_weights, orig_biases = self._normalize(orig_weights, orig_biases)
            weights, biases = self._normalize(weights, biases)

        # if self.permutation:
        weights, biases, perms = self._permute(weights, biases, return_permutation=True)

        return dict(
            sd_weights=sd_weights,
            sd_biases=sd_biases,
            weights=orig_weights,
            biases=orig_biases,
            # augmented and permuted
            perm_weights=weights,
            perm_biases=biases,
            perms=perms,
            label=-1,  # todo: just for compatibility
            kernel_sizes=kernel_sizes,
            unpadded_weights=unpadded_weights,
            unpadded_biases=unpadded_biases,
            weight_names=weight_names,
            bias_names=bias_names,
            original_weight_shapes=original_weight_shapes,
            original_bias_shapes=original_bias_shapes,
        )

    def __getitem__(self, item):
        path0 = self.dataset[item]
        # sample index
        path1 = random.choice(self.dataset)

        data_dict_0 = self._prepare_data(path0)
        data_dict_1 = self._prepare_data(path1)

        return MultiViewCNNMatchingBatch(
            # state dict
            # sd_weights_view_0=data_dict_0["sd_weights"],
            # sd_biases_view_0=data_dict_0["sd_biases"],
            # sd_weights_view_1=data_dict_1["sd_weights"],
            # sd_biases_view_1=data_dict_1["sd_biases"],
            # orig
            weights_view_0=data_dict_0["weights"],
            biases_view_0=data_dict_0["biases"],
            weights_view_1=data_dict_1["weights"],
            biases_view_1=data_dict_1["biases"],
            # augmented and permuted
            perm_weights_view_0=data_dict_0["perm_weights"],
            perm_biases_view_0=data_dict_0["perm_biases"],
            # perm_weights_view_1=data_dict_1["perm_weights"],
            # perm_biases_view_1=data_dict_1["perm_biases"],
            # permutation labels
            perms_view_0=data_dict_0["perms"],
            # perms_view_1=data_dict_1["perms"],
            # class label from first view
            label=data_dict_0["label"],  # todo: add label for seconds view (do this also for the INR datasets?)
            kernel_sizes=torch.tensor(data_dict_0["kernel_sizes"]),
            # unpadded
            unpadded_weights_view_0=data_dict_0["unpadded_weights"],
            unpadded_biases_view_0=data_dict_0["unpadded_biases"],
            unpadded_weights_view_1=data_dict_1["unpadded_weights"],
            unpadded_biases_view_1=data_dict_1["unpadded_biases"],
            weight_names=data_dict_0["weight_names"],
            bias_names=data_dict_0["bias_names"],
            # shapes
            original_weight_shapes=data_dict_0["original_weight_shapes"],
            original_bias_shapes=data_dict_0["original_bias_shapes"],
        )


class MatchingCNNModelsDatasetExtended(MatchingCNNModelsDataset):
    def __getitem__(self, item):
        path0 = self.dataset[item]
        # sample index
        path1 = random.choice(self.dataset)

        data_dict_0 = self._prepare_data(path0)
        data_dict_1 = self._prepare_data(path1)

        return MultiViewCNNMatchingBatchExtended(
            # state dict
            sd_weights_view_0=data_dict_0["sd_weights"],
            sd_biases_view_0=data_dict_0["sd_biases"],
            sd_weights_view_1=data_dict_1["sd_weights"],
            sd_biases_view_1=data_dict_1["sd_biases"],

            # orig
            weights_view_0=data_dict_0["weights"],
            biases_view_0=data_dict_0["biases"],
            weights_view_1=data_dict_1["weights"],
            biases_view_1=data_dict_1["biases"],

            # augmented and permuted
            perm_weights_view_0=data_dict_0["perm_weights"],
            perm_biases_view_0=data_dict_0["perm_biases"],

            # permutation labels
            perms_view_0=data_dict_0["perms"],
            # perms_view_1=data_dict_1["perms"],

            # class label from first view
            label=data_dict_0["label"],  # todo: add label for seconds view (do this also for the INR datasets?)
            kernel_sizes=torch.tensor(data_dict_0["kernel_sizes"]),

            # unpadded
            unpadded_weights_view_0=data_dict_0["unpadded_weights"],
            unpadded_biases_view_0=data_dict_0["unpadded_biases"],
            unpadded_weights_view_1=data_dict_1["unpadded_weights"],
            unpadded_biases_view_1=data_dict_1["unpadded_biases"],
            weight_names=data_dict_0["weight_names"],
            bias_names=data_dict_0["bias_names"],

            # shapes
            original_weight_shapes=data_dict_0["original_weight_shapes"],
            original_bias_shapes=data_dict_0["original_bias_shapes"],
        )


class SineINRDataset(INRDataset):
    """This is specific for the 2 coef Sine-INR dataset from the DWS paper
    (/cortex/data/c-nets/model_embedding/artifacts/sine_2_coef_asinbx_splits.json).

    """
    def __init__(
        self,
        path,
        split="train",
        normalize=False,
        augmentation=True,  # NOTE: always true, it's here for compatibility
        permutation=True,  # NOTE: always true, it's here for compatibility
        statistics_path=None,
        translation_scale=0.5,
        rotation_degree=90,
        noise_scale=2e-1,
        drop_rate=2e-1,
        resize_scale=0.5,
        pos_scale=0.,
        quantile_dropout=.8,
    ):
        super().__init__(
            path=path,
            split=split,
            normalize=normalize,
            augmentation=augmentation,
            permutation=permutation,
            statistics_path=statistics_path,
            translation_scale=translation_scale,
            rotation_degree=rotation_degree,
            noise_scale=noise_scale,
            drop_rate=drop_rate,
            resize_scale=resize_scale,
            pos_scale=pos_scale,
            quantile_dropout=quantile_dropout,
        )
        self.model_to_views = defaultdict(list)

    def __len__(self):
        return len(self.dataset)

    def _prepare_data(self, path):
        state_dict = torch.load(path, map_location=lambda storage, loc: storage)
        label = state_dict.pop("coef")

        weights = tuple(
            [v.permute(1, 0) for w, v in state_dict.items() if "weight" in w]
        )
        biases = tuple([v for w, v in state_dict.items() if "bias" in w])

        orig_weights = weights
        orig_biases = biases
        # augmentation
        weights, biases = self._augment(weights, biases)
        if self.augmentation:
            # todo: if self.augmentation is true, maybe also aug. the original weights and biases?
            pass

        # add feature dim
        orig_weights = tuple([w.unsqueeze(-1) for w in orig_weights])
        orig_biases = tuple([b.unsqueeze(-1) for b in orig_biases])
        weights = tuple([w.unsqueeze(-1) for w in weights])
        biases = tuple([b.unsqueeze(-1) for b in biases])

        if self.normalize:
            orig_weights, orig_biases = self._normalize(orig_weights, orig_biases)
            weights, biases = self._normalize(weights, biases)

        # if self.permutation:
        weights, biases, perms = self._permute(weights, biases, return_permutation=True)

        return dict(
            weights=orig_weights,
            biases=orig_biases,
            # augmented and permuted
            perm_weights=weights,
            perm_biases=biases,
            perms=perms,
            label=torch.tensor(label),
        )

    def __getitem__(self, item):
        path0 = self.dataset[item]
        # sample index
        path1 = random.choice(self.dataset)

        data_dict_0 = self._prepare_data(path0)
        data_dict_1 = self._prepare_data(path1)

        return TwoCoefSineMatchingBatch(
            # orig
            weights_view_0=data_dict_0["weights"],
            biases_view_0=data_dict_0["biases"],
            weights_view_1=data_dict_1["weights"],
            biases_view_1=data_dict_1["biases"],
            # augmented and permuted
            perm_weights_view_0=data_dict_0["perm_weights"],
            perm_biases_view_0=data_dict_0["perm_biases"],
            perm_weights_view_1=data_dict_1["perm_weights"],
            perm_biases_view_1=data_dict_1["perm_biases"],
            # permutation labels
            perms_view_0=data_dict_0["perms"],
            perms_view_1=data_dict_1["perms"],
            # class label from first view
            label_view_0=data_dict_0["label"],
            label_view_1=data_dict_0["label"],
        )


if __name__ == '__main__':
    from experiments.cnn_image_classifier.models import BatchFunctionalCNN, FmnistCNN
    from experiments.utils.data.image_data import get_fashion_mnist_dataloaders, get_cifar10_dataloaders


    test = MatchingCNNModelsDataset(
        # "/Users/avivnavon/Desktop/datasets/tune_zoo_f_mnist_uniform_splits.json",
        "/Users/avivshamsian/Desktop/data/cifar_cls/split.json",
        split="test",
        normalize=False,
        noise_scale=0,
        drop_rate=0,
        pos_scale=0,
        quantile_dropout=0,
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    batch = next(iter(test_loader))
    model = BatchFunctionalCNN(act="gelu")

    # _, _, test_loader = get_fashion_mnist_dataloaders(
    #     data_path="/Users/avivnavon/Desktop/datasets", batch_size=128, num_workers=0
    # )
    _, _, test_loader = get_cifar10_dataloaders(
        "/Users/avivnavon/Desktop/datasets/", batch_size=128, num_workers=0
    )
    image_batch = next(iter(test_loader))
    image, label = image_batch

    # unfolded_weights, unfolded_biases = unfold_matrices(
    #     batch.unpadded_weights_view_0,
    #     batch.unpadded_biases_view_0,
    #     batch.original_weight_shapes,
    #     batch.original_bias_shapes,
    # )

    out = model(
        x=image,
        weights_and_biases=(batch.unpadded_weights_view_0, batch.unpadded_biases_view_0)
    )

    # out is (bs models, bs images, classes)
    preds = out.argmax(-1).permute(1, 0)  # (bs_image, bs_model)
    labels = label.unsqueeze(1).repeat(1, out.size(0))  # (bs_image, bs_model)

    print(f"Accuracy: {preds.eq(labels).float().mean()}")

    # cnn_model = FmnistCNN()
    # # sd = torch.load(
    # #     "/Users/avivnavon/Desktop/datasets/tune_zoo_f_mnist_uniform/NN_tune_trainable_ff43b_00097_97_seed=98_2021-07-12_12-44-44/checkpoint_000050/checkpoints"
    # # )
    # sd = torch.load(
    #     "/Users/avivnavon/Desktop/datasets/tune_zoo_cifar10_uniform_large/NN_tune_trainable_da045_00028_28_seed=29_2021-09-25_13-17-06/checkpoint_000050/checkpoints"
    # )
    # cnn_model.load_state_dict(sd)
    # cnn_out = cnn_model(image)
    # cnn_acc = cnn_out.argmax(-1).eq(label).float().mean()
    # print(f"CNN Accuracy: {cnn_acc}")

