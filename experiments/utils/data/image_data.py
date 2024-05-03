import random
from typing import List

import torch.utils.data
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST, CIFAR10, SVHN, FashionMNIST, CIFAR100, STL10
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
from torchvision.transforms import InterpolationMode


def get_svhn_dataloaders(path, batch_size, val_size=5000, train_size=None, num_workers=4):
    """Builds and returns Dataloader for SVHN dataset."""

    transform = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize(28),
        transforms.ToTensor(),
        transforms.Normalize((0.5, ), (0.5, )),
        # transforms.Normalize((0.1307,), (0.3081,)),  # same as mnist
    ])

    train_dataset = SVHN(root=path, download=False, transform=transform, split="train")
    val_dataset = SVHN(root=path, download=False, transform=transform, split="train")
    test_dataset = SVHN(root=path, download=False, transform=transform, split="test")

    train_indices, val_indices = train_test_split(range(len(train_dataset)), test_size=val_size)

    if train_size is not None:
        train_indices = train_indices[:train_size]

    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )
    return train_loader, val_loader, test_loader


def get_mnist_dataloaders(data_path, batch_size, val_size=0.1, num_workers=4):
    normalization = transforms.Normalize(
        (0.1307,), (0.3081,)
    )
    # define transforms
    trans = [transforms.ToTensor()]
    trans.append(normalization)
    transform = transforms.Compose(trans)

    dataset = MNIST(
        data_path,
        train=True,
        download=False,
        transform=transform,
    )

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=val_size, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # val
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # test
    test_loader = DataLoader(
        MNIST(
            data_path,
            train=False,
            download=False,
            transform=transform
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader, val_loader, test_loader


def get_fashion_mnist_dataloaders(data_path, batch_size, val_size=0.1, num_workers=4):
    normalization = transforms.Normalize(
        (0.5,), (0.5,)
    )
    # define transforms
    trans = [transforms.ToTensor()]
    trans.append(normalization)
    transform = transforms.Compose(trans)

    dataset = FashionMNIST(
        data_path,
        train=True,
        download=False,
        transform=transform,
    )

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=val_size, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # val
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # test
    test_loader = DataLoader(
        FashionMNIST(
            data_path,
            train=False,
            download=False,
            transform=transform
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader, val_loader, test_loader


def get_cifar100_dataloaders(data_path, batch_size, val_size=0.1, num_workers=4):
    normalization = transforms.Normalize(
        (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)
    )
    # define transforms
    trans = [transforms.ToTensor()]
    trans.append(normalization)
    transform = transforms.Compose(trans)

    dataset = CIFAR100(
        data_path,
        train=True,
        download=False,
        transform=transform,
    )

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=val_size, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # val
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # test
    test_loader = DataLoader(
        CIFAR100(
            data_path,
            train=False,
            download=False,
            transform=transform
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader, val_loader, test_loader


def get_stl10_dataloaders(data_path, batch_size, val_size=0.1, num_workers=4):
    transform = [
        transforms.RandomCrop(64),
        transforms.Resize(32, antialias=None),
        transforms.RandomRotation(20),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    transform = transforms.Compose(transform)

    test_transform = [
        transforms.Resize(32, antialias=None),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ]
    test_transform = transforms.Compose(test_transform)

    dataset = STL10(
        data_path,
        split="train",
        download=False,
        transform=transform,
    )

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=val_size, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # val
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # test
    test_loader = DataLoader(
        STL10(
            data_path,
            split="test",
            download=False,
            transform=test_transform
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader, val_loader, test_loader


class CIFAR10Subset(CIFAR10):
    def __init__(self, subset: List[int], indices=None, **kwargs):
        super().__init__(**kwargs)
        self.subset = subset
        if indices is None:
            indices = list(range(len(self.targets)))
        self.indices = indices
        self.aligned_indices = []
        self.set_classes(subset)

    def set_classes(self, subset):
        self.subset = subset
        assert max(subset) <= max(self.targets)
        assert min(subset) >= min(self.targets)

        self.aligned_indices = []
        for idx, label in enumerate(self.targets):
            if label in subset and idx in self.indices:
                self.aligned_indices.append(idx)

    def get_class_names(self):
        return [self.classes[i] for i in self.subset]

    def __len__(self):
        return len(self.aligned_indices)

    def __getitem__(self, item):
        return super().__getitem__(self.aligned_indices[item])


TRAIN_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

TEST_TRANSFORMS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

ROTATE_TRANSFORMS = transforms.Compose([
    transforms.RandomRotation(degrees=[-45, 45],),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

corruption = [
    transforms.RandomRotation(degrees=[-75, 75], interpolation=InterpolationMode.NEAREST),
    transforms.ColorJitter(brightness=0.4, hue=0.25),
    transforms.GaussianBlur(kernel_size=(3, 3)),
    transforms.RandomVerticalFlip(),
    transforms.RandomHorizontalFlip(),
    # transforms.AugMix(severity=3),
]


TRAIN_TRANSFORMS_C = transforms.Compose([
    *corruption,
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


TEST_TRANSFORMS_C = transforms.Compose([
    *corruption,
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])


def get_cifar10_rotate_dataset(path, subset=None, val_size=0.1):
    if subset is None:
        subset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    train_dataset = CIFAR10Subset(
        subset=subset,
        root=path,
        train=True,
        transform=ROTATE_TRANSFORMS
    )
    val_dataset = CIFAR10Subset(
        subset=subset,
        root=path,
        train=True,
        transform=ROTATE_TRANSFORMS
    )

    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_size)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    test_dataset = CIFAR10Subset(
        subset=subset,
        root=path,
        train=False,
        download=False,
        transform=ROTATE_TRANSFORMS
    )

    return train_dataset, val_dataset, test_dataset


def get_cifar10_dataset(path, subset=None, val_size=0.1, corrupt=False):
    if subset is None:
        subset = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

    train_dataset = CIFAR10Subset(
        subset=subset,
        root=path,
        train=True,
        transform=TRAIN_TRANSFORMS if not corrupt else TRAIN_TRANSFORMS_C
    )
    val_dataset = CIFAR10Subset(
        subset=subset,
        root=path,
        train=True,
        transform=TRAIN_TRANSFORMS if not corrupt else TRAIN_TRANSFORMS_C
    )

    indices = list(range(len(train_dataset)))
    train_indices, val_indices = train_test_split(indices, test_size=val_size)
    train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(val_dataset, val_indices)

    test_dataset = CIFAR10Subset(
        subset=subset,
        root=path,
        train=False,
        download=False,
        transform=TEST_TRANSFORMS if not corrupt else TEST_TRANSFORMS_C
    )

    return train_dataset, val_dataset, test_dataset


def get_cifar10_dataloaders(path, batch_size, val_size=0.1, corrupt=False, subset=None, num_workers=4):
    train_dataset, val_dataset, test_dataset = get_cifar10_dataset(path, subset=subset, val_size=val_size, corrupt=corrupt)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # val
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    return train_loader, val_loader, test_loader


def get_cifar10_rotate_dataloaders(path, batch_size, val_size=0.1, subset=None, num_workers=4):
    train_dataset, val_dataset, test_dataset = get_cifar10_rotate_dataset(path, subset=subset, val_size=val_size)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # val
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # test
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )
    return train_loader, val_loader, test_loader


class MultiChannelFMNIST(FashionMNIST):
    def __getitem__(self, item):
        image, label = super().__getitem__(item)
        return image.repeat(3, 1, 1), label


def get_fmnist_cnn_dataloaders(data_path, batch_size, val_size=0.1, num_workers=4):
    # define transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            (0.5,), (0.5,)
        ),
        transforms.Pad(2),
    ])

    dataset = MultiChannelFMNIST(
        data_path,
        train=True,
        download=False,
        transform=transform,
    )

    train_idx, val_idx = train_test_split(range(len(dataset)), test_size=val_size, random_state=42)

    train_dataset = torch.utils.data.Subset(dataset, train_idx)
    val_dataset = torch.utils.data.Subset(dataset, val_idx)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # val
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    # test
    test_loader = DataLoader(
        MultiChannelFMNIST(
            data_path,
            train=False,
            download=False,
            transform=transform
        ),
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=True
    )

    return train_loader, val_loader, test_loader
