"""
Data loading utilities for the current VehicleTypeRecognition pipeline.

Canonical dataset root:
    data/augmented/
      train/<class>/<bucket>/           # training only
      valid_unseen/<class>/             # primary validation
      valid_traincopy/<class>/          # auxiliary check only, optional
      test/<class>/                     # official final evaluation

Images are expected to be preprocessed offline by the base pipeline:
resize with preserved aspect ratio, then zero-pad to 224x224. No morphology is
applied in this module.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import v2


IMAGENET_MEAN: List[float] = [0.485, 0.456, 0.406]
IMAGENET_STD: List[float] = [0.229, 0.224, 0.225]

NUM_CLASSES: int = 10
PRIMARY_VALID_SPLIT = "valid_unseen"
AUX_VALID_SPLIT = "valid_traincopy"
TEST_SPLIT = "test"
TRAIN_SPLIT = "train"


def get_transforms(split: str) -> v2.Compose:
    """
    Return the tensor/normalization transform for an already-preprocessed split.

    Offline preprocessing owns resize and zero padding. Training images can have
    quota-based offline augmentation; validation and test images are not
    augmented here.
    """
    _ = split.lower().strip()
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def get_mixup_cutmix(
    num_classes: int = NUM_CLASSES,
    cutmix_alpha: float = 1.0,
    mixup_alpha: float = 0.2,
) -> v2.RandomChoice:
    """Create the optional online MixUp/CutMix transform for training batches."""
    return v2.RandomChoice(
        [
            v2.CutMix(num_classes=num_classes, alpha=cutmix_alpha),
            v2.MixUp(num_classes=num_classes, alpha=mixup_alpha),
        ]
    )


def _build_mixup_collate_fn(
    mixup_cutmix: v2.RandomChoice,
) -> Callable[[List[Tuple[Tensor, int]]], Tuple[Tensor, Tensor]]:
    default_collate = torch.utils.data.default_collate

    def collate_fn(batch: List[Tuple[Tensor, int]]) -> Tuple[Tensor, Tensor]:
        images, labels = default_collate(batch)
        images, labels = mixup_cutmix(images, labels)
        return images, labels

    return collate_fn


def _loader_kwargs(
    num_workers: int,
    pin_memory: Optional[bool],
    prefetch_factor: Optional[int],
) -> Dict[str, object]:
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    kwargs: Dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers > 0 and prefetch_factor:
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def _require_split(data_dir: str | os.PathLike[str], split: str) -> Path:
    split_path = Path(data_dir) / split
    if not split_path.is_dir():
        raise FileNotFoundError(
            f"Missing required split '{split}': {split_path}. "
            "Run the data split and offline augmentation pipeline first."
        )
    return split_path


def _optional_split(data_dir: str | os.PathLike[str], split: str) -> Optional[Path]:
    split_path = Path(data_dir) / split
    return split_path if split_path.is_dir() else None


def _assert_same_classes(
    reference: datasets.ImageFolder,
    candidate: datasets.ImageFolder,
    split_name: str,
) -> None:
    if candidate.class_to_idx != reference.class_to_idx:
        raise RuntimeError(
            f"class_to_idx mismatch between train and {split_name}.\n"
            f"train: {reference.class_to_idx}\n"
            f"{split_name}: {candidate.class_to_idx}"
        )


def create_dataloaders(
    data_dir: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    mixup_alpha: float = 0.2,
    cutmix_alpha: float = 1.0,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, DataLoader, Optional[DataLoader], DataLoader, List[str]]:
    """
    Create DataLoaders for the canonical split strategy.

    Returns:
        train_loader: uses data/augmented/train only, with MixUp/CutMix collate.
        valid_unseen_loader: primary validation, used for early stopping.
        valid_traincopy_loader: auxiliary validation if the split exists.
        test_loader: official final evaluation split.
        class_names: class names in ImageFolder index order.
    """
    train_path = _require_split(data_dir, TRAIN_SPLIT)
    valid_unseen_path = _require_split(data_dir, PRIMARY_VALID_SPLIT)
    test_path = _require_split(data_dir, TEST_SPLIT)
    valid_traincopy_path = _optional_split(data_dir, AUX_VALID_SPLIT)

    train_ds = datasets.ImageFolder(root=str(train_path), transform=get_transforms(TRAIN_SPLIT))
    valid_unseen_ds = datasets.ImageFolder(
        root=str(valid_unseen_path),
        transform=get_transforms(PRIMARY_VALID_SPLIT),
    )
    test_ds = datasets.ImageFolder(root=str(test_path), transform=get_transforms(TEST_SPLIT))
    valid_traincopy_ds = (
        datasets.ImageFolder(root=str(valid_traincopy_path), transform=get_transforms(AUX_VALID_SPLIT))
        if valid_traincopy_path is not None
        else None
    )

    _assert_same_classes(train_ds, valid_unseen_ds, PRIMARY_VALID_SPLIT)
    _assert_same_classes(train_ds, test_ds, TEST_SPLIT)
    if valid_traincopy_ds is not None:
        _assert_same_classes(train_ds, valid_traincopy_ds, AUX_VALID_SPLIT)

    loader_args = _loader_kwargs(num_workers, pin_memory, prefetch_factor)
    train_collate_fn = _build_mixup_collate_fn(
        get_mixup_cutmix(
            num_classes=len(train_ds.classes),
            cutmix_alpha=cutmix_alpha,
            mixup_alpha=mixup_alpha,
        )
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=train_collate_fn,
        drop_last=True,
        **loader_args,
    )
    valid_unseen_loader = DataLoader(
        valid_unseen_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )
    valid_traincopy_loader = (
        DataLoader(
            valid_traincopy_ds,
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
            **loader_args,
        )
        if valid_traincopy_ds is not None
        else None
    )
    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **loader_args,
    )

    return train_loader, valid_unseen_loader, valid_traincopy_loader, test_loader, train_ds.classes


def create_eval_loader(
    data_dir: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 0,
    pin_memory: Optional[bool] = None,
    prefetch_factor: Optional[int] = None,
) -> Tuple[DataLoader, List[str]]:
    """Create a no-shuffle evaluation loader for one split."""
    split_path = _require_split(data_dir, split)
    ds = datasets.ImageFolder(root=str(split_path), transform=get_transforms(split))
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        **_loader_kwargs(num_workers, pin_memory, prefetch_factor),
    )
    return loader, ds.classes


def get_class_distribution(dataset: datasets.ImageFolder) -> Dict[str, int]:
    """Count images per class for an ImageFolder dataset."""
    dist: Dict[str, int] = {cls: 0 for cls in dataset.classes}
    for _, label_idx in dataset.samples:
        dist[dataset.classes[label_idx]] += 1
    return dict(sorted(dist.items()))


def describe_loader(loader: Optional[DataLoader], split_name: str) -> None:
    if loader is None:
        print(f"  {split_name:<15}: not available")
        return
    print(
        f"  {split_name:<15}: {len(loader.dataset):>8,} samples | "
        f"{len(loader):>5,} batches | batch_size={loader.batch_size}"
    )


if __name__ == "__main__":
    import argparse

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(
        description="Smoke test for the canonical training/validation/test loaders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data_dir", type=str, default="data/augmented")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=0)
    args = parser.parse_args()

    train_loader, valid_unseen_loader, valid_traincopy_loader, test_loader, class_names = (
        create_dataloaders(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    )

    print("VehicleTypeRecognition dataset smoke test")
    print(f"Classes ({len(class_names)}): {class_names}")
    describe_loader(train_loader, TRAIN_SPLIT)
    describe_loader(valid_unseen_loader, PRIMARY_VALID_SPLIT)
    describe_loader(valid_traincopy_loader, AUX_VALID_SPLIT)
    describe_loader(test_loader, TEST_SPLIT)

    images, labels = next(iter(train_loader))
    print(f"Train batch images: {tuple(images.shape)}")
    print(f"Train batch labels: {tuple(labels.shape)}")
