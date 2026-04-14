"""Downstream datasets: EuroSAT, AID, NWPU-RESISC45 and BDD100K."""

from __future__ import annotations

from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .corruptions import apply_corruption


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_EVAL_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


class ImageFolderSplit(Dataset):
    """ImageFolder-style dataset over a single train / val / test split.

    Expected layout:
        root/
            class_a/*.jpg
            class_b/*.jpg
            ...
    """

    def __init__(self, root: str, transform=_EVAL_TRANSFORM) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"split directory not found: {self.root}")
        classes = sorted(d.name for d in self.root.iterdir() if d.is_dir())
        if not classes:
            raise RuntimeError(f"no class subdirectories under {self.root}")
        self.class_to_idx = {c: i for i, c in enumerate(classes)}
        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        self.samples: list[tuple[str, int]] = []
        for cls in classes:
            for p in sorted((self.root / cls).iterdir()):
                if p.suffix.lower() in exts:
                    self.samples.append((str(p), self.class_to_idx[cls]))
        self.transform = transform

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        return self.transform(img), label

    @property
    def num_classes(self) -> int:
        return len(self.class_to_idx)


class CorruptedTestSet(Dataset):
    """Wrap an ImageFolderSplit and apply a named corruption at fixed severity.

    Normalization is done post-corruption so the corruption operates in
    [0, 1] pixel space as in the paper.
    """

    def __init__(self, base: ImageFolderSplit, corruption: str, severity: int) -> None:
        self.base = base
        self.corruption = corruption
        self.severity = severity
        # Pull the underlying split transform apart so we can inject corruption.
        self.to_tensor = T.Compose([T.Resize((224, 224)), T.ToTensor()])
        self.normalize = T.Normalize(mean=_MEAN, std=_STD)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        path, label = self.base.samples[idx]
        img = Image.open(path).convert("RGB")
        x = self.to_tensor(img)
        x = apply_corruption(self.corruption, x, self.severity)
        return self.normalize(x), label


def build_linear_probe_loaders(
    dataset_root: str,
    batch_size: int = 256,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Return train/val/test DataLoaders for one downstream dataset.

    Expected layout:
        dataset_root/train, dataset_root/val, dataset_root/test
    """
    root = Path(dataset_root)
    out = {}
    for split in ("train", "val", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            raise FileNotFoundError(f"missing split: {split_dir}")
        ds = ImageFolderSplit(str(split_dir))
        out[split] = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
        )
    return out


def build_corruption_loader(
    dataset_root: str,
    corruption: str,
    severity: int,
    batch_size: int = 256,
    num_workers: int = 4,
) -> DataLoader:
    """Return a DataLoader for the test split with one corruption applied."""
    test_dir = Path(dataset_root) / "test"
    base = ImageFolderSplit(str(test_dir), transform=_EVAL_TRANSFORM)
    corrupted = CorruptedTestSet(base, corruption, severity)
    return DataLoader(
        corrupted,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
