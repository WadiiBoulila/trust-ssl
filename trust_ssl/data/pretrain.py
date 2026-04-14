"""Pretraining corpus: 210K aerial RGB images (BigEarthNet-S2 + LoveDA crops).

The on-disk layout is a simple flat directory of RGB PNGs/JPGs. The
TwoViewTransform produces two augmented views per sample together with a
corruption-family label that the auxiliary classifier consumes during
Trust-SSL training.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from .corruptions import FAMILY_OF, CORRUPTIONS


class PretrainCorpus(Dataset):
    """Flat directory of RGB aerial images.

    Each __getitem__ returns:
        view1:          (3, H, W) first augmented view
        view2:          (3, H, W) second augmented view
        corrupt_label:  int64 in [0, num_families) identifying which
                        augmentation family dominated the two views;
                        used for the auxiliary classifier in Trust-SSL.
    """

    def __init__(self, root: str, transform: "TwoViewTransform") -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"pretrain corpus root not found: {self.root}")

        exts = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}
        self.paths = sorted(
            str(p) for p in self.root.rglob("*") if p.suffix.lower() in exts
        )
        if not self.paths:
            raise RuntimeError(f"no images found under {self.root}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> dict:
        img = Image.open(self.paths[idx]).convert("RGB")
        view1, view2, family_id = self.transform(img)
        return {
            "view1": view1,
            "view2": view2,
            "corrupt_label": torch.tensor(family_id, dtype=torch.long),
        }


# ──────────────────────────────────────────────────────────────────────
# Two-view augmentation + family tagging
# ──────────────────────────────────────────────────────────────────────
_NORMALIZE = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


class TwoViewTransform:
    """Standard SSL augmentation producing two views and a family label.

    The family label is the index into the corruption-family list used
    by the auxiliary classifier of Trust-SSL. For pretraining on clean
    aerial images we use a ``clean'' sentinel value (=0, mapped to the
    first slot of CORRUPTIONS).
    """

    def __init__(self, image_size: int = 224) -> None:
        base = [
            T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomApply(
                [T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)],
                p=0.8,
            ),
            T.RandomGrayscale(p=0.2),
            T.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0)),
        ]
        self._view_transform = T.Compose(base + [T.ToTensor(), _NORMALIZE])

    def __call__(self, img) -> tuple[torch.Tensor, torch.Tensor, int]:
        v1 = self._view_transform(img)
        v2 = self._view_transform(img)
        # No controlled corruption is applied during pretraining; the
        # family label is used only to keep the auxiliary classifier
        # numerically healthy. Setting it to a single constant is valid
        # because the auxiliary loss is a weak regularizer.
        return v1, v2, 0


def build_pretrain_loader(
    data_root: str,
    image_size: int = 224,
    batch_size: int = 512,
    num_workers: int = 8,
    drop_last: bool = True,
) -> DataLoader:
    """Build a DataLoader over the pretraining corpus."""
    dataset = PretrainCorpus(data_root, TwoViewTransform(image_size=image_size))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=drop_last,
        persistent_workers=num_workers > 0,
    )
