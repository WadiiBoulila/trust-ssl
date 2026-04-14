"""Controlled K–I trajectory analysis.

Reproduces the experiment behind Figure 4 in the paper. For each of
the nine corruption types, we form N clean/corrupted pairs at severity
s ∈ {1,...,5}, push them through the trained Trust-SSL encoder, and
record the mean per-factor conflict K and ignorance I. Results are
grouped into three corruption families (erasure, contradiction,
weather) and written as a JSON file.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from trust_ssl.data.corruptions import CORRUPTIONS, apply_corruption
from trust_ssl.data.downstream import ImageFolderSplit
from trust_ssl.models import build_model
from trust_ssl.utils import load_checkpoint, setup_logger


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]


def _to_tensor_raw() -> T.Compose:
    return T.Compose([T.Resize((224, 224)), T.ToTensor()])


def _normalize(x: torch.Tensor) -> torch.Tensor:
    mean = torch.tensor(_MEAN, device=x.device).view(3, 1, 1)
    std = torch.tensor(_STD, device=x.device).view(3, 1, 1)
    return (x - mean) / std


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="K-I trajectory analysis on EuroSAT")
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--data-root", required=True, type=str,
                   help="path to EuroSAT test split (ImageFolder layout)")
    p.add_argument("--method", type=str, default="trust_ssl",
                   help="method name; must have an evidential gate")
    p.add_argument("--n-samples", type=int, default=500)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--output", type=str, default="results/ki_trajectory.json")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("trust_ssl.eval.ki")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    model = build_model(cfg, method=args.method).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    if not hasattr(model, "compute_ki"):
        raise RuntimeError("this model has no compute_ki method; must be Trust-SSL with evidential gate")

    # Gather N random images from the split
    base = ImageFolderSplit(args.data_root, transform=_to_tensor_raw())
    g = torch.Generator().manual_seed(0)
    indices = torch.randperm(len(base), generator=g)[: args.n_samples].tolist()

    # Load raw tensors once (unnormalized), keep in CPU memory
    logger.info("loading %d raw images", len(indices))
    clean_raw = []
    for idx in indices:
        path, _ = base.samples[idx]
        img = Image.open(path).convert("RGB")
        clean_raw.append(base.transform(img))
    clean_raw = torch.stack(clean_raw, dim=0)   # (N, 3, 224, 224) in [0, 1]

    # Baseline clean-clean
    logger.info("computing clean-clean baseline")
    K_base, I_base = _forward_pairs(model, clean_raw, clean_raw, args.batch_size, device)
    baseline = {"K_mean": float(K_base.mean()), "I_mean": float(I_base.mean())}

    # Per-corruption, per-severity
    family_traj: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"K": [0.0] * 5, "I": [0.0] * 5, "counts": [0] * 5}
    )
    per_corruption: dict = {}

    for corruption in CORRUPTIONS:
        logger.info("corruption: %s (%s)", corruption.name, corruption.family)
        per_corruption[corruption.name] = {"family": corruption.family, "severities": {}}
        for s in (1, 2, 3, 4, 5):
            corrupted = torch.stack([
                apply_corruption(corruption.name, clean_raw[i], s)
                for i in range(clean_raw.shape[0])
            ], dim=0)
            K_vals, I_vals = _forward_pairs(
                model, clean_raw, corrupted, args.batch_size, device
            )
            k_mean = float(K_vals.mean())
            i_mean = float(I_vals.mean())
            per_corruption[corruption.name]["severities"][str(s)] = {
                "K_mean": k_mean,
                "I_mean": i_mean,
            }
            fam = family_traj[corruption.family]
            fam["K"][s - 1] += k_mean
            fam["I"][s - 1] += i_mean
            fam["counts"][s - 1] += 1

    # Finalize per-family averages
    family_out: dict[str, dict[str, list[float]]] = {}
    for family, rec in family_traj.items():
        K_avg = [rec["K"][i] / rec["counts"][i] for i in range(5)]
        I_avg = [rec["I"][i] / rec["counts"][i] for i in range(5)]
        family_out[family] = {"K": K_avg, "I": I_avg}
        logger.info(
            "  %-13s  dK=%+.4f  dI=%+.4f",
            family, K_avg[-1] - K_avg[0], I_avg[-1] - I_avg[0],
        )

    results = {
        "checkpoint": args.checkpoint,
        "method": args.method,
        "n_samples": args.n_samples,
        "baseline_clean_clean": baseline,
        "family_trajectory": family_out,
        "per_corruption": per_corruption,
    }

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote %s", args.output)


@torch.no_grad()
def _forward_pairs(model, raw1: torch.Tensor, raw2: torch.Tensor,
                   batch_size: int, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    n = raw1.shape[0]
    Ks, Is = [], []
    for i in range(0, n, batch_size):
        x1 = _normalize(raw1[i:i + batch_size].to(device))
        x2 = _normalize(raw2[i:i + batch_size].to(device))
        K, I = model.compute_ki(x1, x2)
        Ks.append(K.cpu().numpy())
        Is.append(I.cpu().numpy())
    return np.concatenate(Ks), np.concatenate(Is)


if __name__ == "__main__":
    main()
