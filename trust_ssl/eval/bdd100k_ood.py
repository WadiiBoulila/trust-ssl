"""Zero-shot OOD detection on BDD100K weather splits.

For a given pretrained backbone, this script:
  1. Extracts features on the in-distribution split (clear daytime).
  2. Extracts features on four OOD splits (rain, night, fog, snow).
  3. Computes three standard detector scores on backbone features:
       - Mahalanobis distance fit on ID features
       - Energy score (negative logsumexp of projector logits)
       - Feature-norm score
     and reports the per-split AUROC.
  4. For Trust-SSL (full), additionally computes a native K+I score
     directly from the evidential heads.

Usage:
    python -m trust_ssl.eval.bdd100k_ood \
        --method trust_ssl \
        --checkpoint checkpoints/trust_ssl_ep199.pth \
        --bdd-root datasets/bdd100k \
        --results-dir results/

Expected layout under --bdd-root:
    id/clear_daytime/*.jpg
    ood/rain/*.jpg
    ood/night/*.jpg
    ood/fog/*.jpg
    ood/snow/*.jpg
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T

from trust_ssl.models import build_model
from trust_ssl.utils import load_checkpoint, setup_logger


_MEAN = [0.485, 0.456, 0.406]
_STD = [0.229, 0.224, 0.225]

_BDD_TRANSFORM = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=_MEAN, std=_STD),
])


class BDDSplit(Dataset):
    """Flat directory of BDD100K images for one split."""

    def __init__(self, root: str, max_items: int | None = None) -> None:
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"BDD split not found: {self.root}")
        exts = {".jpg", ".jpeg", ".png"}
        self.paths = sorted(
            str(p) for p in self.root.rglob("*") if p.suffix.lower() in exts
        )
        if max_items:
            self.paths = self.paths[:max_items]

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        img = Image.open(self.paths[idx]).convert("RGB")
        return _BDD_TRANSFORM(img)


# ─────────────────────────────────────────────────────────────────
# Detectors
# ─────────────────────────────────────────────────────────────────
def fit_mahalanobis(id_feats: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (mean, inv_cov_sqrt) for the Mahalanobis detector."""
    mu = id_feats.mean(axis=0)
    centered = id_feats - mu
    cov = (centered.T @ centered) / max(1, id_feats.shape[0] - 1)
    cov += 1e-4 * np.eye(cov.shape[0])
    inv = np.linalg.pinv(cov)
    return mu, inv


def mahalanobis_score(feats: np.ndarray, mu: np.ndarray, inv: np.ndarray) -> np.ndarray:
    diff = feats - mu
    return np.einsum("nd,dk,nk->n", diff, inv, diff)


def energy_score(feats: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Energy-style score using the feature magnitudes.

    For an unsupervised backbone we do not have class logits; we use
    -T * logsumexp(features / T) as a lightweight stand-in, which
    captures the same intuition as the supervised energy score.
    """
    scaled = feats / temperature
    return -temperature * np.log(np.exp(scaled - scaled.max(axis=1, keepdims=True)).sum(axis=1))


def feature_norm_score(feats: np.ndarray) -> np.ndarray:
    return -np.linalg.norm(feats, axis=1)


def auroc(scores_id: np.ndarray, scores_ood: np.ndarray) -> float:
    """Compute AUROC for a detector whose score is *higher* for OOD.

    Convention: we pass detector scores that are larger on OOD, so the
    AUROC here is computed over (ood_is_positive == 1).
    """
    y = np.concatenate([np.zeros_like(scores_id), np.ones_like(scores_ood)])
    s = np.concatenate([scores_id, scores_ood])
    order = np.argsort(-s)
    y_sorted = y[order]
    tps = np.cumsum(y_sorted)
    fps = np.cumsum(1 - y_sorted)
    P = max(1, int(y.sum()))
    N = max(1, int((1 - y).sum()))
    tpr = tps / P
    fpr = fps / N
    tpr = np.concatenate([[0.0], tpr, [1.0]])
    fpr = np.concatenate([[0.0], fpr, [1.0]])
    return float(np.trapezoid(tpr, fpr))


# ─────────────────────────────────────────────────────────────────
# Feature extraction
# ─────────────────────────────────────────────────────────────────
@torch.no_grad()
def extract_backbone(model, loader: DataLoader, device: torch.device) -> torch.Tensor:
    model.eval()
    feats = []
    for x in loader:
        x = x.to(device, non_blocking=True)
        f = model.extract(x) if hasattr(model, "extract") else model(x)
        feats.append(f.detach().cpu())
    return torch.cat(feats, dim=0)


@torch.no_grad()
def extract_ki(model, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    """Return (K_mean_per_sample, I_mean_per_sample) averaged over factors.

    Because OOD test is a single-view regime, we use the same image for
    both inputs of compute_ki: K measures per-factor self-conflict
    (essentially zero on ID) and I measures per-sample ignorance.
    """
    model.eval()
    Ks, Is = [], []
    for x in loader:
        x = x.to(device, non_blocking=True)
        K, I = model.compute_ki(x, x)
        Ks.append(K.detach().cpu().numpy())
        Is.append(I.detach().cpu().numpy())
    return np.concatenate(Ks), np.concatenate(Is)


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Zero-shot OOD on BDD100K")
    p.add_argument("--method", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--bdd-root", required=True, type=str)
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--max-id", type=int, default=5000)
    p.add_argument("--max-ood-per-split", type=int, default=3000)
    p.add_argument("--tag", type=str, default=None)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logger = setup_logger("trust_ssl.eval.ood")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    bdd_root = Path(args.bdd_root)
    id_dir = bdd_root / "id" / "clear_daytime"
    ood_dirs = {
        "rain":  bdd_root / "ood" / "rain",
        "night": bdd_root / "ood" / "night",
        "fog":   bdd_root / "ood" / "fog",
        "snow":  bdd_root / "ood" / "snow",
    }

    # Load model
    ckpt = load_checkpoint(args.checkpoint, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    model = build_model(cfg, method=args.method).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)

    tag = args.tag or Path(args.checkpoint).stem
    out_dir = Path(args.results_dir) / "bdd100k_ood"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ─── ID features ──────────────────────────────────────
    id_ds = BDDSplit(str(id_dir), max_items=args.max_id)
    id_loader = DataLoader(id_ds, batch_size=args.batch_size,
                           num_workers=args.num_workers, pin_memory=True)
    logger.info("[ID] clear_daytime: %d images", len(id_ds))
    id_feats = extract_backbone(model, id_loader, device).numpy()
    logger.info("     features: %s", id_feats.shape)

    # Fit ID detectors
    mu, inv = fit_mahalanobis(id_feats)
    id_maha = mahalanobis_score(id_feats, mu, inv)
    id_energy = energy_score(id_feats)
    id_fn = feature_norm_score(id_feats)

    has_ki = args.method == "trust_ssl"
    id_ki = None
    if has_ki:
        id_K, id_I = extract_ki(model, id_loader, device)
        id_ki = id_K + id_I

    # ─── Per-split AUROC ─────────────────────────────────
    per_split: dict[str, dict] = {}
    for name, path in ood_dirs.items():
        if not path.is_dir():
            logger.warning("  skipping %s (missing dir %s)", name, path)
            continue
        ood_ds = BDDSplit(str(path), max_items=args.max_ood_per_split)
        ood_loader = DataLoader(ood_ds, batch_size=args.batch_size,
                                num_workers=args.num_workers, pin_memory=True)
        logger.info("[OOD] %s: %d images", name, len(ood_ds))
        ood_feats = extract_backbone(model, ood_loader, device).numpy()

        ood_maha = mahalanobis_score(ood_feats, mu, inv)
        ood_energy = energy_score(ood_feats)
        ood_fn = feature_norm_score(ood_feats)

        split_out: dict = {
            "mahalanobis": {"auroc": auroc(id_maha, ood_maha)},
            "energy": {"auroc": auroc(id_energy, ood_energy)},
            "feature_norm": {"auroc": auroc(id_fn, ood_fn)},
        }

        if has_ki:
            ood_K, ood_I = extract_ki(model, ood_loader, device)
            ood_ki = ood_K + ood_I
            split_out["trust_ssl_K_plus_I"] = {"auroc": auroc(id_ki, ood_ki)}

        logger.info(
            "  AUROC: maha=%.2f energy=%.2f featnorm=%.2f",
            split_out["mahalanobis"]["auroc"] * 100,
            split_out["energy"]["auroc"] * 100,
            split_out["feature_norm"]["auroc"] * 100,
        )
        per_split[f"ood_{name}"] = split_out

    # ─── Aggregate means ─────────────────────────────────
    if not per_split:
        logger.error("no OOD splits evaluated; aborting")
        return

    detector_keys = list(next(iter(per_split.values())).keys())
    mean_across: dict[str, float] = {}
    for k in detector_keys:
        vals = [split[k]["auroc"] for split in per_split.values() if k in split]
        if vals:
            mean_across[k] = float(sum(vals) / len(vals))

    logger.info("=" * 60)
    logger.info("MEAN AUROC across OOD splits:")
    for k, v in mean_across.items():
        logger.info("  %-20s : %.2f", k, v * 100)

    results = {
        "method": args.method,
        "tag": tag,
        "per_split": per_split,
        "mean_across_splits": mean_across,
        "id_split": "clear_daytime",
        "id_count": int(id_feats.shape[0]),
    }
    out_path = out_dir / f"{tag}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info("wrote %s", out_path)


if __name__ == "__main__":
    main()
