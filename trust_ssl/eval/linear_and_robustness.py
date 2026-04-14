"""Linear probe and controlled corruption robustness evaluation.

Given a pretrained backbone, this script:
  1. Trains a 100-epoch linear head on the frozen features of
     EuroSAT, AID and NWPU-RESISC45 and reports clean top-1 accuracy.
  2. Evaluates the same backbone under nine corruption types at five
     severities per dataset, applied to the test set only.
  3. Writes per-dataset JSON files containing both clean and corrupted
     accuracies.

Usage:
    python -m trust_ssl.eval.linear_and_robustness \
        --method trust_ssl \
        --checkpoint checkpoints/trust_ssl_ep199.pth \
        --data-root datasets \
        --results-dir results/

The layout under --data-root is expected to be:
    datasets/eurosat/{train,val,test}/<class>/*.jpg
    datasets/aid/...
    datasets/nwpu/...
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from trust_ssl.data import CORRUPTIONS, build_corruption_loader, build_linear_probe_loaders
from trust_ssl.eval.linear_probe import evaluate_head, extract_features, train_linear_head
from trust_ssl.models import build_model
from trust_ssl.utils import load_checkpoint, setup_logger


DATASETS = ("eurosat", "aid", "nwpu")
SEVERITIES = (1, 2, 3, 4, 5)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Linear probe + corruption robustness")
    p.add_argument("--method", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--data-root", type=str, default="datasets")
    p.add_argument("--results-dir", type=str, default="results")
    p.add_argument("--linear-epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--tag", type=str, default=None,
                   help="optional filename tag (defaults to <method>_<ckpt_stem>)")
    p.add_argument("--skip-linear", action="store_true")
    p.add_argument("--skip-robustness", action="store_true")
    return p.parse_args()


def _load_backbone(method: str, ckpt_path: str, device: torch.device):
    ckpt = load_checkpoint(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {}) or {}
    model = build_model(cfg, method=method)
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def main() -> None:
    args = parse_args()
    logger = setup_logger("trust_ssl.eval.lr")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    out_dir = Path(args.results_dir)
    (out_dir / "linear_eval").mkdir(parents=True, exist_ok=True)
    (out_dir / "robustness").mkdir(parents=True, exist_ok=True)
    tag = args.tag or f"{args.method}_{Path(args.checkpoint).stem}"

    logger.info("method      : %s", args.method)
    logger.info("checkpoint  : %s", args.checkpoint)
    logger.info("tag         : %s", tag)

    model = _load_backbone(args.method, args.checkpoint, device)
    logger.info("backbone loaded onto %s", device)

    for ds_name in DATASETS:
        dataset_root = Path(args.data_root) / ds_name
        if not dataset_root.is_dir():
            logger.warning("skipping %s (not found at %s)", ds_name, dataset_root)
            continue

        logger.info("=" * 70)
        logger.info("dataset: %s", ds_name)

        loaders = build_linear_probe_loaders(
            str(dataset_root),
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
        num_classes = loaders["train"].dataset.num_classes

        # ── Extract features for train/val/test ───────────
        logger.info("  extracting features (train/val/test)")
        tr_f, tr_y = extract_features(model, loaders["train"], device)
        va_f, va_y = extract_features(model, loaders["val"], device)
        te_f, te_y = extract_features(model, loaders["test"], device)

        record: dict = {
            "method": args.method,
            "dataset": ds_name,
            "tag": tag,
        }

        if not args.skip_linear:
            head, summary = train_linear_head(
                tr_f, tr_y, va_f, va_y,
                num_classes=num_classes,
                epochs=args.linear_epochs,
                base_lr=0.1,
                device=device,
            )
            test_acc = evaluate_head(head, te_f, te_y, device=device)
            record["test_accuracy"] = float(100.0 * test_acc)
            record["val_accuracy"] = float(100.0 * summary["best_val_accuracy"])
            record["linear_epochs"] = args.linear_epochs
            logger.info("  top-1: %.2f%%  val: %.2f%%",
                        record["test_accuracy"], record["val_accuracy"])

            out = out_dir / "linear_eval" / f"{tag}_{ds_name}.json"
            with open(out, "w") as f:
                json.dump(record, f, indent=2)
            logger.info("  wrote %s", out)

        if not args.skip_robustness:
            # Train a robustness-phase linear head (also fitted on clean
            # train features) to decouple probe training from evaluation.
            rob_head, _ = train_linear_head(
                tr_f, tr_y, va_f, va_y,
                num_classes=num_classes,
                epochs=50,
                base_lr=0.1,
                device=device,
            )
            clean_acc = evaluate_head(rob_head, te_f, te_y, device=device)

            rob_record = {
                "method": args.method,
                "dataset": ds_name,
                "tag": tag,
                "clean": {"accuracy": float(100.0 * clean_acc)},
                "corrupted": {},
            }

            for corruption in CORRUPTIONS:
                per_severity: dict[str, float] = {}
                for s in SEVERITIES:
                    loader = build_corruption_loader(
                        str(dataset_root),
                        corruption=corruption.name,
                        severity=s,
                        batch_size=args.batch_size,
                        num_workers=args.num_workers,
                    )
                    c_f, c_y = extract_features(model, loader, device)
                    acc = evaluate_head(rob_head, c_f, c_y, device=device)
                    per_severity[str(s)] = float(100.0 * acc)
                logger.info(
                    "  %-22s s1=%.1f s2=%.1f s3=%.1f s4=%.1f s5=%.1f",
                    corruption.name,
                    per_severity["1"], per_severity["2"],
                    per_severity["3"], per_severity["4"], per_severity["5"],
                )
                rob_record["corrupted"][corruption.name] = per_severity

            out = out_dir / "robustness" / f"{tag}_{ds_name}.json"
            with open(out, "w") as f:
                json.dump(rob_record, f, indent=2)
            logger.info("  wrote %s", out)

    logger.info("done")


if __name__ == "__main__":
    main()
