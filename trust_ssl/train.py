"""Pretraining entry point for all six methods reported in the paper.

Usage:
    python -m trust_ssl.train \
        --method trust_ssl \
        --config configs/trust_ssl.yaml \
        --data-root datasets/pretrain_210k \
        --epochs 200 \
        --output checkpoints/trust_ssl_ep199.pth

The same script is used for SimCLR, BYOL, VICReg, Trust-SSL (full),
trust_ssl_scalar, trust_ssl_cosine. Method-specific behaviour lives
inside the model classes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn as nn

from trust_ssl.data import build_pretrain_loader
from trust_ssl.models import build_model
from trust_ssl.utils import (
    LARS,
    cosine_schedule,
    linear_ramp,
    load_config,
    merge_configs,
    save_checkpoint,
    setup_logger,
)


VALID_METHODS = (
    "simclr",
    "byol",
    "vicreg",
    "trust_ssl",
    "trust_ssl_scalar",
    "trust_ssl_cosine",
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Trust-SSL pretraining")
    p.add_argument("--method", required=True, choices=VALID_METHODS)
    p.add_argument("--config", required=True, type=str,
                   help="YAML config file with method-specific hyperparameters")
    p.add_argument("--data-root", required=True, type=str,
                   help="path to pretraining corpus root")
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--num-workers", type=int, default=8)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--base-lr", type=float, default=0.3)
    p.add_argument("--weight-decay", type=float, default=1e-6)
    p.add_argument("--warmup-epochs", type=int, default=10)
    p.add_argument("--output", required=True, type=str,
                   help="destination for the final checkpoint")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--print-every", type=int, default=50,
                   help="log every N iterations")
    p.add_argument("--save-every", type=int, default=0,
                   help="also save a checkpoint every N epochs (0 = final only)")
    p.add_argument("--history", type=str, default=None,
                   help="optional path to write an epoch-by-epoch JSON history")
    return p.parse_args()


def _excluded_from_lars(p: nn.Parameter) -> bool:
    """LARS typically excludes biases and BatchNorm parameters."""
    if p.ndim <= 1:
        return True
    return False


def main() -> None:
    args = parse_args()
    logger = setup_logger("trust_ssl.train")

    torch.manual_seed(args.seed)

    if not torch.cuda.is_available():
        logger.warning("CUDA not available; running on CPU will be extremely slow")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ─── Config ───────────────────────────────────────────────
    base_cfg = load_config(args.config)
    cfg = merge_configs(base_cfg, {
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "base_lr": args.base_lr,
    })

    # ─── Data ────────────────────────────────────────────────
    loader = build_pretrain_loader(
        data_root=args.data_root,
        image_size=args.image_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        drop_last=True,
    )
    steps_per_epoch = len(loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = steps_per_epoch * args.warmup_epochs
    logger.info("dataset: %d samples, %d batches/epoch", len(loader.dataset), steps_per_epoch)

    # ─── Model ───────────────────────────────────────────────
    model = build_model(cfg, method=args.method).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("model %s: %.1fM trainable parameters", args.method, n_params / 1e6)

    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optim = LARS(
        model.parameters(),
        lr=args.base_lr,
        weight_decay=args.weight_decay,
        momentum=0.9,
        exclude_from_adaptation=_excluded_from_lars,
    )

    # ─── Trust-SSL schedules ─────────────────────────────────
    is_trust_family = args.method in ("trust_ssl", "trust_ssl_scalar", "trust_ssl_cosine")
    sel_max = float(cfg.get("lambda_sel_max", 0.2))
    sel_start_ep = int(cfg.get("lambda_sel_start_epoch", 100))
    sel_end_ep = int(cfg.get("lambda_sel_end_epoch", 150))
    lm_hi = float(cfg.get("lambda_min_start", 0.5))
    lm_lo = float(cfg.get("lambda_min_end", 0.05))

    # ─── Loop ───────────────────────────────────────────────
    history: list[dict] = []
    global_step = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        model.train()
        epoch_losses: list[float] = []
        epoch_K: list[float] = []
        epoch_I: list[float] = []

        # Trust-SSL schedules are indexed by epoch, not step
        lambda_sel_e = linear_ramp(sel_max, epoch, sel_start_ep, sel_end_ep) if is_trust_family else 0.0
        lambda_min_e = cosine_schedule(
            base=lm_hi, current_step=epoch, total_steps=args.epochs,
            warmup_steps=0, min_value=lm_lo,
        ) if is_trust_family else 0.0

        for it, batch in enumerate(loader):
            view1 = batch["view1"].to(device, non_blocking=True)
            view2 = batch["view2"].to(device, non_blocking=True)
            corrupt = batch.get("corrupt_label")
            if corrupt is not None:
                corrupt = corrupt.to(device, non_blocking=True)

            # LR schedule at step granularity
            lr = cosine_schedule(
                base=args.base_lr,
                current_step=global_step,
                total_steps=total_steps,
                warmup_steps=warmup_steps,
                min_value=0.0,
            )
            for g in optim.param_groups:
                g["lr"] = lr

            # Forward
            if is_trust_family:
                out = _call(model)(
                    view1, view2,
                    corrupt_labels=corrupt,
                    lambda_sel=lambda_sel_e,
                    lambda_min=lambda_min_e,
                )
            else:
                out = _call(model)(view1, view2)

            loss = out["loss"]
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()

            epoch_losses.append(float(loss.detach().cpu().item()))
            if is_trust_family and out.get("K_mean") is not None:
                epoch_K.append(float(out["K_mean"].cpu().item()))
                epoch_I.append(float(out["I_mean"].cpu().item()))

            if (global_step % args.print_every) == 0:
                msg = (
                    f"epoch {epoch:3d}/{args.epochs} "
                    f"it {it:4d}/{steps_per_epoch} "
                    f"loss={float(loss):.4f} "
                    f"lr={lr:.4f}"
                )
                if is_trust_family:
                    msg += f" λ_sel={lambda_sel_e:.3f} λ_min={lambda_min_e:.3f}"
                logger.info(msg)
            global_step += 1

        # End-of-epoch summary
        rec = {
            "epoch": epoch,
            "loss_mean": sum(epoch_losses) / max(1, len(epoch_losses)),
            "lambda_sel": lambda_sel_e,
            "lambda_min": lambda_min_e,
        }
        if epoch_K:
            rec["K_mean"] = sum(epoch_K) / len(epoch_K)
            rec["I_mean"] = sum(epoch_I) / len(epoch_I)
        history.append(rec)
        logger.info("epoch %d complete | loss=%.4f", epoch, rec["loss_mean"])

        # Periodic checkpointing
        if args.save_every and (epoch + 1) % args.save_every == 0:
            out_path = Path(args.output).with_name(f"{Path(args.output).stem}_ep{epoch}.pth")
            save_checkpoint(out_path, _unwrap(model), optim, epoch=epoch,
                            extra={"method": args.method, "config": cfg})
            logger.info("saved intermediate checkpoint -> %s", out_path)

    # Final checkpoint
    save_checkpoint(args.output, _unwrap(model), optim, epoch=args.epochs - 1,
                    extra={"method": args.method, "config": cfg})
    logger.info("saved final checkpoint -> %s", args.output)

    # Optional training history
    if args.history:
        Path(args.history).parent.mkdir(parents=True, exist_ok=True)
        with open(args.history, "w") as f:
            json.dump({"method": args.method, "history": history}, f, indent=2)
        logger.info("wrote history -> %s", args.history)

    total_minutes = (time.time() - t0) / 60.0
    logger.info("done in %.1f minutes", total_minutes)


def _unwrap(m: nn.Module) -> nn.Module:
    return m.module if isinstance(m, nn.DataParallel) else m


def _call(m: nn.Module):
    return m  # alias for readability


if __name__ == "__main__":
    main()
