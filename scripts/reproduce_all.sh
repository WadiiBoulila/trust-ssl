#!/usr/bin/env bash
#
# End-to-end reproduction driver.
#
# Launches pretraining for all six methods reported in the paper, then
# runs linear probe, corruption robustness, BDD100K OOD and the K-I
# trajectory analysis on every checkpoint. Produces the JSON files
# from which every table and figure in the paper is derived.
#
# Usage:
#   bash scripts/reproduce_all.sh                 # full run (~3 days)
#   SKIP_PRETRAIN=1 bash scripts/reproduce_all.sh # eval only
#
# Expected layout:
#   datasets/pretrain_210k/
#   datasets/eurosat/{train,val,test}/
#   datasets/aid/{train,val,test}/
#   datasets/nwpu/{train,val,test}/
#   datasets/bdd100k/{id/clear_daytime, ood/{rain,night,fog,snow}}/

set -euo pipefail

ROOT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
cd "$ROOT"

CKPT_DIR="${CKPT_DIR:-checkpoints}"
RESULTS_DIR="${RESULTS_DIR:-results}"
DATA_ROOT="${DATA_ROOT:-datasets}"
EPOCHS="${EPOCHS:-200}"

mkdir -p "$CKPT_DIR" "$RESULTS_DIR/linear_eval" "$RESULTS_DIR/robustness" "$RESULTS_DIR/bdd100k_ood"

METHODS=(simclr byol vicreg trust_ssl_scalar trust_ssl_cosine trust_ssl)

# ─────────────────────────────────────────────────────────────
# Phase 1: pretraining
# ─────────────────────────────────────────────────────────────
if [[ "${SKIP_PRETRAIN:-0}" != "1" ]]; then
    for method in "${METHODS[@]}"; do
        ckpt="$CKPT_DIR/${method}_ep$((EPOCHS-1)).pth"
        if [[ -f "$ckpt" ]]; then
            echo "[skip] checkpoint already exists: $ckpt"
            continue
        fi
        echo "[pretrain] $method"
        python -m trust_ssl.train \
            --method "$method" \
            --config "configs/${method}.yaml" \
            --data-root "$DATA_ROOT/pretrain_210k" \
            --epochs "$EPOCHS" \
            --output "$ckpt" \
            --history "$RESULTS_DIR/history_${method}.json"
    done
fi

# ─────────────────────────────────────────────────────────────
# Phase 2: linear probe + corruption robustness
# ─────────────────────────────────────────────────────────────
for method in "${METHODS[@]}"; do
    ckpt="$CKPT_DIR/${method}_ep$((EPOCHS-1)).pth"
    if [[ ! -f "$ckpt" ]]; then
        echo "[warn] missing checkpoint: $ckpt"
        continue
    fi
    echo "[eval-lr] $method"
    python -m trust_ssl.eval.linear_and_robustness \
        --method "$method" \
        --checkpoint "$ckpt" \
        --data-root "$DATA_ROOT" \
        --results-dir "$RESULTS_DIR" \
        --tag "${method}"
done

# ─────────────────────────────────────────────────────────────
# Phase 3: zero-shot BDD100K OOD
# ─────────────────────────────────────────────────────────────
for method in "${METHODS[@]}"; do
    ckpt="$CKPT_DIR/${method}_ep$((EPOCHS-1)).pth"
    if [[ ! -f "$ckpt" ]]; then
        continue
    fi
    echo "[eval-ood] $method"
    python -m trust_ssl.eval.bdd100k_ood \
        --method "$method" \
        --checkpoint "$ckpt" \
        --bdd-root "$DATA_ROOT/bdd100k" \
        --results-dir "$RESULTS_DIR" \
        --tag "${method}"
done

# ─────────────────────────────────────────────────────────────
# Phase 4: K-I trajectory analysis (Trust-SSL only)
# ─────────────────────────────────────────────────────────────
ckpt="$CKPT_DIR/trust_ssl_ep$((EPOCHS-1)).pth"
if [[ -f "$ckpt" ]]; then
    echo "[eval-ki] trust_ssl"
    python -m trust_ssl.eval.ki_trajectory \
        --checkpoint "$ckpt" \
        --data-root "$DATA_ROOT/eurosat/test" \
        --n-samples 500 \
        --output "$RESULTS_DIR/ki_trajectory.json"
fi

echo "done. all JSON results in $RESULTS_DIR"
