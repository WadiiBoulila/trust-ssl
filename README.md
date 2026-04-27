# Trust-SSL

**Additive-Residual Selective Invariance for Robust Self-Supervised Representation Learning in Aerial Imagery**

This repository contains the reference implementation for the paper. The code reproduces every experiment reported in the manuscript: pretraining of six self-supervised backbones on an aerial corpus, linear-probe evaluation on three aerial scene classification benchmarks, controlled corruption-robustness evaluation, controlled K–I trajectory analysis, and zero-shot out-of-distribution transfer to BDD100K weather splits.


---

## What's in the box

| Component | Location |
|---|---|
| Pretraining (SimCLR / BYOL / VICReg / Trust-SSL + ablations) | `trust_ssl/train.py` |
| Linear-probe + corruption-robustness evaluation | `trust_ssl/eval/linear_and_robustness.py` |
| Zero-shot BDD100K OOD transfer | `trust_ssl/eval/bdd100k_ood.py` |
| K–I trajectory analysis | `trust_ssl/eval/ki_trajectory.py` |
| Six-method reproduction driver | `scripts/reproduce_all.sh` |

All hyperparameters used in the paper are in `configs/`. Every script has a `--help` that prints every flag.

---

## Quick start

### 1. Environment

Python 3.10 with PyTorch 2.5 and CUDA 12.1 was used for all results in the paper. A conda environment file is provided:

```bash
conda env create -f environment.yml
conda activate trust_ssl
```

Or, with pip:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Data

The pretraining corpus combines BigEarthNet-S2 and LoveDA:

```bash
# Follow the instructions in data/README.md to download and prepare the
# 210K RGB aerial corpus used for pretraining, and the three downstream
# benchmarks (EuroSAT, AID, NWPU-RESISC45). BDD100K is prepared separately.
```

Expected layout under `datasets/`:

```
datasets/
├── pretrain_210k/           # 210,178 images, 256x256 RGB
├── eurosat/                 # train/val/test splits
├── aid/                     # train/val/test splits
├── nwpu/                    # train/val/test splits
└── bdd100k/                 # clear_daytime + rain/night/fog/snow splits
```

### 3. Pretrain a backbone

Single-method pretraining (one of `simclr | byol | vicreg | trust_ssl | trust_ssl_scalar | trust_ssl_cosine`):

```bash
python -m trust_ssl.train \
    --method trust_ssl \
    --config configs/trust_ssl.yaml \
    --data-root datasets/pretrain_210k \
    --epochs 200 \
    --output checkpoints/trust_ssl_ep199.pth
```

### 4. Evaluate a checkpoint

Linear probe and corruption robustness across the three aerial benchmarks:

```bash
python -m trust_ssl.eval.linear_and_robustness \
    --method trust_ssl \
    --checkpoint checkpoints/trust_ssl_ep199.pth \
    --results-dir results/
```

Zero-shot OOD transfer to BDD100K:

```bash
python -m trust_ssl.eval.bdd100k_ood \
    --method trust_ssl \
    --checkpoint checkpoints/trust_ssl_ep199.pth \
    --bdd-root datasets/bdd100k \
    --results-dir results/
```

K–I trajectory analysis on EuroSAT (only for Trust-SSL family):

```bash
python -m trust_ssl.eval.ki_trajectory \
    --checkpoint checkpoints/trust_ssl_ep199.pth \
    --data-root datasets/eurosat \
    --n-samples 500 \
    --output results/ki_trajectory.json
```

### 5. Reproduce everything

The driver `scripts/reproduce_all.sh` launches all six pretraining runs, then all four evaluations, and collects the numbers reported in the paper:

```bash
bash scripts/reproduce_all.sh
```

Expect ~50 hours per pretraining run on a single 8xA100 node and ~3 hours of evaluation per checkpoint.

---

## Reported results

The numbers below are the ones in the paper. They are produced by the code in this repository under identical 200-epoch pretraining and identical linear evaluation.

**Linear probe on three aerial benchmarks (Top-1 %):**

| Method | EuroSAT | AID | NWPU-RESISC45 | Mean |
|---|---|---|---|---|
| SimCLR | 96.39 | 86.07 | 82.92 | 88.46 |
| BYOL | 96.89 | 84.77 | 79.87 | 87.18 |
| VICReg | 97.06 | 88.20 | 84.19 | 89.82 |
| Scalar uncert. | 97.11 | 88.47 | 83.89 | 89.82 |
| Cosine gate | 97.20 | 88.30 | 85.01 | 89.84 |
| **Trust-SSL** | 97.11 | **88.63** | 84.86 | **90.20** |

**Zero-shot Mahalanobis AUROC on BDD100K (%):**

| Method | Rain | Night | Fog | Snow | Mean |
|---|---|---|---|---|---|
| SimCLR | 97.9 | 99.9 | 97.0 | 94.0 | 97.21 |
| BYOL | 97.7 | 99.9 | 96.6 | 89.7 | 95.96 |
| VICReg | 98.3 | 99.9 | 97.6 | 93.8 | 97.41 |
| Scalar uncert. | 99.3 | 99.9 | 98.9 | 96.0 | 98.54 |
| Cosine gate | **99.5** | **100.0** | 98.7 | **97.4** | **98.86** |
| Trust-SSL | 99.0 | **100.0** | **98.3** | 95.1 | 98.09 |

Full robustness, per-family and per-detector tables are in `results/paper_tables/` after running the reproduction driver.

---

## Citation

If you use this code or our reported numbers, please cite:

```bibtex
@article{boulila2026trust,
  title={Trust-SSL: Additive-Residual Selective Invariance for Robust Aerial Self-Supervised Learning},
  author={Boulila, Wadii and Ammar, Adel and Benjdira, Bilel and Driss, Maha},
  journal={arXiv preprint arXiv:2604.21349},
  year={2026}
}

```


