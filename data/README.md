# Data preparation

This repository does not ship any images. The scripts below describe
how to obtain and arrange each dataset so that the training and
evaluation code finds it.

All paths below are relative to the repository root and assume the
default `datasets/` layout. If you keep data elsewhere, pass the
appropriate `--data-root` flag.

## Expected layout

```
datasets/
├── pretrain_210k/                 # flat directory of 210,178 RGB aerial images
├── eurosat/
│   ├── train/<class_name>/*.jpg
│   ├── val/<class_name>/*.jpg
│   └── test/<class_name>/*.jpg
├── aid/
│   ├── train/<class_name>/*.jpg
│   ├── val/<class_name>/*.jpg
│   └── test/<class_name>/*.jpg
├── nwpu/
│   ├── train/<class_name>/*.jpg
│   ├── val/<class_name>/*.jpg
│   └── test/<class_name>/*.jpg
└── bdd100k/
    ├── id/
    │   └── clear_daytime/*.jpg
    └── ood/
        ├── rain/*.jpg
        ├── night/*.jpg
        ├── fog/*.jpg
        └── snow/*.jpg
```

## Pretraining corpus: BigEarthNet-S2 + LoveDA

The 210K image corpus is constructed from two sources:

1. **BigEarthNet-S2** (RGB bands B04, B03, B02). Download the full
   archive from `https://bigearth.net/`. After unpacking, convert
   each `.tif` tile to an 8-bit RGB JPEG and save it under
   `datasets/pretrain_210k/`. Min-max normalize each tile
   independently before saving. We use 200,000 tiles.

2. **LoveDA**. Download from `https://github.com/Junjue-Wang/LoveDA`.
   Extract 10,000 crops of size 256×256 from the provided urban and
   rural scenes and save them alongside the BigEarthNet tiles in
   the same flat directory.

After preparation:

```bash
ls datasets/pretrain_210k | wc -l
# should print 210178
```

## EuroSAT / AID / NWPU-RESISC45

Download the standard splits:

- **EuroSAT**: `https://github.com/phelber/EuroSAT` (RGB version, 27,000 images, 10 classes).
- **AID**: `https://captain-whu.github.io/AID/` (10,000 images, 30 classes).
- **NWPU-RESISC45**: `https://gcheng-nwpu.github.io/` (31,500 images, 45 classes).

The scripts expect an ImageFolder-style `train/val/test` layout. A
typical 50/20/30 split (random, seeded) works well; if you want to
match the paper exactly use the same split seeds as listed in the
README header comments of each configuration file.

## BDD100K

Download the full BDD100K image archive from `https://bdd-data.berkeley.edu/`
and use the provided `labels/attributes.json` metadata to sort images
by `weather` and `time_of_day` into the required splits:

- `id/clear_daytime/`: weather = `clear` AND time_of_day = `daytime`
- `ood/rain/`:   weather = `rainy`
- `ood/night/`:  time_of_day = `night`
- `ood/fog/`:    weather = `foggy`
- `ood/snow/`:   weather = `snowy`

No resizing is necessary; the evaluator resizes on load.

## Sanity check

After preparation, run the small dataset sanity script:

```bash
python -c "
from trust_ssl.data import build_pretrain_loader, build_linear_probe_loaders
pl = build_pretrain_loader('datasets/pretrain_210k', batch_size=8, num_workers=0)
print('pretrain corpus:', len(pl.dataset), 'images')
for name in ('eurosat', 'aid', 'nwpu'):
    loaders = build_linear_probe_loaders(f'datasets/{name}', batch_size=8, num_workers=0)
    print(f'{name}: train={len(loaders[\"train\"].dataset)} val={len(loaders[\"val\"].dataset)} test={len(loaders[\"test\"].dataset)}')
"
```

If this runs without error and prints reasonable counts, the dataset
preparation is complete.
