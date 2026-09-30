# FloodSight

FloodSight is a semantic segmentation project for analyzing post-flood aerial imagery from the [FloodNet dataset](https://github.com/BinaLab/FloodNet-Challenge-EARTHVISION2021). It is being rewritten in PyTorch from an earlier TensorFlow and OpenCV implementation.

The project is a work in progress. The repository contains the model, dataset loader, transforms, loss and metric utilities, and visualization helpers. The training pipeline is still incomplete, so this code is not yet a ready-to-run end-to-end training application.

## Current implementation

- A U-Net-style PyTorch model that predicts 10 semantic classes.
- A FloodNet dataset loader for paired RGB images and grayscale label masks.
- Training and validation image transforms, including resizing, normalization, and horizontal flipping / affine augmentation for training.
- A class-weighted focal loss and confusion-matrix counts for segmentation metrics.
- A helper for visualizing images, ground-truth masks, and predictions.

The class labels are defined in `src/config.py`: Background, Building Flooded, Building Non-Flooded, Road Flooded, Road Non-Flooded, Water, Tree, Vehicle, Pool, and Grass.

## Repository layout

```text
src/
├── config.py          # Class names, colors, class weights, normalization values
├── dataset.py         # FloodNet image and mask loader
├── losses.py          # Focal loss; Dice loss is not implemented yet
├── metric.py          # Per-class true-positive, false-positive, false-negative counts
├── model.py           # U-Net-style segmentation model
├── train.py           # In-progress training loop
├── transforms.py      # Training, validation, and test transforms
└── visualizations.py  # Image and segmentation-mask plotting
utils/
└── download_dataset.py
```

## Setup

The project uses [uv](https://docs.astral.sh/uv/) for dependency management and requires Python 3.13 or later, as specified in `pyproject.toml`.

```bash
uv sync
```

## Dataset

The dataset loader defaults to this root directory:

```text
FloodNet/FloodNet-Supervised_v1.0/
```

It expects the following split layout:

```text
FloodNet/FloodNet-Supervised_v1.0/
├── train/
│   ├── train-org-img/
│   └── train-label-img/
└── val/
    ├── val-org-img/
    └── val-label-img/
```

Images are paired with masks by filename: for an image named `123.png`, the loader looks for a mask named `123_lab.png`. The loader currently handles `train` and `val` splits. Set the `FLOODSIGHT_DATASET_URL` environment variable to an archive URL you are authorized to use, then run `uv run python utils/download_dataset.py` to download and extract it under `FloodNet/`. Check that the resulting folders match the expected layout above before using the loader.

## Development status

This repository is an active rewrite, not a finished training package. In particular, `src/train.py` is currently a short development loop rather than a complete training entry point, and Dice loss in `src/losses.py` is a stub. Model setup, device selection, optimizer and criterion setup, checkpointing, and a complete metric/evaluation workflow still need to be connected before training can be run end to end from this repository.

## Previous implementation results

The earlier TensorFlow and OpenCV implementation was trained for semantic segmentation on FloodNet. Its reported results were **57% accuracy** and **55% mean Intersection over Union (mIoU) on the validation set**. These are results from the earlier implementation; the PyTorch rewrite has not yet been evaluated end to end.
