"""Data-wide assertion for exact NDVI/plain boundary mass matching."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data_semantics import torch_ndvi
from revised_models import ControlledPixelLoss


STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "experiments/raw/hr_gldd"
BATCH_SIZE = 32


def main() -> None:
    images = np.load(DATA / "trainX.npy", mmap_mode="r")
    targets = np.load(DATA / "trainY.npy", mmap_mode="r")
    criterion = ControlledPixelLoss(mode="biophysical")
    checked = 0
    fallback = 0
    maximum_error = 0.0

    for start in range(0, len(images), BATCH_SIZE):
        stop = min(start + BATCH_SIZE, len(images))
        batch_images = (
            torch.from_numpy(np.asarray(images[start:stop]).copy())
            .permute(0, 3, 1, 2)
            .float()
        )
        batch_targets = (
            torch.from_numpy(np.asarray(targets[start:stop]).copy())
            .squeeze(-1)
            .float()
        )
        boundary = criterion.boundary(batch_targets)
        ndvi = torch_ndvi(batch_images)
        gx = F.conv2d(ndvi, criterion.sobel_x, padding=1)
        gy = F.conv2d(ndvi, criterion.sobel_y, padding=1)
        gradient = torch.sqrt(gx.square() + gy.square() + 1e-8).squeeze(1)
        maxima = gradient.amax(dim=(1, 2), keepdim=True).clamp_min(1e-6)
        signal = boundary * (gradient / maxima)
        signal_mass = signal.mean(dim=(1, 2))
        modulation = criterion.match_boundary_mass(boundary, signal)
        target_mass = boundary.mean(dim=(1, 2))
        actual_mass = modulation.mean(dim=(1, 2))
        errors = torch.abs(actual_mass - target_mass)
        maximum_error = max(maximum_error, float(errors.max()))
        fallback += int(torch.sum(signal_mass <= 1e-6))
        checked += len(batch_images)
        torch.testing.assert_close(
            actual_mass, target_mass, rtol=0, atol=1e-7
        )

    assert checked == 1119, checked
    print(
        "PASS: exact equal-mass modulation for "
        f"{checked} HR-GLDD training images; fallbacks={fallback}; "
        f"max_abs_error={maximum_error:.3e}"
    )


if __name__ == "__main__":
    main()
