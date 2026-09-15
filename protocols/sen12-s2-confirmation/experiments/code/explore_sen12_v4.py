#!/usr/bin/env python3
"""Rejected development-only input/loss exploration after the Sen12 v3 gate."""

from __future__ import annotations

import ast
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import xarray as xr
from torch.utils.data import DataLoader, TensorDataset

from revised_models import ControlledPixelLoss, ControlledS3Net


STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "experiments/raw/external/sen12-s2/development"
OUTPUT = (
    STUDY
    / "experiments/derived/results/sen12_v4_development_exploration.json"
)
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
BANDS = ("B02", "B03", "B04", "B08")


def pre_post(value: object) -> tuple[int, int]:
    if isinstance(value, str):
        value = ast.literal_eval(value)
    if isinstance(value, (list, tuple)):
        value = value[0]
    if not isinstance(value, dict):
        raise ValueError("pre_post_dates must contain dictionaries")
    return int(value["pre"]), int(value["post"])


def load() -> tuple[list[str], np.ndarray, np.ndarray]:
    inventories: list[str] = []
    inputs = []
    targets = []
    for inventory_dir in sorted(DATA.iterdir()):
        for path in sorted(inventory_dir.glob("*.nc")):
            with xr.open_dataset(path, engine="h5netcdf") as dataset:
                pre, post = pre_post(dataset.attrs["pre_post_dates"])
                before = np.stack(
                    [
                        np.clip(
                            np.asarray(
                                dataset[band].isel(time=pre).values,
                                dtype=np.float32,
                            )
                            / 10000,
                            0,
                            1,
                        )
                        for band in BANDS
                    ]
                )
                after = np.stack(
                    [
                        np.clip(
                            np.asarray(
                                dataset[band].isel(time=post).values,
                                dtype=np.float32,
                            )
                            / 10000,
                            0,
                            1,
                        )
                        for band in BANDS
                    ]
                )
                dem = np.clip(
                    np.asarray(dataset["DEM"].isel(time=0).values, dtype=np.float32)
                    / 8800,
                    0,
                    1,
                )[None]
                target = np.asarray(
                    dataset["MASK"].isel(time=0).values > 0, dtype=np.float32
                )
            inventories.append(inventory_dir.name)
            inputs.append(np.concatenate([after, before, after - before, dem]))
            targets.append(target)
    return inventories, np.stack(inputs), np.stack(targets)


class ExtendedS3Net(ControlledS3Net):
    def controls(self, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return super().controls(inputs[:, :4])


class ImbalanceAwareLoss(ControlledPixelLoss):
    def __init__(self, mode: str, positive_weight: float):
        super().__init__(mode=mode, alpha=0.95)
        self.positive_weight = positive_weight

    def base_map(
        self, logits: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        positive = F.softplus(-logits) * targets * self.positive_weight
        negative = F.softplus(logits) * (1 - targets)
        bce = positive + negative
        probabilities = torch.sigmoid(logits)
        pt = targets * probabilities + (1 - targets) * (1 - probabilities)
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return 0.5 * bce + 0.5 * alpha_t * (1 - pt).pow(2) * bce

    def forward(
        self,
        logits: torch.Tensor,
        targets: torch.Tensor,
        images: torch.Tensor,
    ) -> torch.Tensor:
        return super().forward(logits, targets, images[:, :4])


def f1(probabilities: np.ndarray, target: np.ndarray, threshold: float) -> float:
    prediction = probabilities >= threshold
    truth = target >= 0.5
    tp = np.sum(prediction & truth)
    fp = np.sum(prediction & ~truth)
    fn = np.sum(~prediction & truth)
    return float(2 * tp / (2 * tp + fp + fn + 1e-12))


def run_one(
    name: str,
    channels: int,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_validation: np.ndarray,
    y_validation: np.ndarray,
) -> dict[str, object]:
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    model = ExtendedS3Net(in_ch=channels, gating_mode="none").to(DEVICE)
    positive_fraction = float(np.mean(y_train))
    positive_weight = min(40.0, (1 - positive_fraction) / positive_fraction)
    criterion = ImbalanceAwareLoss("base", positive_weight).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=16,
        shuffle=True,
        generator=torch.Generator().manual_seed(42),
    )
    history = []
    best_probabilities = None
    best_f1 = -1.0
    for epoch in range(1, 21):
        model.train()
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(batch_x), batch_y, batch_x)
            loss.backward()
            optimizer.step()
        model.eval()
        outputs = []
        with torch.no_grad():
            for start in range(0, len(x_validation), 16):
                outputs.append(
                    torch.sigmoid(
                        model(
                            torch.from_numpy(
                                x_validation[start : start + 16]
                            ).to(DEVICE)
                        )
                    )
                    .cpu()
                    .numpy()
                )
        probabilities = np.concatenate(outputs)
        score = f1(probabilities, y_validation, 0.5)
        history.append({"epoch": epoch, "f1_at_0_5": score})
        if score > best_f1:
            best_f1 = score
            best_probabilities = probabilities.copy()
    assert best_probabilities is not None
    thresholds = np.linspace(0.05, 0.95, 181)
    threshold_scores = [
        f1(best_probabilities, y_validation, float(threshold))
        for threshold in thresholds
    ]
    best_index = int(np.argmax(threshold_scores))
    return {
        "name": name,
        "channels": channels,
        "positive_weight": positive_weight,
        "history": history,
        "best_f1_at_0_5": best_f1,
        "best_validation_threshold": float(thresholds[best_index]),
        "best_threshold_f1": float(threshold_scores[best_index]),
    }


def main() -> None:
    inventories, inputs, targets = load()
    validation = np.asarray([value == "china" for value in inventories])
    representations = {
        "post": 4,
        "post_pre": 8,
        "post_pre_delta": 12,
        "post_pre_delta_dem": 13,
    }
    results = [
        run_one(
            name,
            channels,
            inputs[~validation, :channels],
            targets[~validation],
            inputs[validation, :channels],
            targets[validation],
        )
        for name, channels in representations.items()
    ]
    payload = {
        "status": "development-only-after-v3-kill",
        "disposition": (
            "rejected; not used by the v4 fit because no cross-inventory "
            "configuration exceeded F1 0.0762"
        ),
        "device": str(DEVICE),
        "train_inventories": ["chimanimani", "dominicamaria"],
        "validation_inventory": "china",
        "train_count": int(np.sum(~validation)),
        "validation_count": int(np.sum(validation)),
        "train_positive_fraction": float(np.mean(targets[~validation])),
        "validation_positive_fraction": float(np.mean(targets[validation])),
        "results": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                row["name"]: {
                    "f1_at_0_5": row["best_f1_at_0_5"],
                    "threshold_f1": row["best_threshold_f1"],
                }
                for row in results
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
