#!/usr/bin/env python3
"""Run the controlled experiment matrix requested by the GRSL reviewers.

The released HR-GLDD arrays contain no event IDs or coordinates. This script
therefore reports descriptive multi-seed results only; it does not compute
tile-level p-values or claim geographic/event independence.
"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import binary_dilation, binary_erosion
from torch.utils.data import DataLoader, TensorDataset

from data_semantics import BAND_ORDER, BAND_ORDER_ID, validate_nhwc
from revised_models import (
    ControlledPixelLoss,
    ControlledS3Net,
    ResUNet,
    UNet,
)


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent.parent
DATA_DIR = STUDY_DIR / "experiments/raw/hr_gldd"
RESULT_VARIANT = os.environ.get(
    "REMEDIATION_VARIANT", BAND_ORDER_ID.lower()
).lower()
if RESULT_VARIANT == "rgbn":
    RESULTS_DIR = STUDY_DIR / "experiments/derived/results/reviewer_remediation"
    CHECKPOINT_DIR = (
        STUDY_DIR / "experiments/derived/checkpoints/reviewer_remediation"
    )
else:
    RESULTS_DIR = (
        STUDY_DIR
        / f"experiments/derived/results/reviewer_remediation_{RESULT_VARIANT}"
    )
    CHECKPOINT_DIR = (
        STUDY_DIR
        / f"experiments/derived/checkpoints/reviewer_remediation_{RESULT_VARIANT}"
    )
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

SEEDS = tuple(
    int(value)
    for value in os.environ.get("REMEDIATION_SEEDS", "42,43,44").split(",")
    if value.strip()
)
EPOCHS = int(os.environ.get("REMEDIATION_EPOCHS", "15"))
BATCH_SIZE = int(os.environ.get("REMEDIATION_BATCH_SIZE", "32"))
DEVICE = torch.device(
    "mps"
    if torch.backends.mps.is_available()
    else ("cuda" if torch.cuda.is_available() else "cpu")
)

ARMS = {
    "unet_base": {"model": "unet", "gating": "none", "loss": "base"},
    "resunet_base": {"model": "resunet", "gating": "none", "loss": "base"},
    "s3_none_base": {"model": "s3", "gating": "none", "loss": "base"},
    "s3_raw_base": {"model": "s3", "gating": "raw", "loss": "base"},
    "s3_ndvi_base": {"model": "s3", "gating": "ndvi", "loss": "base"},
    "s3_ndvi_boundary": {
        "model": "s3",
        "gating": "ndvi",
        "loss": "boundary",
    },
    "s3_none_boundary": {
        "model": "s3",
        "gating": "none",
        "loss": "boundary",
    },
    "s3_raw_boundary": {
        "model": "s3",
        "gating": "raw",
        "loss": "boundary",
    },
    "s3_ndvi_biophysical": {
        "model": "s3",
        "gating": "ndvi",
        "loss": "biophysical",
    },
}
ACTIVE_ARMS = tuple(
    arm.strip()
    for arm in os.environ.get("REMEDIATION_ARMS", ",".join(ARMS)).split(",")
    if arm.strip()
)
unknown_arms = set(ACTIVE_ARMS) - set(ARMS)
if unknown_arms:
    raise ValueError(f"Unknown REMEDIATION_ARMS: {sorted(unknown_arms)}")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_data() -> dict[str, np.ndarray]:
    data = {
        f"{split}{kind}": np.load(DATA_DIR / f"{split}{kind}.npy")
        for split in ("train", "val", "test")
        for kind in ("X", "Y")
    }
    for split in ("train", "val", "test"):
        validate_nhwc(data[f"{split}X"])
    return data


def tensors(data: dict[str, np.ndarray]) -> dict[str, torch.Tensor]:
    output: dict[str, torch.Tensor] = {}
    for split in ("train", "val", "test"):
        output[f"x_{split}"] = (
            torch.from_numpy(data[f"{split}X"]).permute(0, 3, 1, 2).float()
        )
        output[f"y_{split}"] = (
            torch.from_numpy(data[f"{split}Y"])
            .permute(0, 3, 1, 2)
            .squeeze(1)
            .float()
        )
    return output


def make_model(config: dict[str, str]) -> torch.nn.Module:
    if config["model"] == "unet":
        return UNet()
    if config["model"] == "resunet":
        return ResUNet()
    return ControlledS3Net(gating_mode=config["gating"])


def loader(
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x, y),
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        generator=generator,
    )


def confusion(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float]:
    predictions = probabilities >= threshold
    truth = targets >= 0.5
    tp = int(np.sum(predictions & truth))
    fp = int(np.sum(predictions & ~truth))
    fn = int(np.sum(~predictions & truth))
    tn = int(np.sum(~predictions & ~truth))
    precision = tp / (tp + fp + 1e-12)
    recall = tp / (tp + fn + 1e-12)
    f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
    iou = tp / (tp + fp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    return {
        "threshold": float(threshold),
        "f1": float(f1),
        "iou": float(iou),
        "precision": float(precision),
        "recall": float(recall),
        "background_fpr": float(fpr),
        "false_positive_pixels": fp,
        "false_positive_area_km2": float(fp * 9e-6),
    }


def tile_f1(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    predictions = probabilities >= threshold
    values = []
    for prediction, truth in zip(predictions, targets >= 0.5):
        tp = np.sum(prediction & truth)
        fp = np.sum(prediction & ~truth)
        fn = np.sum(~prediction & truth)
        denominator = 2 * tp + fp + fn
        values.append(1.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def boundary_mask(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def boundary_f1(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    tolerance_pixels: int = 1,
) -> float:
    predictions = probabilities >= threshold
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for prediction, truth in zip(predictions, targets >= 0.5):
        pred_boundary = boundary_mask(prediction)
        true_boundary = boundary_mask(truth)
        if tolerance_pixels:
            true_match = binary_dilation(
                true_boundary, structure=structure, iterations=tolerance_pixels
            )
            pred_match = binary_dilation(
                pred_boundary, structure=structure, iterations=tolerance_pixels
            )
        else:
            true_match = true_boundary
            pred_match = pred_boundary
        if not pred_boundary.any() and not true_boundary.any():
            values.append(1.0)
            continue
        precision = np.sum(pred_boundary & true_match) / (
            np.sum(pred_boundary) + 1e-12
        )
        recall = np.sum(true_boundary & pred_match) / (
            np.sum(true_boundary) + 1e-12
        )
        values.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(values))


def evaluate(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    metrics = confusion(probabilities, targets, threshold)
    metrics["macro_tile_f1"] = tile_f1(probabilities, targets, threshold)
    metrics["boundary_f1_tolerance_1px"] = boundary_f1(
        probabilities, targets, threshold, tolerance_pixels=1
    )
    return metrics


def choose_threshold_for_recall(
    probabilities: np.ndarray, targets: np.ndarray, target_recall: float
) -> dict[str, float | str]:
    """Select the exact score breakpoint nearest the target validation recall.

    Only positive-label scores determine recall. Candidate breakpoints around the
    desired positive rank are evaluated exactly; ties prefer the higher threshold.
    """
    positive_scores = np.asarray(probabilities[targets >= 0.5], dtype=np.float64)
    if not len(positive_scores):
        raise ValueError("Cannot match recall without positive validation labels")
    ordered = np.sort(positive_scores)
    desired_tp = int(round(target_recall * len(ordered)))
    pivot = min(max(len(ordered) - desired_tp, 0), len(ordered) - 1)
    candidates = {0.0, 1.0}
    for index in range(max(0, pivot - 2), min(len(ordered), pivot + 3)):
        score = float(ordered[index])
        candidates.add(score)
        candidates.add(float(np.nextafter(score, np.inf)))
    scored = []
    for threshold in candidates:
        achieved = float(np.mean(positive_scores >= threshold))
        scored.append((abs(achieved - target_recall), -threshold, achieved))
    mismatch, negative_threshold, achieved = min(scored)
    return {
        "threshold": float(-negative_threshold),
        "target_recall": float(target_recall),
        "achieved_validation_recall": achieved,
        "absolute_recall_mismatch": float(mismatch),
        "tie_policy": "minimum recall mismatch, then highest threshold",
    }


def infer(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    *,
    degraded_resolution: bool = False,
) -> np.ndarray:
    model.eval()
    batches = []
    data_loader = loader(x, y, shuffle=False, seed=0)
    with torch.no_grad():
        for batch_x, _ in data_loader:
            batch_x = batch_x.to(DEVICE)
            if degraded_resolution:
                batch_x = F.interpolate(
                    batch_x,
                    size=(38, 38),
                    mode="bilinear",
                    align_corners=False,
                    antialias=True,
                )
                batch_x = F.interpolate(
                    batch_x,
                    size=(128, 128),
                    mode="bilinear",
                    align_corners=False,
                )
            batches.append(torch.sigmoid(model(batch_x)).cpu().numpy())
    return np.concatenate(batches).astype(np.float32)


def train_one(
    arm: str,
    config: dict[str, str],
    seed: int,
    tensor_data: dict[str, torch.Tensor],
) -> dict[str, Any]:
    set_seed(seed)
    model = make_model(config).to(DEVICE)
    criterion = ControlledPixelLoss(mode=config["loss"]).to(DEVICE)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=1.5e-3, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS, eta_min=1e-5
    )
    train_loader = loader(
        tensor_data["x_train"],
        tensor_data["y_train"],
        shuffle=True,
        seed=seed,
    )
    val_loader = loader(
        tensor_data["x_val"],
        tensor_data["y_val"],
        shuffle=False,
        seed=seed,
    )
    best_f1 = -1.0
    best_state = None
    started = time.time()
    for _epoch in range(EPOCHS):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y, batch_x)
            loss.backward()
            optimizer.step()
        scheduler.step()

        model.eval()
        tp = fp = fn = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(DEVICE)
                prediction = torch.sigmoid(model(batch_x)) >= 0.5
                truth = batch_y.to(DEVICE) >= 0.5
                tp += int(torch.sum(prediction & truth))
                fp += int(torch.sum(prediction & ~truth))
                fn += int(torch.sum(~prediction & truth))
        val_f1 = 2 * tp / (2 * tp + fp + fn + 1e-12)
        if val_f1 > best_f1:
            best_f1 = float(val_f1)
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
    if best_state is None:
        raise RuntimeError(f"No checkpoint selected for {arm}, seed {seed}")
    model.load_state_dict(
        {key: value.to(DEVICE) for key, value in best_state.items()}
    )
    checkpoint = CHECKPOINT_DIR / f"{arm}_seed{seed}.pt"
    torch.save(best_state, checkpoint)
    val_probabilities = infer(
        model, tensor_data["x_val"], tensor_data["y_val"]
    )
    test_probabilities = infer(
        model, tensor_data["x_test"], tensor_data["y_test"]
    )
    degraded_probabilities = infer(
        model,
        tensor_data["x_test"],
        tensor_data["y_test"],
        degraded_resolution=True,
    )
    np.save(
        RESULTS_DIR / f"val_probabilities_{arm}_seed{seed}.npy",
        val_probabilities,
    )
    np.save(
        RESULTS_DIR / f"test_probabilities_{arm}_seed{seed}.npy",
        test_probabilities,
    )
    np.save(
        RESULTS_DIR / f"test_probabilities_10m_{arm}_seed{seed}.npy",
        degraded_probabilities,
    )
    return {
        "arm": arm,
        "seed": seed,
        "config": config,
        "parameters": sum(parameter.numel() for parameter in model.parameters()),
        "best_validation_f1": best_f1,
        "duration_seconds": time.time() - started,
        "checkpoint": str(checkpoint.relative_to(STUDY_DIR)),
        "validation_default": evaluate(
            val_probabilities,
            tensor_data["y_val"].numpy(),
            threshold=0.5,
        ),
        "test_default": evaluate(
            test_probabilities,
            tensor_data["y_test"].numpy(),
            threshold=0.5,
        ),
        "test_controlled_10m": evaluate(
            degraded_probabilities,
            tensor_data["y_test"].numpy(),
            threshold=0.5,
        ),
    }


def aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    metric_groups = (
        "test_default",
        "test_matched_validation_recall",
        "test_controlled_10m",
    )
    arms_present = [
        arm for arm in ARMS if any(run["arm"] == arm for run in runs)
    ]
    for arm in arms_present:
        arm_runs = [run for run in runs if run["arm"] == arm]
        output[arm] = {
            "config": ARMS[arm],
            "parameters": arm_runs[0]["parameters"],
            "n_seeds": len(arm_runs),
        }
        for group in metric_groups:
            keys = arm_runs[0][group]
            output[arm][group] = {
                key: {
                    "mean": float(np.mean([run[group][key] for run in arm_runs])),
                    "std": float(np.std([run[group][key] for run in arm_runs])),
                }
                for key in keys
                if isinstance(arm_runs[0][group][key], (int, float))
            }
    return output


def main() -> None:
    print(
        f"Device={DEVICE}; epochs={EPOCHS}; batch={BATCH_SIZE}; "
        f"bands={BAND_ORDER}",
        flush=True,
    )
    data = load_data()
    tensor_data = tensors(data)
    output = RESULTS_DIR / "reviewer_remediation_summary.json"
    append = os.environ.get("REMEDIATION_APPEND", "0") == "1"
    runs = []
    if append and output.is_file():
        previous = json.loads(output.read_text())
        active_pairs = {
            (arm, seed) for arm in ACTIVE_ARMS for seed in SEEDS
        }
        runs = [
            run
            for run in previous["runs"]
            if (run["arm"], run["seed"]) not in active_pairs
        ]
    for seed in SEEDS:
        for arm in ACTIVE_ARMS:
            config = ARMS[arm]
            print(f"Training {arm}, seed {seed}", flush=True)
            run = train_one(arm, config, seed, tensor_data)
            runs.append(run)
            print(
                f"  F1={run['test_default']['f1']:.4f}; "
                f"boundary={run['test_default']['boundary_f1_tolerance_1px']:.4f}; "
                f"FPR={run['test_default']['background_fpr']:.4f}; "
                f"{run['duration_seconds']:.1f}s",
                flush=True,
            )

    targets_val = tensor_data["y_val"].numpy()
    targets_test = tensor_data["y_test"].numpy()
    recall_reference_arm = "unet_base"
    if not all(
        (
            RESULTS_DIR
            / f"val_probabilities_{recall_reference_arm}_seed{seed}.npy"
        ).is_file()
        for seed in SEEDS
    ):
        recall_reference_arm = ACTIVE_ARMS[0]
    for seed in SEEDS:
        reference_val = np.load(
            RESULTS_DIR
            / f"val_probabilities_{recall_reference_arm}_seed{seed}.npy"
        ).astype(np.float32)
        target_recall = confusion(reference_val, targets_val, 0.5)["recall"]
        for arm in ACTIVE_ARMS:
            val_probabilities = np.load(
                RESULTS_DIR / f"val_probabilities_{arm}_seed{seed}.npy"
            ).astype(np.float32)
            test_probabilities = np.load(
                RESULTS_DIR / f"test_probabilities_{arm}_seed{seed}.npy"
            ).astype(np.float32)
            selection = choose_threshold_for_recall(
                val_probabilities, targets_val, target_recall
            )
            run = next(
                item
                for item in runs
                if item["arm"] == arm and item["seed"] == seed
            )
            run["matched_validation_recall_target"] = target_recall
            run["matched_validation_recall_achieved"] = selection[
                "achieved_validation_recall"
            ]
            run["matched_validation_recall_mismatch"] = selection[
                "absolute_recall_mismatch"
            ]
            run["matched_validation_recall_tie_policy"] = selection["tie_policy"]
            run["test_matched_validation_recall"] = evaluate(
                test_probabilities, targets_test, float(selection["threshold"])
            )

    payload = {
        "protocol": {
            "dataset": "HR-GLDD official mixed arrays",
            "band_order": list(BAND_ORDER),
            "seeds": list(SEEDS),
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "device": str(DEVICE),
            "event_ids_available": False,
            "geographic_coordinates_available": False,
            "inferential_statistics": (
                "Not reported: the released arrays do not identify independent "
                "events or spatial blocks."
            ),
            "boundary_metric": (
                "Per-tile boundary F1 with one-pixel Chebyshev "
                "(3x3, 8-neighbour) symmetric matching tolerance."
            ),
            "matched_recall": (
                "Each model threshold is selected at the exact validation-score "
                "breakpoint nearest U-Net validation recall at threshold 0.5; "
                "ties prefer the higher threshold, which is then applied to test."
            ),
            "controlled_resolution": (
                "PlanetScope test arrays downsampled 128->38 and bilinearly "
                "restored to 128, holding sensor, region, labels and model fixed."
            ),
        },
        "arms": {
            arm: ARMS[arm]
            for arm in ARMS
            if any(run["arm"] == arm for run in runs)
        },
        "runs": runs,
        "aggregate": aggregate(runs),
    }
    output.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Saved {output}", flush=True)


if __name__ == "__main__":
    main()
