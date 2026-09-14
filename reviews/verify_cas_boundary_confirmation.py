#!/usr/bin/env python3
"""Independently recompute the preregistered CAS confirmation metrics."""

from __future__ import annotations

import hashlib
import json
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import numpy as np
from PIL import Image
from scipy.ndimage import binary_dilation, binary_erosion


STUDY = Path(__file__).resolve().parent.parent
CAS_ROOT = STUDY / "experiments/raw/external/cas"
RESULTS = (
    STUDY / "experiments/derived/results/cas_boundary_confirmation"
)
SUMMARY = RESULTS / "cas_boundary_confirmation_summary.json"
FOLDS = (
    STUDY
    / "research/dataset-metadata/cas-boundary-confirmation/folds.json"
)
CROP_SIZE = 128
EVAL_TILE_LIMIT = 160
SELECTION_SALT = "r010-cas-boundary-crops-v1"
SEEDS = (42, 43, 44)
EVENT_ARCHIVES = {
    "Mengdong": "Mengdong Township.zip",
    "Moxi-UAV-1m": "Moxi town（UAV-1m）.zip",
    "Tiburon-Planet": "Tiburon Peninsula（planet）.zip",
}


def normalized_stem(path: str) -> str:
    return Path(path).stem.lower().replace("_mask", "").replace("-mask", "")


def selected_masks(event: str) -> np.ndarray:
    path = CAS_ROOT / EVENT_ARCHIVES[event]
    with ZipFile(path) as archive:
        image_members: dict[str, str] = {}
        mask_members: dict[str, str] = {}
        for name in archive.namelist():
            if name.endswith("/"):
                continue
            parts = [part.lower() for part in Path(name).parts]
            key = normalized_stem(name)
            if "img" in parts:
                image_members[key] = name
            elif "mask" in parts:
                mask_members[key] = name
        shared = sorted(set(image_members) & set(mask_members))
        shared.sort(
            key=lambda key: hashlib.sha256(
                f"{SELECTION_SALT}:{event}:{key}".encode()
            ).hexdigest()
        )
        crops = []
        for key in shared[:EVAL_TILE_LIMIT]:
            array = np.asarray(
                Image.open(BytesIO(archive.read(mask_members[key])))
            )
            while array.ndim > 2:
                array = array[..., 0]
            array = array > 0
            for y in range(0, 512, CROP_SIZE):
                for x in range(0, 512, CROP_SIZE):
                    crops.append(
                        array[y : y + CROP_SIZE, x : x + CROP_SIZE]
                    )
    return np.stack(crops)


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def boundary_f1(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    predictions = probabilities >= threshold
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for prediction, truth in zip(predictions, targets):
        pred_boundary = boundary(prediction)
        true_boundary = boundary(truth)
        true_match = binary_dilation(true_boundary, structure=structure)
        pred_match = binary_dilation(pred_boundary, structure=structure)
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


def metrics(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> dict[str, float | int]:
    prediction = probabilities >= threshold
    tp = int(np.sum(prediction & targets))
    fp = int(np.sum(prediction & ~targets))
    fn = int(np.sum(~prediction & targets))
    tn = int(np.sum(~prediction & ~targets))
    return {
        "threshold": float(threshold),
        "f1": 2 * tp / (2 * tp + fp + fn + 1e-12),
        "iou": tp / (tp + fp + fn + 1e-12),
        "precision": tp / (tp + fp + 1e-12),
        "recall": tp / (tp + fn + 1e-12),
        "background_fpr": fp / (fp + tn + 1e-12),
        "false_positive_pixels": fp,
        "boundary_f1": boundary_f1(probabilities, targets, threshold),
    }


def close(first: float | int, second: float | int) -> bool:
    return bool(np.isclose(first, second, rtol=0, atol=1e-10))


def main() -> None:
    payload = json.loads(SUMMARY.read_text(encoding="utf-8"))
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    if set(folds["protected_test_events"]) != set(EVENT_ARCHIVES):
        raise RuntimeError("Verifier event list differs from frozen protected events")
    targets = {event: selected_masks(event) for event in EVENT_ARCHIVES}

    checks = []
    for run in payload["training_records"]:
        for event, reported in run["test_events"].items():
            probabilities = np.load(
                STUDY / reported["probabilities"], mmap_mode="r"
            )
            recomputed = metrics(
                np.asarray(probabilities),
                targets[event],
                float(run["validation_threshold"]),
            )
            for metric, value in recomputed.items():
                checks.append(
                    {
                        "loss_mode": run["loss_mode"],
                        "seed": run["seed"],
                        "event": event,
                        "metric": metric,
                        "reported": reported[metric],
                        "recomputed": value,
                        "matched": close(reported[metric], value),
                    }
                )

    effects = {}
    for seed in SEEDS:
        base = next(
            run
            for run in payload["training_records"]
            if run["seed"] == seed and run["loss_mode"] == "base"
        )
        boundary_run = next(
            run
            for run in payload["training_records"]
            if run["seed"] == seed and run["loss_mode"] == "boundary"
        )
        per_event = {}
        for event in EVENT_ARCHIVES:
            per_event[event] = {
                "delta_f1": (
                    boundary_run["test_events"][event]["f1"]
                    - base["test_events"][event]["f1"]
                ),
                "delta_boundary_f1": (
                    boundary_run["test_events"][event]["boundary_f1"]
                    - base["test_events"][event]["boundary_f1"]
                ),
            }
        effects[str(seed)] = {
            "event_macro_delta_f1": float(
                np.mean([value["delta_f1"] for value in per_event.values()])
            ),
            "event_macro_delta_boundary_f1": float(
                np.mean(
                    [
                        value["delta_boundary_f1"]
                        for value in per_event.values()
                    ]
                )
            ),
        }
    for seed in SEEDS:
        for metric, value in effects[str(seed)].items():
            reported = payload["effects"][str(seed)][metric]
            checks.append(
                {
                    "loss_mode": "contrast",
                    "seed": seed,
                    "event": "event-macro",
                    "metric": metric,
                    "reported": reported,
                    "recomputed": value,
                    "matched": close(reported, value),
                }
            )

    failed = [check for check in checks if not check["matched"]]
    output = {
        "run": "R011b-cas-boundary-confirmation",
        "independence_unit": "CAS subdataset region/event",
        "protected_events": list(EVENT_ARCHIVES),
        "checks": checks,
        "failed": failed,
        "status": "passed" if not failed else "failed",
    }
    destination = STUDY / "reviews/verified-cas-boundary-confirmation.json"
    destination.write_text(json.dumps(output, indent=2) + "\n")
    if failed:
        raise SystemExit(f"{len(failed)} CAS metric checks failed")
    print(f"PASS: {len(checks)} CAS metrics independently recomputed")


if __name__ == "__main__":
    main()
