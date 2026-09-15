#!/usr/bin/env python3
"""Diagnose empty-crop, threshold, and event-family sensitivity in LRD."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile
from scipy.ndimage import binary_dilation, binary_erosion


STUDY = Path(__file__).resolve().parents[2]
DATA = STUDY / "experiments/raw/external/lrd-optical"
SUMMARY = (
    STUDY
    / "experiments/derived/results/lrd_boundary_confirmation/"
    "lrd_confirmation_summary.json"
)
FAMILIES = (
    STUDY
    / "research/dataset-metadata/lrd-prospective-confirmation/"
    "event_family_audit.json"
)
OUTPUT = (
    STUDY
    / "experiments/derived/results/lrd_boundary_confirmation/"
    "endpoint_sensitivity.json"
)


def one(root: Path, pattern: str) -> Path:
    matches = list(root.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"{root} {pattern}: expected 1, got {len(matches)}")
    return matches[0]


def selected_masks(
    fold: str, eid: str, crop_size: int, limit: int, salt: str
) -> np.ndarray:
    mask = tifffile.imread(one(DATA / fold / eid, "*__MASK.tif"))
    while mask.ndim > 2:
        mask = mask[..., 0]
    truth = mask > 0
    height, width = truth.shape
    padded_height = ((height + crop_size - 1) // crop_size) * crop_size
    padded_width = ((width + crop_size - 1) // crop_size) * crop_size
    truth = np.pad(
        truth,
        ((0, padded_height - height), (0, padded_width - width)),
        mode="constant",
        constant_values=False,
    )
    coordinates = [
        (y, x)
        for y in range(0, padded_height, crop_size)
        for x in range(0, padded_width, crop_size)
    ]
    coordinates.sort(
        key=lambda row: hashlib.sha256(
            f"{salt}:{eid}:{row[0]}:{row[1]}".encode()
        ).hexdigest()
    )
    return np.stack(
        [
            truth[y : y + crop_size, x : x + crop_size]
            for y, x in coordinates[:limit]
        ]
    )


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def crop_macro_boundary_f1(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
    empty_empty_score: float,
) -> float:
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for prediction, truth in zip(probabilities >= threshold, targets):
        predicted_boundary = boundary(prediction)
        truth_boundary = boundary(truth)
        if not predicted_boundary.any() and not truth_boundary.any():
            values.append(empty_empty_score)
            continue
        precision = np.sum(
            predicted_boundary
            & binary_dilation(truth_boundary, structure=structure)
        ) / (np.sum(predicted_boundary) + 1e-12)
        recall = np.sum(
            truth_boundary
            & binary_dilation(predicted_boundary, structure=structure)
        ) / (np.sum(truth_boundary) + 1e-12)
        values.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(values))


def pooled_boundary_f1(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    structure = np.ones((3, 3), dtype=bool)
    matched_prediction = 0
    predicted_count = 0
    matched_truth = 0
    truth_count = 0
    for prediction, truth in zip(probabilities >= threshold, targets):
        predicted_boundary = boundary(prediction)
        truth_boundary = boundary(truth)
        matched_prediction += int(
            np.sum(
                predicted_boundary
                & binary_dilation(truth_boundary, structure=structure)
            )
        )
        predicted_count += int(np.sum(predicted_boundary))
        matched_truth += int(
            np.sum(
                truth_boundary
                & binary_dilation(predicted_boundary, structure=structure)
            )
        )
        truth_count += int(np.sum(truth_boundary))
    precision = matched_prediction / (predicted_count + 1e-12)
    recall = matched_truth / (truth_count + 1e-12)
    return float(2 * precision * recall / (precision + recall + 1e-12))


def binary_f1(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    prediction = probabilities >= threshold
    truth = targets >= 0.5
    tp = int(np.sum(prediction & truth))
    fp = int(np.sum(prediction & ~truth))
    fn = int(np.sum(~prediction & truth))
    return float(2 * tp / (2 * tp + fp + fn + 1e-12))


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    config = payload["config"]
    family = json.loads(FAMILIES.read_text())
    fold_keys = {
        "development": "development_eids",
        "validation": "validation_eids",
        "protected": "protected_test_eids",
    }
    masks = {
        fold: {
            eid: selected_masks(
                fold,
                eid,
                config["crop_size"],
                config["crops_per_event"],
                config["crop_salt"],
            )
            for eid in config[key]
        }
        for fold, key in fold_keys.items()
    }
    empty_counts = {
        fold: {
            "empty": int(
                sum(np.sum(~np.any(rows, axis=(1, 2))) for rows in by_eid.values())
            ),
            "total": int(sum(len(rows) for rows in by_eid.values())),
        }
        for fold, by_eid in masks.items()
    }
    for row in empty_counts.values():
        row["fraction"] = row["empty"] / row["total"]

    validation_targets = np.concatenate(list(masks["validation"].values()))
    absolute_performance = {}
    run_lookup = {}
    for run in payload["training_records"]:
        key = f"{run['loss_mode']}-seed{run['seed']}"
        run_lookup[(run["loss_mode"], run["seed"])] = run
        val_probabilities = np.load(
            STUDY / run["validation_probabilities"], mmap_mode="r"
        )
        protected_f1 = np.mean(
            [event["f1"] for event in run["test_events"].values()]
        )
        protected_bf1 = np.mean(
            [event["boundary_f1"] for event in run["test_events"].values()]
        )
        absolute_performance[key] = {
            "threshold": run["validation_threshold"],
            "maximum_validation_f1": binary_f1(
                val_probabilities,
                validation_targets,
                run["validation_threshold"],
            ),
            "protected_event_macro_f1": float(protected_f1),
            "protected_event_macro_boundary_f1_frozen": float(protected_bf1),
        }

    variants = {
        "frozen_crop_macro_empty_one_selected_threshold": (
            "crop",
            1.0,
            "selected",
        ),
        "crop_macro_empty_zero_selected_threshold": ("crop", 0.0, "selected"),
        "pooled_boundary_selected_threshold": ("pooled", 0.0, "selected"),
        "pooled_boundary_common_0_5": ("pooled", 0.0, "common"),
    }
    sensitivity = {}
    protected_eids = config["protected_test_eids"]
    for variant, (aggregation, empty_score, threshold_mode) in variants.items():
        effects = {}
        for seed in config["seeds"]:
            base = run_lookup[("base", seed)]
            weighted = run_lookup[("boundary", seed)]
            per_event = {}
            for eid in protected_eids:
                target = masks["protected"][eid]
                base_prob = np.load(
                    STUDY / base["test_events"][eid]["probabilities"],
                    mmap_mode="r",
                )
                weighted_prob = np.load(
                    STUDY / weighted["test_events"][eid]["probabilities"],
                    mmap_mode="r",
                )
                base_threshold = (
                    0.5
                    if threshold_mode == "common"
                    else base["validation_threshold"]
                )
                weighted_threshold = (
                    0.5
                    if threshold_mode == "common"
                    else weighted["validation_threshold"]
                )
                function = (
                    pooled_boundary_f1
                    if aggregation == "pooled"
                    else lambda p, t, x: crop_macro_boundary_f1(
                        p, t, x, empty_score
                    )
                )
                per_event[eid] = float(
                    function(weighted_prob, target, weighted_threshold)
                    - function(base_prob, target, base_threshold)
                )
            effects[str(seed)] = per_event
        event_means = {
            eid: float(
                np.mean([effects[str(seed)][eid] for seed in config["seeds"]])
            )
            for eid in protected_eids
        }
        valid_eids = family["valid_nonoverlapping_protected_eids"]
        sensitivity[variant] = {
            "seed_macro_effects_points": {
                seed: 100 * float(np.mean(list(rows.values())))
                for seed, rows in effects.items()
            },
            "event_mean_effects_points": {
                eid: 100 * value for eid, value in event_means.items()
            },
            "all_eid_mean_effect_points": 100
            * float(np.mean(list(event_means.values()))),
            "valid_nonoverlap_eid_mean_effect_points": 100
            * float(np.mean([event_means[eid] for eid in valid_eids])),
            "positive_all_eids": all(value > 0 for value in event_means.values()),
            "positive_valid_nonoverlap_eids": all(
                event_means[eid] > 0 for eid in valid_eids
            ),
        }
    output = {
        "schema_version": 1,
        "analysis_classification": (
            "post-hoc diagnostic of a failed internally precommitted LRD run"
        ),
        "empty_crop_counts": empty_counts,
        "absolute_performance": absolute_performance,
        "event_family_audit": {
            "compromised_protected_eids": family[
                "compromised_protected_eids"
            ],
            "valid_nonoverlapping_protected_eids": family[
                "valid_nonoverlapping_protected_eids"
            ],
        },
        "boundary_endpoint_sensitivity": sensitivity,
        "frozen_f1_effect_points": 100
        * payload["mean_event_macro_delta_f1"],
        "conclusion": (
            "The frozen rule fails arithmetically, but the boundary magnitude "
            "and EID direction are not stable to the empty-crop convention, "
            "threshold, or trigger-family audit. Treat external evidence as "
            "failed/indeterminate, not a robust transfer rejection."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
