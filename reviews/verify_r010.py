#!/usr/bin/env python3
"""Independent recomputation of an HR-GLDD order-sensitivity result set."""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion


STUDY = Path(__file__).resolve().parent.parent
RESULT_VARIANT = os.environ.get("HRGLDD_RESULTS_VARIANT", "rgbn").lower()
RESULTS = (
    STUDY / "experiments/derived/results/reviewer_remediation"
    if RESULT_VARIANT == "rgbn"
    else STUDY
    / f"experiments/derived/results/reviewer_remediation_{RESULT_VARIANT}"
)
TEST_TARGETS = np.load(
    STUDY / "experiments/raw/hr_gldd/testY.npy", mmap_mode="r"
)[..., 0].astype(bool)
VAL_TARGETS = np.load(
    STUDY / "experiments/raw/hr_gldd/valY.npy", mmap_mode="r"
)[..., 0].astype(bool)


def boundary(mask: np.ndarray) -> np.ndarray:
    structure = np.ones((3, 3), dtype=bool)
    return binary_dilation(mask, structure=structure) ^ binary_erosion(
        mask, structure=structure, border_value=0
    )


def macro_f1(prediction: np.ndarray, targets: np.ndarray) -> float:
    values = []
    for pred, truth in zip(prediction, targets):
        tp = np.sum(pred & truth)
        fp = np.sum(pred & ~truth)
        fn = np.sum(~pred & truth)
        denominator = 2 * tp + fp + fn
        values.append(1.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(values))


def boundary_f1(prediction: np.ndarray, targets: np.ndarray) -> float:
    structure = np.ones((3, 3), dtype=bool)
    values = []
    for pred, truth in zip(prediction, targets):
        pred_boundary = boundary(pred)
        truth_boundary = boundary(truth)
        truth_match = binary_dilation(truth_boundary, structure=structure)
        pred_match = binary_dilation(pred_boundary, structure=structure)
        if not pred_boundary.any() and not truth_boundary.any():
            values.append(1.0)
            continue
        precision = np.sum(pred_boundary & truth_match) / (
            np.sum(pred_boundary) + 1e-12
        )
        recall = np.sum(truth_boundary & pred_match) / (
            np.sum(truth_boundary) + 1e-12
        )
        values.append(2 * precision * recall / (precision + recall + 1e-12))
    return float(np.mean(values))


def metrics(
    probabilities: np.ndarray,
    targets: np.ndarray,
    threshold: float,
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
        "false_positive_area_km2": fp * 9e-6,
        "macro_tile_f1": macro_f1(prediction, targets),
        "boundary_f1_tolerance_1px": boundary_f1(prediction, targets),
    }


def close(left: float | int, right: float | int, tolerance: float = 1e-10) -> bool:
    return abs(float(left) - float(right)) <= tolerance


def record_group(
    checks: list[dict],
    run: dict,
    group: str,
    probabilities: np.ndarray,
    threshold: float,
) -> None:
    recomputed = metrics(probabilities, TEST_TARGETS, threshold)
    for key, value in recomputed.items():
        reported = run[group][key]
        checks.append(
            {
                "arm": run["arm"],
                "seed": run["seed"],
                "group": group,
                "metric": key,
                "reported": reported,
                "recomputed": value,
                "matched": close(reported, value),
            }
        )


def main() -> None:
    summary = json.loads((RESULTS / "reviewer_remediation_summary.json").read_text())
    checks: list[dict] = []
    for run in summary["runs"]:
        arm = run["arm"]
        seed = run["seed"]
        test = np.load(
            RESULTS / f"test_probabilities_{arm}_seed{seed}.npy"
        ).astype(np.float32)
        degraded = np.load(
            RESULTS / f"test_probabilities_10m_{arm}_seed{seed}.npy"
        ).astype(np.float32)
        validation = np.load(
            RESULTS / f"val_probabilities_{arm}_seed{seed}.npy"
        ).astype(np.float32)

        record_group(checks, run, "test_default", test, 0.5)
        matched_threshold = run["test_matched_validation_recall"]["threshold"]
        record_group(
            checks,
            run,
            "test_matched_validation_recall",
            test,
            matched_threshold,
        )
        record_group(
            checks,
            run,
            "test_controlled_10m",
            degraded,
            0.5,
        )
        achieved = metrics(
            validation,
            VAL_TARGETS,
            matched_threshold,
        )["recall"]
        checks.append(
            {
                "arm": arm,
                "seed": seed,
                "group": "threshold_selection",
                "metric": "achieved_validation_recall",
                "reported": run["matched_validation_recall_achieved"],
                "recomputed": achieved,
                "matched": close(
                    run["matched_validation_recall_achieved"], achieved
                ),
            }
        )
        mismatch = abs(achieved - run["matched_validation_recall_target"])
        checks.append(
            {
                "arm": arm,
                "seed": seed,
                "group": "threshold_selection",
                "metric": "absolute_recall_mismatch",
                "reported": run["matched_validation_recall_mismatch"],
                "recomputed": mismatch,
                "matched": close(
                    run["matched_validation_recall_mismatch"], mismatch
                ),
            }
        )

    for arm, aggregate in summary["aggregate"].items():
        arm_runs = [run for run in summary["runs"] if run["arm"] == arm]
        for group in (
            "test_default",
            "test_matched_validation_recall",
            "test_controlled_10m",
        ):
            for metric, reported in aggregate[group].items():
                values = [float(run[group][metric]) for run in arm_runs]
                for statistic, value in (
                    ("mean", float(np.mean(values))),
                    ("std", float(np.std(values))),
                ):
                    checks.append(
                        {
                            "arm": arm,
                            "seed": "aggregate",
                            "group": group,
                            "metric": f"{metric}.{statistic}",
                            "reported": reported[statistic],
                            "recomputed": value,
                            "matched": close(reported[statistic], value),
                        }
                    )

    failed = [check for check in checks if not check["matched"]]
    output = {
        "run": f"order-sensitivity-{RESULT_VARIANT}",
        "probability_precision": "float32 saved; tolerance 1e-10",
        "coverage": (
            "default, matched-recall, controlled-resolution, threshold-selection, "
            "and aggregate metrics"
        ),
        "checks": checks,
        "failed": failed,
        "status": "passed" if not failed else "failed",
    }
    destination = (
        STUDY / "reviews/verified-r010.json"
        if RESULT_VARIANT == "rgbn"
        else STUDY / f"reviews/verified-{RESULT_VARIANT}-sensitivity.json"
    )
    destination.write_text(
        json.dumps(output, indent=2) + "\n"
    )
    if failed:
        raise SystemExit(f"{len(failed)} metric checks failed")
    print(
        f"PASS: {len(checks)} {RESULT_VARIANT.upper()} metrics "
        "independently recomputed"
    )


if __name__ == "__main__":
    main()
