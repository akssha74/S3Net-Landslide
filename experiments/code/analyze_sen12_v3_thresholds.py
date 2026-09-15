#!/usr/bin/env python3
"""Reproduce the development-only threshold diagnostic after the v3 gate failed."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sen12_s2_confirmation as run


STUDY = Path(__file__).resolve().parents[2]
V3_FIT = (
    STUDY / "experiments/derived/results/sen12_s2_confirmation/fit_decisions.json"
)
OUTPUT = (
    STUDY
    / "experiments/derived/results/sen12_v3_threshold_diagnostic.json"
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def score(
    probabilities: np.ndarray, targets: np.ndarray, threshold: float
) -> float:
    prediction = probabilities >= threshold
    truth = targets >= 0.5
    true_positive = np.sum(prediction & truth)
    false_positive = np.sum(prediction & ~truth)
    false_negative = np.sum(~prediction & truth)
    return float(
        2
        * true_positive
        / (2 * true_positive + false_positive + false_negative + 1e-12)
    )


def main() -> None:
    data = run.load_fold("development")
    selected = np.asarray(
        [value == "china" for value in data["inventories"]], dtype=bool
    )
    targets = data["y"][selected]
    fit = json.loads(V3_FIT.read_text())
    thresholds = np.linspace(0.005, 0.5, 100)
    rows = []
    for fit_row in fit["runs"]:
        prediction_path = STUDY / fit_row["validation_predictions"]
        probabilities = np.load(prediction_path, allow_pickle=False)
        values = [
            score(probabilities, targets, float(threshold))
            for threshold in thresholds
        ]
        best = int(np.argmax(values))
        rows.append(
            {
                "arm": fit_row["arm"],
                "seed": fit_row["seed"],
                "prediction_path": fit_row["validation_predictions"],
                "prediction_sha256": file_sha256(prediction_path),
                "f1_at_0_5": score(probabilities, targets, 0.5),
                "best_grid_threshold": float(thresholds[best]),
                "best_grid_f1": float(values[best]),
            }
        )
    payload = {
        "status": "development-only-after-v3-kill",
        "v3_fit_decisions_sha256": file_sha256(V3_FIT),
        "validation_inventory": "china",
        "validation_count": int(np.sum(selected)),
        "threshold_grid": {
            "minimum": float(thresholds[0]),
            "maximum": float(thresholds[-1]),
            "count": len(thresholds),
        },
        "maximum_best_grid_f1": max(row["best_grid_f1"] for row in rows),
        "all_best_grid_f1_below_0_25": all(
            row["best_grid_f1"] < 0.25 for row in rows
        ),
        "runs": rows,
    }
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "run_count": len(rows),
                "maximum_best_grid_f1": payload["maximum_best_grid_f1"],
                "all_best_grid_f1_below_0_25": payload[
                    "all_best_grid_f1_below_0_25"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
