#!/usr/bin/env python3
"""Development-only mixed-inventory headroom matrix after the v3 gate failed."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run_sen12_s2_confirmation as run


STUDY = Path(__file__).resolve().parents[2]
OUTPUT = (
    STUDY
    / "experiments/derived/results/sen12_v4_mixed_matrix_exploration.json"
)
CHECKPOINTS = (
    STUDY / "experiments/derived/checkpoints/sen12_v4_mixed_matrix_exploration"
)
PREDICTIONS = (
    STUDY
    / "experiments/derived/results/sen12_v4_mixed_matrix_predictions"
)


def main() -> None:
    if OUTPUT.exists() or (
        CHECKPOINTS.exists() and any(CHECKPOINTS.iterdir())
    ) or (PREDICTIONS.exists() and any(PREDICTIONS.iterdir())):
        raise RuntimeError("exploration destination is not pristine")
    data = run.load_fold("development")
    validation = np.asarray(
        [
            int(
                hashlib.sha256(f"v4-mixed:{name}".encode()).hexdigest(),
                16,
            )
            % 5
            == 0
            for name in data["filenames"]
        ]
    )
    train = {"x": data["x"][~validation], "y": data["y"][~validation]}
    held_out = {"x": data["x"][validation], "y": data["y"][validation]}
    run.CHECKPOINTS = CHECKPOINTS
    run.VALIDATION_PREDICTIONS = PREDICTIONS
    results = []
    for arm, config in run.ARMS.items():
        for seed in run.SEEDS:
            row = run.train_one(arm, config, seed, train, held_out)
            results.append(row)
            print(
                json.dumps(
                    {
                        "arm": arm,
                        "seed": seed,
                        "validation_f1": row["validation_f1"],
                    }
                ),
                flush=True,
            )
    inventory_array = np.asarray(data["inventories"])
    payload = {
        "status": "development-only-after-v3-kill",
        "split_salt": "v4-mixed",
        "split_rule": "sha256(salt:filename) modulo 5 equals zero",
        "train_count": int(np.sum(~validation)),
        "validation_count": int(np.sum(validation)),
        "validation_by_inventory": {
            inventory: int(np.sum(validation & (inventory_array == inventory)))
            for inventory in sorted(set(data["inventories"]))
        },
        "all_runs_at_least_0_25": all(
            row["validation_f1"] >= 0.25 for row in results
        ),
        "all_runs_at_least_0_20": all(
            row["validation_f1"] >= 0.20 for row in results
        ),
        "all_runs_at_least_0_15": all(
            row["validation_f1"] >= 0.15 for row in results
        ),
        "arm_seed_mean": {
            arm: float(
                np.mean(
                    [
                        row["validation_f1"]
                        for row in results
                        if row["arm"] == arm
                    ]
                )
            )
            for arm in run.ARMS
        },
        "runs": results,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(
        json.dumps(
            {
                "all_runs_at_least_0_25": payload["all_runs_at_least_0_25"],
                "all_runs_at_least_0_20": payload["all_runs_at_least_0_20"],
                "all_runs_at_least_0_15": payload["all_runs_at_least_0_15"],
                "arm_seed_mean": payload["arm_seed_mean"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
