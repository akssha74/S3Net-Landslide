#!/usr/bin/env python3
"""Freeze compact bootstrap outputs needed for stdlib-only claim verification."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[2]
RESULTS = STUDY / "experiments/derived/results"
SEN12 = (
    RESULTS
    / "sen12_s2_confirmation_v5/sen12_s2_confirmation_summary.json"
)
OUTPUT = RESULTS / "public_claim_sufficient_statistics.json"
SOURCE_PATHS = [
    RESULTS / "reviewer_remediation/reviewer_remediation_summary.json",
    RESULTS
    / "reviewer_remediation_bgrn/reviewer_remediation_summary.json",
    RESULTS
    / "cas_boundary_confirmation/cas_boundary_confirmation_summary.json",
    RESULTS / "lrd_boundary_confirmation/endpoint_sensitivity.json",
    SEN12,
]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def bootstrap_draws(values: list[float], salt: str) -> list[float]:
    rng = np.random.default_rng(
        int(hashlib.sha256(salt.encode()).hexdigest()[:16], 16)
    )
    array = np.asarray(values, dtype=float)
    draws = np.mean(
        array[rng.integers(0, len(array), size=(10000, len(array)))],
        axis=1,
    )
    return sorted(float(value) for value in draws)


def main() -> None:
    sen12 = json.loads(SEN12.read_text())
    inventories = sen12["inventories"]
    bootstrap = {}
    for name, row in sen12["contrasts"].items():
        values = [float(row["per_inventory"][key]) for key in inventories]
        bootstrap[name] = {
            "salt": name,
            "draws_sorted": bootstrap_draws(values, name),
        }

    generic = sen12["generic_boundary_average"]
    generic_values = [
        float(generic["per_inventory"][key]) for key in inventories
    ]
    bootstrap["generic_boundary_average"] = {
        "salt": "generic-average",
        "draws_sorted": bootstrap_draws(generic_values, "generic-average"),
    }

    payload = {
        "schema_version": 1,
        "purpose": (
            "Compact frozen bootstrap outputs for dependency-free recomputation "
            "of all manuscript point estimates, envelopes, and conditions."
        ),
        "source_artifacts": {
            str(path.relative_to(STUDY)): sha256(path)
            for path in SOURCE_PATHS
        },
        "sen12_bootstrap": bootstrap,
    }
    OUTPUT.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    )
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(OUTPUT.relative_to(STUDY)),
                "sha256": sha256(OUTPUT),
                "bootstrap_series": len(bootstrap),
                "draws_per_series": 10000,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
