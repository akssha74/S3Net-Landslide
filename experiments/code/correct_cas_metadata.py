#!/usr/bin/env python3
"""Correct CAS provenance/config metadata without changing numeric outputs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from run_cas_boundary_confirmation import canonical_config


STUDY = Path(__file__).resolve().parents[2]
FOLDS = (
    STUDY / "research/dataset-metadata/cas-boundary-confirmation/folds.json"
)
RESULTS = (
    STUDY
    / "experiments/derived/results/cas_boundary_confirmation/"
    "cas_boundary_confirmation_summary.json"
)
CONFIG = (
    STUDY
    / "experiments/derived/results/cas_boundary_confirmation/frozen_config.json"
)
NUMERIC_KEYS = (
    "archive_hashes",
    "training_records",
    "effects",
    "event_mean_boundary_effects",
    "mean_event_macro_delta_f1",
    "mean_event_macro_delta_boundary_f1",
    "pass_conditions",
)


def canonical_hash(value: object) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode()).hexdigest()


def main() -> None:
    folds = json.loads(FOLDS.read_text(encoding="utf-8"))
    summary = json.loads(RESULTS.read_text(encoding="utf-8"))
    numeric_before = canonical_hash(
        {key: summary[key] for key in NUMERIC_KEYS}
    )

    config = canonical_config(folds)
    config_id = canonical_hash(config)[:16]
    process_condition = {
        "condition": (
            "No post-access code, threshold, or reporting-rule change."
        ),
        "status": "not-independently-time-verifiable",
        "reason": (
            "The only surviving protocol was first committed after execution; "
            "run logs are retained but cannot establish a pre-run immutable lock."
        ),
        "included_in_quantitative_verdict": False,
    }
    summary.update(
        {
            "config_id": config_id,
            "config": config,
            "config_identity_scope": (
                "Execution-only fields; post-run provenance prose is excluded "
                "from the hash."
            ),
            "pass_conditions_scope": "four quantitative conditions",
            "process_condition": process_condition,
            "analysis_classification": (
                "post-hoc designated-region sensitivity analysis"
            ),
            "verdict": "failed-sensitivity-criteria",
            "verdict_basis": (
                "Three of four quantitative conditions failed; the separate "
                "process condition is not independently time-verifiable."
            ),
            "scope": (
                "Post-hoc sensitivity analysis on three designated CAS "
                "regions; no confirmation, population, sensor-invariant, "
                "operational, or global claim."
            ),
        }
    )
    numeric_after = canonical_hash(
        {key: summary[key] for key in NUMERIC_KEYS}
    )
    if numeric_after != numeric_before:
        raise RuntimeError("CAS numeric payload changed during metadata repair")

    frozen = {
        "schema_version": 2,
        "artifact_status": (
            "post-execution record of the executed configuration; "
            "filename retained for compatibility"
        ),
        "config_id": config_id,
        "config_identity_scope": (
            "Execution-only fields; post-run provenance prose is excluded "
            "from the hash."
        ),
        **config,
    }
    CONFIG.write_text(json.dumps(frozen, indent=2) + "\n", encoding="utf-8")
    RESULTS.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(
        f"PASS: corrected CAS metadata; config_id={config_id}; "
        f"numeric_sha256={numeric_after}"
    )


if __name__ == "__main__":
    main()
