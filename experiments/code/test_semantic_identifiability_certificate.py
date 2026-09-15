#!/usr/bin/env python3
"""Independently check the semantic-identifiability certificate."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[2]
RGBN = json.loads(
    (
        STUDY
        / "experiments/derived/results/reviewer_remediation/"
        "reviewer_remediation_summary.json"
    ).read_text()
)
BGRN = json.loads(
    (
        STUDY
        / "experiments/derived/results/reviewer_remediation_bgrn/"
        "reviewer_remediation_summary.json"
    ).read_text()
)
CERTIFICATE = json.loads(
    (
        STUDY
        / "experiments/derived/results/"
        "semantic_identifiability_certificate.json"
    ).read_text()
)
OUTPUT = STUDY / "reviews/verified-semantic-identifiability.json"
SEEDS = (42, 43, 44)


def runs(payload: dict, arm: str) -> list[dict]:
    return [
        row
        for row in payload["runs"]
        if row["arm"] == arm and row["seed"] in SEEDS
    ]


def mean_metric(payload: dict, arm: str, metric: str) -> float:
    return float(
        np.mean([row["test_default"][metric] for row in runs(payload, arm)])
    )


def main() -> None:
    by_id = {
        row["contrast_id"]: row for row in CERTIFICATE["contrasts"]
    }
    expected = {
        "index-input-vs-raw-f1": {
            "RGBN": 100
            * (
                mean_metric(RGBN, "s3_ndvi_base", "f1")
                - mean_metric(RGBN, "s3_raw_base", "f1")
            ),
            "BGRN": 100
            * (
                mean_metric(BGRN, "s3_ndvi_base", "f1")
                - mean_metric(BGRN, "s3_raw_base", "f1")
            ),
        },
        "index-modulation-vs-plain-f1": {
            "RGBN": 100
            * (
                mean_metric(RGBN, "s3_ndvi_biophysical", "f1")
                - mean_metric(RGBN, "s3_ndvi_boundary", "f1")
            ),
            "BGRN": 100
            * (
                mean_metric(BGRN, "s3_ndvi_biophysical", "f1")
                - mean_metric(BGRN, "s3_ndvi_boundary", "f1")
            ),
        },
        "index-modulation-vs-plain-boundary-f1": {
            order: 100
            * (
                mean_metric(
                    payload,
                    "s3_ndvi_biophysical",
                    "boundary_f1_tolerance_1px",
                )
                - mean_metric(
                    payload,
                    "s3_ndvi_boundary",
                    "boundary_f1_tolerance_1px",
                )
            )
            for order, payload in (("RGBN", RGBN), ("BGRN", BGRN))
        },
        "plain-boundary-vs-base-f1": {
            order: float(
                np.mean(
                    [
                        100
                        * (
                            next(
                                row
                                for row in runs(payload, boundary_arm)
                                if row["seed"] == seed
                            )["test_default"]["f1"]
                            - next(
                                row
                                for row in runs(payload, base_arm)
                                if row["seed"] == seed
                            )["test_default"]["f1"]
                        )
                        for base_arm, boundary_arm in (
                            ("s3_none_base", "s3_none_boundary"),
                            ("s3_raw_base", "s3_raw_boundary"),
                            ("s3_ndvi_base", "s3_ndvi_boundary"),
                        )
                        for seed in SEEDS
                    ]
                )
            )
            for order, payload in (("RGBN", RGBN), ("BGRN", BGRN))
        },
    }
    for contrast_id, values in expected.items():
        actual = by_id[contrast_id]["effect_points_by_schema"]
        assert all(
            np.isclose(actual[order], value, atol=1e-12, rtol=0)
            for order, value in values.items()
        ), (contrast_id, actual, values)
        envelope = [min(values.values()), max(values.values())]
        assert np.allclose(
            by_id[contrast_id]["semantic_envelope_points"],
            envelope,
            atol=1e-12,
            rtol=0,
        )
    generic = by_id["plain-boundary-vs-base-f1"]
    individual_effects = [
        100
        * (
            next(
                row
                for row in runs(payload, boundary_arm)
                if row["seed"] == seed
            )["test_default"]["f1"]
            - next(
                row
                for row in runs(payload, base_arm)
                if row["seed"] == seed
            )["test_default"]["f1"]
        )
        for payload in (RGBN, BGRN)
        for base_arm, boundary_arm in (
            ("s3_none_base", "s3_none_boundary"),
            ("s3_raw_base", "s3_raw_boundary"),
            ("s3_ndvi_base", "s3_ndvi_boundary"),
        )
        for seed in SEEDS
    ]
    assert generic["individual_effects_positive"] == sum(
        value > 0 for value in individual_effects
    )
    assert generic["individual_effects_total"] == 18
    assert np.allclose(
        generic["individual_effect_envelope_points"],
        [min(individual_effects), max(individual_effects)],
        atol=1e-12,
        rtol=0,
    )
    assert generic["direction_status"] == "positive-identified"
    assert (
        by_id["index-modulation-vs-plain-boundary-f1"][
            "direction_status"
        ]
        == "direction-unidentified"
    )
    OUTPUT.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "certificate": (
                    "experiments/derived/results/"
                    "semantic_identifiability_certificate.json"
                ),
                "contrasts_recomputed": len(expected),
                "generic_individual_effects_checked": 18,
                "failures": 0,
                "status": "passed",
            },
            indent=2,
        )
        + "\n"
    )
    print("PASS: semantic-identifiability certificate independently verified")


if __name__ == "__main__":
    main()
