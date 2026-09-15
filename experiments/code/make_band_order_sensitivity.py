#!/usr/bin/env python3
"""Compare the complete RGBN and BGRN candidate-order result surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


STUDY = Path(__file__).resolve().parents[2]
RGBN_PATH = (
    STUDY
    / "experiments/derived/results/reviewer_remediation/"
    "reviewer_remediation_summary.json"
)
BGRN_PATH = (
    STUDY
    / "experiments/derived/results/reviewer_remediation_bgrn/"
    "reviewer_remediation_summary.json"
)
OUTPUT = STUDY / "experiments/derived/results/band_order_sensitivity.json"
TABLE = STUDY / "paper/tables/tab_band_order_sensitivity.tex"
SEEDS = (42, 43, 44)
ARMS = (
    "unet_base",
    "resunet_base",
    "s3_none_base",
    "s3_raw_base",
    "s3_ndvi_base",
    "s3_none_boundary",
    "s3_raw_boundary",
    "s3_ndvi_boundary",
    "s3_ndvi_biophysical",
)
LABELS = {
    "unet_base": "U-Net/base",
    "resunet_base": "ResU-Net/base",
    "s3_none_base": "Attn-zero/base",
    "s3_raw_base": "Attn-raw/base",
    "s3_ndvi_base": "Attn-index/base",
    "s3_none_boundary": "Attn-zero/plain-boundary",
    "s3_raw_boundary": "Attn-raw/plain-boundary",
    "s3_ndvi_boundary": "Attn-index/plain-boundary",
    "s3_ndvi_biophysical": "Attn-index/index-boundary",
}


def run(payload: dict[str, Any], arm: str, seed: int) -> dict[str, Any]:
    return next(
        item
        for item in payload["runs"]
        if item["arm"] == arm and item["seed"] == seed
    )


def values(
    payload: dict[str, Any], arm: str, metric: str
) -> list[float]:
    return [
        float(run(payload, arm, seed)["test_default"][metric])
        for seed in SEEDS
    ]


def mean(payload: dict[str, Any], arm: str, metric: str) -> float:
    return float(np.mean(values(payload, arm, metric)))


def order_summary(payload: dict[str, Any]) -> dict[str, Any]:
    base_controls = ("s3_none_base", "s3_raw_base", "s3_ndvi_base")
    paired = {
        "s3_none": ("s3_none_base", "s3_none_boundary"),
        "s3_raw": ("s3_raw_base", "s3_raw_boundary"),
        "s3_index": ("s3_ndvi_base", "s3_ndvi_boundary"),
    }
    boundary_seed_effects = {
        name: [
            100
            * (
                run(payload, boundary_arm, seed)["test_default"]["f1"]
                - run(payload, base_arm, seed)["test_default"]["f1"]
            )
            for seed in SEEDS
        ]
        for name, (base_arm, boundary_arm) in paired.items()
    }
    index_f1 = mean(payload, "s3_ndvi_base", "f1")
    raw_f1 = mean(payload, "s3_raw_base", "f1")
    zero_f1 = mean(payload, "s3_none_base", "f1")
    return {
        "band_order": payload["protocol"]["band_order"],
        "base_control_mean_f1": {
            arm: mean(payload, arm, "f1") for arm in base_controls
        },
        "index_input_specificity_supported_on_f1": (
            index_f1 > raw_f1 and index_f1 > zero_f1
        ),
        "plain_boundary_f1_effects_by_seed_points": boundary_seed_effects,
        "plain_boundary_positive_all_seeds": all(
            effect > 0
            for effects in boundary_seed_effects.values()
            for effect in effects
        ),
        "index_modulated_vs_plain_index": {
            "mean_f1_points": 100
            * (
                mean(payload, "s3_ndvi_biophysical", "f1")
                - mean(payload, "s3_ndvi_boundary", "f1")
            ),
            "mean_boundary_f1_points": 100
            * (
                mean(
                    payload,
                    "s3_ndvi_biophysical",
                    "boundary_f1_tolerance_1px",
                )
                - mean(
                    payload,
                    "s3_ndvi_boundary",
                    "boundary_f1_tolerance_1px",
                )
            ),
            "f1_effects_by_seed_points": [
                100
                * (
                    run(payload, "s3_ndvi_biophysical", seed)[
                        "test_default"
                    ]["f1"]
                    - run(payload, "s3_ndvi_boundary", seed)["test_default"][
                        "f1"
                    ]
                )
                for seed in SEEDS
            ],
            "boundary_f1_effects_by_seed_points": [
                100
                * (
                    run(payload, "s3_ndvi_biophysical", seed)[
                        "test_default"
                    ]["boundary_f1_tolerance_1px"]
                    - run(payload, "s3_ndvi_boundary", seed)["test_default"][
                        "boundary_f1_tolerance_1px"
                    ]
                )
                for seed in SEEDS
            ],
        },
    }


def table(rgbn: dict[str, Any], bgrn: dict[str, Any]) -> None:
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.5pt}",
        r"\begin{tabular}{@{}lrrrr@{}}",
        r"\toprule",
        r"Configuration & RGBN F1 & BGRN F1 & RGBN BF1 & BGRN BF1 \\",
        r"\midrule",
    ]
    for arm in ARMS:
        row = [
            LABELS[arm],
            f"{100*mean(rgbn, arm, 'f1'):.2f}",
            f"{100*mean(bgrn, arm, 'f1'):.2f}",
            f"{100*mean(rgbn, arm, 'boundary_f1_tolerance_1px'):.2f}",
            f"{100*mean(bgrn, arm, 'boundary_f1_tolerance_1px'):.2f}",
        ]
        lines.append(" & ".join(row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TABLE.write_text("\n".join(lines) + "\n")


def combined_range(
    payloads: tuple[dict[str, Any], ...],
    arms: tuple[str, ...],
    extractor: Any,
) -> list[float]:
    arm_means = [
        float(
            np.mean(
                [
                    extractor(run(payload, arm, seed))
                    for seed in SEEDS
                ]
            )
        )
        for payload in payloads
        for arm in arms
    ]
    return [round(min(arm_means), 2), round(max(arm_means), 2)]


def main() -> None:
    rgbn = json.loads(RGBN_PATH.read_text())
    bgrn = json.loads(BGRN_PATH.read_text())
    if rgbn["protocol"]["band_order"] != ["red", "green", "blue", "nir"]:
        raise RuntimeError("RGBN result does not declare RGBN")
    if bgrn["protocol"]["band_order"] != ["blue", "green", "red", "nir"]:
        raise RuntimeError("BGRN result does not declare BGRN")
    summaries = {"RGBN": order_summary(rgbn), "BGRN": order_summary(bgrn)}
    payloads = (rgbn, bgrn)
    controlled_arms = tuple(arm for arm in ARMS if arm.startswith("s3_"))
    output = {
        "schema_version": 1,
        "question": (
            "Are mechanism conclusions stable to the two plausible HR-GLDD "
            "visible-band orders?"
        ),
        "orders": summaries,
        "order_robust": {
            "index_input_specificity_not_supported_on_f1": all(
                not row["index_input_specificity_supported_on_f1"]
                for row in summaries.values()
            ),
            "plain_boundary_positive_all_seeds": all(
                row["plain_boundary_positive_all_seeds"]
                for row in summaries.values()
            ),
            "index_modulated_f1_not_better_than_plain": all(
                row["index_modulated_vs_plain_index"]["mean_f1_points"] < 0
                for row in summaries.values()
            ),
            "index_modulated_boundary_effect_direction_stable": bool(
                np.sign(
                    summaries["RGBN"]["index_modulated_vs_plain_index"][
                        "mean_boundary_f1_points"
                    ]
                )
                == np.sign(
                    summaries["BGRN"]["index_modulated_vs_plain_index"][
                        "mean_boundary_f1_points"
                    ]
                )
            ),
        },
        "combined_operating_ranges_percent": {
            "matched_fpr": combined_range(
                payloads,
                controlled_arms,
                lambda row: 100
                * row["test_matched_validation_recall"]["background_fpr"],
            ),
            "matched_precision": combined_range(
                payloads,
                controlled_arms,
                lambda row: 100
                * row["test_matched_validation_recall"]["precision"],
            ),
            "controlled_resolution_f1_drop_points": combined_range(
                payloads,
                ARMS,
                lambda row: 100
                * (
                    row["test_default"]["f1"]
                    - row["test_controlled_10m"]["f1"]
                ),
            ),
        },
        "scope": (
            "Corrective sensitivity analysis; does not establish which order "
            "is physically authoritative."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")
    table(rgbn, bgrn)
    print(f"Wrote {OUTPUT} and {TABLE}")


if __name__ == "__main__":
    main()
