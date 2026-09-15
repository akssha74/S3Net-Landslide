#!/usr/bin/env python3
"""Certify which mechanism-effect directions survive admissible schemas."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


STUDY = Path(__file__).resolve().parents[2]
SOURCE = STUDY / "experiments/derived/results/band_order_sensitivity.json"
OUTPUT = (
    STUDY
    / "experiments/derived/results/semantic_identifiability_certificate.json"
)
TABLE = STUDY / "paper/tables/tab_semantic_identifiability.tex"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def status(values: list[float]) -> str:
    if min(values) > 0:
        return "positive-identified"
    if max(values) < 0:
        return "negative-identified"
    return "direction-unidentified"


def certificate(
    contrast_id: str,
    estimand: str,
    rgbn: float,
    bgrn: float,
    interpretation: str,
    **extra: Any,
) -> dict[str, Any]:
    values = [float(rgbn), float(bgrn)]
    return {
        "contrast_id": contrast_id,
        "estimand": estimand,
        "effect_points_by_schema": {
            "RGBN": values[0],
            "BGRN": values[1],
        },
        "semantic_envelope_points": [min(values), max(values)],
        "direction_status": status(values),
        "interpretation": interpretation,
        **extra,
    }


def main() -> None:
    payload = json.loads(SOURCE.read_text())
    orders = payload["orders"]
    rgbn = orders["RGBN"]
    bgrn = orders["BGRN"]

    def input_effect(order: dict[str, Any]) -> float:
        controls = order["base_control_mean_f1"]
        return 100 * (
            controls["s3_ndvi_base"] - controls["s3_raw_base"]
        )

    def modulation_effect(order: dict[str, Any], metric: str) -> float:
        return float(order["index_modulated_vs_plain_index"][metric])

    def boundary_effect(order: dict[str, Any]) -> tuple[float, list[float]]:
        effects = [
            float(value)
            for values in order["plain_boundary_f1_effects_by_seed_points"].values()
            for value in values
        ]
        return float(np.mean(effects)), effects

    rgbn_boundary_mean, rgbn_boundary_runs = boundary_effect(rgbn)
    bgrn_boundary_mean, bgrn_boundary_runs = boundary_effect(bgrn)
    contrasts = [
        certificate(
            "index-input-vs-raw-f1",
            "candidate-index attention minus raw-edge attention mean F1",
            input_effect(rgbn),
            input_effect(bgrn),
            (
                "The candidate index is not superior to the same-capacity raw "
                "edge control under either admissible schema."
            ),
        ),
        certificate(
            "index-modulation-vs-plain-f1",
            "equal-mass index modulation minus plain boundary weighting mean F1",
            modulation_effect(rgbn, "mean_f1_points"),
            modulation_effect(bgrn, "mean_f1_points"),
            (
                "Index modulation is inferior on mean F1 under both admissible "
                "schemas."
            ),
        ),
        certificate(
            "index-modulation-vs-plain-boundary-f1",
            (
                "equal-mass index modulation minus plain boundary weighting "
                "mean boundary F1"
            ),
            modulation_effect(rgbn, "mean_boundary_f1_points"),
            modulation_effect(bgrn, "mean_boundary_f1_points"),
            (
                "The boundary-effect direction is not semantically "
                "identifiable because its envelope crosses zero."
            ),
        ),
        certificate(
            "plain-boundary-vs-base-f1",
            (
                "plain boundary weighting minus base loss, mean across three "
                "attention controls and three seeds"
            ),
            rgbn_boundary_mean,
            bgrn_boundary_mean,
            (
                "The generic boundary effect is positive under both schemas "
                "and in all 18 individual comparisons."
            ),
            individual_effect_envelope_points=[
                min(rgbn_boundary_runs + bgrn_boundary_runs),
                max(rgbn_boundary_runs + bgrn_boundary_runs),
            ],
            individual_effects_positive=sum(
                value > 0
                for value in rgbn_boundary_runs + bgrn_boundary_runs
            ),
            individual_effects_total=len(
                rgbn_boundary_runs + bgrn_boundary_runs
            ),
        ),
    ]
    output = {
        "schema_version": 1,
        "name": "semantic-identifiability certificate",
        "definition": (
            "For admissible metadata-consistent schemas O and a signed "
            "contrast delta(o), the semantic envelope is "
            "[min_o delta(o), max_o delta(o)]. Effect direction is identified "
            "only when this interval excludes zero."
        ),
        "admissible_schemas": ["RGBN", "BGRN"],
        "schema_basis": (
            "The released arrays do not authoritatively bind visible columns "
            "0 and 2; the official display and native product order support "
            "the two candidates."
        ),
        "source_artifact": str(SOURCE.relative_to(STUDY)),
        "source_sha256": sha256(SOURCE),
        "contrasts": contrasts,
        "certificate_summary": {
            row["contrast_id"]: row["direction_status"] for row in contrasts
        },
        "scope": (
            "Finite sensitivity certificate for the stated candidate schemas; "
            "not a population interval or proof that no other schema exists."
        ),
    }
    OUTPUT.write_text(json.dumps(output, indent=2) + "\n")

    labels = {
        "index-input-vs-raw-f1": "Index input vs raw, F1",
        "index-modulation-vs-plain-f1": "Index vs plain loss, F1",
        "index-modulation-vs-plain-boundary-f1": "Index vs plain loss, BF1",
        "plain-boundary-vs-base-f1": "Plain boundary vs base, F1",
    }
    status_labels = {
        "positive-identified": "positive",
        "negative-identified": "negative",
        "direction-unidentified": "unidentified",
    }
    lines = [
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.2pt}",
        r"\begin{tabular}{@{}lrrrl@{}}",
        r"\toprule",
        r"Contrast (points) & RGBN & BGRN & Envelope & Direction \\",
        r"\midrule",
    ]
    for row in contrasts:
        lower, upper = row["semantic_envelope_points"]
        lines.append(
            f"{labels[row['contrast_id']]} & "
            f"{row['effect_points_by_schema']['RGBN']:+.2f} & "
            f"{row['effect_points_by_schema']['BGRN']:+.2f} & "
            f"[{lower:+.2f}, {upper:+.2f}] & "
            f"{status_labels[row['direction_status']]} "
            + r"\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {OUTPUT} and {TABLE}")


if __name__ == "__main__":
    main()
