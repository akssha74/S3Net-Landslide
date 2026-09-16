#!/usr/bin/env python3
"""Recompute manuscript claims from compact release records using stdlib only."""

from __future__ import annotations

import hashlib
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
RESULTS = STUDY / "experiments/derived/results"
SUFFICIENT = RESULTS / "public_claim_sufficient_statistics.json"
OUTPUT = STUDY / "reviews/verified-public-claims-stdlib.json"


def load(relative: str) -> dict:
    return json.loads((STUDY / relative).read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def close(left: float, right: float, tolerance: float = 1e-12) -> bool:
    return math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance)


def quantile(sorted_values: list[float], q: float) -> float:
    position = (len(sorted_values) - 1) * q
    lower = int(math.floor(position))
    upper = int(math.ceil(position))
    if lower == upper:
        return sorted_values[lower]
    weight = position - lower
    return (
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def grouped_runs(payload: dict) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in payload["runs"]:
        grouped[row["arm"]].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: row["seed"])
    return grouped


def metric_mean(grouped: dict[str, list[dict]], arm: str, metric: str) -> float:
    return statistics.fmean(
        float(row["test_default"][metric]) for row in grouped[arm]
    )


def verify_hrgldd() -> dict:
    paths = {
        "RGBN": (
            "experiments/derived/results/reviewer_remediation/"
            "reviewer_remediation_summary.json"
        ),
        "BGRN": (
            "experiments/derived/results/reviewer_remediation_bgrn/"
            "reviewer_remediation_summary.json"
        ),
    }
    payloads = {name: load(path) for name, path in paths.items()}
    grouped = {name: grouped_runs(payload) for name, payload in payloads.items()}
    arm_labels = {
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
    table = {}
    for arm, label in arm_labels.items():
        table[label] = [
            round(100 * metric_mean(grouped["RGBN"], arm, "f1"), 2),
            round(100 * metric_mean(grouped["BGRN"], arm, "f1"), 2),
            round(
                100
                * metric_mean(
                    grouped["RGBN"], arm, "boundary_f1_tolerance_1px"
                ),
                2,
            ),
            round(
                100
                * metric_mean(
                    grouped["BGRN"], arm, "boundary_f1_tolerance_1px"
                ),
                2,
            ),
        ]
    table_text = (STUDY / "paper/tables/tab_band_order_sensitivity.tex").read_text()
    for label, values in table.items():
        rendered = f"{label} & " + " & ".join(f"{value:.2f}" for value in values)
        assert rendered in table_text, rendered

    def contrast(order: str, treatment: str, control: str, metric: str) -> float:
        return 100 * (
            metric_mean(grouped[order], treatment, metric)
            - metric_mean(grouped[order], control, metric)
        )

    def boundary_effects(order: str) -> list[float]:
        rows = {
            (row["arm"], int(row["seed"])): row
            for row in payloads[order]["runs"]
        }
        return [
            100
            * (
                rows[(boundary, seed)]["test_default"]["f1"]
                - rows[(base, seed)]["test_default"]["f1"]
            )
            for base, boundary in (
                ("s3_none_base", "s3_none_boundary"),
                ("s3_raw_base", "s3_raw_boundary"),
                ("s3_ndvi_base", "s3_ndvi_boundary"),
            )
            for seed in (42, 43, 44)
        ]

    certificate = {
        "index_input_vs_raw_f1": {
            order: contrast(order, "s3_ndvi_base", "s3_raw_base", "f1")
            for order in grouped
        },
        "index_vs_plain_f1": {
            order: contrast(
                order, "s3_ndvi_biophysical", "s3_ndvi_boundary", "f1"
            )
            for order in grouped
        },
        "index_vs_plain_boundary_f1": {
            order: contrast(
                order,
                "s3_ndvi_biophysical",
                "s3_ndvi_boundary",
                "boundary_f1_tolerance_1px",
            )
            for order in grouped
        },
        "plain_boundary_vs_base_f1": {
            order: statistics.fmean(boundary_effects(order))
            for order in grouped
        },
    }
    registered_payload = load(
        "experiments/derived/results/semantic_identifiability_certificate.json"
    )
    registered = {
        row["contrast_id"]: row for row in registered_payload["contrasts"]
    }
    registered_ids = {
        "index_input_vs_raw_f1": "index-input-vs-raw-f1",
        "index_vs_plain_f1": "index-modulation-vs-plain-f1",
        "index_vs_plain_boundary_f1": (
            "index-modulation-vs-plain-boundary-f1"
        ),
        "plain_boundary_vs_base_f1": "plain-boundary-vs-base-f1",
    }
    for key, values in certificate.items():
        record = registered[registered_ids[key]]
        for order, value in values.items():
            assert close(value, record["effect_points_by_schema"][order])
        assert close(
            min(values.values()), record["semantic_envelope_points"][0]
        )
        assert close(
            max(values.values()), record["semantic_envelope_points"][1]
        )

    effects = boundary_effects("RGBN") + boundary_effects("BGRN")
    assert len(effects) == 18 and all(value > 0 for value in effects)

    matched_fpr = []
    matched_precision = []
    drops = []
    default_boundary_fpr = []
    exact_recall = 0
    for order, groups in grouped.items():
        del order
        for arm, rows in groups.items():
            drops.append(
                100
                * (
                    statistics.fmean(
                        row["test_default"]["f1"] for row in rows
                    )
                    - statistics.fmean(
                        row["test_controlled_10m"]["f1"] for row in rows
                    )
                )
            )
            exact_recall += sum(
                row["matched_validation_recall_mismatch"] == 0 for row in rows
            )
            if arm.startswith("s3_"):
                matched_fpr.append(
                    statistics.fmean(
                        row["test_matched_validation_recall"]["background_fpr"]
                        for row in rows
                    )
                )
                matched_precision.append(
                    statistics.fmean(
                        row["test_matched_validation_recall"]["precision"]
                        for row in rows
                    )
                )
            if rows[0]["config"]["loss"] != "base":
                default_boundary_fpr.append(
                    100
                    * statistics.fmean(
                        row["test_default"]["background_fpr"] for row in rows
                    )
                )
    operating = {
        "matched_fpr_percent": [
            round(100 * min(matched_fpr), 2),
            round(100 * max(matched_fpr), 2),
        ],
        "matched_precision_percent": [
            round(100 * min(matched_precision), 2),
            round(100 * max(matched_precision), 2),
        ],
        "degradation_drop_points": [round(min(drops), 2), round(max(drops), 2)],
        "boundary_weighted_default_fpr_percent": [
            round(min(default_boundary_fpr), 2),
            round(max(default_boundary_fpr), 2),
        ],
        "exact_recall_matches": exact_recall,
        "runs": sum(len(rows) for groups in grouped.values() for rows in groups.values()),
    }
    assert operating == {
        "matched_fpr_percent": [1.97, 2.01],
        "matched_precision_percent": [78.41, 78.74],
        "degradation_drop_points": [4.33, 5.2],
        "boundary_weighted_default_fpr_percent": [2.47, 2.5],
        "exact_recall_matches": 53,
        "runs": 54,
    }
    return {
        "table_cells_verified": 36,
        "certificate": certificate,
        "generic_boundary_effects": {
            "positive": sum(value > 0 for value in effects),
            "total": len(effects),
            "range_points": [min(effects), max(effects)],
        },
        "operating": operating,
    }


def verify_cas_lrd() -> dict:
    cas = load(
        "experiments/derived/results/cas_boundary_confirmation/"
        "cas_boundary_confirmation_summary.json"
    )
    cas_f1 = [
        float(row["event_macro_delta_f1"]) for row in cas["effects"].values()
    ]
    cas_bf1 = [
        float(row["event_macro_delta_boundary_f1"])
        for row in cas["effects"].values()
    ]
    assert close(statistics.fmean(cas_f1), cas["mean_event_macro_delta_f1"])
    assert close(
        statistics.fmean(cas_bf1), cas["mean_event_macro_delta_boundary_f1"]
    )

    lrd = load(
        "experiments/derived/results/lrd_boundary_confirmation/"
        "endpoint_sensitivity.json"
    )
    lrd_result = {
        "protected_empty_percent": round(
            100 * lrd["empty_crop_counts"]["protected"]["fraction"], 1
        ),
        "validation_f1_range": [
            min(
                row["maximum_validation_f1"]
                for row in lrd["absolute_performance"].values()
            ),
            max(
                row["maximum_validation_f1"]
                for row in lrd["absolute_performance"].values()
            ),
        ],
        "selected_f1_effect_points": lrd["frozen_f1_effect_points"],
        "boundary_alternatives_points": [
            row["all_eid_mean_effect_points"]
            for row in lrd["boundary_endpoint_sensitivity"].values()
        ],
        "nonoverlap_eids": len(
            lrd["event_family_audit"]["valid_nonoverlapping_protected_eids"]
        ),
    }
    assert lrd_result["protected_empty_percent"] == 84.1
    assert round(lrd_result["selected_f1_effect_points"], 2) == -4.73
    assert lrd_result["nonoverlap_eids"] == 4
    return {
        "cas": {
            "mean_f1_points": 100 * statistics.fmean(cas_f1),
            "mean_boundary_f1_points": 100 * statistics.fmean(cas_bf1),
            "nonnegative_seed_f1": sum(value >= 0 for value in cas_f1),
        },
        "lrd": lrd_result,
    }


def verify_sen12(sufficient: dict) -> dict:
    sen12 = load(
        "experiments/derived/results/sen12_s2_confirmation_v5/"
        "sen12_s2_confirmation_summary.json"
    )
    lookup = {
        (row["arm"], int(row["seed"])): row for row in sen12["runs"]
    }
    inventories = sorted(sen12["inventories"])

    def per_inventory(treatment: str, control: str, metric: str) -> dict[str, float]:
        return {
            inventory: statistics.fmean(
                float(lookup[(treatment, seed)]["protected"][inventory][metric])
                - float(lookup[(control, seed)]["protected"][inventory][metric])
                for seed in (42, 43, 44)
            )
            for inventory in inventories
        }

    recomputed = {}
    for name, row in sen12["contrasts"].items():
        values = per_inventory(row["treatment"], row["control"], row["metric"])
        for inventory, value in values.items():
            assert close(value, row["per_inventory"][inventory])
        mean = statistics.fmean(values.values())
        assert close(mean, row["mean"])
        draws = sufficient["sen12_bootstrap"][name]["draws_sorted"]
        interval = [quantile(draws, 0.025), quantile(draws, 0.975)]
        assert all(close(a, b) for a, b in zip(interval, row["interval"]))
        recomputed[name] = {
            "mean": mean,
            "interval": interval,
            "inventories": len(values),
        }

    generic_by_control = {}
    for control, row in sen12["generic_boundary_by_control"].items():
        values = per_inventory(row["treatment"], row["control"], row["metric"])
        mean = statistics.fmean(values.values())
        assert close(mean, row["mean"])
        generic_by_control[control] = {"per_inventory": values, "mean": mean}
    generic_average = {
        inventory: statistics.fmean(
            generic_by_control[control]["per_inventory"][inventory]
            for control in generic_by_control
        )
        for inventory in inventories
    }
    pooled = sen12["generic_boundary_average"]
    for inventory, value in generic_average.items():
        assert close(value, pooled["per_inventory"][inventory])
    generic_mean = statistics.fmean(generic_average.values())
    assert close(generic_mean, pooled["mean"])
    generic_draws = sufficient["sen12_bootstrap"]["generic_boundary_average"][
        "draws_sorted"
    ]
    generic_interval = [
        quantile(generic_draws, 0.025),
        quantile(generic_draws, 0.975),
    ]
    assert all(close(a, b) for a, b in zip(generic_interval, pooled["interval"]))

    absolute_f1 = {}
    for arm in sen12["absolute_performance"]:
        values = {
            inventory: statistics.fmean(
                float(lookup[(arm, seed)]["protected"][inventory]["f1"])
                for seed in (42, 43, 44)
            )
            for inventory in inventories
        }
        mean = statistics.fmean(values.values())
        assert close(mean, sen12["absolute_performance"][arm]["f1"]["mean"])
        absolute_f1[arm] = mean

    conditions = {
        "ndvi_attention_vs_raw_edge_upper_ci_below_1_point": (
            recomputed["ndvi_attention_vs_raw_edge_f1"]["interval"][1] < 0.01
        ),
        "ndvi_modulated_vs_plain_upper_ci_below_1_point": (
            recomputed["ndvi_modulated_vs_plain_f1"]["interval"][1] < 0.01
        ),
        "generic_boundary_all_control_means_positive_lower_ci_above_zero_8_of_10": (
            all(row["mean"] > 0 for row in generic_by_control.values())
            and generic_interval[0] > 0
            and sum(value > 0 for value in generic_average.values()) >= 8
        ),
        "all_arm_absolute_mean_f1_at_least_0_10": all(
            value >= 0.10 for value in absolute_f1.values()
        ),
        "all_integrity_checks_pass": all(
            check["passed"] for check in sen12["integrity_checks"]
        ),
    }
    assert conditions == sen12["conditions"]
    assert [key for key, value in conditions.items() if value] == [
        "ndvi_modulated_vs_plain_upper_ci_below_1_point"
    ]
    return {
        "contrasts": recomputed,
        "generic_boundary_average": {
            "mean": generic_mean,
            "interval": generic_interval,
            "positive_inventories": sum(
                value > 0 for value in generic_average.values()
            ),
        },
        "absolute_f1_range": [
            min(absolute_f1.values()),
            max(absolute_f1.values()),
        ],
        "conditions": conditions,
        "protected_cells_verified": (
            len(sen12["runs"]) * len(inventories) * 2
        ),
    }


def verify_ledger_graph() -> dict:
    counts = {"claims": 0, "artifacts": 0, "runs": 0, "references": 0}

    claim_path = STUDY / "evidence/claim-ledger.jsonl"
    for line_number, line in enumerate(claim_path.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("status") != "verified":
            continue
        counts["claims"] += 1
        for reference in row.get("source_artifacts", []):
            path = STUDY / reference["path"]
            assert path.is_file(), (claim_path, line_number, path)
            assert sha256(path) == reference["sha256"], (
                claim_path,
                line_number,
                path,
            )
            counts["references"] += 1

    artifact_path = STUDY / "evidence/artifact-ledger.jsonl"
    for line_number, line in enumerate(
        artifact_path.read_text().splitlines(), 1
    ):
        row = json.loads(line)
        counts["artifacts"] += 1
        references = [
            {"path": row["path"], "sha256": row["sha256"]},
            *row.get("source_artifacts", []),
        ]
        for reference in references:
            path = STUDY / reference["path"]
            assert path.is_file(), (artifact_path, line_number, path)
            assert sha256(path) == reference["sha256"], (
                artifact_path,
                line_number,
                path,
            )
            counts["references"] += 1

    run_path = STUDY / "experiments/run-ledger.jsonl"
    for line_number, line in enumerate(run_path.read_text().splitlines(), 1):
        row = json.loads(line)
        if row.get("status") != "succeeded":
            continue
        # Exclude this verifier's own row to avoid a self-hash cycle.
        if row.get("run_id") == "R084-public-claims-ledger-verification":
            continue
        counts["runs"] += 1
        references = [
            {"path": row["log_path"], "sha256": row["log_sha256"]},
            *row.get("output_artifacts", []),
        ]
        for reference in references:
            path = STUDY / reference["path"]
            assert path.is_file(), (run_path, line_number, path)
            assert sha256(path) == reference["sha256"], (
                run_path,
                line_number,
                path,
            )
            counts["references"] += 1

    return counts


def main() -> None:
    sufficient = json.loads(SUFFICIENT.read_text())
    for relative, expected in sufficient["source_artifacts"].items():
        assert sha256(STUDY / relative) == expected, relative
    result = {
        "schema_version": 1,
        "status": "pass",
        "runtime": "python-standard-library-only",
        "source_artifacts": sufficient["source_artifacts"],
        "ledger_graph": verify_ledger_graph(),
        "hrgldd": verify_hrgldd(),
        "external": verify_cas_lrd(),
        "sen12": verify_sen12(sufficient),
    }
    OUTPUT.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "status": "pass",
                "output": str(OUTPUT.relative_to(STUDY)),
                "sha256": sha256(OUTPUT),
                "hrgldd_table_cells": result["hrgldd"]["table_cells_verified"],
                "ledger_references": result["ledger_graph"]["references"],
                "sen12_protected_cells": result["sen12"][
                    "protected_cells_verified"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
