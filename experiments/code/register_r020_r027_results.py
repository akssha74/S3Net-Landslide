#!/usr/bin/env python3
"""Register dual-order HR-GLDD and prospective LRD result identities."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STUDY = Path(__file__).resolve().parents[2]
CLAIMS = STUDY / "evidence/claim-ledger.jsonl"
REGISTRY = STUDY / "evidence/result-registry.jsonl"
ARTIFACTS = STUDY / "evidence/artifact-ledger.jsonl"
RUNS = STUDY / "experiments/run-ledger.jsonl"
TREE = STUDY / "experiments/tree.jsonl"
BGRN = (
    STUDY
    / "experiments/derived/results/reviewer_remediation_bgrn/"
    "reviewer_remediation_summary.json"
)
BAND_SENSITIVITY = (
    STUDY / "experiments/derived/results/band_order_sensitivity.json"
)
LRD = (
    STUDY
    / "experiments/derived/results/lrd_boundary_confirmation/"
    "lrd_confirmation_summary.json"
)
LRD_VERIFIED = STUDY / "reviews/verified-lrd-confirmation.json"
LRD_ENDPOINT = (
    STUDY
    / "experiments/derived/results/lrd_boundary_confirmation/"
    "endpoint_sensitivity.json"
)
LRD_FAMILIES = (
    STUDY
    / "research/dataset-metadata/lrd-prospective-confirmation/"
    "event_family_audit.json"
)
INVENTORY = STUDY / "experiments/derived/results/frozen-output-inventory.json"
CAS_SUMMARY = (
    STUDY
    / "experiments/derived/results/cas_boundary_confirmation/"
    "cas_boundary_confirmation_summary.json"
)
CAS_CONFIG = (
    STUDY
    / "experiments/derived/results/cas_boundary_confirmation/frozen_config.json"
)
CAS_VERIFIED = STUDY / "reviews/verified-cas-boundary-confirmation.json"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n"
            for row in rows
        )
    )


def run_record(
    run_id: str,
    node_id: str,
    purpose: str,
    command: str,
    log_path: str,
    outputs: list[str],
) -> dict[str, Any]:
    output_artifacts = [
        {"path": path, "sha256": digest(STUDY / path)} for path in outputs
    ]
    if not output_artifacts:
        output_artifacts = [
            {"path": log_path, "sha256": digest(STUDY / log_path)}
        ]
    return {
        "run_id": run_id,
        "node_id": node_id,
        "purpose": purpose,
        "command": command,
        "cwd": "studies/disaster-hrgldd-landslide",
        "status": "succeeded",
        "exit_code": 0,
        "started_at": "2026-09-15T04:43:00Z",
        "completed_at": "2026-09-15T05:59:59Z",
        "log_path": log_path,
        "log_sha256": digest(STUDY / log_path),
        "output_artifacts": output_artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--band-order-code-commit", required=True)
    parser.add_argument("--lrd-protocol-commit", required=True)
    args = parser.parse_args()
    bgrn = json.loads(BGRN.read_text())
    sensitivity = json.loads(BAND_SENSITIVITY.read_text())
    lrd = json.loads(LRD.read_text())
    inventory_sha = digest(INVENTORY)

    claims = [
        row
        for row in read_jsonl(CLAIMS)
        if not str(row.get("claim_id", "")).startswith(
            ("C-r020", "C-r027", "C-r032", "C-r033")
        )
    ]
    claims.extend(
        [
            {
                "claim_id": "C-r020-band-order-ambiguity",
                "claim": (
                    "Released HR-GLDD metadata do not authoritatively distinguish "
                    "RGBN from BGRN; both candidate orders are evaluated."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/audit_hrgldd_band_order.py"
                ),
                "run_ids": ["R019-band-order-ambiguity-audit"],
                "source_artifacts": [
                    {
                        "path": (
                            "research/dataset-metadata/hrgldd-official/"
                            "band-order-audit.json"
                        ),
                        "sha256": digest(
                            STUDY
                            / "research/dataset-metadata/hrgldd-official/"
                            "band-order-audit.json"
                        ),
                    }
                ],
            },
            {
                "claim_id": "C-r020-order-robust-controls",
                "claim": (
                    "Under both RGBN and BGRN candidates, index-input specificity "
                    "is unsupported and plain boundary weighting improves F1 in "
                    "every seed/control comparison."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/make_band_order_sensitivity.py"
                ),
                "run_ids": [
                    "R020-bgrn-order-sensitivity",
                    "R020b-bgrn-independent-verification",
                    "R020c-band-order-comparison",
                ],
                "source_artifacts": [
                    {
                        "path": str(BAND_SENSITIVITY.relative_to(STUDY)),
                        "sha256": digest(BAND_SENSITIVITY),
                    }
                ],
            },
            {
                "claim_id": "C-r020-spectral-loss-order-instability",
                "claim": (
                    "Index-modulated minus plain boundary F1 changes from +0.38 "
                    "points under RGBN to -0.05 under BGRN; spectral-loss "
                    "specificity is not order-robust."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/make_band_order_sensitivity.py"
                ),
                "run_ids": ["R020c-band-order-comparison"],
                "source_artifacts": [
                    {
                        "path": str(BAND_SENSITIVITY.relative_to(STUDY)),
                        "sha256": digest(BAND_SENSITIVITY),
                    }
                ],
            },
            {
                "claim_id": "C-r027-lrd-prospective-failure",
                "claim": (
                    "Under the frozen local rule, the LRD run failed all four "
                    "criteria across six released EIDs."
                ),
                "status": "superseded",
                "superseded_by": "C-r033-lrd-indeterminate",
                "supersession_reason": (
                    "No independent pre-access timestamp exists; two protected "
                    "EIDs share a trigger family with development; validation "
                    "is weak; and boundary magnitude changes with empty-crop "
                    "and threshold conventions."
                ),
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "reviews/verify_lrd_confirmation.py"
                ),
                "run_ids": [
                    "R025-lrd-prospective-fit",
                    "R026-lrd-protected-fetch",
                    "R027-lrd-prospective-evaluation",
                    "R027b-lrd-independent-verification",
                ],
                "source_artifacts": [
                    {
                        "path": str(LRD.relative_to(STUDY)),
                        "sha256": digest(LRD),
                    },
                    {
                        "path": str(LRD_VERIFIED.relative_to(STUDY)),
                        "sha256": digest(LRD_VERIFIED),
                    },
                    {
                        "path": (
                            "research/dataset-metadata/lrd-prospective-confirmation/"
                            "protected_access_log.json"
                        ),
                        "sha256": digest(
                            STUDY
                            / "research/dataset-metadata/"
                            "lrd-prospective-confirmation/protected_access_log.json"
                        ),
                    },
                ],
            },
            {
                "claim_id": "C-r032-lrd-event-family-leakage",
                "claim": (
                    "PH0003 development and PH0001/PH0004 protected share the "
                    "Philippines 2022-04-10 trigger family; only four protected "
                    "EIDs remain non-overlapping under the minimum family rule."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/audit_lrd_event_families.py"
                ),
                "run_ids": ["R032-lrd-event-family-audit"],
                "source_artifacts": [
                    {
                        "path": str(LRD_FAMILIES.relative_to(STUDY)),
                        "sha256": digest(LRD_FAMILIES),
                    }
                ],
            },
            {
                "claim_id": "C-r033-lrd-indeterminate",
                "claim": (
                    "The internally precommitted LRD run is failed/indeterminate: "
                    "validation F1 is 0.008-0.151, the boundary effect changes "
                    "from -24.10 to -0.15 points under empty/empty scoring, and "
                    "trigger-family leakage reduces valid protected EIDs to four."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/analyze_lrd_endpoint_sensitivity.py"
                ),
                "run_ids": [
                    "R032-lrd-event-family-audit",
                    "R033-lrd-endpoint-sensitivity",
                ],
                "source_artifacts": [
                    {
                        "path": str(LRD_ENDPOINT.relative_to(STUDY)),
                        "sha256": digest(LRD_ENDPOINT),
                    },
                    {
                        "path": str(LRD_FAMILIES.relative_to(STUDY)),
                        "sha256": digest(LRD_FAMILIES),
                    },
                ],
            },
        ]
    )
    write_jsonl(CLAIMS, claims)

    registry = [
        row
        for row in read_jsonl(REGISTRY)
        if not str(row.get("result_id", "")).startswith(
            ("R020", "R027", "R032", "R033")
        )
    ]
    arm_metadata = {
        "unet_base": ("unet", "base"),
        "resunet_base": ("resunet", "base"),
        "s3_none_base": ("controlled-attention-zero", "base"),
        "s3_raw_base": ("controlled-attention-raw", "base"),
        "s3_ndvi_base": ("controlled-attention-index", "base"),
        "s3_none_boundary": ("controlled-attention-zero", "plain-boundary"),
        "s3_raw_boundary": ("controlled-attention-raw", "plain-boundary"),
        "s3_ndvi_boundary": ("controlled-attention-index", "plain-boundary"),
        "s3_ndvi_biophysical": (
            "controlled-attention-index",
            "equal-mass-index-boundary",
        ),
    }
    for arm, (method_id, loss_id) in arm_metadata.items():
        registry.append(
            {
                "result_id": f"R020-bgrn-{arm}",
                "configuration_id": f"bgrn-{arm}",
                "method_id": method_id,
                "loss_id": loss_id,
                "code_commit": args.band_order_code_commit,
                "dataset_id": "hr-gldd-planetscope-v1-bgrn-candidate",
                "split": "official released arrays; descriptive corrective sensitivity",
                "generator": (
                    "HRGLDD_ARRAY_ORDER=BGRN REMEDIATION_VARIANT=bgrn "
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_reviewer_remediation.py"
                ),
                "metrics_artifact": str(BGRN.relative_to(STUDY)),
                "artifact_sha256": digest(BGRN),
                "frozen_output_inventory": str(INVENTORY.relative_to(STUDY)),
                "frozen_output_inventory_sha256": inventory_sha,
                "manuscript_quantitative_claims": [
                    {
                        "path": "paper/tables/tab_band_order_sensitivity.tex",
                        "anchor": f"configuration row {arm}",
                    }
                ],
                "status": "current-candidate",
            }
        )
    registry.append(
        {
            "result_id": "R020-order-sensitivity",
            "configuration_id": "rgbn-vs-bgrn-complete-surface",
            "method_id": "dual-order-semantic-sensitivity",
            "loss_id": "multiple",
            "code_commit": args.band_order_code_commit,
            "dataset_id": "hr-gldd-planetscope-v1-order-ambiguous",
            "split": "official released arrays; descriptive corrective sensitivity",
            "generator": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_band_order_sensitivity.py"
            ),
            "metrics_artifact": str(BAND_SENSITIVITY.relative_to(STUDY)),
            "artifact_sha256": digest(BAND_SENSITIVITY),
            "manuscript_quantitative_claims": [
                {
                    "path": "paper/sections/results.tex",
                    "anchor": "Band-order sensitivity and matched controls",
                }
            ],
            "status": "current",
        }
    )
    for loss_mode in ("base", "boundary"):
        registry.append(
            {
                "result_id": f"R027-lrd-{loss_mode}",
                "configuration_id": f"{lrd['config_id']}-{loss_mode}",
                "method_id": "rgb-zero-control-attention",
                "loss_id": loss_mode,
                "code_commit": args.lrd_protocol_commit,
                "dataset_id": "lrd-v3-28-eid-prospective",
                "split": "16 development, 6 validation, 6 protected EIDs",
                "generator": (
                    "LRD_PROTOCOL_COMMIT="
                    f"{args.lrd_protocol_commit} "
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_lrd_prospective_confirmation.py"
                ),
                "metrics_artifact": str(LRD.relative_to(STUDY)),
                "artifact_sha256": digest(LRD),
                "frozen_output_inventory": str(INVENTORY.relative_to(STUDY)),
                "frozen_output_inventory_sha256": inventory_sha,
                "manuscript_quantitative_claims": [
                    {
                        "path": "paper/sections/results.tex",
                        "anchor": "Preregistered LRD confirmation paragraph",
                    },
                    {
                        "path": "paper/tables/tab_lrd_confirmation.tex",
                        "anchor": f"{loss_mode} comparator effects",
                    },
                ],
                "status": "superseded-as-confirmation",
            }
        )
    registry.append(
        {
            "result_id": "R033-lrd-endpoint-family-sensitivity",
            "configuration_id": f"{lrd['config_id']}-posthoc-sensitivity",
            "method_id": "lrd-external-evidence-diagnostic",
            "loss_id": "base-vs-plain-boundary",
            "code_commit": args.lrd_protocol_commit,
            "dataset_id": "lrd-v3-eid-family-audited",
            "split": (
                "16 development, 6 validation, 6 designated EIDs; "
                "4 protected EIDs nonoverlapping by minimum trigger family"
            ),
            "generator": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/analyze_lrd_endpoint_sensitivity.py"
            ),
            "metrics_artifact": str(LRD_ENDPOINT.relative_to(STUDY)),
            "artifact_sha256": digest(LRD_ENDPOINT),
            "manuscript_quantitative_claims": [
                {
                    "path": "paper/sections/results.tex",
                    "anchor": "Internally precommitted LRD stress-test paragraph",
                },
                {
                    "path": "paper/supplement.tex",
                    "anchor": "Supplementary Method S2",
                },
            ],
            "status": "current",
        }
    )
    write_jsonl(REGISTRY, registry)

    new_runs = [
        run_record(
            "R019-band-order-ambiguity-audit",
            "N010-rgbn-correction",
            "Record evidence and counterevidence for two candidate HR-GLDD orders.",
            "/opt/homebrew/bin/python3.10 experiments/code/audit_hrgldd_band_order.py",
            "experiments/logs/R019-band-order-ambiguity-audit.log",
            [
                "research/dataset-metadata/hrgldd-official/band-order-audit.json"
            ],
        ),
        run_record(
            "R020-bgrn-order-sensitivity",
            "N010-rgbn-correction",
            "Run all nine configurations and three seeds under BGRN.",
            (
                "HRGLDD_ARRAY_ORDER=BGRN REMEDIATION_VARIANT=bgrn "
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/run_reviewer_remediation.py"
            ),
            "experiments/logs/R020-bgrn-order-sensitivity.log",
            [str(BGRN.relative_to(STUDY))],
        ),
        run_record(
            "R020b-bgrn-independent-verification",
            "N010-rgbn-correction",
            "Independently recompute all 1,404 BGRN result fields.",
            (
                "HRGLDD_RESULTS_VARIANT=bgrn /opt/homebrew/bin/python3.10 "
                "reviews/verify_r010.py"
            ),
            "experiments/logs/R020b-bgrn-independent-verification.log",
            ["reviews/verified-bgrn-sensitivity.json"],
        ),
        run_record(
            "R020c-band-order-comparison",
            "N010-rgbn-correction",
            "Generate the dual-order comparison and publication table.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_band_order_sensitivity.py"
            ),
            "experiments/logs/R020c-band-order-comparison.log",
            [
                str(BAND_SENSITIVITY.relative_to(STUDY)),
                "paper/tables/tab_band_order_sensitivity.tex",
            ],
        ),
        run_record(
            "R021-equal-mass-both-orders",
            "N010-rgbn-correction",
            "Assert exact mass matching under both candidate orders.",
            "HRGLDD_ARRAY_ORDER={RGBN,BGRN} test_equal_mass_dataset.py",
            "experiments/logs/R021-equal-mass-both-orders.log",
            [],
        ),
        run_record(
            "R022-cas-posthoc-reclassification",
            "N011-cas-external",
            "Reclassify CAS as post-hoc sensitivity without changing metrics.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/correct_cas_metadata.py"
            ),
            "experiments/logs/R022-cas-posthoc-reclassification.log",
            [
                str(CAS_SUMMARY.relative_to(STUDY)),
                str(CAS_CONFIG.relative_to(STUDY)),
            ],
        ),
        run_record(
            "R022b-cas-posthoc-verification",
            "N011-cas-external",
            "Recompute all 150 CAS fields after post-hoc reclassification.",
            (
                "/opt/homebrew/bin/python3.10 "
                "reviews/verify_cas_boundary_confirmation.py"
            ),
            "experiments/logs/R022b-cas-posthoc-verification.log",
            [str(CAS_VERIFIED.relative_to(STUDY))],
        ),
        run_record(
            "R023-lrd-development-fetch",
            "N024-lrd-prospective-confirmation",
            "Fetch development-only named S2 RGB and mask files.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/fetch_lrd_optical_folds.py --fold development"
            ),
            "experiments/logs/R023-lrd-development-fetch.log",
            [
                "research/dataset-metadata/lrd-prospective-confirmation/"
                "development_optical_manifest.json"
            ],
        ),
        run_record(
            "R024-lrd-validation-fetch",
            "N024-lrd-prospective-confirmation",
            "Fetch validation EIDs after protocol freeze.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/fetch_lrd_optical_folds.py --fold validation"
            ),
            "experiments/logs/R024-lrd-validation-fetch.log",
            [
                "research/dataset-metadata/lrd-prospective-confirmation/"
                "validation_optical_manifest.json"
            ],
        ),
        run_record(
            "R025-lrd-prospective-fit",
            "N024-lrd-prospective-confirmation",
            "Fit six LRD models and freeze checkpoints/thresholds.",
            (
                f"LRD_PROTOCOL_COMMIT={args.lrd_protocol_commit} "
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/run_lrd_prospective_confirmation.py --stage fit"
            ),
            "experiments/logs/R025-lrd-prospective-fit.log",
            [
                "experiments/derived/results/lrd_boundary_confirmation/"
                "fit_decisions.json",
                "experiments/derived/results/lrd_boundary_confirmation/"
                "protected_access_authorization.json",
            ],
        ),
        run_record(
            "R026-lrd-protected-fetch",
            "N024-lrd-prospective-confirmation",
            "Fetch protected EIDs only after authorization.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/fetch_lrd_optical_folds.py --fold protected"
            ),
            "experiments/logs/R026-lrd-protected-fetch.log",
            [
                "research/dataset-metadata/lrd-prospective-confirmation/"
                "protected_optical_manifest.json"
            ],
        ),
        run_record(
            "R027-lrd-prospective-evaluation",
            "N024-lrd-prospective-confirmation",
            "Evaluate all frozen models on six protected EIDs exactly once.",
            (
                f"LRD_PROTOCOL_COMMIT={args.lrd_protocol_commit} "
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/run_lrd_prospective_confirmation.py "
                "--stage evaluate"
            ),
            "experiments/logs/R027-lrd-prospective-evaluation.log",
            [str(LRD.relative_to(STUDY))],
        ),
        run_record(
            "R027b-lrd-independent-verification",
            "N024-lrd-prospective-confirmation",
            "Independently recompute 330 protected LRD fields.",
            "/opt/homebrew/bin/python3.10 reviews/verify_lrd_confirmation.py",
            "experiments/logs/R027b-lrd-independent-verification.log",
            [str(LRD_VERIFIED.relative_to(STUDY))],
        ),
        run_record(
            "R027c-lrd-table",
            "N024-lrd-prospective-confirmation",
            "Generate the protected-EID and seed effect table.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_lrd_confirmation_table.py"
            ),
            "experiments/logs/R027c-lrd-table.log",
            ["paper/tables/tab_lrd_confirmation.tex"],
        ),
        run_record(
            "R028-frozen-output-inventory",
            "N024-lrd-prospective-confirmation",
            "Hash all 294 retained outputs across current analyses.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/build_frozen_output_inventory.py"
            ),
            "experiments/logs/R028-frozen-output-inventory.log",
            [str(INVENTORY.relative_to(STUDY))],
        ),
        run_record(
            "R029-rgbn-candidate-artifacts",
            "N010-rgbn-correction",
            "Regenerate detailed RGBN-candidate tables, CAS table, and atlas.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "experiments/logs/R029-rgbn-candidate-artifacts.log",
            [
                "experiments/derived/results/reviewer_remediation/"
                "mechanism_contrasts.json",
                "paper/tables/tab_reviewer_remediation.tex",
                "paper/tables/tab_cas_seed_effects.tex",
                "paper/figures/fig_false_positive_atlas.pdf",
            ],
        ),
        run_record(
            "R029b-rgbn-candidate-verification",
            "N010-rgbn-correction",
            "Independently recompute all 1,404 RGBN-candidate result fields.",
            (
                "HRGLDD_RESULTS_VARIANT=rgbn /opt/homebrew/bin/python3.10 "
                "reviews/verify_r010.py"
            ),
            "experiments/logs/R029b-rgbn-candidate-verification.log",
            ["reviews/verified-r010.json"],
        ),
        run_record(
            "R030-environment-freeze",
            "N010-rgbn-correction",
            "Freeze the transitive installed environment with RECORD hashes.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/freeze_environment.py"
            ),
            "experiments/logs/R030-environment-freeze.log",
            ["environment/environment-freeze.json"],
        ),
        run_record(
            "R031-dual-order-semantics",
            "N010-rgbn-correction",
            "Audit ambiguity and test both explicit candidate-order mappings.",
            (
                "audit_hrgldd_band_order.py; "
                "HRGLDD_ARRAY_ORDER={RGBN,BGRN} test_data_semantics.py"
            ),
            "experiments/logs/R031-dual-order-semantics.log",
            [
                "experiments/code/data_semantics.py",
                "experiments/code/test_data_semantics.py",
                "research/data-schema.json",
                (
                    "research/dataset-metadata/hrgldd-official/"
                    "band-order-audit.json"
                ),
            ],
        ),
        run_record(
            "R032-lrd-event-family-audit",
            "N024-lrd-prospective-confirmation",
            "Group released EIDs by country and trigger date to audit leakage.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/audit_lrd_event_families.py"
            ),
            "experiments/logs/R032-lrd-event-family-audit.log",
            [str(LRD_FAMILIES.relative_to(STUDY))],
        ),
        run_record(
            "R033-lrd-endpoint-sensitivity",
            "N024-lrd-prospective-confirmation",
            "Diagnose empty-crop, threshold, and trigger-family sensitivity.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/analyze_lrd_endpoint_sensitivity.py"
            ),
            "experiments/logs/R033-lrd-endpoint-sensitivity.log",
            [str(LRD_ENDPOINT.relative_to(STUDY))],
        ),
        run_record(
            "R034-figure-title-removal",
            "N010-rgbn-correction",
            "Regenerate publication artifacts without the duplicate figure title.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "experiments/logs/R034-figure-title-removal.log",
            [
                "experiments/derived/results/reviewer_remediation/"
                "mechanism_contrasts.json",
                "paper/tables/tab_reviewer_remediation.tex",
                "paper/tables/tab_cas_seed_effects.tex",
                "paper/figures/fig_false_positive_atlas.pdf",
            ],
        ),
    ]
    for row in new_runs:
        if row["run_id"] == "R029-rgbn-candidate-artifacts":
            row["output_artifacts"] = []
            row["superseded_outputs"] = [
                {
                    "path": path,
                    "rewritten_by": "R034-figure-title-removal",
                }
                for path in (
                    "experiments/derived/results/reviewer_remediation/"
                    "mechanism_contrasts.json",
                    "paper/tables/tab_reviewer_remediation.tex",
                    "paper/tables/tab_cas_seed_effects.tex",
                    "paper/figures/fig_false_positive_atlas.pdf",
                )
            ]
    new_ids = {row["run_id"] for row in new_runs}
    runs = [row for row in read_jsonl(RUNS) if row.get("run_id") not in new_ids]
    runs.extend(new_runs)
    write_jsonl(RUNS, runs)

    artifacts = [
        row
        for row in read_jsonl(ARTIFACTS)
        if row.get("artifact_id")
        not in {"A-tab-band-order-r020", "A-tab-lrd-r027"}
    ]
    artifacts.extend(
        [
            {
                "artifact_id": "A-tab-band-order-r020",
                "claim_ids": [
                    "C-r020-order-robust-controls",
                    "C-r020-spectral-loss-order-instability",
                ],
                "generator_code": (
                    "experiments/code/make_band_order_sensitivity.py"
                ),
                "generator_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/make_band_order_sensitivity.py"
                ),
                "path": "paper/tables/tab_band_order_sensitivity.tex",
                "latex_reference": "tables/tab_band_order_sensitivity.tex",
                "justification": (
                    "Places all nine configurations under both plausible "
                    "orders so only order-robust conclusions are retained."
                ),
                "run_ids": ["R020c-band-order-comparison"],
                "sha256": digest(
                    STUDY / "paper/tables/tab_band_order_sensitivity.tex"
                ),
                "source_artifacts": [
                    {
                        "path": str(BAND_SENSITIVITY.relative_to(STUDY)),
                        "sha256": digest(BAND_SENSITIVITY),
                    }
                ],
                "type": "table",
                "verified_at": "2026-09-15T05:35:00Z",
            },
            {
                "artifact_id": "A-tab-lrd-r027",
                "claim_ids": [
                    "C-r027-lrd-prospective-failure",
                    "C-r033-lrd-indeterminate",
                ],
                "generator_code": (
                    "experiments/code/make_lrd_confirmation_table.py"
                ),
                "generator_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/make_lrd_confirmation_table.py"
                ),
                "path": "paper/tables/tab_lrd_confirmation.tex",
                "latex_reference": "tables/tab_lrd_confirmation.tex",
                "justification": (
                    "Reports every designated-EID and seed-macro effect under "
                    "the frozen local LRD rule; sensitivity is separate."
                ),
                "run_ids": ["R027c-lrd-table"],
                "sha256": digest(
                    STUDY / "paper/tables/tab_lrd_confirmation.tex"
                ),
                "source_artifacts": [
                    {
                        "path": str(LRD.relative_to(STUDY)),
                        "sha256": digest(LRD),
                    }
                ],
                "type": "table",
                "verified_at": "2026-09-15T05:35:00Z",
            },
        ]
    )
    write_jsonl(ARTIFACTS, artifacts)

    tree = read_jsonl(TREE)
    for node in tree:
        if node.get("node_id") == "N010-rgbn-correction":
            node["run_ids"] = sorted(
                set(node.get("run_ids", []))
                | {
                    "R019-band-order-ambiguity-audit",
                    "R020-bgrn-order-sensitivity",
                    "R020b-bgrn-independent-verification",
                    "R020c-band-order-comparison",
                    "R021-equal-mass-both-orders",
                    "R029-rgbn-candidate-artifacts",
                    "R029b-rgbn-candidate-verification",
                    "R030-environment-freeze",
                    "R031-dual-order-semantics",
                    "R034-figure-title-removal",
                }
            )
            node["outcome"] = (
                "Both candidate orders reject index-input specificity; plain "
                "boundary F1 gains are positive in every seed, while the "
                "index-modulated boundary contrast changes sign across orders."
            )
        if node.get("node_id") == "N024-lrd-prospective-confirmation":
            node["status"] = "succeeded"
            node["confirmatory"] = False
            node["stage"] = "internally-precommitted-failed-indeterminate"
            node["decision"] = "failed-indeterminate-no-transfer-inference"
            node["outcome"] = (
                "All four frozen criteria failed arithmetically, but no public "
                "pre-access timestamp exists, trigger-family leakage reduces "
                "nonoverlapping protected EIDs to four, and the boundary effect "
                "is unstable to empty-crop and threshold conventions."
            )
            node["run_ids"] = [
                "R023-lrd-development-fetch",
                "R024-lrd-validation-fetch",
                "R025-lrd-prospective-fit",
                "R026-lrd-protected-fetch",
                "R027-lrd-prospective-evaluation",
                "R027b-lrd-independent-verification",
                "R027c-lrd-table",
                "R028-frozen-output-inventory",
                "R032-lrd-event-family-audit",
                "R033-lrd-endpoint-sensitivity",
            ]
            node["artifacts"] = [
                "research/lrd-boundary-confirmation-preregistration.md",
                str(LRD.relative_to(STUDY)),
                str(LRD_VERIFIED.relative_to(STUDY)),
                "paper/tables/tab_lrd_confirmation.tex",
                (
                    "research/dataset-metadata/lrd-prospective-confirmation/"
                    "protected_access_log.json"
                ),
                str(INVENTORY.relative_to(STUDY)),
            ]
        if node.get("node_id") == "N011-cas-external":
            node["stage"] = "post-hoc-designated-region-sensitivity"
            node["outcome"] = (
                "The post-hoc sensitivity analysis failed three of four "
                "criteria; it receives no confirmation credit."
            )
            node["run_ids"] = sorted(
                set(node.get("run_ids", []))
                | {
                    "R022-cas-posthoc-reclassification",
                    "R022b-cas-posthoc-verification",
                }
            )
    write_jsonl(TREE, tree)
    print(
        f"Registered {len(claims)} claims, {len(registry)} results, "
        f"{len(runs)} runs, and {len(artifacts)} artifacts"
    )


if __name__ == "__main__":
    main()
