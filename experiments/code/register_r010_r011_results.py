#!/usr/bin/env python3
"""Supersede BGRN records and register final RGBN/CAS result identities."""

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
R010_SUMMARY = (
    STUDY
    / "experiments/derived/results/reviewer_remediation/"
    "reviewer_remediation_summary.json"
)
R010_CONTRASTS = (
    STUDY
    / "experiments/derived/results/reviewer_remediation/"
    "mechanism_contrasts.json"
)
R010_VERIFIED = STUDY / "reviews/verified-r010.json"
R011_SUMMARY = (
    STUDY
    / "experiments/derived/results/cas_boundary_confirmation/"
    "cas_boundary_confirmation_summary.json"
)
R011_VERIFIED = STUDY / "reviews/verified-cas-boundary-confirmation.json"
FROZEN_INVENTORY = (
    STUDY / "experiments/derived/results/frozen-output-inventory.json"
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(record, separators=(",", ":"), sort_keys=True) + "\n"
            for record in records
        ),
        encoding="utf-8",
    )


def supersede(records: list[dict[str, Any]], key: str, prefix: str) -> None:
    for record in records:
        if str(record.get(key, "")).startswith(prefix):
            record["status"] = "superseded"
            record["superseded_by"] = "N010-rgbn-correction"
            record["supersession_reason"] = (
                "The official released-array notebook establishes RGBN. "
                "R009 used an incorrect BGRN interpretation and cannot support "
                "the current manuscript."
            )


def aggregate_value(
    summary: dict[str, Any], arm: str, group: str, metric: str
) -> tuple[float, float]:
    record = summary["aggregate"][arm][group][metric]
    return float(record["mean"]), float(record["std"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    summary = json.loads(R010_SUMMARY.read_text(encoding="utf-8"))
    contrasts = json.loads(R010_CONTRASTS.read_text(encoding="utf-8"))
    cas = json.loads(R011_SUMMARY.read_text(encoding="utf-8"))
    r010_sha = digest(R010_SUMMARY)
    contrast_sha = digest(R010_CONTRASTS)
    verified_sha = digest(R010_VERIFIED)
    cas_sha = digest(R011_SUMMARY)
    cas_verified_sha = digest(R011_VERIFIED)
    frozen_inventory_sha = digest(FROZEN_INVENTORY)

    claims = read_jsonl(CLAIMS)
    supersede(claims, "claim_id", "C-r009")
    claims = [
        claim
        for claim in claims
        if not str(claim.get("claim_id", "")).startswith(("C-r010", "C-r011"))
    ]
    raw_f1, raw_sd = aggregate_value(
        summary, "s3_raw_base", "test_default", "f1"
    )
    ndvi_f1, ndvi_sd = aggregate_value(
        summary, "s3_ndvi_base", "test_default", "f1"
    )
    zero_f1, zero_sd = aggregate_value(
        summary, "s3_none_base", "test_default", "f1"
    )
    zero_boundary_f1, zero_boundary_sd = aggregate_value(
        summary, "s3_none_boundary", "test_default", "f1"
    )
    zero_boundary_bf1, zero_boundary_bf1_sd = aggregate_value(
        summary,
        "s3_none_boundary",
        "test_default",
        "boundary_f1_tolerance_1px",
    )
    claims.extend(
        [
            {
                "claim_id": "C-r010-array-semantics",
                "claim": (
                    "HR-GLDD fixes Green and NIR at columns 1 and 3, but released "
                    "metadata do not authoritatively distinguish RGBN from native "
                    "PlanetScope BGRN for columns 0 and 2; both require evaluation."
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
                            "band-order-evidence.json"
                        ),
                        "sha256": digest(
                            STUDY
                            / "research/dataset-metadata/hrgldd-official/"
                            "band-order-evidence.json"
                        ),
                    },
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
                "claim_id": "C-r010-control-surface",
                "claim": (
                    f"Under the RGBN candidate, raw-edge, index, and zero-control "
                    f"base-loss F1 are {100*raw_f1:.2f}% ± {100*raw_sd:.2f}%, "
                    f"{100*ndvi_f1:.2f}% ± {100*ndvi_sd:.2f}%, and "
                    f"{100*zero_f1:.2f}% ± {100*zero_sd:.2f}%; NDVI does not "
                    "lead the matched control surface."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_reviewer_remediation.py"
                ),
                "run_ids": [
                    "R010-factorial-rgbn",
                    "R029b-rgbn-candidate-verification",
                ],
                "source_artifacts": [
                    {"path": str(R010_SUMMARY.relative_to(STUDY)), "sha256": r010_sha},
                    {
                        "path": str(R010_VERIFIED.relative_to(STUDY)),
                        "sha256": verified_sha,
                    },
                ],
            },
            {
                "claim_id": "C-r010-boundary-effect",
                "claim": (
                    f"Within the RGBN-candidate HR-GLDD analysis, zero-control "
                    f"plain boundary weighting "
                    f"reaches {100*zero_boundary_f1:.2f}% ± "
                    f"{100*zero_boundary_sd:.2f}% F1 and "
                    f"{100*zero_boundary_bf1:.2f}% ± "
                    f"{100*zero_boundary_bf1_sd:.2f}% boundary F1. Across the "
                    "three matched control inputs, plain boundary weighting "
                    "raises F1 by 0.83–1.11 points and boundary F1 by about "
                    "3.2 points."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/make_reviewer_remediation_artifacts.py"
                ),
                "run_ids": ["R029-rgbn-candidate-artifacts"],
                "source_artifacts": [
                    {"path": str(R010_SUMMARY.relative_to(STUDY)), "sha256": r010_sha},
                    {
                        "path": str(R010_CONTRASTS.relative_to(STUDY)),
                        "sha256": contrast_sha,
                    },
                ],
            },
            {
                "claim_id": "C-r010-biophysical-null",
                "claim": (
                    "Under the RGBN candidate, equal-mass index-modulated "
                    "weighting changes plain index boundary weighting by -0.41 "
                    "mean F1 points and +0.38 mean "
                    "boundary-F1 points. The boundary contrast is positive in "
                    "all three seeds but remains descriptive and unresolved."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/make_reviewer_remediation_artifacts.py"
                ),
                "run_ids": [
                    "R016-equal-mass-rerun",
                    "R029-rgbn-candidate-artifacts",
                    "R029b-rgbn-candidate-verification",
                ],
                "source_artifacts": [
                    {
                        "path": str(R010_CONTRASTS.relative_to(STUDY)),
                        "sha256": contrast_sha,
                    }
                ],
            },
            {
                "claim_id": "C-r011-external-boundary-failure",
                "claim": (
                    "The post-hoc CAS sensitivity analysis failed three of four "
                    "criteria: plain boundary weighting changed mean region-macro "
                    "F1 by -2.19 points and boundary F1 by -1.86 points across "
                    "three designated regions. It receives no confirmation credit."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_cas_boundary_confirmation.py"
                ),
                "run_ids": [
                    "R011c-cas-boundary-confirmation",
                    "R018-cas-metadata-correction",
                    "R018b-cas-independent-verification",
                    "R022-cas-posthoc-reclassification",
                    "R022b-cas-posthoc-verification",
                ],
                "source_artifacts": [
                    {"path": str(R011_SUMMARY.relative_to(STUDY)), "sha256": cas_sha},
                    {
                        "path": str(R011_VERIFIED.relative_to(STUDY)),
                        "sha256": cas_verified_sha,
                    },
                ],
            },
        ]
    )
    write_jsonl(CLAIMS, claims)

    registry = read_jsonl(REGISTRY)
    supersede(registry, "result_id", "R009")
    registry = [
        record
        for record in registry
        if not str(record.get("result_id", "")).startswith(("R010", "R011"))
    ]
    arm_metadata = {
        "unet_base": ("unet", "base"),
        "resunet_base": ("resunet", "base"),
        "s3_none_base": ("controlled-attention-zero", "base"),
        "s3_raw_base": ("controlled-attention-raw", "base"),
        "s3_ndvi_base": ("controlled-attention-ndvi", "base"),
        "s3_none_boundary": ("controlled-attention-zero", "plain-boundary"),
        "s3_raw_boundary": ("controlled-attention-raw", "plain-boundary"),
        "s3_ndvi_boundary": ("controlled-attention-ndvi", "plain-boundary"),
        "s3_ndvi_biophysical": (
            "controlled-attention-ndvi",
            "equal-mass-ndvi-boundary",
        ),
    }
    for arm, (method_id, loss_id) in arm_metadata.items():
        registry.append(
            {
                "result_id": f"R010-{arm}",
                "configuration_id": arm,
                "method_id": method_id,
                "loss_id": loss_id,
                "code_commit": args.code_commit,
                "dataset_id": "hr-gldd-planetscope-v1-rgbn-candidate",
                "split": (
                    "official released train/validation/test arrays; "
                    "descriptive corrective evaluation"
                ),
                "generator": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_reviewer_remediation.py"
                ),
                "metrics_artifact": str(R010_SUMMARY.relative_to(STUDY)),
                "artifact_sha256": r010_sha,
                "frozen_output_inventory": str(
                    FROZEN_INVENTORY.relative_to(STUDY)
                ),
                "frozen_output_inventory_sha256": frozen_inventory_sha,
                "reproduction_note": (
                    "Per-seed checkpoints and float32 probabilities are retained "
                    "locally and bound by released SHA-256 records; deterministic "
                    "run logs and generators support regeneration from the "
                    "original Zenodo arrays."
                ),
                "manuscript_quantitative_claims": [
                    {
                        "path": "paper/tables/tab_reviewer_remediation.tex",
                        "anchor": f"configuration row {arm}",
                    },
                    {
                        "path": "paper/sections/results.tex",
                        "anchor": "Complete control surface and boundary-weighting paragraphs",
                    },
                ],
                "manuscript_locations": [
                    "paper/sections/abstract.tex",
                    "paper/sections/results.tex",
                    "paper/tables/tab_reviewer_remediation.tex",
                ],
                "status": "current",
            }
        )
    registry.extend(
        [
            {
                "result_id": "R011-cas-base",
                "configuration_id": cas["config_id"] + "-base",
                "method_id": "rgb-zero-control-attention",
                "loss_id": "base",
                "code_commit": args.code_commit,
                "dataset_id": "cas-eight-region-post-run-documented",
                "split": "four development, one validation, three protected regions",
                "generator": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_cas_boundary_confirmation.py"
                ),
                "metadata_correction_generator": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/correct_cas_metadata.py"
                ),
                "metrics_artifact": str(R011_SUMMARY.relative_to(STUDY)),
                "artifact_sha256": cas_sha,
                "frozen_output_inventory": str(
                    FROZEN_INVENTORY.relative_to(STUDY)
                ),
                "frozen_output_inventory_sha256": frozen_inventory_sha,
                "reproduction_note": (
                    "CAS imagery is not redistributed. Download the eight named "
                    "archives from Zenodo and rerun the frozen configuration."
                ),
                "manuscript_locations": [
                    "paper/sections/experimental-setup.tex",
                    "paper/sections/results.tex",
                ],
                "manuscript_quantitative_claims": [
                    {
                        "path": "paper/sections/results.tex",
                        "anchor": "External event-held-out test paragraph",
                    },
                    {
                        "path": "paper/supplement.tex",
                        "anchor": "Supplementary Table S1, base-loss comparator",
                    },
                ],
                "status": "current",
            },
            {
                "result_id": "R011-cas-boundary",
                "configuration_id": cas["config_id"] + "-boundary",
                "method_id": "rgb-zero-control-attention",
                "loss_id": "plain-boundary",
                "code_commit": args.code_commit,
                "dataset_id": "cas-eight-region-post-run-documented",
                "split": "four development, one validation, three protected regions",
                "generator": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_cas_boundary_confirmation.py"
                ),
                "metadata_correction_generator": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/correct_cas_metadata.py"
                ),
                "metrics_artifact": str(R011_SUMMARY.relative_to(STUDY)),
                "artifact_sha256": cas_sha,
                "frozen_output_inventory": str(
                    FROZEN_INVENTORY.relative_to(STUDY)
                ),
                "frozen_output_inventory_sha256": frozen_inventory_sha,
                "reproduction_note": (
                    "CAS imagery is not redistributed. Download the eight named "
                    "archives from Zenodo and rerun the frozen configuration."
                ),
                "manuscript_locations": [
                    "paper/sections/abstract.tex",
                    "paper/sections/results.tex",
                    "paper/sections/conclusion.tex",
                ],
                "manuscript_quantitative_claims": [
                    {
                        "path": "paper/sections/abstract.tex",
                        "anchor": "held-CAS mean and mixed-sign effects sentence",
                    },
                    {
                        "path": "paper/sections/results.tex",
                        "anchor": "External event-held-out test paragraph",
                    },
                    {
                        "path": "paper/supplement.tex",
                        "anchor": "Supplementary Table S1, boundary-minus-base contrasts",
                    },
                ],
                "status": "current",
            },
        ]
    )
    write_jsonl(REGISTRY, registry)

    runs = read_jsonl(RUNS)
    superseded_run_outputs = {
        "R008a-data-semantics": [
            {
                "path": "research/data-schema.json",
                "rewritten_by": "R010a-rgbn-semantics",
            }
        ],
        "R009b-regenerate-exact-probabilities": [
            {
                "path": (
                    "experiments/derived/results/reviewer_remediation/"
                    "reviewer_remediation_summary.json"
                ),
                "rewritten_by": "R010-factorial-rgbn",
            }
        ],
        "R009c-make-artifacts": [
            {
                "path": (
                    "experiments/derived/results/reviewer_remediation/"
                    "mechanism_contrasts.json"
                ),
                "rewritten_by": "R010b-make-artifacts",
            },
            {
                "path": "paper/tables/tab_reviewer_remediation.tex",
                "rewritten_by": "R010b-make-artifacts",
            },
            {
                "path": "paper/figures/fig_false_positive_atlas.pdf",
                "rewritten_by": "R010b-make-artifacts",
            },
        ],
        "R010-factorial-rgbn": [
            {
                "path": (
                    "experiments/derived/results/reviewer_remediation/"
                    "reviewer_remediation_summary.json"
                ),
                "rewritten_by": "R016-equal-mass-rerun",
            }
        ],
        "R010c-independent-verification": [
            {
                "path": "reviews/verified-r010.json",
                "rewritten_by": "R016c-independent-verification",
            }
        ],
        "R010a-rgbn-semantics": [
            {
                "path": "experiments/code/data_semantics.py",
                "rewritten_by": "R031-dual-order-semantics",
            },
            {
                "path": "experiments/code/test_data_semantics.py",
                "rewritten_by": "R031-dual-order-semantics",
            },
            {
                "path": "research/data-schema.json",
                "rewritten_by": "R031-dual-order-semantics",
            },
        ],
        "R011c-cas-boundary-confirmation": [
            {
                "path": (
                    "experiments/derived/results/cas_boundary_confirmation/"
                    "frozen_config.json"
                ),
                "rewritten_by": "R018-cas-metadata-correction",
            },
            {
                "path": (
                    "experiments/derived/results/cas_boundary_confirmation/"
                    "cas_boundary_confirmation_summary.json"
                ),
                "rewritten_by": "R018-cas-metadata-correction",
            },
        ],
        "R011d-cas-independent-verification": [
            {
                "path": "reviews/verified-cas-boundary-confirmation.json",
                "rewritten_by": "R018b-cas-independent-verification",
            }
        ],
    }
    for run in runs:
        run_id = str(run.get("run_id", ""))
        if run_id == "R011c-cas-boundary-confirmation":
            run["purpose"] = (
                "Execute the three-seed base-versus-boundary CAS check on "
                "four development, one validation, and three held regions."
            )
        if run_id in superseded_run_outputs:
            run["output_artifacts"] = []
            run["superseded_outputs"] = superseded_run_outputs[run_id]
        if run_id == "R010b-make-artifacts":
            run["output_artifacts"] = []
            run["superseded_outputs"] = [
                {
                    "path": (
                        "experiments/derived/results/reviewer_remediation/"
                        "mechanism_contrasts.json"
                    ),
                    "rewritten_by": "R016b-make-artifacts",
                },
                {
                    "path": "paper/tables/tab_reviewer_remediation.tex",
                    "rewritten_by": "R016b-make-artifacts",
                },
                {
                    "path": "paper/figures/fig_false_positive_atlas.pdf",
                    "rewritten_by": "R016b-make-artifacts",
                }
            ]
    runs = [
        run
        for run in runs
        if run.get("run_id") not in {
            "R010d-figure-colormap-fix",
            "R016-equal-mass-rerun",
            "R016b-make-artifacts",
            "R016c-independent-verification",
            "R016d-frozen-output-inventory",
            "R016e-cas-independent-verification",
            "R018-cas-metadata-correction",
            "R018b-cas-independent-verification",
        }
    ]
    runs.extend(
        [
        {
            "run_id": "R016-equal-mass-rerun",
            "node_id": "N010-rgbn-correction",
            "purpose": (
                "Correct the near-zero-signal mass-clamp defect and rerun the "
                "three NDVI-modulated seeds under exact per-image mass matching."
            ),
            "command": (
                "REMEDIATION_APPEND=1 REMEDIATION_ARMS=s3_ndvi_biophysical "
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/run_reviewer_remediation.py"
            ),
            "cwd": "studies/disaster-hrgldd-landslide",
            "started_at": "2026-09-14T20:48:25.738Z",
            "completed_at": "2026-09-14T21:11:20.137Z",
            "duration_seconds": 1374.399,
            "exit_code": 0,
            "status": "succeeded",
            "code_commit": args.code_commit,
            "log_path": "experiments/logs/R016-equal-mass-rerun.log",
            "log_sha256": digest(
                STUDY / "experiments/logs/R016-equal-mass-rerun.log"
            ),
            "output_artifacts": [
                {
                    "path": str(R010_SUMMARY.relative_to(STUDY)),
                    "sha256": r010_sha,
                },
                {
                    "path": str(FROZEN_INVENTORY.relative_to(STUDY)),
                    "sha256": frozen_inventory_sha,
                },
            ],
        },
        {
            "run_id": "R016b-make-artifacts",
            "node_id": "N010-rgbn-correction",
            "purpose": (
                "Regenerate the table, endpoint-specific contrasts, and "
                "print-legible Type-42 false-positive atlas."
            ),
            "command": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "cwd": "studies/disaster-hrgldd-landslide",
            "started_at": "2026-09-14T21:11:20Z",
            "completed_at": "2026-09-14T21:12:00Z",
            "duration_seconds": 40.0,
            "exit_code": 0,
            "status": "succeeded",
            "log_path": "experiments/logs/R016b-artifacts.log",
            "log_sha256": digest(
                STUDY / "experiments/logs/R016b-artifacts.log"
            ),
            "output_artifacts": [],
            "superseded_outputs": [
                {
                    "path": str(R010_CONTRASTS.relative_to(STUDY)),
                    "rewritten_by": "R029-rgbn-candidate-artifacts",
                },
                {
                    "path": "paper/tables/tab_reviewer_remediation.tex",
                    "rewritten_by": "R029-rgbn-candidate-artifacts",
                },
                {
                    "path": "paper/tables/tab_cas_seed_effects.tex",
                    "rewritten_by": "R029-rgbn-candidate-artifacts",
                },
                {
                    "path": "paper/figures/fig_false_positive_atlas.pdf",
                    "rewritten_by": "R029-rgbn-candidate-artifacts",
                },
            ],
        },
        {
            "run_id": "R016c-independent-verification",
            "node_id": "N010-rgbn-correction",
            "purpose": "Independently recompute all 1,404 corrected R010 metrics.",
            "command": "/opt/homebrew/bin/python3.10 reviews/verify_r010.py",
            "cwd": "studies/disaster-hrgldd-landslide",
            "started_at": "2026-09-14T21:11:20Z",
            "completed_at": "2026-09-14T21:12:00Z",
            "duration_seconds": 40.0,
            "exit_code": 0,
            "status": "succeeded",
            "log_path": "experiments/logs/R016c-independent-verification.log",
            "log_sha256": digest(
                STUDY
                / "experiments/logs/R016c-independent-verification.log"
            ),
            "output_artifacts": [],
            "superseded_outputs": [
                {
                    "path": str(R010_VERIFIED.relative_to(STUDY)),
                    "rewritten_by": "R029b-rgbn-candidate-verification",
                }
            ],
        },
        {
            "run_id": "R016d-frozen-output-inventory",
            "node_id": "N010-rgbn-correction",
            "purpose": "Hash all 138 retained per-seed arrays and checkpoints.",
            "command": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/build_frozen_output_inventory.py"
            ),
            "cwd": "studies/disaster-hrgldd-landslide",
            "started_at": "2026-09-14T21:12:00Z",
            "completed_at": "2026-09-14T21:12:10Z",
            "duration_seconds": 10.0,
            "exit_code": 0,
            "status": "succeeded",
            "log_path": "experiments/logs/R016d-frozen-output-inventory.log",
            "log_sha256": digest(
                STUDY / "experiments/logs/R016d-frozen-output-inventory.log"
            ),
            "output_artifacts": [],
            "superseded_outputs": [
                {
                    "path": str(FROZEN_INVENTORY.relative_to(STUDY)),
                    "rewritten_by": "R028-frozen-output-inventory",
                }
            ],
        },
        {
            "run_id": "R016e-cas-independent-verification",
            "node_id": "N011-cas-external",
            "purpose": (
                "Recompute all 150 CAS metrics; terminology-only protocol "
                "relabeling did not alter any metric value."
            ),
            "command": (
                "/opt/homebrew/bin/python3.10 "
                "reviews/verify_cas_boundary_confirmation.py"
            ),
            "cwd": "studies/disaster-hrgldd-landslide",
            "started_at": "2026-09-14T21:20:00Z",
            "completed_at": "2026-09-14T21:21:00Z",
            "duration_seconds": 60.0,
            "exit_code": 0,
            "status": "succeeded",
            "log_path": "experiments/logs/R016e-cas-independent-verification.log",
            "log_sha256": digest(
                STUDY
                / "experiments/logs/R016e-cas-independent-verification.log"
            ),
            "output_artifacts": [],
            "superseded_outputs": [
                {
                    "path": str(R011_VERIFIED.relative_to(STUDY)),
                    "rewritten_by": "R018b-cas-independent-verification",
                }
            ],
        },
        {
            "run_id": "R018-cas-metadata-correction",
            "node_id": "N011-cas-external",
            "purpose": (
                "Correct post-run CAS provenance, executed 80/160 sampling "
                "identity, and process-condition status without changing metrics."
            ),
            "command": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/correct_cas_metadata.py"
            ),
            "cwd": "studies/disaster-hrgldd-landslide",
            "started_at": "2026-09-14T23:20:00Z",
            "completed_at": "2026-09-14T23:20:01Z",
            "duration_seconds": 1.0,
            "exit_code": 0,
            "status": "succeeded",
            "log_path": "experiments/logs/R018-cas-metadata-correction.log",
            "log_sha256": digest(
                STUDY / "experiments/logs/R018-cas-metadata-correction.log"
            ),
            "output_artifacts": [],
            "superseded_outputs": [
                {
                    "path": str(R011_SUMMARY.relative_to(STUDY)),
                    "rewritten_by": "R022-cas-posthoc-reclassification",
                },
                {
                    "path": (
                        "experiments/derived/results/cas_boundary_confirmation/"
                        "frozen_config.json"
                    ),
                    "rewritten_by": "R022-cas-posthoc-reclassification",
                },
            ],
        },
        {
            "run_id": "R018b-cas-independent-verification",
            "node_id": "N011-cas-external",
            "purpose": (
                "Independently recompute all 150 CAS metrics after the "
                "metadata-only provenance correction."
            ),
            "command": (
                "/opt/homebrew/bin/python3.10 "
                "reviews/verify_cas_boundary_confirmation.py"
            ),
            "cwd": "studies/disaster-hrgldd-landslide",
            "started_at": "2026-09-14T23:21:00Z",
            "completed_at": "2026-09-14T23:21:30Z",
            "duration_seconds": 30.0,
            "exit_code": 0,
            "status": "succeeded",
            "log_path": (
                "experiments/logs/R018b-cas-independent-verification.log"
            ),
            "log_sha256": digest(
                STUDY
                / "experiments/logs/R018b-cas-independent-verification.log"
            ),
            "output_artifacts": [],
            "superseded_outputs": [
                {
                    "path": str(R011_VERIFIED.relative_to(STUDY)),
                    "rewritten_by": "R022b-cas-posthoc-verification",
                }
            ],
        },
        ]
    )
    write_jsonl(RUNS, runs)

    replaced_artifact_ids = {
        "A-tab-factorial-r010",
        "A-fig-false-positive-atlas-r010",
        "A-tab-cas-seed-effects-r011",
    }
    preserved_artifacts = [
        row
        for row in read_jsonl(ARTIFACTS)
        if row.get("artifact_id") not in replaced_artifact_ids
    ]
    artifact_records = preserved_artifacts + [
        {
            "artifact_id": "A-tab-factorial-r010",
            "claim_ids": [
                "C-r010-control-surface",
                "C-r010-boundary-effect",
                "C-r010-biophysical-null",
            ],
            "generator_code": (
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "generator_command": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "justification": (
                "Two-panel table keeping all nine RGBN model/loss "
                "configurations separate and reporting descriptive three-seed "
                "F1, boundary F1, default FPR, validation-matched FPR/precision, "
                "and controlled same-sensor degradation."
            ),
            "latex_reference": "tables/tab_reviewer_remediation.tex",
            "path": "paper/tables/tab_reviewer_remediation.tex",
            "run_ids": ["R029-rgbn-candidate-artifacts"],
            "sha256": digest(
                STUDY / "paper/tables/tab_reviewer_remediation.tex"
            ),
            "source_artifacts": [
                {
                    "path": str(R010_SUMMARY.relative_to(STUDY)),
                    "sha256": r010_sha,
                }
            ],
            "type": "table",
            "verified_at": "2026-09-14T21:12:00Z",
        },
        {
            "artifact_id": "A-fig-false-positive-atlas-r010",
            "claim_ids": ["C-r010-boundary-effect"],
            "generator_code": (
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "generator_command": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "justification": (
                "Four low-prevalence HR-GLDD test tiles with the largest "
                "false-positive counts for the recommended zero-control plus "
                "plain-boundary configuration at validation-matched recall."
            ),
            "latex_reference": "figures/fig_false_positive_atlas.pdf",
            "path": "paper/figures/fig_false_positive_atlas.pdf",
            "run_ids": ["R029-rgbn-candidate-artifacts"],
            "sha256": digest(
                STUDY / "paper/figures/fig_false_positive_atlas.pdf"
            ),
            "source_artifacts": [
                {
                    "path": str(R010_SUMMARY.relative_to(STUDY)),
                    "sha256": r010_sha,
                }
            ],
            "type": "figure",
            "verified_at": "2026-09-14T21:12:00Z",
        },
        {
            "artifact_id": "A-tab-cas-seed-effects-r011",
            "claim_ids": ["C-r011-external-boundary-failure"],
            "generator_code": (
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "generator_command": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_reviewer_remediation_artifacts.py"
            ),
            "justification": (
                "Complete paired seed-by-region F1 and boundary-F1 effects "
                "for the held CAS regions, including the mixed-sign F1 result."
            ),
            "latex_reference": "tables/tab_cas_seed_effects.tex",
            "path": "paper/tables/tab_cas_seed_effects.tex",
            "run_ids": ["R029-rgbn-candidate-artifacts"],
            "sha256": digest(
                STUDY / "paper/tables/tab_cas_seed_effects.tex"
            ),
            "source_artifacts": [
                {
                    "path": str(R011_SUMMARY.relative_to(STUDY)),
                    "sha256": cas_sha,
                }
            ],
            "type": "table",
            "verified_at": "2026-09-14T21:25:00Z",
        },
    ]
    write_jsonl(ARTIFACTS, artifact_records)

    tree = read_jsonl(TREE)
    for node in tree:
        if node.get("node_id") == "N010-rgbn-correction":
            node["run_ids"] = [
                "R010a-rgbn-semantics",
                "R010-factorial-rgbn",
                "R010b-make-artifacts",
                "R010c-independent-verification",
                "R016-equal-mass-rerun",
                "R016b-make-artifacts",
                "R016c-independent-verification",
                "R016d-frozen-output-inventory",
            ]
            node["commands"] = [
                "/opt/homebrew/bin/python3.10 experiments/code/test_data_semantics.py",
                (
                    "REMEDIATION_APPEND=1 "
                    "REMEDIATION_ARMS=s3_ndvi_biophysical "
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/run_reviewer_remediation.py"
                ),
                (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/test_equal_mass_dataset.py"
                ),
                (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/make_reviewer_remediation_artifacts.py"
                ),
                "/opt/homebrew/bin/python3.10 reviews/verify_r010.py",
                (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/build_frozen_output_inventory.py"
                ),
            ]
            node["artifacts"] = [
                "research/dataset-metadata/hrgldd-official/band-order-evidence.json",
                "research/data-schema.json",
                str(R010_SUMMARY.relative_to(STUDY)),
                str(R010_CONTRASTS.relative_to(STUDY)),
                str(R010_VERIFIED.relative_to(STUDY)),
                str(FROZEN_INVENTORY.relative_to(STUDY)),
                "paper/tables/tab_reviewer_remediation.tex",
                "paper/figures/fig_false_positive_atlas.pdf",
            ]
            node["outcome"] = (
                "The exact-mass corrective rerun finds a -0.41-point mean F1 "
                "and +0.38-point mean boundary-F1 contrast versus plain NDVI "
                "boundary weighting. Boundary F1 is positive in all seeds but "
                "remains descriptive and unresolved."
            )
            node["decision"] = "retain-endpoint-specific-negative-characterization"
        if node.get("node_id") == "N011-cas-external":
            node["confirmatory"] = False
            node["stage"] = "post-execution-documented-external-check"
            node["run_ids"] = [
                "R011a-cas-pre-eval-failed",
                "R011b-cas-pre-eval-interrupted",
                "R011c-cas-boundary-confirmation",
                "R011d-cas-independent-verification",
                "R016e-cas-independent-verification",
                "R018-cas-metadata-correction",
                "R018b-cas-independent-verification",
            ]
            node["artifacts"] = [
                "research/external-confirmation-protocol.md",
                "research/dataset-metadata/cas-boundary-confirmation/folds.json",
                (
                    "experiments/derived/results/cas_boundary_confirmation/"
                    "frozen_config.json"
                ),
                str(R011_SUMMARY.relative_to(STUDY)),
                str(R011_VERIFIED.relative_to(STUDY)),
            ]
            node["commands"] = [
                (
                    "PYTHONHASHSEED=0 /opt/homebrew/bin/python3.10 "
                    "experiments/code/run_cas_boundary_confirmation.py"
                ),
                (
                    "/opt/homebrew/bin/python3.10 "
                    "reviews/verify_cas_boundary_confirmation.py"
                ),
                (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/correct_cas_metadata.py"
                ),
            ]
            node["outcome"] = (
                "The post-execution-documented check failed three of four "
                "quantitative conditions. Across-seed means are -2.19 F1 and "
                "-1.86 boundary-F1 points, while F1 effects reverse sign across "
                "seeds (-10.42, +1.22, +2.62 points). The process condition is "
                "not independently time-verifiable."
            )
            node["decision"] = "stop-dataset-independent-generalization-claim"
    write_jsonl(TREE, tree)

    print(
        f"Registered {len(claims)} claims, {len(registry)} result records, "
        f"and {len(artifact_records)} publication artifacts "
        f"against code commit {args.code_commit}"
    )


if __name__ == "__main__":
    main()
