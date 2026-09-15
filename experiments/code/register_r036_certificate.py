#!/usr/bin/env python3
"""Register semantic-identifiability and word-count artifacts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


STUDY = Path(__file__).resolve().parents[2]
CLAIMS = STUDY / "evidence/claim-ledger.jsonl"
REGISTRY = STUDY / "evidence/result-registry.jsonl"
RUNS = STUDY / "experiments/run-ledger.jsonl"
TREE = STUDY / "experiments/tree.jsonl"
ARTIFACTS = STUDY / "evidence/artifact-ledger.jsonl"
CERTIFICATE = (
    STUDY
    / "experiments/derived/results/semantic_identifiability_certificate.json"
)
WORD_COUNT = (
    STUDY / "experiments/derived/results/manuscript_word_count.json"
)
TABLE = STUDY / "paper/tables/tab_semantic_identifiability.tex"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text().splitlines()
        if line.strip()
    ]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(
            json.dumps(row, separators=(",", ":"), sort_keys=True) + "\n"
            for row in rows
        )
    )


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_record(
    run_id: str,
    purpose: str,
    command: str,
    log_path: str,
    outputs: list[str],
) -> dict[str, Any]:
    log = STUDY / log_path
    return {
        "run_id": run_id,
        "node_id": "N025-semantic-identifiability-certificate",
        "purpose": purpose,
        "command": command,
        "cwd": "studies/disaster-hrgldd-landslide",
        "started_at": "2026-09-15T08:55:00Z",
        "completed_at": "2026-09-15T09:05:00Z",
        "exit_code": 0,
        "status": "succeeded",
        "log_path": log_path,
        "log_sha256": digest(log),
        "output_artifacts": [
            {"path": path, "sha256": digest(STUDY / path)}
            for path in outputs
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-commit", required=True)
    args = parser.parse_args()
    certificate = json.loads(CERTIFICATE.read_text())
    contrasts = {
        row["contrast_id"]: row for row in certificate["contrasts"]
    }

    claims = [
        row
        for row in read_jsonl(CLAIMS)
        if row.get("claim_id")
        not in {
            "C-r036-semantic-identifiability",
            "C-r037-word-count",
        }
    ]
    claims.extend(
        [
            {
                "claim_id": "C-r036-semantic-identifiability",
                "claim": (
                    "Across the two metadata-consistent HR-GLDD schemas, "
                    "index-input-versus-raw F1 has envelope [-0.21,-0.01] "
                    "points, equal-mass index-versus-plain F1 "
                    "[-0.41,-0.33], index-versus-plain boundary F1 "
                    "[-0.05,+0.38] (direction-unidentified), and generic "
                    "plain-boundary F1 [+0.96,+1.13] with all 18 individual "
                    "effects positive."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/test_semantic_identifiability_certificate.py"
                ),
                "run_ids": [
                    "R036-semantic-identifiability-certificate",
                    "R036b-semantic-identifiability-verification",
                ],
                "source_artifacts": [
                    {
                        "path": str(CERTIFICATE.relative_to(STUDY)),
                        "sha256": digest(CERTIFICATE),
                    },
                    {
                        "path": (
                            "experiments/logs/"
                            "R036b-semantic-identifiability-verification.log"
                        ),
                        "sha256": digest(
                            STUDY
                            / "experiments/logs/"
                            "R036b-semantic-identifiability-verification.log"
                        ),
                    },
                    {
                        "path": "reviews/verified-semantic-identifiability.json",
                        "sha256": digest(
                            STUDY
                            / "reviews/verified-semantic-identifiability.json"
                        ),
                    },
                ],
                "contrast_statuses": {
                    key: row["direction_status"]
                    for key, row in contrasts.items()
                },
            },
            {
                "claim_id": "C-r037-word-count",
                "claim": (
                    "The Pandoc token count over abstract through conclusion "
                    "is 2,788 under the printed exclusion scope."
                ),
                "status": "verified",
                "analysis_command": (
                    "/opt/homebrew/bin/python3.10 "
                    "experiments/code/count_manuscript_words.py"
                ),
                "run_ids": ["R041-cycle25-presentation-word-count"],
                "source_artifacts": [
                    {
                        "path": str(WORD_COUNT.relative_to(STUDY)),
                        "sha256": digest(WORD_COUNT),
                    }
                ],
            },
        ]
    )
    write_jsonl(CLAIMS, claims)

    registry = [
        row
        for row in read_jsonl(REGISTRY)
        if row.get("result_id")
        != "R036-semantic-identifiability-certificate"
    ]
    registry.append(
        {
            "result_id": "R036-semantic-identifiability-certificate",
            "configuration_id": "rgbn-bgrn-four-contrast-certificate-v1",
            "method_id": "finite-schema-effect-envelope",
            "loss_id": "multiple-frozen-contrasts",
            "code_commit": args.code_commit,
            "dataset_id": "hr-gldd-planetscope-v1-two-admissible-schemas",
            "split": "official released train/validation/test arrays; descriptive",
            "generator": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_semantic_identifiability_certificate.py"
            ),
            "metrics_artifact": str(CERTIFICATE.relative_to(STUDY)),
            "artifact_sha256": digest(CERTIFICATE),
            "manuscript_quantitative_claims": [
                {
                    "path": "paper/sections/abstract.tex",
                    "anchor": "semantic-identifiability envelope sentence",
                },
                {
                    "path": "paper/sections/results.tex",
                    "anchor": "Semantic-identifiability certificate subsection",
                },
                {
                    "path": "paper/tables/tab_semantic_identifiability.tex",
                    "anchor": "all four certificate rows",
                },
            ],
            "status": "current",
        }
    )
    write_jsonl(REGISTRY, registry)

    superseded_word_run = run_record(
        "R037-manuscript-word-count",
        "Generate the earlier word count before scope correction.",
        (
            "/opt/homebrew/bin/python3.10 "
            "experiments/code/count_manuscript_words.py"
        ),
        "experiments/logs/R037-manuscript-word-count.log",
        [],
    )
    superseded_word_run["superseded_outputs"] = [
        {
            "path": "experiments/derived/results/manuscript_word_count.json",
            "rewritten_by": "R038-scope-correct-word-count",
        }
    ]
    superseded_scope_run = run_record(
        "R038-scope-correct-word-count",
        "Generate the scope-correct count before cycle-24 prose repairs.",
        (
            "/opt/homebrew/bin/python3.10 "
            "experiments/code/count_manuscript_words.py"
        ),
        "experiments/logs/R038-scope-correct-word-count.log",
        [],
    )
    superseded_scope_run["superseded_outputs"] = [
        {
            "path": "experiments/derived/results/manuscript_word_count.json",
            "rewritten_by": "R039-cycle24-word-count",
        }
    ]
    superseded_cycle24_run = run_record(
        "R039-cycle24-word-count",
        "Regenerate the word count before the full cycle-24 union.",
        (
            "/opt/homebrew/bin/python3.10 "
            "experiments/code/count_manuscript_words.py"
        ),
        "experiments/logs/R039-cycle24-word-count.log",
        [],
    )
    superseded_cycle24_run["superseded_outputs"] = [
        {
            "path": "experiments/derived/results/manuscript_word_count.json",
            "rewritten_by": "R040-cycle24-union-word-count",
        }
    ]
    superseded_union_run = run_record(
        "R040-cycle24-union-word-count",
        "Generate the word count before cycle-25 presentation repairs.",
        (
            "/opt/homebrew/bin/python3.10 "
            "experiments/code/count_manuscript_words.py"
        ),
        "experiments/logs/R040-cycle24-union-word-count.log",
        [],
    )
    superseded_union_run["superseded_outputs"] = [
        {
            "path": "experiments/derived/results/manuscript_word_count.json",
            "rewritten_by": "R041-cycle25-presentation-word-count",
        }
    ]
    new_runs = [
        run_record(
            "R036-semantic-identifiability-certificate",
            "Generate finite schema envelopes and the certificate table.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_semantic_identifiability_certificate.py"
            ),
            "experiments/logs/R036-semantic-identifiability-certificate.log",
            [
                "experiments/derived/results/"
                "semantic_identifiability_certificate.json",
                "paper/tables/tab_semantic_identifiability.tex",
            ],
        ),
        run_record(
            "R036b-semantic-identifiability-verification",
            "Independently recompute certificate contrasts from run summaries.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/test_semantic_identifiability_certificate.py"
            ),
            (
                "experiments/logs/"
                "R036b-semantic-identifiability-verification.log"
            ),
            ["reviews/verified-semantic-identifiability.json"],
        ),
        superseded_word_run,
        superseded_scope_run,
        superseded_cycle24_run,
        superseded_union_run,
        run_record(
            "R041-cycle25-presentation-word-count",
            "Regenerate the count after cycle-25 presentation repairs.",
            (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/count_manuscript_words.py"
            ),
            "experiments/logs/R041-cycle25-presentation-word-count.log",
            ["experiments/derived/results/manuscript_word_count.json"],
        ),
    ]
    runs = [
        row
        for row in read_jsonl(RUNS)
        if row.get("run_id")
        not in {
            "R036-semantic-identifiability-certificate",
            "R036b-semantic-identifiability-verification",
            "R037-manuscript-word-count",
            "R038-scope-correct-word-count",
            "R039-cycle24-word-count",
            "R040-cycle24-union-word-count",
            "R041-cycle25-presentation-word-count",
        }
    ]
    runs.extend(new_runs)
    write_jsonl(RUNS, runs)

    tree = [
        row
        for row in read_jsonl(TREE)
        if row.get("node_id") != "N025-semantic-identifiability-certificate"
    ]
    tree.append(
        {
            "node_id": "N025-semantic-identifiability-certificate",
            "parent_id": "N010-rgbn-correction",
            "stage": "retrospective-new-measurement-formalization",
            "status": "succeeded",
            "decision": "continue-as-schema-robust-negative-characterization",
            "confirmatory": False,
            "exploratory": True,
            "hypothesis": (
                "Propagating every metadata-consistent schema into signed "
                "mechanism contrasts distinguishes schema-invariant directions "
                "from a zero-crossing unidentified direction."
            ),
            "commands": [row["command"] for row in new_runs],
            "artifacts": [
                "experiments/derived/results/"
                "semantic_identifiability_certificate.json",
                "reviews/verified-semantic-identifiability.json",
                "paper/tables/tab_semantic_identifiability.tex",
                "experiments/derived/results/manuscript_word_count.json",
            ],
            "run_ids": [row["run_id"] for row in new_runs],
            "outcome": (
                "Three effect directions are schema-invariant over RGBN/BGRN; "
                "index-modulated boundary F1 crosses zero and is unidentified."
            ),
        }
    )
    write_jsonl(TREE, tree)

    artifacts = [
        row
        for row in read_jsonl(ARTIFACTS)
        if row.get("artifact_id") != "A-tab-semantic-identifiability-r036"
    ]
    artifacts.append(
        {
            "artifact_id": "A-tab-semantic-identifiability-r036",
            "claim_ids": ["C-r036-semantic-identifiability"],
            "generator_code": (
                "experiments/code/"
                "make_semantic_identifiability_certificate.py"
            ),
            "generator_command": (
                "/opt/homebrew/bin/python3.10 "
                "experiments/code/make_semantic_identifiability_certificate.py"
            ),
            "path": "paper/tables/tab_semantic_identifiability.tex",
            "latex_reference": "tables/tab_semantic_identifiability.tex",
            "justification": (
                "Reports all schema-specific contrasts, finite envelopes, and "
                "direction statuses."
            ),
            "run_ids": ["R036-semantic-identifiability-certificate"],
            "sha256": digest(TABLE),
            "source_artifacts": [
                {
                    "path": str(CERTIFICATE.relative_to(STUDY)),
                    "sha256": digest(CERTIFICATE),
                }
            ],
            "type": "table",
            "verified_at": "2026-09-15T09:05:00Z",
        }
    )
    write_jsonl(ARTIFACTS, artifacts)
    print(
        f"Registered {len(claims)} claims, {len(registry)} results, "
        f"{len(runs)} runs, {len(tree)} tree nodes, and "
        f"{len(artifacts)} artifacts"
    )


if __name__ == "__main__":
    main()
