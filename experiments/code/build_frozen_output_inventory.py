#!/usr/bin/env python3
"""Hash every frozen HR-GLDD, CAS, and LRD probability/checkpoint artifact."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
OUTPUT = (
    STUDY / "experiments/derived/results/frozen-output-inventory.json"
)
ROOTS = (
    STUDY / "experiments/derived/checkpoints/reviewer_remediation",
    STUDY / "experiments/derived/results/reviewer_remediation",
    STUDY / "experiments/derived/checkpoints/reviewer_remediation_bgrn",
    STUDY / "experiments/derived/results/reviewer_remediation_bgrn",
    STUDY / "experiments/derived/checkpoints/cas_boundary_confirmation",
    STUDY / "experiments/derived/results/cas_boundary_confirmation",
    STUDY / "experiments/derived/checkpoints/lrd_boundary_confirmation",
    STUDY / "experiments/derived/results/lrd_boundary_confirmation",
)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def include(path: Path) -> bool:
    return path.suffix == ".pt" or (
        path.suffix == ".npy" and "probabilities" in path.name
    )


def main() -> None:
    files = sorted(
        path
        for root in ROOTS
        for path in root.rglob("*")
        if path.is_file() and include(path)
    )
    records = [
        {
            "path": path.relative_to(STUDY).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": digest(path),
        }
        for path in files
    ]
    counts = {
        "hr_gldd_rgbn_checkpoints": sum(
            record["path"].startswith(
                "experiments/derived/checkpoints/reviewer_remediation/"
            )
            for record in records
        ),
        "hr_gldd_rgbn_probability_arrays": sum(
            record["path"].startswith(
                "experiments/derived/results/reviewer_remediation/"
            )
            for record in records
        ),
        "hr_gldd_bgrn_checkpoints": sum(
            record["path"].startswith(
                "experiments/derived/checkpoints/reviewer_remediation_bgrn/"
            )
            for record in records
        ),
        "hr_gldd_bgrn_probability_arrays": sum(
            record["path"].startswith(
                "experiments/derived/results/reviewer_remediation_bgrn/"
            )
            for record in records
        ),
        "cas_checkpoints": sum(
            record["path"].startswith(
                "experiments/derived/checkpoints/cas_boundary_confirmation/"
            )
            for record in records
        ),
        "cas_probability_arrays": sum(
            record["path"].startswith(
                "experiments/derived/results/cas_boundary_confirmation/"
            )
            for record in records
        ),
        "lrd_checkpoints": sum(
            record["path"].startswith(
                "experiments/derived/checkpoints/lrd_boundary_confirmation/"
            )
            for record in records
        ),
        "lrd_probability_arrays": sum(
            record["path"].startswith(
                "experiments/derived/results/lrd_boundary_confirmation/"
            )
            for record in records
        ),
    }
    expected = {
        "hr_gldd_rgbn_checkpoints": 27,
        "hr_gldd_rgbn_probability_arrays": 81,
        "hr_gldd_bgrn_checkpoints": 27,
        "hr_gldd_bgrn_probability_arrays": 81,
        "cas_checkpoints": 6,
        "cas_probability_arrays": 24,
        "lrd_checkpoints": 6,
        "lrd_probability_arrays": 42,
    }
    if counts != expected:
        raise RuntimeError(
            f"Frozen-output inventory incomplete: {counts} != {expected}"
        )
    payload = {
        "schema_version": 1,
        "purpose": (
            "Portable identity for every frozen per-seed probability array "
            "and checkpoint supporting the dual-order HR-GLDD, CAS, and "
            "prospective LRD summaries."
        ),
        "assets_redistributed": False,
        "redistribution_note": (
            "The large binary outputs are not stored in the source release. "
            "These hashes bind locally retained outputs; deterministic run "
            "logs and generators are released for regeneration."
        ),
        "counts": counts,
        "records": records,
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"Wrote {OUTPUT} with {len(records)} frozen artifacts")


if __name__ == "__main__":
    main()
