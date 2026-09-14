#!/usr/bin/env python3
"""Hash every frozen R016/R011 probability array and checkpoint."""

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
    STUDY / "experiments/derived/checkpoints/cas_boundary_confirmation",
    STUDY / "experiments/derived/results/cas_boundary_confirmation",
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
        "hr_gldd_checkpoints": sum(
            "checkpoints/reviewer_remediation" in record["path"]
            for record in records
        ),
        "hr_gldd_probability_arrays": sum(
            "results/reviewer_remediation" in record["path"]
            for record in records
        ),
        "cas_checkpoints": sum(
            "checkpoints/cas_boundary_confirmation" in record["path"]
            for record in records
        ),
        "cas_probability_arrays": sum(
            "results/cas_boundary_confirmation" in record["path"]
            for record in records
        ),
    }
    expected = {
        "hr_gldd_checkpoints": 27,
        "hr_gldd_probability_arrays": 81,
        "cas_checkpoints": 6,
        "cas_probability_arrays": 24,
    }
    if counts != expected:
        raise RuntimeError(
            f"Frozen-output inventory incomplete: {counts} != {expected}"
        )
    payload = {
        "schema_version": 1,
        "purpose": (
            "Portable identity for every frozen per-seed probability array "
            "and checkpoint supporting the R016/R011 summaries."
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
