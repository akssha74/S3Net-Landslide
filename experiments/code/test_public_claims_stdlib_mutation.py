#!/usr/bin/env python3
"""Ensure observation-level corruption makes the stdlib verifier fail."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
VERIFIER = STUDY / "experiments/code/verify_public_claims_stdlib.py"
SUFFICIENT = (
    STUDY / "experiments/derived/results/public_claim_sufficient_statistics.json"
)


def main() -> None:
    spec = importlib.util.spec_from_file_location("public_claim_verifier", VERIFIER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load verifier")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    payload = json.loads(SUFFICIENT.read_text())
    arm = sorted(payload["sen12_boundary_f1_per_patch"])[0]
    seed = sorted(payload["sen12_boundary_f1_per_patch"][arm])[0]
    inventory = sorted(
        payload["sen12_boundary_f1_per_patch"][arm][seed]
    )[0]
    payload["sen12_boundary_f1_per_patch"][arm][seed][inventory][0] += 0.25

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        mutated = root / "mutated.json"
        mutated.write_text(json.dumps(payload))
        module.SUFFICIENT = mutated
        module.OUTPUT = root / "should-not-exist.json"
        try:
            module.main()
        except AssertionError:
            pass
        else:
            raise RuntimeError("mutated boundary contribution was not rejected")

    print(
        json.dumps(
            {
                "status": "pass",
                "mutation": {
                    "arm": arm,
                    "seed": seed,
                    "inventory": inventory,
                    "field": "boundary_f1_per_patch[0]",
                    "delta": 0.25,
                },
                "expected_verifier_outcome": "non-zero",
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
