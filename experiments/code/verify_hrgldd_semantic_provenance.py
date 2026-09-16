#!/usr/bin/env python3
"""Verify the final HR-GLDD channel-order evidence hierarchy."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
METADATA = STUDY / "research/dataset-metadata/hrgldd-official"
EVIDENCE = METADATA / "band-order-evidence.json"
AUDIT = METADATA / "band-order-audit.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    evidence = json.loads(EVIDENCE.read_text())
    audit = json.loads(AUDIT.read_text())

    assert evidence["array_construction_code_available"] is False
    assert "RGBN has stronger release-internal support" in evidence["conclusion"]
    assert "BGRN is retained only as a conservative" in evidence["conclusion"]
    assert audit["status"] == "rgbn-supported-bgrn-conservative-sensitivity"
    assert audit["candidate_orders"]["RGBN"]["order"] == [
        "red",
        "green",
        "blue",
        "nir",
    ]
    assert audit["candidate_orders"]["BGRN"]["order"] == [
        "blue",
        "green",
        "red",
        "nir",
    ]
    assert audit["fixed_semantics"] == {"green_index": 1, "nir_index": 3}

    print(
        json.dumps(
            {
                "status": "pass",
                "interpretation": "RGBN-supported-BGRN-conservative-sensitivity",
                "artifacts": {
                    str(EVIDENCE.relative_to(STUDY)): sha256(EVIDENCE),
                    str(AUDIT.relative_to(STUDY)): sha256(AUDIT),
                },
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
