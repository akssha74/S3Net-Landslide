#!/usr/bin/env python3
"""Verify that manuscript prose names the registered Sen12 passing condition."""

from __future__ import annotations

import json
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
SUMMARY = (
    STUDY
    / "experiments/derived/results/sen12_s2_confirmation_v5/"
    "sen12_s2_confirmation_summary.json"
)
ABSTRACT = STUDY / "paper/sections/abstract.tex"
RESULTS = STUDY / "paper/sections/results.tex"
EXPECTED = "ndvi_modulated_vs_plain_upper_ci_below_1_point"
EXPECTED_TEXT = "NDVI-modulated-minus-plain-boundary"


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    passing = [name for name, value in summary["conditions"].items() if value]
    assert passing == [EXPECTED], passing

    for path in (ABSTRACT, RESULTS):
        text = path.read_text()
        assert EXPECTED_TEXT in text, path
        assert "passes only its NDVI-attention-minus-raw-edge" not in text, path
        assert "sole passing condition is the NDVI-attention-minus-raw-edge" not in text, path

    print(
        json.dumps(
            {
                "status": "pass",
                "sole_passing_condition": EXPECTED,
                "manuscript_label": EXPECTED_TEXT,
                "checked_files": [
                    str(ABSTRACT.relative_to(STUDY)),
                    str(RESULTS.relative_to(STUDY)),
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
