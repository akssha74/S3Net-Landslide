"""Tests for exact validation-recall threshold selection."""

from __future__ import annotations

import numpy as np

from run_reviewer_remediation import choose_threshold_for_recall, confusion


def main() -> None:
    probabilities = np.array([0.95, 0.80, 0.60, 0.60, 0.30, 0.10])
    targets = np.ones_like(probabilities)
    selection = choose_threshold_for_recall(probabilities, targets, 0.5)
    achieved = confusion(
        probabilities,
        targets,
        float(selection["threshold"]),
    )["recall"]
    assert abs(achieved - selection["achieved_validation_recall"]) < 1e-10
    assert selection["absolute_recall_mismatch"] <= 1 / len(targets)
    assert selection["tie_policy"] == (
        "minimum recall mismatch, then highest threshold"
    )
    print(
        "PASS: exact score-breakpoint recall matching; "
        f"target=0.5 achieved={achieved:.6f}"
    )


if __name__ == "__main__":
    main()
