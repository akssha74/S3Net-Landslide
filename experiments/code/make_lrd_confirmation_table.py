#!/usr/bin/env python3
"""Generate the preregistered LRD event/seed effect table."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


STUDY = Path(__file__).resolve().parents[2]
SUMMARY = (
    STUDY
    / "experiments/derived/results/lrd_boundary_confirmation/"
    "lrd_confirmation_summary.json"
)
TABLE = STUDY / "paper/tables/tab_lrd_confirmation.tex"
SEEDS = (42, 43, 44)


def main() -> None:
    payload = json.loads(SUMMARY.read_text())
    lines = [
        r"\small",
        r"\begin{tabular}{lrr}",
        r"\toprule",
        r"Unit & $\Delta$F1 (points) & $\Delta$BF1 (points) \\",
        r"\midrule",
    ]
    for eid in payload["config"]["protected_test_eids"]:
        f1 = 100 * np.mean(
            [
                payload["effects"][str(seed)]["per_event"][eid]["delta_f1"]
                for seed in SEEDS
            ]
        )
        bf1 = 100 * payload["event_mean_boundary_effects"][eid]
        lines.append(f"{eid} & ${f1:+.2f}$ & ${bf1:+.2f}$ " + r"\\")
    lines.append(r"\midrule")
    for seed in SEEDS:
        row = payload["effects"][str(seed)]
        lines.append(
            f"Seed {seed} macro & "
            f"${100*row['event_macro_delta_f1']:+.2f}$ & "
            f"${100*row['event_macro_delta_boundary_f1']:+.2f}$ "
            + r"\\"
        )
    lines.extend(
        [
            r"\midrule",
            (
                "Across-seed macro & "
                f"${100*payload['mean_event_macro_delta_f1']:+.2f}$ & "
                f"${100*payload['mean_event_macro_delta_boundary_f1']:+.2f}$ "
                + r"\\"
            ),
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    TABLE.write_text("\n".join(lines) + "\n")
    print(f"Wrote {TABLE}")


if __name__ == "__main__":
    main()
