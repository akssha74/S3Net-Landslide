#!/usr/bin/env python3
"""Generate Supplementary Table S3 from the frozen Sen12 summary."""

from __future__ import annotations

import json
from pathlib import Path


STUDY = Path(__file__).resolve().parents[2]
SUMMARY = (
    STUDY
    / "experiments/derived/results/sen12_s2_confirmation_v5/"
    "sen12_s2_confirmation_summary.json"
)
OUTPUT = STUDY / "paper/tables/tab_sen12_conditions.tex"


def points(value: float) -> str:
    return f"{100 * value:+.2f}"


def estimate(row: dict) -> str:
    return (
        f"${points(row['mean'])}$ "
        f"[${points(row['interval'][0])},{points(row['interval'][1])}$] pt"
    )


def main() -> None:
    summary = json.loads(SUMMARY.read_text())
    contrasts = summary["contrasts"]
    generic = summary["generic_boundary_average"]
    positive = sum(
        value > 0 for value in generic["per_inventory"].values()
    )
    arm_means = [
        row["f1"]["mean"]
        for row in summary["absolute_performance"].values()
    ]
    overlaps = len(summary["cross_inventory_spatial_overlaps"])
    text = rf"""\begin{{tabular}}{{@{{}}p{{0.25\textwidth}}p{{0.24\textwidth}}p{{0.31\textwidth}}c@{{}}}}
\toprule
Condition & Estimate & Registered requirement & Result \\
\midrule
NDVI attention minus raw edge F1 &
{estimate(contrasts["ndvi_attention_vs_raw_edge_f1"])} &
95\% upper endpoint $<+1.00$ pt & Fail \\
NDVI-modulated minus plain-boundary F1 &
{estimate(contrasts["ndvi_modulated_vs_plain_f1"])} &
95\% upper endpoint $<+1.00$ pt & Pass \\
Generic plain-boundary F1 &
{estimate(generic)}; {positive}/10 positive &
All control means $>0$; pooled lower endpoint $>0$; at least 8/10 positive &
Fail \\
Absolute transfer &
Arm means ${100 * min(arm_means):.2f}$--${100 * max(arm_means):.2f}\%$ F1 &
Every arm mean at least $10\%$ F1 & Fail \\
Registered spatial integrity &
{overlaps} EPSG:4326 transformed-envelope overlap &
Zero cross-inventory envelope overlaps & Fail \\
\bottomrule
\end{{tabular}}
"""
    OUTPUT.write_text(text)
    print(f"Wrote {OUTPUT.relative_to(STUDY)}")


if __name__ == "__main__":
    main()
