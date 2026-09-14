#!/usr/bin/env python3
"""Generate corrected tables, mechanism contrasts, and a false-positive atlas."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42

from data_semantics import BLUE, GREEN, RED


SCRIPT_DIR = Path(__file__).resolve().parent
STUDY_DIR = SCRIPT_DIR.parent.parent
RESULTS_DIR = STUDY_DIR / "experiments/derived/results/reviewer_remediation"
CAS_SUMMARY = (
    STUDY_DIR
    / "experiments/derived/results/cas_boundary_confirmation/"
    "cas_boundary_confirmation_summary.json"
)
PAPER_TABLES = STUDY_DIR / "paper/tables"
PAPER_FIGURES = STUDY_DIR / "paper/figures"
DATA_DIR = STUDY_DIR / "experiments/raw/hr_gldd"
PAPER_TABLES.mkdir(parents=True, exist_ok=True)
PAPER_FIGURES.mkdir(parents=True, exist_ok=True)

LABELS = {
    "unet_base": "U-Net/base",
    "resunet_base": "ResU-Net/base",
    "s3_none_base": "Attn-zero/base",
    "s3_raw_base": "Attn-raw/base",
    "s3_ndvi_base": "Attn-NDVI/base",
    "s3_none_boundary": "Attn-zero/plain-boundary",
    "s3_raw_boundary": "Attn-raw/plain-boundary",
    "s3_ndvi_boundary": "Attn-NDVI/plain-boundary",
    "s3_ndvi_biophysical": "Attn-NDVI/NDVI-boundary",
}


def pm(group: dict, metric: str, scale: float = 100.0) -> str:
    value = group[metric]
    return f"{value['mean'] * scale:.2f} ({value['std'] * scale:.2f})"


def delta(
    aggregate: dict, left: str, right: str, group: str, metric: str
) -> float:
    return (
        aggregate[left][group][metric]["mean"]
        - aggregate[right][group][metric]["mean"]
    )


def write_table(payload: dict) -> None:
    aggregate = payload["aggregate"]
    lines = [
        "\\footnotesize",
        "\\setlength{\\tabcolsep}{5pt}",
        "\\begin{tabular}{@{}lrrr@{}}",
        "\\toprule",
        "Configuration & F1 (\\%) & Boundary F1 (\\%) & Default FPR (\\%) \\\\",
        "\\midrule",
    ]
    for arm in payload["arms"]:
        default = aggregate[arm]["test_default"]
        lines.append(
            f"{LABELS[arm]} & {pm(default, 'f1')} & "
            f"{pm(default, 'boundary_f1_tolerance_1px')} & "
            f"{pm(default, 'background_fpr')} \\\\"
        )
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabular}",
            "\\par\\vspace{0.6em}",
            "\\begin{tabular}{@{}lrrr@{}}",
            "\\toprule",
            "Configuration & Matched FPR (\\%) & Matched precision (\\%) & 10-m F1 (\\%) \\\\",
            "\\midrule",
        ]
    )
    for arm in payload["arms"]:
        matched = aggregate[arm]["test_matched_validation_recall"]
        degraded = aggregate[arm]["test_controlled_10m"]
        lines.append(
            f"{LABELS[arm]} & {pm(matched, 'background_fpr')} & "
            f"{pm(matched, 'precision')} & {pm(degraded, 'f1')} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    (PAPER_TABLES / "tab_reviewer_remediation.tex").write_text(
        "\n".join(lines)
    )


def write_contrasts(payload: dict) -> str:
    aggregate = payload["aggregate"]
    contrasts = {
        "ndvi_vs_zero_control_f1": delta(
            aggregate, "s3_ndvi_base", "s3_none_base", "test_default", "f1"
        ),
        "ndvi_vs_raw_edge_f1": delta(
            aggregate, "s3_ndvi_base", "s3_raw_base", "test_default", "f1"
        ),
        "plain_boundary_vs_base_f1": delta(
            aggregate,
            "s3_ndvi_boundary",
            "s3_ndvi_base",
            "test_default",
            "f1",
        ),
        "plain_boundary_vs_base_boundary_f1": delta(
            aggregate,
            "s3_ndvi_boundary",
            "s3_ndvi_base",
            "test_default",
            "boundary_f1_tolerance_1px",
        ),
        "zero_control_boundary_vs_base_f1": delta(
            aggregate,
            "s3_none_boundary",
            "s3_none_base",
            "test_default",
            "f1",
        ),
        "raw_edge_boundary_vs_base_f1": delta(
            aggregate,
            "s3_raw_boundary",
            "s3_raw_base",
            "test_default",
            "f1",
        ),
        "ndvi_boundary_vs_raw_boundary_f1": delta(
            aggregate,
            "s3_ndvi_boundary",
            "s3_raw_boundary",
            "test_default",
            "f1",
        ),
        "biophysical_vs_plain_boundary_f1": delta(
            aggregate,
            "s3_ndvi_biophysical",
            "s3_ndvi_boundary",
            "test_default",
            "f1",
        ),
        "biophysical_vs_plain_boundary_boundary_f1": delta(
            aggregate,
            "s3_ndvi_biophysical",
            "s3_ndvi_boundary",
            "test_default",
            "boundary_f1_tolerance_1px",
        ),
    }
    for arm in payload["arms"]:
        contrasts[f"{arm}_controlled_resolution_f1_drop"] = (
            aggregate[arm]["test_controlled_10m"]["f1"]["mean"]
            - aggregate[arm]["test_default"]["f1"]["mean"]
        )

    recommended = max(
        (
            "s3_none_boundary",
            "s3_raw_boundary",
            "s3_ndvi_boundary",
            "s3_ndvi_biophysical",
        ),
        key=lambda arm: (
            aggregate[arm]["test_default"]["f1"]["mean"],
            aggregate[arm]["test_default"]["boundary_f1_tolerance_1px"]["mean"],
        ),
    )
    output = {
        "recommended_arm_by_mean_f1_then_boundary_f1": recommended,
        "contrasts": contrasts,
        "interpretation_rule": (
            "Input specificity requires NDVI to exceed the raw-edge control on "
            "the same endpoint. Segmentation-F1 loss specificity requires the "
            "NDVI-modulated arm to exceed plain boundary weighting on F1; "
            "boundary F1 is reported separately. No equivalence margin or "
            "tile-level inference is used because it was not prespecified and "
            "event/spatial IDs are absent."
        ),
    }
    (RESULTS_DIR / "mechanism_contrasts.json").write_text(
        json.dumps(output, indent=2) + "\n"
    )
    return recommended


def normalize_rgb(image: np.ndarray) -> np.ndarray:
    rgb = image[..., [RED, GREEN, BLUE]].astype(float)
    low, high = np.percentile(rgb, [2, 98])
    return np.clip((rgb - low) / (high - low + 1e-12), 0, 1)


def write_cas_seed_table() -> None:
    payload = json.loads(CAS_SUMMARY.read_text())
    display_names = {
        "Mengdong": "Mengdong",
        "Moxi-UAV-1m": "Moxi town",
        "Tiburon-Planet": "Tiburon Peninsula",
    }

    def pair(f1: float, boundary_f1: float) -> str:
        return f"${100*f1:+.2f}/{100*boundary_f1:+.2f}$"

    lines = [
        r"\small",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Region & Seed 42 & Seed 43 & Seed 44 \\",
        r"\midrule",
    ]
    for event, label in display_names.items():
        cells = [
            pair(
                payload["effects"][str(seed)]["per_event"][event]["delta_f1"],
                payload["effects"][str(seed)]["per_event"][event][
                    "delta_boundary_f1"
                ],
            )
            for seed in (42, 43, 44)
        ]
        lines.append(f"{label} & {' & '.join(cells)} " + r"\\")
    lines.extend([r"\midrule"])
    macro_cells = [
        pair(
            payload["effects"][str(seed)]["event_macro_delta_f1"],
            payload["effects"][str(seed)]["event_macro_delta_boundary_f1"],
        )
        for seed in (42, 43, 44)
    ]
    lines.extend(
        [
            f"Event macro & {' & '.join(macro_cells)} " + r"\\",
            r"\bottomrule",
            r"\end{tabular}",
        ]
    )
    (PAPER_TABLES / "tab_cas_seed_effects.tex").write_text(
        "\n".join(lines) + "\n"
    )


def make_false_positive_atlas(payload: dict, arm: str) -> None:
    seed = 42
    run = next(
        item
        for item in payload["runs"]
        if item["arm"] == arm and item["seed"] == seed
    )
    threshold = run["test_matched_validation_recall"]["threshold"]
    probabilities = np.load(
        RESULTS_DIR / f"test_probabilities_{arm}_seed{seed}.npy"
    ).astype(np.float32)
    test_x = np.load(DATA_DIR / "testX.npy", mmap_mode="r")
    test_y = np.load(DATA_DIR / "testY.npy", mmap_mode="r")[..., 0] >= 0.5
    predictions = probabilities >= threshold
    prevalence = test_y.mean(axis=(1, 2))
    false_positive_counts = np.sum(predictions & ~test_y, axis=(1, 2))
    eligible = np.where(prevalence < 0.05)[0]
    selected = eligible[
        np.argsort(false_positive_counts[eligible])[-4:][::-1]
    ]

    figure, axes = plt.subplots(4, 4, figsize=(6.27, 7.0), dpi=220)
    for position, index in enumerate(selected):
        row = position
        rgb = normalize_rgb(np.asarray(test_x[index]))
        truth = test_y[index]
        prediction = predictions[index]
        false_positive = prediction & ~truth
        overlay = rgb.copy()
        overlay[false_positive] = 0.65 * overlay[false_positive] + 0.35 * np.array(
            [1.0, 0.0, 0.0]
        )
        images = (rgb, truth, prediction, overlay)
        titles = (
            f"RGB tile {index}",
            "Ground truth",
            f"Prediction ($t={threshold:.2f}$)",
            f"False positives: {false_positive.sum()} px",
        )
        for offset, (image, title) in enumerate(zip(images, titles)):
            column = offset
            axes[row, column].imshow(
                image,
                cmap="gray" if offset in (1, 2) else None,
                vmin=0 if offset in (1, 2) else None,
                vmax=1 if offset in (1, 2) else None,
            )
            axes[row, column].set_title(title, fontsize=8)
            axes[row, column].axis("off")
    figure.suptitle(
        "High-false-positive, low-prevalence HR-GLDD tiles at validation-matched recall\n"
        "Red overlay marks false-positive pixels; background categories are not annotated.",
        fontsize=9,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(
        PAPER_FIGURES / "fig_false_positive_atlas.pdf",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def main() -> None:
    payload = json.loads(
        (RESULTS_DIR / "reviewer_remediation_summary.json").read_text()
    )
    if len(payload["protocol"]["seeds"]) < 3:
        raise RuntimeError("Refusing publication artifacts from fewer than 3 seeds")
    write_table(payload)
    write_cas_seed_table()
    recommended = write_contrasts(payload)
    make_false_positive_atlas(payload, recommended)
    print(f"Generated remediation artifacts; recommended arm: {recommended}")


if __name__ == "__main__":
    main()
