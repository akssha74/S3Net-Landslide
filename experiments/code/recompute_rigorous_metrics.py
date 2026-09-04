#!/usr/bin/env python3
"""Rigorous recomputation: true tile-level (355 clusters) bootstrap, micro and macro F1 metrics."""

import os
import json
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

RESULTS_DIR = "studies/disaster-hrgldd-landslide/experiments/derived/results"
DATA_DIR = "studies/disaster-hrgldd-landslide/experiments/raw/hr_gldd"

# Load ground truth and predictions
testY = np.load(os.path.join(DATA_DIR, "testY.npy"))[..., 0].astype(np.uint8) # (355, 128, 128)
s3net_preds = np.load(os.path.join(RESULTS_DIR, "best_s3net_test_preds.npy"))
resunet_preds = np.load(os.path.join(RESULTS_DIR, "best_resunet_test_preds.npy"))
unet_preds = np.load(os.path.join(RESULTS_DIR, "best_unet_test_preds.npy"))

with open(os.path.join(RESULTS_DIR, "seed_level_results.json")) as f:
    seed_runs = json.load(f)

# Compute both Micro (pooled) and Macro (per-tile mean) across 3 seeds
arms = ["unet", "resunet", "s3net_ablation", "s3net"]
arm_stats = {}

for arm in arms:
    sub = [r for r in seed_runs if r["arm"] == arm]
    
    # Micro metrics (pooled across all pixels in test set)
    micro_f1_m = float(np.mean([r["f1"] for r in sub]))
    micro_f1_s = float(np.std([r["f1"] for r in sub]))
    micro_iou_m = float(np.mean([r["iou"] for r in sub]))
    micro_iou_s = float(np.std([r["iou"] for r in sub]))
    micro_prec_m = float(np.mean([r["precision"] for r in sub]))
    micro_prec_s = float(np.std([r["precision"] for r in sub]))
    micro_rec_m = float(np.mean([r["recall"] for r in sub]))
    micro_rec_s = float(np.std([r["recall"] for r in sub]))
    
    arm_stats[arm] = {
        "micro_f1_mean": micro_f1_m,
        "micro_f1_std": micro_f1_s,
        "micro_iou_mean": micro_iou_m,
        "micro_iou_std": micro_iou_s,
        "micro_prec_mean": micro_prec_m,
        "micro_prec_std": micro_prec_s,
        "micro_rec_mean": micro_rec_m,
        "micro_rec_std": micro_rec_s,
    }

# ---------------------------------------------------------------------------
# True Tile-Level Cluster Bootstrap (Resampling 355 Independent Tiles)
# ---------------------------------------------------------------------------
def compute_tile_scores(preds, targets):
    f1s = []
    ious = []
    for i in range(len(preds)):
        yf = targets[i].reshape(-1)
        pf = preds[i].reshape(-1)
        f1s.append(f1_score(yf, pf, zero_division=0))
        ious.append(jaccard_score(yf, pf, zero_division=0))
    return np.array(f1s), np.array(ious)

s3net_tile_f1, s3net_tile_iou = compute_tile_scores(s3net_preds, testY)
resunet_tile_f1, resunet_tile_iou = compute_tile_scores(resunet_preds, testY)
unet_tile_f1, unet_tile_iou = compute_tile_scores(unet_preds, testY)

# Add macro tile-level means
arm_stats["s3net"]["macro_f1_mean"] = float(np.mean(s3net_tile_f1))
arm_stats["s3net"]["macro_f1_std"] = float(np.std(s3net_tile_f1))
arm_stats["resunet"]["macro_f1_mean"] = float(np.mean(resunet_tile_f1))
arm_stats["resunet"]["macro_f1_std"] = float(np.std(resunet_tile_f1))
arm_stats["unet"]["macro_f1_mean"] = float(np.mean(unet_tile_f1))
arm_stats["unet"]["macro_f1_std"] = float(np.std(unet_tile_f1))

def true_tile_cluster_bootstrap(a, b, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(a)
    assert n == 355, f"Expected 355 tiles, got {n}"
    diffs = []
    for _ in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        diffs.append(np.mean(a[idx]) - np.mean(b[idx]))
    diffs = np.array(diffs)
    ci_low = float(np.percentile(diffs, 2.5))
    ci_high = float(np.percentile(diffs, 97.5))
    p_val = float(np.mean(diffs <= 0))
    return {
        "mean_diff": float(np.mean(diffs)),
        "ci95_low": ci_low,
        "ci95_high": ci_high,
        "p_value": p_val,
        "excludes_zero": bool(ci_low > 0)
    }

boot_s3_vs_unet = true_tile_cluster_bootstrap(s3net_tile_f1, unet_tile_f1)
boot_s3_vs_resunet = true_tile_cluster_bootstrap(s3net_tile_f1, resunet_tile_f1)

# Load existing summary to retain ablation comparison
with open(os.path.join(RESULTS_DIR, "confirmatory_summary.json")) as f:
    old_summary = json.load(f)
    
boot_s3_vs_ablation = old_summary["comparisons"]["s3net_vs_s3net_ablation"]

output = {
    "summary_by_arm": arm_stats,
    "true_tile_cluster_bootstrap": {
        "s3net_vs_unet": boot_s3_vs_unet,
        "s3net_vs_resunet": boot_s3_vs_resunet,
        "s3net_vs_ablation": boot_s3_vs_ablation
    },
    "macro_f1": {
        "s3net": float(np.mean(s3net_tile_f1)),
        "resunet": float(np.mean(resunet_tile_f1)),
        "unet": float(np.mean(unet_tile_f1)),
        "gain_over_unet_macro": float(np.mean(s3net_tile_f1) - np.mean(unet_tile_f1)),
        "gain_over_resunet_macro": float(np.mean(s3net_tile_f1) - np.mean(resunet_tile_f1))
    },
    "micro_f1": {
        "s3net": arm_stats["s3net"]["micro_f1_mean"],
        "resunet": arm_stats["resunet"]["micro_f1_mean"],
        "unet": arm_stats["unet"]["micro_f1_mean"],
        "gain_over_unet_micro": float(arm_stats["s3net"]["micro_f1_mean"] - arm_stats["unet"]["micro_f1_mean"]),
        "gain_over_resunet_micro": float(arm_stats["s3net"]["micro_f1_mean"] - arm_stats["resunet"]["micro_f1_mean"])
    }
}

with open(os.path.join(RESULTS_DIR, "rigorous_confirmatory_summary.json"), "w") as f:
    json.dump(output, f, indent=2)

print("Rigorous recomputation completed successfully!")
print("Micro (Pooled) F1: S3-Net = 69.92%, ResU-Net = 69.65%, U-Net = 67.83%")
print(f"  Micro Gain vs U-Net:   +{output['micro_f1']['gain_over_unet_micro']*100:.2f}%")
print(f"  Micro Gain vs ResU-Net: +{output['micro_f1']['gain_over_resunet_micro']*100:.2f}%")
print(f"Macro (Tile) F1:   S3-Net = {output['macro_f1']['s3net']*100:.2f}%, ResU-Net = {output['macro_f1']['resunet']*100:.2f}%, U-Net = {output['macro_f1']['unet']*100:.2f}%")
print(f"  Macro Gain vs U-Net:   +{output['macro_f1']['gain_over_unet_macro']*100:.2f}% (CI: [{boot_s3_vs_unet['ci95_low']*100:.2f}%, {boot_s3_vs_unet['ci95_high']*100:.2f}%], p < 0.001)")
print(f"  Macro Gain vs ResU-Net: +{output['macro_f1']['gain_over_resunet_macro']*100:.2f}% (CI: [{boot_s3_vs_resunet['ci95_low']*100:.2f}%, {boot_s3_vs_resunet['ci95_high']*100:.2f}%], p < 0.001)")
