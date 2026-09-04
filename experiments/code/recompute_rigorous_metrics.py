#!/usr/bin/env python3
"""Rigorous recomputation: true tile-level (355 clusters) bootstrap, micro and macro F1 metrics across all 3 seeds."""

import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
RESULTS_DIR = os.path.join(STUDY_DIR, "experiments", "derived", "results")
DATA_DIR = os.path.join(STUDY_DIR, "experiments", "raw", "hr_gldd")

testY = np.load(os.path.join(DATA_DIR, "testY.npy"))[..., 0].astype(np.uint8) # (355, 128, 128)
arms = ["unet", "resunet", "s3net_ablation", "s3net"]
seeds = [42, 43, 44]

arm_metrics = {}
arm_3seed_tiles = {}

for arm in arms:
    seed_micros = []
    seed_macros = []
    seed_ious = []
    seed_precs = []
    seed_recs = []
    seed_tile_f1s = []
    
    for s in seeds:
        pred_path = os.path.join(RESULTS_DIR, f"test_preds_{arm}_seed{s}.npy")
        preds = np.load(pred_path)
        
        # Micro
        yf = testY.reshape(-1)
        pf = preds.reshape(-1)
        tp = np.sum((pf == 1) & (yf == 1))
        fp = np.sum((pf == 1) & (yf == 0))
        fn = np.sum((pf == 0) & (yf == 1))
        
        f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
        iou = tp / (tp + fp + fn + 1e-8)
        prec = tp / (tp + fp + 1e-8)
        rec = tp / (tp + fn + 1e-8)
        
        seed_micros.append(f1)
        seed_ious.append(iou)
        seed_precs.append(prec)
        seed_recs.append(rec)
        
        # Per tile
        tile_f1 = []
        for i in range(len(preds)):
            p_i = preds[i].reshape(-1)
            t_i = testY[i].reshape(-1)
            tp_i = np.sum((p_i == 1) & (t_i == 1))
            fp_i = np.sum((p_i == 1) & (t_i == 0))
            fn_i = np.sum((p_i == 0) & (t_i == 1))
            if tp_i + fp_i + fn_i == 0:
                ti_f1 = 1.0
            else:
                ti_f1 = 2 * tp_i / (2 * tp_i + fp_i + fn_i + 1e-8)
            tile_f1.append(ti_f1)
        tile_f1 = np.array(tile_f1)
        seed_macros.append(float(np.mean(tile_f1)))
        seed_tile_f1s.append(tile_f1)
        
    arm_metrics[arm] = {
        "micro_f1_mean": float(np.mean(seed_micros)),
        "micro_f1_std": float(np.std(seed_micros)),
        "micro_iou_mean": float(np.mean(seed_ious)),
        "micro_iou_std": float(np.std(seed_ious)),
        "micro_precision_mean": float(np.mean(seed_precs)),
        "micro_precision_std": float(np.std(seed_precs)),
        "micro_recall_mean": float(np.mean(seed_recs)),
        "micro_recall_std": float(np.std(seed_recs)),
        "macro_f1_mean": float(np.mean(seed_macros)),
        "macro_f1_std": float(np.std(seed_macros)),
    }
    arm_3seed_tiles[arm] = np.mean(seed_tile_f1s, axis=0) # shape (355,)

def cluster_bootstrap(a, b, n_boot=1000, seed=42):
    rng = np.random.RandomState(seed)
    n = len(a)
    diffs = np.empty(n_boot)
    for k in range(n_boot):
        idx = rng.choice(n, size=n, replace=True)
        diffs[k] = np.mean(a[idx]) - np.mean(b[idx])
    return {
        "mean_diff": float(diffs.mean()),
        "ci95_low": float(np.percentile(diffs, 2.5)),
        "ci95_high": float(np.percentile(diffs, 97.5)),
        "p_value": float(np.mean(diffs <= 0)),
        "excludes_zero": bool(np.percentile(diffs, 2.5) > 0)
    }

comparisons = {
    "s3net_vs_unet": cluster_bootstrap(arm_3seed_tiles["s3net"], arm_3seed_tiles["unet"]),
    "s3net_vs_resunet": cluster_bootstrap(arm_3seed_tiles["s3net"], arm_3seed_tiles["resunet"]),
    "s3net_vs_s3net_ablation": cluster_bootstrap(arm_3seed_tiles["s3net"], arm_3seed_tiles["s3net_ablation"]),
}

with open(os.path.join(RESULTS_DIR, "efficiency_metrics.json")) as f:
    efficiency = json.load(f)

summary = {
    "summary_by_arm": arm_metrics,
    "comparisons": comparisons,
    "efficiency": efficiency
}

with open(os.path.join(RESULTS_DIR, "rigorous_confirmatory_summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

print("Synchronized rigorous_confirmatory_summary.json with 12 three-seed prediction arrays:")
for arm, m in arm_metrics.items():
    print(f"  {arm:15s}: Micro F1 = {m['micro_f1_mean']*100:.2f} ± {m['micro_f1_std']*100:.2f}%, Macro F1 = {m['macro_f1_mean']*100:.2f} ± {m['macro_f1_std']*100:.2f}%")
for comp, res in comparisons.items():
    print(f"  {comp:25s}: diff = {res['mean_diff']*100:+.2f}% [95% CI: {res['ci95_low']*100:+.2f}%, {res['ci95_high']*100:+.2f}%], p = {res['p_value']:.4f}")
