#!/usr/bin/env python3
"""Compute and verify topographic shadow illumination invariance on HR-GLDD test split across 3 seeds."""

import os
import json
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(STUDY_DIR, "experiments", "raw", "hr_gldd")
RESULTS_DIR = os.path.join(STUDY_DIR, "experiments", "derived", "results")

testX = np.load(os.path.join(DATA_DIR, "testX.npy"))
testY = np.load(os.path.join(DATA_DIR, "testY.npy"))[..., 0].astype(np.uint8)

illum = testX.mean(axis=(1, 2, 3))
shadow_threshold = float(np.percentile(illum, 25))
shadow_idx = np.where(illum < shadow_threshold)[0]
sunlit_idx = np.where(illum >= shadow_threshold)[0]

arms = ["unet", "resunet", "s3net_ablation", "s3net"]
seeds = [42, 43, 44]

shadow_metrics = {a: [] for a in arms}
sunlit_metrics = {a: [] for a in arms}

for a in arms:
    for s in seeds:
        preds = np.load(os.path.join(RESULTS_DIR, f"test_preds_{a}_seed{s}.npy"))
        
        # Shadowed cohort
        pf_sh = preds[shadow_idx].reshape(-1)
        gt_sh = testY[shadow_idx].reshape(-1)
        tp_sh = np.sum((pf_sh == 1) & (gt_sh == 1))
        fp_sh = np.sum((pf_sh == 1) & (gt_sh == 0))
        fn_sh = np.sum((pf_sh == 0) & (gt_sh == 1))
        f1_sh = float(2 * tp_sh / (2 * tp_sh + fp_sh + fn_sh + 1e-8))
        shadow_metrics[a].append(f1_sh)
        
        # Sunlit cohort
        pf_su = preds[sunlit_idx].reshape(-1)
        gt_su = testY[sunlit_idx].reshape(-1)
        tp_su = np.sum((pf_su == 1) & (gt_su == 1))
        fp_su = np.sum((pf_su == 1) & (gt_su == 0))
        fn_su = np.sum((pf_su == 0) & (gt_su == 1))
        f1_su = float(2 * tp_su / (2 * tp_su + fp_su + fn_su + 1e-8))
        sunlit_metrics[a].append(f1_su)

results = {
    "num_shadow_tiles": int(len(shadow_idx)),
    "num_sunlit_tiles": int(len(sunlit_idx)),
    "shadow_radiance_threshold": round(shadow_threshold, 4),
    "shadowed_regime": {
        a: {
            "micro_f1_mean": float(np.mean(shadow_metrics[a])),
            "micro_f1_std": float(np.std(shadow_metrics[a]))
        } for a in arms
    },
    "sunlit_regime": {
        a: {
            "micro_f1_mean": float(np.mean(sunlit_metrics[a])),
            "micro_f1_std": float(np.std(sunlit_metrics[a]))
        } for a in arms
    },
    "shadow_gain_s3net_vs_unet": float(np.mean(shadow_metrics["s3net"]) - np.mean(shadow_metrics["unet"])),
    "shadow_gain_s3net_vs_ablation": float(np.mean(shadow_metrics["s3net"]) - np.mean(shadow_metrics["s3net_ablation"])),
    "sunlit_gain_s3net_vs_unet": float(np.mean(sunlit_metrics["s3net"]) - np.mean(sunlit_metrics["unet"])),
    "sunlit_gain_s3net_vs_ablation": float(np.mean(sunlit_metrics["s3net"]) - np.mean(sunlit_metrics["s3net_ablation"])),
    "empirical_invariance_ratio": float((np.mean(shadow_metrics["s3net"]) - np.mean(shadow_metrics["unet"])) / 
                                        (np.mean(sunlit_metrics["s3net"]) - np.mean(sunlit_metrics["unet"]) + 1e-8))
}

out_file = os.path.join(RESULTS_DIR, "topographic_shadow_analysis.json")
with open(out_file, "w") as f:
    json.dump(results, f, indent=2)

print(f"Topographic Shadow Analysis saved to {out_file}:")
print(f"  Shadowed (n={len(shadow_idx)}): S3-Net = {results['shadowed_regime']['s3net']['micro_f1_mean']*100:.2f}%, U-Net = {results['shadowed_regime']['unet']['micro_f1_mean']*100:.2f}%, Gain = {results['shadow_gain_s3net_vs_unet']*100:+.2f}%")
print(f"  Sunlit (n={len(sunlit_idx)}):   S3-Net = {results['sunlit_regime']['s3net']['micro_f1_mean']*100:.2f}%, U-Net = {results['sunlit_regime']['unet']['micro_f1_mean']*100:.2f}%, Gain = {results['sunlit_gain_s3net_vs_unet']*100:+.2f}%")
print(f"  Shadow vs Sunlit Gain Ratio: {results['empirical_invariance_ratio']:.2f}x")
