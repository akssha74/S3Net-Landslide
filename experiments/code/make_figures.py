#!/usr/bin/env python3
"""Generate publication-quality figures for IEEE GRSL landslide manuscript on real HR-GLDD data."""

import os
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = "studies/disaster-hrgldd-landslide/experiments/derived/results"
DATA_DIR = "studies/disaster-hrgldd-landslide/experiments/raw/hr_gldd"
FIGURES_DIR = "studies/disaster-hrgldd-landslide/paper/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 8,
    "axes.labelsize": 8,
    "legend.fontsize": 7,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.titlesize": 9,
    "text.usetex": False
})

def make_performance_chart():
    summary_file = os.path.join(RESULTS_DIR, "rigorous_confirmatory_summary.json")
    with open(summary_file) as f:
        data = json.load(f)["summary_by_arm"]
        
    arms = ["unet", "resunet", "s3net_ablation", "s3net"]
    labels = ["U-Net", "ResU-Net", "S$^3$-Net (w/o G)", "S$^3$-Net (Ours)"]
    
    f1_m = [data[a]["micro_f1_mean"] * 100 for a in arms]
    f1_s = [data[a]["micro_f1_std"] * 100 for a in arms]
    iou_m = [data[a]["micro_iou_mean"] * 100 for a in arms]
    iou_s = [data[a]["micro_iou_std"] * 100 for a in arms]
    prec_m = [data[a]["micro_precision_mean"] * 100 for a in arms]
    prec_s = [data[a]["micro_precision_std"] * 100 for a in arms]
    rec_m = [data[a]["micro_recall_mean"] * 100 for a in arms]
    rec_s = [data[a]["micro_recall_std"] * 100 for a in arms]
    
    x = np.arange(len(arms))
    width = 0.20
    
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    
    # All 4 series have valid standard deviation error bars
    r1 = ax.bar(x - 1.5*width, f1_m, width, yerr=f1_s, capsize=2, label="F1-Score", color="#1f77b4", edgecolor="black", linewidth=0.5)
    r2 = ax.bar(x - 0.5*width, iou_m, width, yerr=iou_s, capsize=2, label="IoU", color="#2ca02c", edgecolor="black", linewidth=0.5)
    r3 = ax.bar(x + 0.5*width, prec_m, width, yerr=prec_s, capsize=2, label="Precision", color="#ff7f0e", edgecolor="black", linewidth=0.5)
    r4 = ax.bar(x + 1.5*width, rec_m, width, yerr=rec_s, capsize=2, label="Recall", color="#9467bd", edgecolor="black", linewidth=0.5)
    
    ax.set_ylabel("Metric Score (%)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=10, ha="right", fontsize=6.5)
    ax.set_ylim(45, 85)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(loc="upper left", framealpha=0.9, ncol=2)
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_performance_comparison.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved performance figure to {out_path}")

def make_mechanism_diagram():
    fig, ax = plt.subplots(figsize=(3.4, 2.2), dpi=300)
    
    ax.text(0.5, 0.93, "Physics-Guided Spectral-Spatial Scarp Attention", ha="center", weight="bold", color="#1f77b4", fontsize=7.5)
    
    # Terrain profile
    x_terrain = np.linspace(0.05, 0.95, 100)
    y_terrain = 0.70 - 0.40 * (x_terrain - 0.05)
    ax.plot(x_terrain, y_terrain, color="#444444", lw=1.5)
    
    # Landslide scar (exposed soil / sheared scarp)
    ax.fill_between(x_terrain[30:70], 0.10, y_terrain[30:70], color="#d95f02", alpha=0.3, label="Landslide Scar")
    # Intact vegetation canopy
    ax.fill_between(x_terrain[0:30], 0.10, y_terrain[0:30], color="#2ca02c", alpha=0.3, label="Vegetated Slopes")
    ax.fill_between(x_terrain[70:100], 0.10, y_terrain[70:100], color="#2ca02c", alpha=0.3)
    
    # Scarp sharp boundaries
    ax.scatter([x_terrain[30], x_terrain[70]], [y_terrain[30], y_terrain[70]], color="#d62728", s=30, zorder=5, label="Scarp Discontinuity")
    ax.annotate("Crown Scarp\n(High |∇NDVI|)", xy=(x_terrain[30], y_terrain[30]), xytext=(0.08, 0.75),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0), fontsize=6, weight="bold")
    ax.annotate("Toe Boundary\n(High |∇NDVI|)", xy=(x_terrain[70], y_terrain[70]), xytext=(0.65, 0.58),
                arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.0), fontsize=6, weight="bold")
                
    # SSSA block formula exactly matching Equations (4) and (5)
    formula_str = r"$\mathbf{M}_l = \sigma(\text{Conv}_{1 \times 1}(\delta(\text{BN}(\text{Conv}_{3 \times 3}([\mathbf{F}_l, \mathbf{V}_l, \mathbf{G}_l]))))); \quad \mathbf{F}_{l,\text{out}} = \mathbf{F}_l \odot \mathbf{M}_l + \mathbf{F}_l$"
    ax.text(0.5, 0.04, formula_str,
            ha="center", fontsize=5.8, bbox=dict(boxstyle="round,pad=0.2", fc="#f0f0f0", ec="#aaaaaa", lw=0.5))

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.0)
    ax.axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_scattering_mechanism.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved mechanism diagram to {out_path}")

def make_qualitative_patches():
    testX = np.load(os.path.join(DATA_DIR, "testX.npy")) # (355, 128, 128, 4)
    testY = np.load(os.path.join(DATA_DIR, "testY.npy")) # (355, 128, 128, 1)
    s3net_preds = np.load(os.path.join(RESULTS_DIR, "best_s3net_test_preds.npy"))
    unet_preds = np.load(os.path.join(RESULTS_DIR, "best_unet_test_preds.npy"))
    
    # Pick patch index with clean landslide scar and surrounding vegetation
    idx_candidates = [i for i in range(len(testY)) if 0.12 <= testY[i].mean() <= 0.35]
    idx = idx_candidates[2] if len(idx_candidates) > 2 else 0
    print(f"Plotting qualitative patch index: {idx}")
    
    patch_rgb = testX[idx][..., [2, 1, 0]] # Red, Green, Blue
    patch_rgb = (patch_rgb - patch_rgb.min()) / (patch_rgb.max() - patch_rgb.min() + 1e-6)
    
    red = testX[idx, :, :, 2]
    nir = testX[idx, :, :, 3]
    ndvi = (nir - red) / (nir + red + 1e-6)
    
    dx = ndvi[:, 1:] - ndvi[:, :-1]
    dx = np.pad(dx, ((0, 0), (0, 1)), mode='edge')
    dy = ndvi[1:, :] - ndvi[:-1, :]
    dy = np.pad(dy, ((0, 1), (0, 0)), mode='edge')
    grad = np.sqrt(dx**2 + dy**2)
    
    gt = testY[idx, :, :, 0]
    p_unet = unet_preds[idx]
    p_s3net = s3net_preds[idx]
    
    fig, axes = plt.subplots(1, 5, figsize=(7.0, 1.6), dpi=300)
    
    axes[0].imshow(patch_rgb)
    axes[0].set_title("(a) True Color (RGB)", fontsize=6.5)
    axes[0].axis("off")
    
    axes[1].imshow(grad, cmap="magma")
    axes[1].set_title("(b) Scarp |∇NDVI|", fontsize=6.5)
    axes[1].axis("off")
    
    axes[2].imshow(gt, cmap="Blues", vmin=0, vmax=1)
    axes[2].set_title("(c) Ground Truth", fontsize=6.5)
    axes[2].axis("off")
    
    axes[3].imshow(p_unet, cmap="Oranges", vmin=0, vmax=1)
    axes[3].set_title("(d) Vanilla U-Net", fontsize=6.5)
    axes[3].axis("off")
    
    axes[4].imshow(p_s3net, cmap="Greens", vmin=0, vmax=1)
    axes[4].set_title("(e) S$^3$-Net (Ours)", fontsize=6.5)
    axes[4].axis("off")
    
    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig_qualitative_patches.pdf")
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"Saved qualitative patches to {out_path}")

if __name__ == "__main__":
    make_performance_chart()
    make_mechanism_diagram()
    make_qualitative_patches()
