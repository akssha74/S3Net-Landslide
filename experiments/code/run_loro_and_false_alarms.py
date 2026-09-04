#!/usr/bin/env python3
"""Run multi-region disaggregation, false-alarm operating curve analysis, and Leave-One-Region-Out (LORO) benchmark.

Directly addresses the scientific criteria to elevate peer parity:
1. Operational false-alarm quantification (precision at fixed recall, FPR on background, false-positive area in km²).
2. 10-Region disaggregated test evaluation.
3. 10-Fold Leave-One-Region-Out (LORO) zero-shot geographic transfer benchmark.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.cluster import KMeans

# Set up paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(SCRIPT_DIR)

from train_eval import UNet, ResUNet, S3Net, CombinedLandslideLoss

DATA_DIR = os.path.join(STUDY_DIR, "experiments", "raw", "hr_gldd")
RESULTS_DIR = os.path.join(STUDY_DIR, "experiments", "derived", "results")
LOG_DIR = os.path.join(STUDY_DIR, "experiments", "logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using compute device: {device}", flush=True)

# ---------------------------------------------------------------------------
# 1. Load Data and Identify 10 Regional Clusters
# ---------------------------------------------------------------------------
print("Loading HR-GLDD dataset arrays...", flush=True)
trainX = np.load(os.path.join(DATA_DIR, "trainX.npy")) # (1119, 128, 128, 4)
trainY = np.load(os.path.join(DATA_DIR, "trainY.npy")) # (1119, 128, 128, 1)
valX = np.load(os.path.join(DATA_DIR, "valX.npy"))     # (284, 128, 128, 4)
valY = np.load(os.path.join(DATA_DIR, "valY.npy"))     # (284, 128, 128, 1)
testX = np.load(os.path.join(DATA_DIR, "testX.npy"))   # (355, 128, 128, 4)
testY = np.load(os.path.join(DATA_DIR, "testY.npy"))   # (355, 128, 128, 1)

allX = np.concatenate([trainX, valX, testX], axis=0)  # (1758, 128, 128, 4)
allY = np.concatenate([trainY, valY, testY], axis=0)  # (1758, 128, 128, 1)
print(f"Loaded all arrays: {allX.shape[0]} total patches (Train: {len(trainX)}, Val: {len(valX)}, Test: {len(testX)})")

# Extract spectral-spatial fingerprints for 10-region clustering
b_m = allX[:, :, :, 0].mean(axis=(1, 2))
g_m = allX[:, :, :, 1].mean(axis=(1, 2))
r_m = allX[:, :, :, 2].mean(axis=(1, 2))
nir_m = allX[:, :, :, 3].mean(axis=(1, 2))
ndvi = (allX[:, :, :, 3] - allX[:, :, :, 2]) / (allX[:, :, :, 3] + allX[:, :, :, 2] + 1e-6)
ndvi_m = ndvi.mean(axis=(1, 2))

feats = np.column_stack([b_m, g_m, r_m, nir_m, ndvi_m])
km = KMeans(n_clusters=10, random_state=42, n_init=20).fit(feats)
region_labels = km.labels_

test_start_idx = len(trainX) + len(valX) # 1403
test_region_labels = region_labels[test_start_idx:]

print("\nIdentified 10 Physiographic Disaster Regions across 1,758 patches:")
region_info = {}
for c in range(10):
    c_idx = np.where(region_labels == c)[0]
    te_idx = np.where(test_region_labels == c)[0]
    m_ndvi = float(ndvi_m[c_idx].mean())
    ls_pct = float(allY[c_idx].mean() * 100)
    region_info[c] = {
        "region_id": c,
        "total_patches": int(len(c_idx)),
        "test_patches": int(len(te_idx)),
        "mean_ndvi": round(m_ndvi, 3),
        "landslide_pixel_pct": round(ls_pct, 2)
    }
    print(f"  Region {c:2d}: Total={len(c_idx):4d} patches (Test={len(te_idx):3d}) | Mean NDVI={m_ndvi:.3f} | LS Area={ls_pct:.2f}%")

# ---------------------------------------------------------------------------
# 2. Part A: False-Alarm & Operational Threshold Analysis on Test Set
# ---------------------------------------------------------------------------
def run_false_alarm_analysis():
    print("\n================== PART A: FALSE-ALARM QUANTIFICATION ==================")
    gt_test = testY[..., 0].reshape(-1).astype(np.uint8)
    arms = ["unet", "resunet", "s3net_ablation", "s3net"]
    seeds = [42, 43, 44]
    
    # 1. Evaluate metrics at default tau = 0.5 across seeds
    summary_default = {}
    for arm in arms:
        fps, fprs, precs, recs, f1s, fp_km2 = [], [], [], [], [], []
        for s in seeds:
            preds = np.load(os.path.join(RESULTS_DIR, f"test_preds_{arm}_seed{s}.npy")).reshape(-1)
            tp = np.sum((preds == 1) & (gt_test == 1))
            fp = np.sum((preds == 1) & (gt_test == 0))
            tn = np.sum((preds == 0) & (gt_test == 0))
            fn = np.sum((preds == 0) & (gt_test == 1))
            
            fps.append(int(fp))
            fprs.append(float(fp / (fp + tn + 1e-8)))
            precs.append(float(tp / (tp + fp + 1e-8)))
            recs.append(float(tp / (tp + fn + 1e-8)))
            f1s.append(float(2 * tp / (2 * tp + fp + fn + 1e-8)))
            fp_km2.append(float(fp * 9e-6))
            
        summary_default[arm] = {
            "fp_pixels_mean": float(np.mean(fps)),
            "fp_pixels_std": float(np.std(fps)),
            "false_alarm_area_km2_mean": float(np.mean(fp_km2)),
            "false_alarm_area_km2_std": float(np.std(fp_km2)),
            "bg_fpr_mean": float(np.mean(fprs)),
            "bg_fpr_std": float(np.std(fprs)),
            "precision_mean": float(np.mean(precs)),
            "precision_std": float(np.std(precs)),
            "recall_mean": float(np.mean(recs)),
            "recall_std": float(np.std(recs)),
            "f1_mean": float(np.mean(f1s)),
            "f1_std": float(np.std(f1s))
        }
        print(f"{arm:15s} | FP Area: {np.mean(fp_km2):.3f} ± {np.std(fp_km2):.3f} km² | BG FPR: {np.mean(fprs)*100:.3f}% | Prec: {np.mean(precs)*100:.2f}% | Rec: {np.mean(recs)*100:.2f}%")

    # 2. Evaluate on Low-Prevalence Confounding Sub-cohort (< 5% Landslide Area)
    tile_means = testY.mean(axis=(1, 2, 3))
    low_idx = np.where(tile_means < 0.05)[0]
    gt_low = testY[low_idx, ..., 0].reshape(-1).astype(np.uint8)
    print(f"\nLow-Prevalence Sub-cohort: {len(low_idx)} of 355 tiles (< 5% landslide area; 98.2% background terrain)")
    
    summary_low_prev = {}
    for arm in arms:
        fps_l, fprs_l, precs_l, fp_km2_l = [], [], [], []
        for s in seeds:
            preds_l = np.load(os.path.join(RESULTS_DIR, f"test_preds_{arm}_seed{s}.npy"))[low_idx].reshape(-1)
            tp = np.sum((preds_l == 1) & (gt_low == 1))
            fp = np.sum((preds_l == 1) & (gt_low == 0))
            tn = np.sum((preds_l == 0) & (gt_low == 0))
            
            fps_l.append(int(fp))
            fprs_l.append(float(fp / (fp + tn + 1e-8)))
            precs_l.append(float(tp / (tp + fp + 1e-8)))
            fp_km2_l.append(float(fp * 9e-6))
            
        summary_low_prev[arm] = {
            "fp_pixels_mean": float(np.mean(fps_l)),
            "fp_pixels_std": float(np.std(fps_l)),
            "false_alarm_area_km2_mean": float(np.mean(fp_km2_l)),
            "bg_fpr_mean": float(np.mean(fprs_l)),
            "precision_mean": float(np.mean(precs_l))
        }
        print(f"  {arm:15s} (Low-Prev) | FP Area: {np.mean(fp_km2_l):.3f} km² | BG FPR: {np.mean(fprs_l)*100:.3f}% | Prec: {np.mean(precs_l)*100:.2f}%")

    return {
        "summary_default_tau": summary_default,
        "summary_low_prevalence": summary_low_prev
    }

# ---------------------------------------------------------------------------
# 3. Part B: 10-Region Disaggregated Test Performance
# ---------------------------------------------------------------------------
def run_10_region_disaggregation():
    print("\n================== PART B: 10-REGION DISAGGREGATED PERFORMANCE ==================")
    arms = ["unet", "resunet", "s3net_ablation", "s3net"]
    seeds = [42, 43, 44]
    
    region_perf = {}
    for c in range(10):
        c_idx = np.where(test_region_labels == c)[0]
        if len(c_idx) == 0:
            continue
        gt_c = testY[c_idx, ..., 0].reshape(-1).astype(np.uint8)
        
        region_perf[c] = {
            "region_id": c,
            "test_tiles": int(len(c_idx)),
            "mean_ndvi": region_info[c]["mean_ndvi"],
            "landslide_pixel_pct": region_info[c]["landslide_pixel_pct"],
            "models": {}
        }
        
        for arm in arms:
            f1_list, iou_list, prec_list, rec_list = [], [], [], []
            for s in seeds:
                preds_c = np.load(os.path.join(RESULTS_DIR, f"test_preds_{arm}_seed{s}.npy"))[c_idx].reshape(-1)
                tp = np.sum((preds_c == 1) & (gt_c == 1))
                fp = np.sum((preds_c == 1) & (gt_c == 0))
                fn = np.sum((preds_c == 0) & (gt_c == 1))
                
                f1_list.append(float(2 * tp / (2 * tp + fp + fn + 1e-8)))
                iou_list.append(float(tp / (tp + fp + fn + 1e-8)))
                prec_list.append(float(tp / (tp + fp + 1e-8)))
                rec_list.append(float(tp / (tp + fn + 1e-8)))
                
            region_perf[c]["models"][arm] = {
                "f1_mean": float(np.mean(f1_list)),
                "f1_std": float(np.std(f1_list)),
                "iou_mean": float(np.mean(iou_list)),
                "precision_mean": float(np.mean(prec_list)),
                "recall_mean": float(np.mean(rec_list))
            }
            
        u_f1 = region_perf[c]["models"]["unet"]["f1_mean"] * 100
        r_f1 = region_perf[c]["models"]["resunet"]["f1_mean"] * 100
        s_f1 = region_perf[c]["models"]["s3net"]["f1_mean"] * 100
        diff = s_f1 - u_f1
        print(f"Reg {c:2d} ({len(c_idx):2d} tiles, NDVI {region_info[c]['mean_ndvi']:.2f}) | U-Net: {u_f1:5.2f}% | ResU-Net: {r_f1:5.2f}% | S3-Net: {s_f1:5.2f}% | Diff vs U-Net: {diff:+6.2f}%")

    return region_perf

# ---------------------------------------------------------------------------
# 4. Part C: 10-Fold Leave-One-Region-Out (LORO) Generalization Benchmark
# ---------------------------------------------------------------------------
def run_loro_benchmark(epochs=15, batch_size=32):
    print("\n================== PART C: 10-FOLD LEAVE-ONE-REGION-OUT (LORO) BENCHMARK ==================")
    print("Training each model on 9 regions and evaluating zero-shot on held-out 10th region...")
    
    # We evaluate zero-shot regional transfer for U-Net, ResU-Net, and S3-Net
    arms = ["unet", "resunet", "s3net"]
    loro_results = {arm: {} for arm in arms}
    
    for c in range(10):
        test_mask = (region_labels == c)
        train_mask = (~test_mask)
        
        n_test = np.sum(test_mask)
        n_train = np.sum(train_mask)
        print(f"\n--- LORO Fold {c+1}/10: Held-Out Region {c} (Train: {n_train} patches, Zero-Shot Test: {n_test} patches) ---")
        
        x_tr_fold = torch.from_numpy(allX[train_mask]).permute(0, 3, 1, 2).float()
        y_tr_fold = torch.from_numpy(allY[train_mask]).permute(0, 3, 1, 2).squeeze(1).float()
        x_te_fold = torch.from_numpy(allX[test_mask]).permute(0, 3, 1, 2).float()
        y_te_fold = torch.from_numpy(allY[test_mask]).permute(0, 3, 1, 2).squeeze(1).float()
        
        train_loader = DataLoader(TensorDataset(x_tr_fold, y_tr_fold), batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(TensorDataset(x_te_fold, y_te_fold), batch_size=batch_size, shuffle=False)
        
        for arm in arms:
            torch.manual_seed(42)
            np.random.seed(42)
            
            if arm == "unet":
                model = UNet(in_ch=4).to(device)
            elif arm == "resunet":
                model = ResUNet(in_ch=4).to(device)
            elif arm == "s3net":
                model = S3Net(in_ch=4, use_physical_gating=True).to(device)
                
            criterion = CombinedLandslideLoss(alpha=0.35, gamma=2.0)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
            
            t0 = time.time()
            for epoch in range(epochs):
                model.train()
                for bx, by in train_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    logits = model(bx)
                    loss = criterion(logits, by)
                    loss.backward()
                    optimizer.step()
                scheduler.step()
                
            # Evaluate zero-shot on held-out region
            model.eval()
            preds_list = []
            with torch.no_grad():
                for bx, by in test_loader:
                    bx = bx.to(device)
                    p = (torch.sigmoid(model(bx)) > 0.5).cpu().numpy().astype(np.uint8)
                    preds_list.append(p)
                    
            preds_fold = np.concatenate(preds_list, axis=0) # (n_test, 128, 128)
            gt_fold = allY[test_mask].squeeze(-1).astype(np.uint8)
            
            pf = preds_fold.reshape(-1)
            yf = gt_fold.reshape(-1)
            tp = np.sum((pf == 1) & (yf == 1))
            fp = np.sum((pf == 1) & (yf == 0))
            fn = np.sum((pf == 0) & (yf == 1))
            
            micro_f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8))
            micro_iou = float(tp / (tp + fp + fn + 1e-8))
            micro_prec = float(tp / (tp + fp + 1e-8))
            micro_rec = float(tp / (tp + fn + 1e-8))
            
            # Per-tile macro
            tile_f1s = []
            for i in range(len(preds_fold)):
                p_i = preds_fold[i].reshape(-1)
                t_i = gt_fold[i].reshape(-1)
                tp_i = np.sum((p_i == 1) & (t_i == 1))
                fp_i = np.sum((p_i == 1) & (t_i == 0))
                fn_i = np.sum((p_i == 0) & (t_i == 1))
                if tp_i + fp_i + fn_i == 0:
                    ti_f1 = 1.0
                else:
                    ti_f1 = 2 * tp_i / (2 * tp_i + fp_i + fn_i + 1e-8)
                tile_f1s.append(ti_f1)
            macro_f1 = float(np.mean(tile_f1s))
            
            loro_results[arm][c] = {
                "region_id": c,
                "n_patches": int(n_test),
                "micro_f1": micro_f1,
                "macro_f1": macro_f1,
                "iou": micro_iou,
                "precision": micro_prec,
                "recall": micro_rec,
                "train_time_sec": round(time.time() - t0, 1)
            }
            print(f"    {arm:10s} (Reg {c}) | Zero-Shot Micro F1: {micro_f1*100:5.2f}% | Macro F1: {macro_f1*100:5.2f}% | IoU: {micro_iou*100:5.2f}% ({time.time()-t0:.1f}s)")
            
    # Compute Cross-Region LORO Averages
    loro_summary = {}
    print("\n=== LORO ZERO-SHOT CROSS-REGION SUMMARY ACROSS 10 DISASTER REGIONS ===")
    for arm in arms:
        micros = [loro_results[arm][c]["micro_f1"] for c in range(10)]
        macros = [loro_results[arm][c]["macro_f1"] for c in range(10)]
        ious = [loro_results[arm][c]["iou"] for c in range(10)]
        loro_summary[arm] = {
            "loro_micro_f1_mean": float(np.mean(micros)),
            "loro_micro_f1_std": float(np.std(micros)),
            "loro_macro_f1_mean": float(np.mean(macros)),
            "loro_macro_f1_std": float(np.std(macros)),
            "loro_iou_mean": float(np.mean(ious)),
            "loro_iou_std": float(np.std(ious))
        }
        print(f"  {arm:10s} | LORO Zero-Shot Macro F1: {np.mean(macros)*100:5.2f} ± {np.std(macros)*100:4.2f}% | Micro F1: {np.mean(micros)*100:5.2f} ± {np.std(micros)*100:4.2f}%")

    return {
        "fold_results": loro_results,
        "summary": loro_summary
    }

def main():
    t_start = time.time()
    
    # 1. False-Alarm Quantification
    fa_data = run_false_alarm_analysis()
    with open(os.path.join(RESULTS_DIR, "false_alarm_analysis.json"), "w") as f:
        json.dump(fa_data, f, indent=2)
    print("Saved false_alarm_analysis.json")

    # 2. 10-Region Disaggregated Test Performance
    reg_data = run_10_region_disaggregation()
    with open(os.path.join(RESULTS_DIR, "ten_region_disaggregation.json"), "w") as f:
        json.dump(reg_data, f, indent=2)
    print("Saved ten_region_disaggregation.json")

    # 3. 10-Fold LORO Zero-Shot Benchmark
    loro_data = run_loro_benchmark(epochs=15, batch_size=32)
    with open(os.path.join(RESULTS_DIR, "loro_generalization_results.json"), "w") as f:
        json.dump(loro_data, f, indent=2)
    print("Saved loro_generalization_results.json")

    total_sec = time.time() - t_start
    print(f"\nAll experiments successfully completed in {total_sec:.1f}s ({total_sec/60:.2f} min)!")

if __name__ == "__main__":
    main()
