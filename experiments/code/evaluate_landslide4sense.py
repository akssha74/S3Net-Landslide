#!/usr/bin/env python3
"""Cross-Dataset, Cross-Sensor Evaluation on Landslide4Sense (Sentinel-2, 10 m GSD).

Evaluates models trained on HR-GLDD (PlanetScope 3 m, 4-band optical) zero-shot on
an independent global benchmark: Landslide4Sense (245 multi-spectral patches from
diverse global geohazard zones, Ghorbanzadeh et al., IEEE GRSL / IARAI 2022).

Establishes whether physical scarp gradient gating provides genuine cross-sensor,
cross-resolution transferability without fine-tuning.
"""

import os
import sys
import time
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
sys.path.append(SCRIPT_DIR)

from train_eval import UNet, ResUNet, S3Net

DATA_DIR = os.path.join(STUDY_DIR, "experiments", "raw", "landslide4sense")
HR_GLDD_DIR = os.path.join(STUDY_DIR, "experiments", "raw", "hr_gldd")
RESULTS_DIR = os.path.join(STUDY_DIR, "experiments", "derived", "results")

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using compute device: {device}", flush=True)

def evaluate_zero_shot_l4s():
    print("\n================== LANDSLIDE4SENSE ZERO-SHOT BENCHMARK ==================")
    l4s_X = np.load(os.path.join(DATA_DIR, "l4s_testX.npy")) # (245, 128, 128, 4) [B, G, R, NIR]
    l4s_Y = np.load(os.path.join(DATA_DIR, "l4s_testY.npy"))[..., 0].astype(np.uint8) # (245, 128, 128)
    
    gt_flat = l4s_Y.reshape(-1)
    pos_px = np.sum(gt_flat == 1)
    neg_px = np.sum(gt_flat == 0)
    print(f"Loaded Landslide4Sense validation set: {len(l4s_X)} patches (128x128, 10m GSD)")
    print(f"  Total pixels: {len(gt_flat):,}, Landslide pixels: {pos_px:,} ({pos_px/len(gt_flat)*100:.2f}%)")

    x_tensor = torch.from_numpy(l4s_X).permute(0, 3, 1, 2).float()
    loader = DataLoader(TensorDataset(x_tensor), batch_size=32, shuffle=False)

    # We evaluate models trained on HR-GLDD
    # First, train each model on HR-GLDD train split (seeds 42, 43, 44) and evaluate directly on Landslide4Sense
    trainX = np.load(os.path.join(HR_GLDD_DIR, "trainX.npy"))
    trainY = np.load(os.path.join(HR_GLDD_DIR, "trainY.npy"))
    valX = np.load(os.path.join(HR_GLDD_DIR, "valX.npy"))
    valY = np.load(os.path.join(HR_GLDD_DIR, "valY.npy"))

    x_tr = torch.from_numpy(trainX).permute(0, 3, 1, 2).float()
    y_tr = torch.from_numpy(trainY).permute(0, 3, 1, 2).squeeze(1).float()
    x_val = torch.from_numpy(valX).permute(0, 3, 1, 2).float()
    y_val = torch.from_numpy(valY).permute(0, 3, 1, 2).squeeze(1).float()

    from train_eval import CombinedLandslideLoss
    criterion = CombinedLandslideLoss(alpha=0.35, gamma=2.0)
    
    arms = ["unet", "resunet", "s3net_ablation", "s3net"]
    seeds = [42, 43, 44]
    
    results = {}

    for arm in arms:
        f1_list, macro_list, iou_list, prec_list, rec_list, fpr_list = [], [], [], [], [], []
        for s in seeds:
            torch.manual_seed(s)
            np.random.seed(s)

            if arm == "unet":
                model = UNet(4).to(device)
            elif arm == "resunet":
                model = ResUNet(4).to(device)
            elif arm == "s3net_ablation":
                model = S3Net(4, use_physical_gating=False).to(device)
            elif arm == "s3net":
                model = S3Net(4, use_physical_gating=True).to(device)

            optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=15, eta_min=1e-5)
            
            tr_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=32, shuffle=True)
            v_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=32, shuffle=False)

            best_v_f1 = -1.0
            best_w = None

            for epoch in range(15):
                model.train()
                for bx, by in tr_loader:
                    bx, by = bx.to(device), by.to(device)
                    optimizer.zero_grad()
                    out = model(bx)
                    loss = criterion(out, by)
                    loss.backward()
                    optimizer.step()
                scheduler.step()

                model.eval()
                tp, fp, fn = 0, 0, 0
                with torch.no_grad():
                    for bx, by in v_loader:
                        bx, by = bx.to(device), by.to(device)
                        pm = (torch.sigmoid(model(bx)) > 0.5)
                        tm = (by > 0.5)
                        tp += torch.sum(pm & tm).item()
                        fp += torch.sum(pm & (~tm)).item()
                        fn += torch.sum((~pm) & tm).item()
                vf1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
                if vf1 > best_v_f1:
                    best_v_f1 = vf1
                    best_w = {k: v.cpu().clone() for k, v in model.state_dict().items()}

            # Zero-shot evaluation on Landslide4Sense
            model.load_state_dict({k: v.to(device) for k, v in best_w.items()})
            model.eval()

            preds_l = []
            with torch.no_grad():
                for bx, in loader:
                    bx = bx.to(device)
                    p = (torch.sigmoid(model(bx)) > 0.5).cpu().numpy().astype(np.uint8)
                    preds_l.append(p)
            preds = np.concatenate(preds_l, axis=0) # (245, 128, 128)

            pf = preds.reshape(-1)
            tp = np.sum((pf == 1) & (gt_flat == 1))
            fp = np.sum((pf == 1) & (gt_flat == 0))
            fn = np.sum((pf == 0) & (gt_flat == 1))
            tn = np.sum((pf == 0) & (gt_flat == 0))

            f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8))
            iou = float(tp / (tp + fp + fn + 1e-8))
            prec = float(tp / (tp + fp + 1e-8))
            rec = float(tp / (tp + fn + 1e-8))
            fpr = float(fp / (fp + tn + 1e-8))

            tile_f1s = []
            for i in range(len(preds)):
                p_i = preds[i].reshape(-1)
                t_i = l4s_Y[i].reshape(-1)
                tp_i = np.sum((p_i == 1) & (t_i == 1))
                fp_i = np.sum((p_i == 1) & (t_i == 0))
                fn_i = np.sum((p_i == 0) & (t_i == 1))
                if tp_i + fp_i + fn_i == 0:
                    ti_f1 = 1.0
                else:
                    ti_f1 = 2 * tp_i / (2 * tp_i + fp_i + fn_i + 1e-8)
                tile_f1s.append(ti_f1)
            macro_f1 = float(np.mean(tile_f1s))

            f1_list.append(f1)
            macro_list.append(macro_f1)
            iou_list.append(iou)
            prec_list.append(prec)
            rec_list.append(rec)
            fpr_list.append(fpr)
            print(f"  {arm:15s} (seed {s}) -> L4S Micro F1: {f1*100:.2f}%, Macro F1: {macro_f1*100:.2f}%, IoU: {iou*100:.2f}%, Rec: {rec*100:.2f}%, Prec: {prec*100:.2f}%")

        results[arm] = {
            "micro_f1_mean": float(np.mean(f1_list)),
            "micro_f1_std": float(np.std(f1_list)),
            "macro_f1_mean": float(np.mean(macro_list)),
            "macro_f1_std": float(np.std(macro_list)),
            "iou_mean": float(np.mean(iou_list)),
            "precision_mean": float(np.mean(prec_list)),
            "recall_mean": float(np.mean(rec_list)),
            "bg_fpr_mean": float(np.mean(fpr_list))
        }
        print(f"--> {arm.upper()} Zero-Shot 3-Seed Mean on Landslide4Sense:")
        print(f"    Micro F1: {np.mean(f1_list)*100:.2f} ± {np.std(f1_list)*100:.2f}% | Macro F1: {np.mean(macro_list)*100:.2f} ± {np.std(macro_list)*100:.2f}% | Recall: {np.mean(rec_list)*100:.2f}%\n")

    summary_out = {
        "dataset": "Landslide4Sense (Sentinel-2 10m GSD)",
        "num_patches": len(l4s_X),
        "evaluation_protocol": "Zero-shot cross-sensor transfer without fine-tuning",
        "results_by_arm": results,
        "delta_s3net_vs_unet_micro": float(results["s3net"]["micro_f1_mean"] - results["unet"]["micro_f1_mean"]),
        "delta_s3net_vs_unet_macro": float(results["s3net"]["macro_f1_mean"] - results["unet"]["macro_f1_mean"]),
        "delta_s3net_vs_resunet_micro": float(results["s3net"]["micro_f1_mean"] - results["resunet"]["micro_f1_mean"]),
        "delta_s3net_vs_resunet_macro": float(results["s3net"]["macro_f1_mean"] - results["resunet"]["macro_f1_mean"])
    }

    out_file = os.path.join(RESULTS_DIR, "landslide4sense_zero_shot_results.json")
    with open(out_file, "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"Saved Landslide4Sense zero-shot results to {out_file}")

if __name__ == "__main__":
    evaluate_zero_shot_l4s()
