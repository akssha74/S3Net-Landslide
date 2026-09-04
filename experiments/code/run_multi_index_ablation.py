#!/usr/bin/env python3
"""Multi-Index Biophysical Ablation Study on HR-GLDD benchmark across 3 seeds.

Evaluates whether the performance gain is specific to the biophysics of photosynthetic
vegetation canopy displacement (NDVI scarp gradients) or whether alternative optical
indices (SAVI, NDWI) or raw channel gradients provide equivalent gating.

Architectural variants:
1. S3-Net (NDVI): Proposed biophysical scarp gradient (NIR - Red) / (NIR + Red)
2. S3-Net (SAVI): Soil-Adjusted Vegetation Index ((NIR - Red) / (NIR + Red + 0.5)) * 1.5
3. S3-Net (NDWI): Normalized Difference Water Index (Green - NIR) / (Green + NIR)
4. S3-Net (Raw Grad): Raw spectral spatial gradients |∇Red| and |∇NIR| without non-linear ratio
5. S3-Net (No Gating): Capacity-matched baseline without physical skip gating
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

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(STUDY_DIR, "experiments", "raw", "hr_gldd")
RESULTS_DIR = os.path.join(STUDY_DIR, "experiments", "derived", "results")

os.makedirs(RESULTS_DIR, exist_ok=True)
device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using compute device: {device}", flush=True)

# ---------------------------------------------------------------------------
# Load 100% Real HR-GLDD PlanetScope Arrays
# ---------------------------------------------------------------------------
print("Loading real HR-GLDD dataset arrays...", flush=True)
trainX = np.load(os.path.join(DATA_DIR, "trainX.npy")) # (1119, 128, 128, 4)
trainY = np.load(os.path.join(DATA_DIR, "trainY.npy")) # (1119, 128, 128, 1)
valX = np.load(os.path.join(DATA_DIR, "valX.npy"))     # (284, 128, 128, 4)
valY = np.load(os.path.join(DATA_DIR, "valY.npy"))     # (284, 128, 128, 1)
testX = np.load(os.path.join(DATA_DIR, "testX.npy"))   # (355, 128, 128, 4)
testY = np.load(os.path.join(DATA_DIR, "testY.npy"))   # (355, 128, 128, 1)

x_tr = torch.from_numpy(trainX).permute(0, 3, 1, 2).float()
y_tr = torch.from_numpy(trainY).permute(0, 3, 1, 2).squeeze(1).float()
x_val = torch.from_numpy(valX).permute(0, 3, 1, 2).float()
y_val = torch.from_numpy(valY).permute(0, 3, 1, 2).squeeze(1).float()
x_te = torch.from_numpy(testX).permute(0, 3, 1, 2).float()
y_te = torch.from_numpy(testY).permute(0, 3, 1, 2).squeeze(1).float()

# ---------------------------------------------------------------------------
# Composite Loss
# ---------------------------------------------------------------------------
class CombinedLandslideLoss(nn.Module):
    def __init__(self, alpha=0.35, gamma=2.0, bce_weight=0.5):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce_weight = bce_weight

    def forward(self, logits, targets):
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
        probs = torch.sigmoid(logits)
        pt = targets * probs + (1 - targets) * (1 - probs)
        focal_weight = (1 - pt) ** self.gamma
        alpha_t = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        focal = alpha_t * focal_weight * F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        focal_loss = focal.mean()
        return self.bce_weight * bce + (1.0 - self.bce_weight) * focal_loss

# ---------------------------------------------------------------------------
# Residual Blocks & Flexible Attention
# ---------------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )
        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, bias=False),
                nn.BatchNorm2d(out_ch)
            )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))

class MultiIndexScarpAttention(nn.Module):
    def __init__(self, feat_dim, index_channels=2):
        super().__init__()
        self.index_channels = index_channels
        in_ch = feat_dim + index_channels if index_channels > 0 else feat_dim
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, feat, physical_tensors=None):
        if self.index_channels > 0 and physical_tensors is not None:
            resampled = [F.interpolate(t, size=feat.shape[2:], mode='bilinear', align_corners=False) for t in physical_tensors]
            inp = torch.cat([feat] + resampled, dim=1)
        else:
            inp = feat
        gate = self.gate_conv(inp)
        return feat * gate + feat

class MultiIndexS3Net(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, gating_mode="ndvi"):
        super().__init__()
        self.gating_mode = gating_mode
        self.inc = ResBlock(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(128, 256))

        idx_ch = 0 if gating_mode == "none" else 2
        self.attn3 = MultiIndexScarpAttention(128, index_channels=idx_ch)
        self.attn2 = MultiIndexScarpAttention(64, index_channels=idx_ch)
        self.attn1 = MultiIndexScarpAttention(32, index_channels=idx_ch)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = ResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = ResBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = ResBlock(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)

        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def extract_index_and_gradient(self, x):
        blue = x[:, 0:1, :, :]
        green = x[:, 1:2, :, :]
        red = x[:, 2:3, :, :]
        nir = x[:, 3:4, :, :]

        if self.gating_mode == "ndvi":
            idx = (nir - red) / (nir + red + 1e-6)
        elif self.gating_mode == "savi":
            L = 0.5
            idx = ((nir - red) / (nir + red + L)) * (1.0 + L)
        elif self.gating_mode == "ndwi":
            idx = (green - nir) / (green + nir + 1e-6)
        elif self.gating_mode == "raw_grad":
            gx_r = F.conv2d(red, self.sobel_x, padding=1)
            gy_r = F.conv2d(red, self.sobel_y, padding=1)
            grad_r = torch.sqrt(gx_r**2 + gy_r**2 + 1e-8)
            gx_n = F.conv2d(nir, self.sobel_x, padding=1)
            gy_n = F.conv2d(nir, self.sobel_y, padding=1)
            grad_n = torch.sqrt(gx_n**2 + gy_n**2 + 1e-8)
            return [grad_r, grad_n]
        else:
            return None

        gx = F.conv2d(idx, self.sobel_x, padding=1)
        gy = F.conv2d(idx, self.sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return [idx, grad]

    def forward(self, x):
        phys = self.extract_index_and_gradient(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d3 = self.up1(x4)
        x3_g = self.attn3(x3, phys)
        d3 = self.conv_up1(torch.cat([d3, x3_g], dim=1))
        
        d2 = self.up2(d3)
        x2_g = self.attn2(x2, phys)
        d2 = self.conv_up2(torch.cat([d2, x2_g], dim=1))
        
        d1 = self.up3(d2)
        x1_g = self.attn1(x1, phys)
        d1 = self.conv_up3(torch.cat([d1, x1_g], dim=1))
        
        return self.outc(d1).squeeze(1)

# ---------------------------------------------------------------------------
# Training Harness
# ---------------------------------------------------------------------------
def train_and_eval_mode(mode_name, seed, epochs=15, batch_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    
    model = MultiIndexS3Net(in_ch=4, gating_mode=mode_name).to(device)
    criterion = CombinedLandslideLoss(alpha=0.35, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_f1 = -1.0
    best_weights = None
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
        
        # Validation
        model.eval()
        tp, fp, fn = 0, 0, 0
        with torch.no_grad():
            for bx, by in val_loader:
                bx, by = bx.to(device), by.to(device)
                pred_mask = (torch.sigmoid(model(bx)) > 0.5)
                target_mask = (by > 0.5)
                tp += torch.sum(pred_mask & target_mask).item()
                fp += torch.sum(pred_mask & (~target_mask)).item()
                fn += torch.sum((~pred_mask) & target_mask).item()
        val_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-8)
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
    # Test evaluation
    model.load_state_dict({k: v.to(device) for k, v in best_weights.items()})
    model.eval()
    
    t_preds_list = []
    with torch.no_grad():
        for bx, by in test_loader:
            bx = bx.to(device)
            p = (torch.sigmoid(model(bx)) > 0.5).cpu().numpy().astype(np.uint8)
            t_preds_list.append(p)
            
    test_preds = np.concatenate(t_preds_list, axis=0) # (355, 128, 128)
    test_targets = testY.squeeze(-1).astype(np.uint8)  # (355, 128, 128)
    
    # Micro
    pf = test_preds.reshape(-1)
    yf = test_targets.reshape(-1)
    tp = np.sum((pf == 1) & (yf == 1))
    fp = np.sum((pf == 1) & (yf == 0))
    fn = np.sum((pf == 0) & (yf == 1))
    micro_f1 = float(2 * tp / (2 * tp + fp + fn + 1e-8))
    micro_iou = float(tp / (tp + fp + fn + 1e-8))
    micro_prec = float(tp / (tp + fp + 1e-8))
    micro_rec = float(tp / (tp + fn + 1e-8))
    
    # Macro
    tile_f1s = []
    for i in range(len(test_preds)):
        p_i = test_preds[i].reshape(-1)
        t_i = test_targets[i].reshape(-1)
        tp_i = np.sum((p_i == 1) & (t_i == 1))
        fp_i = np.sum((p_i == 1) & (t_i == 0))
        fn_i = np.sum((p_i == 0) & (t_i == 1))
        if tp_i + fp_i + fn_i == 0:
            ti_f1 = 1.0
        else:
            ti_f1 = 2 * tp_i / (2 * tp_i + fp_i + fn_i + 1e-8)
        tile_f1s.append(ti_f1)
    macro_f1 = float(np.mean(tile_f1s))
    
    dt = time.time() - t0
    print(f"Mode: {mode_name:10s} (Seed {seed}) in {dt:.1f}s | Micro F1: {micro_f1*100:.2f}%, Macro F1: {macro_f1*100:.2f}%, IoU: {micro_iou*100:.2f}%, Rec: {micro_rec*100:.2f}%")
    return {
        "mode": mode_name,
        "seed": seed,
        "micro_f1": micro_f1,
        "micro_iou": micro_iou,
        "micro_precision": micro_prec,
        "micro_recall": micro_rec,
        "macro_f1": macro_f1,
        "tile_f1s": tile_f1s
    }

def main():
    modes = ["ndvi", "savi", "ndwi", "raw_grad"]
    seeds = [42, 43, 44]
    results = {}
    
    print("\n================== RUNNING MULTI-INDEX BIOPHYSICAL ABLATION ==================")
    for mode in modes:
        mode_runs = []
        for s in seeds:
            res = train_and_eval_mode(mode, s)
            mode_runs.append(res)
            
        micros = [r["micro_f1"] for r in mode_runs]
        macros = [r["macro_f1"] for r in mode_runs]
        ious = [r["micro_iou"] for r in mode_runs]
        precs = [r["micro_precision"] for r in mode_runs]
        recs = [r["micro_recall"] for r in mode_runs]
        
        results[mode] = {
            "micro_f1_mean": float(np.mean(micros)),
            "micro_f1_std": float(np.std(micros)),
            "macro_f1_mean": float(np.mean(macros)),
            "macro_f1_std": float(np.std(macros)),
            "iou_mean": float(np.mean(ious)),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "all_runs": [{k: v for k, v in r.items() if k != "tile_f1s"} for r in mode_runs]
        }
        print(f"\n---> {mode.upper()} 3-Seed Mean: Micro F1 = {np.mean(micros)*100:.2f} ± {np.std(micros)*100:.2f}% | Macro F1 = {np.mean(macros)*100:.2f} ± {np.std(macros)*100:.2f}% | Recall = {np.mean(recs)*100:.2f}%\n")
        
    out_file = os.path.join(RESULTS_DIR, "multi_index_biophysical_ablation.json")
    with open(out_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved multi-index ablation results to {out_file}")

if __name__ == "__main__":
    main()
