import os
import time
import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import f1_score, precision_score, recall_score, jaccard_score

# Set base directories
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
STUDY_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))
DATA_DIR = os.path.join(STUDY_DIR, "experiments", "raw", "hr_gldd")
RESULTS_DIR = os.path.join(STUDY_DIR, "experiments", "derived", "results")
LOG_DIR = os.path.join(STUDY_DIR, "experiments", "logs")

os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Using compute device: {device}", flush=True)

# ---------------------------------------------------------------------------
# Load 100% Real HR-GLDD PlanetScope Satellite Arrays
# ---------------------------------------------------------------------------
print("Loading real HR-GLDD dataset arrays...", flush=True)
t0 = time.time()
trainX = np.load(os.path.join(DATA_DIR, "trainX.npy")) # (1119, 128, 128, 4)
trainY = np.load(os.path.join(DATA_DIR, "trainY.npy")) # (1119, 128, 128, 1)
valX = np.load(os.path.join(DATA_DIR, "valX.npy"))     # (284, 128, 128, 4)
valY = np.load(os.path.join(DATA_DIR, "valY.npy"))     # (284, 128, 128, 1)
testX = np.load(os.path.join(DATA_DIR, "testX.npy"))   # (355, 128, 128, 4)
testY = np.load(os.path.join(DATA_DIR, "testY.npy"))   # (355, 128, 128, 1)
print(f"Loaded all arrays in {time.time()-t0:.2f}s", flush=True)

# Convert NHWC -> NCHW and PyTorch Tensors
x_tr = torch.from_numpy(trainX).permute(0, 3, 1, 2).float()
y_tr = torch.from_numpy(trainY).permute(0, 3, 1, 2).squeeze(1).float()
x_val = torch.from_numpy(valX).permute(0, 3, 1, 2).float()
y_val = torch.from_numpy(valY).permute(0, 3, 1, 2).squeeze(1).float()
x_te = torch.from_numpy(testX).permute(0, 3, 1, 2).float()
y_te = torch.from_numpy(testY).permute(0, 3, 1, 2).squeeze(1).float()

print(f"Dataset summary:")
print(f"  Train: {x_tr.shape[0]} tiles, landslide px fraction: {torch.mean(y_tr).item()*100:.2f}%")
print(f"  Val:   {x_val.shape[0]} tiles, landslide px fraction: {torch.mean(y_val).item()*100:.2f}%")
print(f"  Test:  {x_te.shape[0]} tiles, landslide px fraction: {torch.mean(y_te).item()*100:.2f}%")

# ---------------------------------------------------------------------------
# Loss Function: Combined BCE + Focal Loss (Lin et al., ICCV 2017)
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
# Architectures
# ---------------------------------------------------------------------------
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return self.conv(x)

# 1. Baseline U-Net (Ronneberger et al., MICCAI 2015; Meena et al., ESSD 2023)
class UNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
        self.inc = DoubleConv(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = DoubleConv(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = DoubleConv(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = DoubleConv(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d3 = self.up1(x4)
        d3 = self.conv_up1(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        d2 = self.conv_up2(torch.cat([d2, x2], dim=1))
        d1 = self.up3(d2)
        d1 = self.conv_up3(torch.cat([d1, x1], dim=1))
        return self.outc(d1).squeeze(1)

# 2. ResU-Net Baseline (Zhang et al., IEEE GRSL 2018; Meena et al., ESSD 2023)
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

class ResUNet(nn.Module):
    def __init__(self, in_ch=4, out_ch=1):
        super().__init__()
        self.inc = ResBlock(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(128, 256))
        
        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = ResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = ResBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = ResBlock(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d3 = self.up1(x4)
        d3 = self.conv_up1(torch.cat([d3, x3], dim=1))
        d2 = self.up2(d3)
        d2 = self.conv_up2(torch.cat([d2, x2], dim=1))
        d1 = self.up3(d2)
        d1 = self.conv_up3(torch.cat([d1, x1], dim=1))
        return self.outc(d1).squeeze(1)

# 3. Proposed S^3-Net: Spectral-Spatial Scarp Network
class SpectralSpatialScarpAttention(nn.Module):
    def __init__(self, feat_dim, use_physical_gating=True):
        super().__init__()
        self.use_physical_gating = use_physical_gating
        in_ch = feat_dim + 2 if use_physical_gating else feat_dim
        self.gate_conv = nn.Sequential(
            nn.Conv2d(in_ch, feat_dim // 2, 3, padding=1),
            nn.BatchNorm2d(feat_dim // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(feat_dim // 2, 1, 1),
            nn.Sigmoid()
        )
    def forward(self, feat, ndvi=None, grad=None):
        if self.use_physical_gating and ndvi is not None and grad is not None:
            n_res = F.interpolate(ndvi, size=feat.shape[2:], mode='bilinear', align_corners=False)
            g_res = F.interpolate(grad, size=feat.shape[2:], mode='bilinear', align_corners=False)
            inp = torch.cat([feat, n_res, g_res], dim=1)
        else:
            inp = feat
        gate = self.gate_conv(inp)
        return feat * gate + feat

class S3Net(nn.Module):
    def __init__(self, in_ch=4, out_ch=1, use_physical_gating=True):
        super().__init__()
        self.use_physical_gating = use_physical_gating
        self.inc = ResBlock(in_ch, 32)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), ResBlock(32, 64))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), ResBlock(64, 128))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), ResBlock(128, 256))
        
        self.attn3 = SpectralSpatialScarpAttention(128, use_physical_gating=use_physical_gating)
        self.attn2 = SpectralSpatialScarpAttention(64, use_physical_gating=use_physical_gating)
        self.attn1 = SpectralSpatialScarpAttention(32, use_physical_gating=use_physical_gating)

        self.up1 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.conv_up1 = ResBlock(256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.conv_up2 = ResBlock(128, 64)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.conv_up3 = ResBlock(64, 32)
        self.outc = nn.Conv2d(32, out_ch, 1)

        # Sobel filters for NDVI scarp gradient extraction
        sobel_x = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        sobel_y = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)

    def extract_scarp_features(self, x):
        red = x[:, 0:1, :, :]
        nir = x[:, 3:4, :, :]
        ndvi = (nir - red) / (nir + red + 1e-6)
        
        gx = F.conv2d(ndvi, self.sobel_x, padding=1)
        gy = F.conv2d(ndvi, self.sobel_y, padding=1)
        grad = torch.sqrt(gx**2 + gy**2 + 1e-8)
        return ndvi, grad

    def forward(self, x):
        ndvi, grad = self.extract_scarp_features(x)
        
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        
        d3 = self.up1(x4)
        x3_gated = self.attn3(x3, ndvi, grad) if self.use_physical_gating else self.attn3(x3)
        d3 = self.conv_up1(torch.cat([d3, x3_gated], dim=1))
        
        d2 = self.up2(d3)
        x2_gated = self.attn2(x2, ndvi, grad) if self.use_physical_gating else self.attn2(x2)
        d2 = self.conv_up2(torch.cat([d2, x2_gated], dim=1))
        
        d1 = self.up3(d2)
        x1_gated = self.attn1(x1, ndvi, grad) if self.use_physical_gating else self.attn1(x1)
        d1 = self.conv_up3(torch.cat([d1, x1_gated], dim=1))
        
        return self.outc(d1).squeeze(1)

# ---------------------------------------------------------------------------
# Training and Evaluation Harness
# ---------------------------------------------------------------------------
def train_and_eval_arm(arm_name, seed, epochs=15, batch_size=32):
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    train_loader = DataLoader(TensorDataset(x_tr, y_tr), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(x_val, y_val), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(TensorDataset(x_te, y_te), batch_size=batch_size, shuffle=False)
    
    if arm_name == "unet":
        model = UNet(in_ch=4).to(device)
    elif arm_name == "resunet":
        model = ResUNet(in_ch=4).to(device)
    elif arm_name == "s3net_ablation":
        model = S3Net(in_ch=4, use_physical_gating=False).to(device)
    elif arm_name == "s3net":
        model = S3Net(in_ch=4, use_physical_gating=True).to(device)
        
    criterion = CombinedLandslideLoss(alpha=0.35, gamma=2.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    best_val_f1 = -1.0
    best_weights = None
    
    print(f"\n--- Training {arm_name} (Seed {seed}) ---", flush=True)
    t_start = time.time()
    
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
        
        # Fast Validation on GPU
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
            
    train_sec = time.time() - t_start
    print(f"Finished {arm_name} (seed {seed}) in {train_sec:.1f}s. Best Val F1: {best_val_f1:.4f}", flush=True)
    
    # Evaluate on held-out test split (355 tiles)
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
    
    # Micro Metrics (Pooled across all test pixels)
    flat_preds = test_preds.reshape(-1)
    flat_targets = test_targets.reshape(-1)
    
    tp = np.sum((flat_preds == 1) & (flat_targets == 1))
    fp = np.sum((flat_preds == 1) & (flat_targets == 0))
    fn = np.sum((flat_preds == 0) & (flat_targets == 1))
    tn = np.sum((flat_preds == 0) & (flat_targets == 0))
    
    micro_f1 = 2 * tp / (2 * tp + fp + fn + 1e-8)
    micro_iou = tp / (tp + fp + fn + 1e-8)
    micro_prec = tp / (tp + fp + 1e-8)
    micro_rec = tp / (tp + fn + 1e-8)
    fpr_background = fp / (fp + tn + 1e-8) # False positive rate on non-landslide background
    
    # Macro Metrics (Per-tile scores across 355 tiles)
    tile_f1s = []
    tile_ious = []
    for i in range(len(test_preds)):
        p_i = test_preds[i].reshape(-1)
        t_i = test_targets[i].reshape(-1)
        tp_i = np.sum((p_i == 1) & (t_i == 1))
        fp_i = np.sum((p_i == 1) & (t_i == 0))
        fn_i = np.sum((p_i == 0) & (t_i == 1))
        if tp_i + fp_i + fn_i == 0:
            # Both pred and target are empty (no landslide present, correct negative)
            ti_f1 = 1.0
            ti_iou = 1.0
        else:
            ti_f1 = 2 * tp_i / (2 * tp_i + fp_i + fn_i + 1e-8)
            ti_iou = tp_i / (tp_i + fp_i + fn_i + 1e-8)
        tile_f1s.append(float(ti_f1))
        tile_ious.append(float(ti_iou))
        
    macro_f1 = float(np.mean(tile_f1s))
    macro_iou = float(np.mean(tile_ious))
    
    print(f"Results for {arm_name} (seed {seed}):")
    print(f"  Micro F1: {micro_f1:.4f}, IoU: {micro_iou:.4f}, Prec: {micro_prec:.4f}, Rec: {micro_rec:.4f}, BG FPR: {fpr_background:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}, Macro IoU: {macro_iou:.4f}", flush=True)
    
    return {
        "arm": arm_name,
        "seed": seed,
        "duration_sec": train_sec,
        "micro_f1": float(micro_f1),
        "micro_iou": float(micro_iou),
        "micro_precision": float(micro_prec),
        "micro_recall": float(micro_rec),
        "fpr_background": float(fpr_background),
        "macro_f1": float(macro_f1),
        "macro_iou": float(macro_iou),
        "tile_f1s": tile_f1s,
        "tile_ious": tile_ious,
        "test_preds": test_preds
    }

# ---------------------------------------------------------------------------
# True Tile-Level Cluster Bootstrap across 3 Seeds (355 independent clusters)
# ---------------------------------------------------------------------------
def run_true_tile_cluster_bootstrap(tile_scores_3seeds_a, tile_scores_3seeds_b, n_boot=1000, seed=42):
    """
    tile_scores_3seeds: (355,) array where each entry is the 3-seed averaged score for that tile.
    Resamples the 355 independent tiles with replacement.
    """
    np.random.seed(seed)
    n = len(tile_scores_3seeds_a)
    assert n == 355, f"Expected 355 tiles, got {n}"
    
    diffs = []
    for _ in range(n_boot):
        idx = np.random.choice(n, size=n, replace=True)
        mean_a = np.mean(tile_scores_3seeds_a[idx])
        mean_b = np.mean(tile_scores_3seeds_b[idx])
        diffs.append(mean_a - mean_b)
        
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

def measure_latency_and_throughput(model, in_ch=4, num_warmup=100, num_runs=500):
    model.eval()
    dummy = torch.randn(1, in_ch, 128, 128, device=device)
    with torch.no_grad():
        for _ in range(num_warmup):
            _ = model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
            
        t0 = time.time()
        for _ in range(num_runs):
            _ = model(dummy)
        if device.type == "mps":
            torch.mps.synchronize()
        elif device.type == "cuda":
            torch.cuda.synchronize()
        dt = time.time() - t0
        
    lat_ms = (dt / num_runs) * 1000.0
    throughput = num_runs / dt
    return lat_ms, throughput

def main():
    arms = ["unet", "resunet", "s3net_ablation", "s3net"]
    seeds = [42, 43, 44]
    all_runs = []
    
    for arm in arms:
        for seed in seeds:
            res = train_and_eval_arm(arm, seed, epochs=15, batch_size=32)
            all_runs.append(res)
            
            # Save individual seed prediction array for 100% reproduction and verification
            pred_file = os.path.join(RESULTS_DIR, f"test_preds_{arm}_seed{seed}.npy")
            np.save(pred_file, res["test_preds"])
            print(f"Saved {pred_file} (shape: {res['test_preds'].shape})", flush=True)

    # Save best / seed-42 prediction arrays for figure generation
    for arm in arms:
        run_42 = next(r for r in all_runs if r["arm"] == arm and r["seed"] == 42)
        np.save(os.path.join(RESULTS_DIR, f"best_{arm}_test_preds.npy"), run_42["test_preds"])

    # Measure Efficiency (Parameters & Latency on current device)
    print("\n================== EFFICIENCY & INFERENCE SPEED ==================")
    efficiency_dict = {}
    model_constructors = {
        "unet": lambda: UNet(4).to(device),
        "resunet": lambda: ResUNet(4).to(device),
        "s3net_ablation": lambda: S3Net(4, use_physical_gating=False).to(device),
        "s3net": lambda: S3Net(4, use_physical_gating=True).to(device)
    }
    for arm, ctor in model_constructors.items():
        m = ctor()
        params = sum(p.numel() for p in m.parameters() if p.requires_grad)
        lat_ms, fps = measure_latency_and_throughput(m)
        efficiency_dict[arm] = {
            "parameters": params,
            "parameters_million": round(params / 1e6, 2),
            "latency_ms_per_tile": round(lat_ms, 2),
            "throughput_tiles_per_sec": round(fps, 1)
        }
        print(f"  {arm:15s} | Params: {params:,} ({params/1e6:.2f}M) | Latency: {lat_ms:.2f} ms/tile | Throughput: {fps:.1f} tiles/s")

    with open(os.path.join(RESULTS_DIR, "efficiency_metrics.json"), "w") as f:
        json.dump(efficiency_dict, f, indent=2)

    # Save seed-level results without heavy arrays
    clean_runs = [{k: v for k, v in r.items() if k not in ["tile_f1s", "tile_ious", "test_preds"]} for r in all_runs]
    with open(os.path.join(RESULTS_DIR, "seed_level_results.json"), "w") as f:
        json.dump(clean_runs, f, indent=2)

    # Aggregate summary by arm across 3 seeds
    summary_by_arm = {}
    for arm in arms:
        subset = [r for r in all_runs if r["arm"] == arm]
        summary_by_arm[arm] = {
            # Micro metrics (pooled pixels)
            "micro_f1_mean": float(np.mean([r["micro_f1"] for r in subset])),
            "micro_f1_std": float(np.std([r["micro_f1"] for r in subset])),
            "micro_iou_mean": float(np.mean([r["micro_iou"] for r in subset])),
            "micro_iou_std": float(np.std([r["micro_iou"] for r in subset])),
            "micro_precision_mean": float(np.mean([r["micro_precision"] for r in subset])),
            "micro_precision_std": float(np.std([r["micro_precision"] for r in subset])),
            "micro_recall_mean": float(np.mean([r["micro_recall"] for r in subset])),
            "micro_recall_std": float(np.std([r["micro_recall"] for r in subset])),
            "fpr_background_mean": float(np.mean([r["fpr_background"] for r in subset])),
            "fpr_background_std": float(np.std([r["fpr_background"] for r in subset])),
            # Macro metrics (per-tile average across 355 tiles)
            "macro_f1_mean": float(np.mean([r["macro_f1"] for r in subset])),
            "macro_f1_std": float(np.std([r["macro_f1"] for r in subset])),
            "macro_iou_mean": float(np.mean([r["macro_iou"] for r in subset])),
            "macro_iou_std": float(np.std([r["macro_iou"] for r in subset])),
            # Legacy keys for backward compatibility
            "f1_mean": float(np.mean([r["micro_f1"] for r in subset])),
            "f1_std": float(np.std([r["micro_f1"] for r in subset])),
            "iou_mean": float(np.mean([r["micro_iou"] for r in subset])),
            "iou_std": float(np.std([r["micro_iou"] for r in subset])),
            "precision_mean": float(np.mean([r["micro_precision"] for r in subset])),
            "precision_std": float(np.std([r["micro_precision"] for r in subset])),
            "recall_mean": float(np.mean([r["micro_recall"] for r in subset])),
            "recall_std": float(np.std([r["micro_recall"] for r in subset]))
        }

    print("\n================== SUMMARY RESULTS ACROSS 3 SEEDS ==================")
    for arm, s in summary_by_arm.items():
        print(f"ARM: {arm}")
        print(f"  Micro F1:  {s['micro_f1_mean']*100:.2f} +- {s['micro_f1_std']*100:.2f}%")
        print(f"  Micro IoU: {s['micro_iou_mean']*100:.2f} +- {s['micro_iou_std']*100:.2f}%")
        print(f"  Micro Rec: {s['micro_recall_mean']*100:.2f} +- {s['micro_recall_std']*100:.2f}%")
        print(f"  Macro F1:  {s['macro_f1_mean']*100:.2f} +- {s['macro_f1_std']*100:.2f}%")
        print(f"  BG FPR:    {s['fpr_background_mean']*100:.3f}%")

    # ---------------------------------------------------------------------------
    # TRUE 3-SEED TILE-LEVEL CLUSTER BOOTSTRAP (Resampling 355 Independent Tiles)
    # ---------------------------------------------------------------------------
    # For each tile i, compute the average score across the 3 seeds
    arm_tile_scores_3seeds = {}
    for arm in arms:
        runs = [r for r in all_runs if r["arm"] == arm]
        tile_arrays = [np.array(r["tile_f1s"]) for r in runs] # 3 arrays of shape (355,)
        mean_tile_f1s = np.mean(tile_arrays, axis=0)          # shape (355,)
        arm_tile_scores_3seeds[arm] = mean_tile_f1s

    print("\n================== TRUE 3-SEED TILE CLUSTER BOOTSTRAP (355 TILES) ==================")
    comparisons = {}
    for baseline in ["unet", "resunet", "s3net_ablation"]:
        comp_key = f"s3net_vs_{baseline}"
        boot_res = run_true_tile_cluster_bootstrap(
            arm_tile_scores_3seeds["s3net"],
            arm_tile_scores_3seeds[baseline],
            n_boot=1000,
            seed=42
        )
        comparisons[comp_key] = boot_res
        diff_pct = boot_res["mean_diff"] * 100
        low_pct = boot_res["ci95_low"] * 100
        high_pct = boot_res["ci95_high"] * 100
        print(f"Comparison: S3-Net vs {baseline:15s}")
        print(f"  Macro Diff: {diff_pct:+.2f}% (95% CI: [{low_pct:+.2f}%, {high_pct:+.2f}%], p = {boot_res['p_value']:.4f}, excludes_zero = {boot_res['excludes_zero']})")

    summary_output = {
        "summary_by_arm": summary_by_arm,
        "comparisons": comparisons,
        "efficiency": efficiency_dict
    }

    # Save to both summary files for 100% harmony
    with open(os.path.join(RESULTS_DIR, "confirmatory_summary.json"), "w") as f:
        json.dump(summary_output, f, indent=2)
    with open(os.path.join(RESULTS_DIR, "rigorous_confirmatory_summary.json"), "w") as f:
        json.dump(summary_output, f, indent=2)

    print("\nAll experiments, metrics, seed arrays, and bootstrap summaries saved successfully!")

if __name__ == "__main__":
    main()
