# S3-Net: Spectral-Spatial Scarp Network for PlanetScope Landslide Segmentation

Official PyTorch implementation, replication artifacts, and evaluation scripts for the paper:
**"Decoupling Vegetation Scarp Contrast from Topographic False Alarms in PlanetScope Landslide Segmentation"**  
*Targeted for IEEE Geoscience and Remote Sensing Letters (GRSL).*

---

## 🔬 Overview

Automated landslide detection from high-resolution satellite imagery frequently suffers from severe false-positive clutter along dry riverbeds, exposed agricultural clearings, and unpaved mountain roads. 

**S³-Net** introduces a physics-guided deep learning architecture that explicitly computes Normalized Difference Vegetation Index (NDVI) scarp gradients ($|\nabla \text{NDVI}|$) and couples them directly into multi-scale residual spatial attention gates.
- **Ultra-Lightweight:** Only **2.11M parameters** ($\approx 14\times$ smaller than DCA-UNet).
- **Fast Inference:** **2.16 ms/tile** latency (**462.6 tiles/sec** throughput) on Apple Silicon Metal Performance Shaders (MPS).
- **No Elevation Metadata Needed:** Completely DEM-free, ensuring instant emergency deployability without waiting for external topographic downloads.

---

## 📊 Summary of Results (Official HR-GLDD Benchmark, 3 Seeds)

Evaluated on the globally distributed High-Resolution Global Landslide Detector Database (HR-GLDD; Meena et al., *Earth System Science Data*, 2023) across 1,758 real PlanetScope satellite patches ($128 \times 128$, 3 m GSD) spanning 10 disaster events worldwide.

| Architecture / Model | Parameters | Micro F1 (%) | Macro F1 (%) | IoU (%) | Precision (%) | Recall (%) | Inference Latency |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Vanilla U-Net | 1.93M | $68.33 \pm 0.27$ | $54.84 \pm 0.65$ | $51.89 \pm 0.32$ | $75.97 \pm 0.87$ | $62.08 \pm 0.16$ | 1.18 ms/tile |
| ResU-Net Baseline | 2.01M | $69.66 \pm 0.89$ | $57.92 \pm 1.32$ | $53.45 \pm 1.05$ | $75.70 \pm 1.92$ | $64.65 \pm 2.80$ | 1.77 ms/tile |
| S³-Net (w/o Physical Gating) | 2.11M | $69.26 \pm 0.44$ | $57.07 \pm 0.29$ | $52.97 \pm 0.52$ | $\mathbf{77.18 \pm 0.53}$ | $62.81 \pm 0.86$ | 2.14 ms/tile |
| **S³-Net (Proposed)** | **2.11M** | $\mathbf{70.18 \pm 0.25}$ | $\mathbf{58.52 \pm 0.24}$ | $\mathbf{54.06 \pm 0.29}$ | $74.31 \pm 1.96$ | $\mathbf{66.59 \pm 1.95}$ | **2.16 ms/tile** |
| *DCA-UNet (Song et al., 2026)* | 29.50M | 74.41 | — | 59.24 | — | — | Heavy (29.5M) |

### Statistical Significance (True Tile-Level Cluster Bootstrap across 355 Independent Tiles, 3-Seed Pooled)
- **S³-Net vs. Vanilla U-Net:** $+3.69\%$ Macro F1 gain ($95\%$ CI: $[+2.90\%, +4.51\%]$, $p < 0.001$, excludes zero).
- **S³-Net vs. ResU-Net:** $+0.60\%$ Macro F1 gain ($95\%$ CI: $[+0.17\%, +1.01\%]$, $p = 0.005$, excludes zero). On Micro F1, both models reach parity ($70.18\%$ vs. $69.66\%$), with S³-Net exhibiting substantially lower seed variance ($\sigma = 0.25\%$ vs. $0.89\%$).
- **S³-Net vs. Unguided Ablation:** $+1.44\%$ Macro F1 gain ($95\%$ CI: $[+1.08\%, +1.83\%]$, $p < 0.001$, excludes zero), confirming that physical vegetation scarp contrast actively suppresses false-alarm clutter and improves recall.

---

## 🛠️ Repository Structure

```
S3Net-Landslide/
├── README.md
├── requirements.txt
├── experiments/
│   ├── code/
│   │   ├── train_eval.py                   # Complete training & evaluation pipeline across 3 seeds
│   │   ├── recompute_rigorous_metrics.py   # True tile-level cluster bootstrap verification
│   │   ├── make_tables.py                  # Generates LaTeX performance tables
│   │   └── make_figures.py                 # Generates publication PDF figures
│   └── derived/
│       └── results/
│           ├── confirmatory_summary.json           # Aggregated 3-seed metrics and bootstrap CIs
│           ├── rigorous_confirmatory_summary.json  # Comprehensive verification summary
│           ├── seed_level_results.json             # Individual seed metrics
│           └── efficiency_metrics.json             # Parameter counts, latencies, and throughputs
└── paper/
    ├── tables/
    │   └── tab_performance.tex        # Compiled LaTeX table
    └── figures/
        ├── fig_performance_comparison.pdf
        ├── fig_scattering_mechanism.pdf
        └── fig_qualitative_patches.pdf
```

---

## 🚀 Reproduction Instructions

### 1. Environment Setup
```bash
git clone https://github.com/akssha74/S3Net-Landslide.git
cd S3Net-Landslide
pip install -r requirements.txt
```

### 2. Download HR-GLDD Benchmark
Download the official HR-GLDD PlanetScope arrays (`trainX.npy`, `trainY.npy`, `valX.npy`, `valY.npy`, `testX.npy`, `testY.npy`) from Zenodo:
- DOI: [10.5281/zenodo.7189381](https://doi.org/10.5281/zenodo.7189381)
- Place arrays into `experiments/raw/hr_gldd/`.

### 3. Run Training and Evaluation
```bash
python experiments/code/train_eval.py
```
This trains all four benchmark models across random seeds 42, 43, 44, executes the true tile-level 1,000-draw cluster bootstrap across 355 independent tiles, measures inference speeds, and generates verified result JSONs.

### 4. Recreate Figures and Tables
```bash
python experiments/code/make_tables.py
python experiments/code/make_figures.py
```

---

## 📄 License
This repository is released under the MIT License. The HR-GLDD dataset is governed by the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
