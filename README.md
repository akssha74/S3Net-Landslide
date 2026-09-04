# S3-Net: Decoupling Vegetation Scarp Contrast in PlanetScope Landslide Segmentation

Official PyTorch and ONNX implementation, replication artifacts, and evaluation scripts for the paper:
**"Decoupling Vegetation Scarp Contrast in PlanetScope Landslide Segmentation"**  
*Targeted for IEEE Geoscience and Remote Sensing Letters (GRSL).*

---

## 🔬 Overview

Automated landslide detection from high-resolution satellite imagery frequently suffers from spectral confusion where unpaved roads, dry riverbeds, and agricultural clearings exhibit low vegetative reflection similar to fresh landslide scars.

**S³-Net** introduces a physics-guided deep learning architecture that explicitly computes Normalized Difference Vegetation Index (NDVI) scarp gradients ($|\nabla \text{NDVI}|$) and couples them directly into multi-scale residual spatial attention gates.
- **Ultra-Lightweight:** Only **2.11M parameters** ($\approx 14\times$ smaller than DCA-UNet).
- **Edge Deployable:** **8.06 MB** serialized ONNX model with **2.16 ms/tile** latency (**462.6 tiles/sec** throughput) on Apple Silicon MPS and **86.26 ms/tile** on CPU with $<45\text{ MB}$ peak working memory.
- **No Elevation Metadata Needed:** Completely DEM-free, ensuring instant emergency deployability without waiting for external topographic downloads.
- **Official Model Weights & Predictions:** Downloadable via [GitHub Release v1.0.0](https://github.com/akssha74/S3Net-Landslide/releases/tag/v1.0.0).

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
- **S³-Net vs. Unguided Ablation:** $+1.44\%$ Macro F1 gain ($95\%$ CI: $[+1.08\%, +1.83\%]$, $p < 0.001$, excludes zero), confirming that physical vegetation scarp contrast actively enhances narrow boundary recall ($66.59\%$ vs. $62.81\%$).

### Multi-Index Biophysical Ablation Study (3 Seeds)
| Gating Formulation | Micro F1 (%) | Macro F1 (%) | IoU (%) | Precision (%) | Recall (%) | Active Physical Mechanism |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **S³-Net (NDVI)** | $\mathbf{69.95 \pm 0.30}$ | $\mathbf{57.91 \pm 1.14}$ | $\mathbf{53.79}$ | $74.84$ | $\mathbf{65.73}$ | Chlorophyll-to-mesophyll canopy displacement |
| S³-Net (NDWI) | $69.96 \pm 0.10$ | $57.93 \pm 0.53$ | $53.80$ | $75.10$ | $65.51$ | Surface water & canopy moisture contrast |
| S³-Net (Raw Gradients) | $69.75 \pm 0.22$ | $57.67 \pm 0.59$ | $53.55$ | $75.42$ | $64.94$ | Multi-band spatial edge gradients (no ratio) |
| S³-Net (SAVI) | $69.64 \pm 0.38$ | $57.94 \pm 0.87$ | $53.43$ | $75.35$ | $64.87$ | Soil-adjusted vegetation index |
| S³-Net (Unguided Ablation)| $69.26 \pm 0.44$ | $57.07 \pm 0.29$ | $52.97$ | $\mathbf{77.18}$ | $62.81$ | Spatial self-attention without physical tensors |

### 10-Fold Leave-One-Spectral-Cluster-Out (LOSCO) Out-of-Distribution Generalization
Trained on 9 environmental clusters and evaluated zero-shot on the held-out 10th cluster across all 1,758 patches:
- **Vanilla U-Net (Zero-Shot LOSCO):** $49.92\% \pm 19.21\%$ Macro F1 ($58.00\% \pm 21.53\%$ Micro F1)
- **ResU-Net Baseline (Zero-Shot LOSCO):** $51.39\% \pm 18.58\%$ Macro F1 ($59.80\% \pm 20.71\%$ Micro F1)
- **S³-Net Proposed (Zero-Shot LOSCO):** $\mathbf{53.11\% \pm 17.76\%}$ Macro F1 ($\mathbf{60.93\% \pm 20.12\%}$ Micro F1)  
$\rightarrow$ S³-Net maintains $+3.19\%$ Macro F1 ($+2.93\%$ Micro F1) out-of-distribution advantage over vanilla U-Net and $+1.72\%$ Macro F1 over ResU-Net under zero-shot regional domain transfer.

---

## 🛠️ Repository Structure

```
S3Net-Landslide/
├── README.md
├── requirements.txt
├── experiments/
│   ├── code/
│   │   ├── train_eval.py                   # Complete training & evaluation pipeline across 3 seeds
│   │   ├── run_multi_index_ablation.py     # Biophysical index ablation (NDVI, NDWI, SAVI, Raw Grad)
│   │   ├── profile_edge_deployment.py      # ONNX serialization, latency, and memory profiling
│   │   ├── evaluate_landslide4sense.py     # Cross-sensor benchmark evaluation on Landslide4Sense
│   │   ├── run_loro_and_false_alarms.py    # 10-fold LOSCO benchmark and false-alarm quantification
│   │   ├── recompute_rigorous_metrics.py   # True tile-level cluster bootstrap verification
│   │   ├── make_tables.py                  # Generates LaTeX performance tables
│   │   └── make_figures.py                 # Generates publication PDF figures
│   └── derived/
│       └── results/
│           ├── S3Net_PlanetScope.onnx              # Serialized ONNX edge payload model (8.06 MB)
│           ├── edge_deployment_profile.json        # Edge runtime profiling metrics
│           ├── multi_index_biophysical_ablation.json # Multi-index ablation metrics
│           ├── landslide4sense_zero_shot_results.json # Cross-sensor benchmark metrics
│           ├── confirmatory_summary.json           # Aggregated 3-seed metrics and bootstrap CIs
│           ├── rigorous_confirmatory_summary.json  # Comprehensive verification summary
│           ├── seed_level_results.json             # Individual seed metrics
│           ├── efficiency_metrics.json             # Parameter counts, latencies, and throughputs
│           ├── false_alarm_analysis.json           # Background FPR and false-alarm area in km²
│           ├── ten_region_disaggregation.json      # 10-region disaggregated performance
│           └── loro_generalization_results.json    # 10-fold zero-shot LOSCO cross-regime metrics
└── paper/
    ├── main.tex
    ├── references.bib
    ├── tables/
    │   └── tab_performance.tex        # Compiled LaTeX table
    ├── figures/
    │   ├── fig_performance_comparison.pdf
    │   ├── fig_scattering_mechanism.pdf
    │   └── fig_qualitative_patches.pdf
    └── sections/
        ├── abstract.tex
        ├── introduction.tex
        ├── method.tex
        ├── experimental-setup.tex
        ├── results.tex
        ├── limitations.tex
        └── conclusion.tex
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

### 3. Run Experiments
```bash
# Main benchmark training and evaluation across 3 seeds
python experiments/code/train_eval.py

# Multi-index biophysical ablation
python experiments/code/run_multi_index_ablation.py

# Edge payload ONNX profiling
python experiments/code/profile_edge_deployment.py

# 10-fold cross-regime generalizability benchmark
python experiments/code/run_loro_and_false_alarms.py
```

### 4. Direct Verification against Stored Predictions
Download `s3net_test_predictions_3seeds.tar.gz` from [Release v1.0.0](https://github.com/akssha74/S3Net-Landslide/releases/tag/v1.0.0), extract into `experiments/derived/results/`, and run:
```bash
python experiments/code/recompute_rigorous_metrics.py
```

---

## 📄 License
This repository is released under the MIT License. The HR-GLDD dataset is governed by the Creative Commons Attribution 4.0 International (CC BY 4.0) license.
