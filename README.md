# Controlled PlanetScope landslide-segmentation evaluation

This is the corrected implementation and result release for:

**Spectral Edge Priors versus Boundary Weighting for PlanetScope Landslide Segmentation**

The study does **not** claim an NDVI-specific or biophysical-loss mechanism.
Across three seeds, raw-edge, NDVI, and zero-control residual-attention models are
closely grouped. Ordinary boundary weighting improves boundary F1 across all three
controls; NDVI-modulated weighting does not improve F1 over the plain control.

## Reproduce

1. Download HR-GLDD arrays from https://doi.org/10.5281/zenodo.7189381.
2. Place `trainX.npy`, `trainY.npy`, `valX.npy`, `valY.npy`, `testX.npy`, and
   `testY.npy` under `experiments/raw/hr_gldd/`.
3. Install `requirements.txt`.
4. Run:

```bash
python experiments/code/test_data_semantics.py
python experiments/code/test_revised_models.py
python experiments/code/test_reviewer_remediation_metrics.py
python experiments/code/run_reviewer_remediation.py
python experiments/code/make_reviewer_remediation_artifacts.py
```

The released HR-GLDD arrays do not include event IDs or coordinates. The study
therefore reports descriptive multi-seed results and makes no tile-independence,
event-held-out, or geographic-transfer inference.
