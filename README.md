# Controlled PlanetScope landslide-segmentation evaluation

This is the corrected implementation and result release for:

**Spectral Edge Priors versus Boundary Weighting for PlanetScope Landslide Segmentation**

The study does **not** claim an NDVI-specific or biophysical-loss mechanism.
Across three seeds, raw-edge, NDVI, and zero-control residual-attention models are
closely grouped. Ordinary boundary weighting improves boundary F1 across all three
controls; NDVI-modulated weighting does not improve F1 over the plain control.
The released HR-GLDD arrays are interpreted as RGBN from the pinned official
notebook that loads the arrays and renders channels 0--2 directly as RGB.

A post-execution-documented event-held-out CAS check separately assesses whether
the generic boundary-weighting result replicates in RGB imagery. The corrected
record reports the executed 80/160 source-tile limits and carries no prospective
or preregistration credit. CAS imagery is not redistributed.

## Reproduce

1. Download HR-GLDD arrays from https://doi.org/10.5281/zenodo.7189381.
2. Place `trainX.npy`, `trainY.npy`, `valX.npy`, `valY.npy`, `testX.npy`, and
   `testY.npy` under `experiments/raw/hr_gldd/`.
3. Install `requirements.txt`.
4. Run:

```bash
python experiments/code/test_data_semantics.py
python experiments/code/test_revised_models.py
python experiments/code/test_equal_mass_dataset.py
python experiments/code/test_reviewer_remediation_metrics.py
python experiments/code/run_reviewer_remediation.py
python experiments/code/make_reviewer_remediation_artifacts.py
python experiments/code/build_frozen_output_inventory.py
```

To reproduce the external check, download the eight CAS archives named in
`run_cas_boundary_confirmation.py` from
https://doi.org/10.5281/zenodo.10294997, place them under
`experiments/raw/external/cas/`, then run:

```bash
python experiments/code/run_cas_boundary_confirmation.py
python reviews/verify_cas_boundary_confirmation.py
```

The released HR-GLDD arrays do not include event IDs or coordinates. The study
therefore reports descriptive multi-seed results and makes no tile-independence,
event-held-out, or geographic-transfer inference.

`experiments/derived/results/frozen-output-inventory.json` records SHA-256
identities for all 138 retained per-seed probability arrays and checkpoints.
The binaries are not redistributed; the run logs and deterministic generators
needed to regenerate them are included.
