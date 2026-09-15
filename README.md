# Controlled PlanetScope landslide-segmentation evaluation

This is the corrected implementation and result release for:

**Band-Order Ambiguity in Spectral Mechanism Attribution for PlanetScope Landslide Segmentation**

The released HR-GLDD arrays do not authoritatively identify columns 0 and 2 as
Red or Blue. The complete nine-configuration, three-seed analysis is therefore
run under both RGBN and BGRN candidates. Index-input specificity fails under
both orders; plain boundary weighting improves every HR-GLDD comparison, while
the boundary effect of index modulation changes sign across orders.

CAS is a post-hoc designated-region sensitivity analysis and carries no
confirmation credit. A separate LRD experiment was preregistered before
validation/protected optical access; all four conditions failed across six
protected EIDs.

## Reproduce

1. Download HR-GLDD arrays from https://doi.org/10.5281/zenodo.7189381.
2. Place `trainX.npy`, `trainY.npy`, `valX.npy`, `valY.npy`, `testX.npy`, and
   `testY.npy` under `experiments/raw/hr_gldd/`.
3. Install `requirements.txt`.
4. Run:

```bash
HRGLDD_ARRAY_ORDER=RGBN python experiments/code/test_data_semantics.py
HRGLDD_ARRAY_ORDER=BGRN python experiments/code/test_data_semantics.py
python experiments/code/test_revised_models.py
python experiments/code/test_equal_mass_dataset.py
python experiments/code/test_reviewer_remediation_metrics.py
HRGLDD_ARRAY_ORDER=RGBN REMEDIATION_VARIANT=rgbn python experiments/code/run_reviewer_remediation.py
HRGLDD_ARRAY_ORDER=BGRN REMEDIATION_VARIANT=bgrn python experiments/code/run_reviewer_remediation.py
python experiments/code/make_band_order_sensitivity.py
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

For the prospective confirmation, download Landslide Reference Data v3 from
https://doi.org/10.5281/zenodo.17007637. Follow the immutable sequence in
`research/lrd-boundary-confirmation-preregistration.md`; the released
`fit_decisions.json`, protected authorization, logs, and verifier preserve the
executed outcome.

The released HR-GLDD arrays do not include event IDs or coordinates. The study
therefore reports descriptive multi-seed results and makes no tile-independence,
event-held-out, or geographic-transfer inference.

`experiments/derived/results/frozen-output-inventory.json` records SHA-256
identities for all 294 retained per-seed probability arrays and checkpoints.
The binaries are not redistributed; the run logs and deterministic generators
needed to regenerate them are included.
