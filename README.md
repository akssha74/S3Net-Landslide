# Controlled PlanetScope landslide-segmentation evaluation

This is the corrected implementation and result release for:

**Semantic Identifiability under Band-Order Ambiguity in PlanetScope Landslide Segmentation**

The released HR-GLDD arrays do not authoritatively identify columns 0 and 2 as
Red or Blue. The complete nine-configuration, three-seed analysis is therefore
run under both RGBN and BGRN candidates. Index-input specificity fails under
both orders; plain boundary weighting improves every HR-GLDD comparison, while
the boundary effect of index modulation changes sign across orders. A generated
semantic-identifiability certificate propagates both admissible schemas into
four signed contrasts and marks each direction positive, negative, or
unidentified.

The Git tag is a minimal runnable release. The complete scientific tree is the
source-snapshot tarball listed under the release's Assets; the companion Git
bundle exposes its two-stage source history.

CAS is a post-hoc designated-region sensitivity analysis and carries no
confirmation credit. The LRD experiment was internally precommitted before
validation/protected optical access, but lacks an external timestamp, contains
trigger-family leakage, and has unstable empty-crop boundary scoring. It is
classified as failed/indeterminate sensitivity evidence.

The publicly locked and authorized Sen12Landslides stress test also does not
confirm transfer: only one of five registered conditions passes and all seven
arm means are below 3.2% F1. A registered conservative transformed-envelope
check fails for two abutting Kyrgyzstan tiles, although their native rectangles
have zero positive-area overlap. A public post-access correction handles one
SCL-255 array only in descriptive cloud denominators; it changed no model input,
metric, or condition. The expanded verifier reproduces 420 primary and 294
high-confidence metric cells and all ten SCL summaries.

## Reproduce

1. Download HR-GLDD arrays from https://doi.org/10.5281/zenodo.7189381.
2. Place `trainX.npy`, `trainY.npy`, `valX.npy`, `valY.npy`, `testX.npy`, and
   `testY.npy` under `experiments/raw/hr_gldd/`.
3. Install `requirements.txt`; Pandoc is additionally required only to
   regenerate the declared manuscript word count.
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
python experiments/code/make_semantic_identifiability_certificate.py
python experiments/code/test_semantic_identifiability_certificate.py
python experiments/code/count_manuscript_words.py
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

For the LRD stress test, download Landslide Reference Data v3 from
https://doi.org/10.5281/zenodo.17007637. The original local protocol, current
status, fit decisions, authorization, endpoint/family sensitivity artifacts,
logs, and standalone numeric verifier are included.

For the Sen12 result, the release includes the pinned membership and extraction
manifests, public authorization/correction receipts, fit decision, summary,
diagnostic logs, and standalone verifier. Raw Sen12 files, checkpoints, and
probability arrays are not redistributed; obtain the public data from
https://huggingface.co/datasets/paulhoehn/Sen12Landslides.

The released HR-GLDD arrays do not include event IDs or coordinates. The study
therefore reports descriptive multi-seed results and makes no tile-independence,
event-held-out, or geographic-transfer inference.

`experiments/derived/results/frozen-output-inventory.json` records SHA-256
identities for all 294 retained per-seed probability arrays and checkpoints.
The binaries are not redistributed; the run logs and deterministic generators
needed to regenerate them are included.
