# Sen12Landslides Sentinel-2 confirmation protocol

## Status and purpose

Version 6 is a post-access semantic execution correction to the publicly locked
and authorized version-5 design. It tests whether the HR-GLDD mechanism
conclusions recur with authoritative named bands and inventory-level evaluation.
It does not test HR-GLDD's physical band order, and it does not claim a new
pristine preregistration.

Version 1 was publicly timestamped at commit
`18d2da8c16b6b45ebee506e09b9b4ea027bd4db4` before protected access, but its
indexer did not enforce membership in the official S12LS-LD task. Version 1 was
therefore never used to fit or evaluate a model. Version 2 added the pinned task
membership and stricter artifact binding and was publicly sealed at commit
`0d9fee766735ff0d062f6ef3c0be077efb4276b0`. Its index attempt failed closed
before writing a member index because the code inferred inventory from filename
prefix while the 28 `usa` task rows use the prefix `usa_puertorico`. No
development extraction, fit, authorization, or protected access occurred under
version 2. Version 3 takes inventory exclusively from the pinned task-membership
row and was publicly sealed at commit
`a37fd741e3268cbbc3f93fdc1c5f93d61fbe3941`. It indexed all 4,988 task members,
selected 2,466, and extracted 577 development files. Its predeclared
cross-inventory fit gate then failed: all 21 `china`-held-out F1 values were
0.0216--0.0520 against a 0.25 floor, and threshold exploration could not raise
any above 0.0791. Version 3 was stopped without authorization or protected
access. Development-only mixed-inventory exploration subsequently found
individual F1 0.2123--0.3290 and arm-level seed means 0.2662--0.3166. Version 4
used that disclosed calibration design and was publicly sealed at commit
`f2f12a7989027ff9daca182be9b125546eb671df`. Its final fit passed both
development floors. Before authorization, independent re-inference found
maximum absolute probability differences of `1.19e-7`--`1.79e-7` in 15 of 21
runs; exact bit-level checking later showed up to 24 local float32 ULPs.
Classifications and F1 were identical, but the frozen absolute tolerance of
`1e-7` rejected them.
Version 4 was stopped without authorization or protected access. Version 5
replaces only that numerical reproducibility criterion with NumPy `allclose`
using absolute tolerance `2*float32_eps`, relative tolerance `1e-6`, and exact
threshold-classification identity, uses pristine v5 output paths, and supersedes
versions 1--4. Version 5 was publicly sealed at commit
`a4fc49723d2305a9acdd580ad22b04cff1a0acf3`, authorized at public commit
`3f390776a37660ccab45d5f16ef6f21585313be2`, and used to extract exactly 1,889
protected files. Its first evaluation stopped during dataset loading, before
model inference, when one selected Indonesia post image contained SCL value 255
at all pixels. Official SCL classes are 0--11 and the dataset supplies no
`_FillValue` metadata. An authorization-verified SCL-only audit found 255 in
exactly this one file. Version 6 retains 255 in the histogram, treats it as
unavailable only for descriptive SCL denominators, and changes no model input,
MASK, prediction, endpoint, condition, or analysis population.

During version-2 remediation, the upstream task file exposed inventory
membership counts and one protected-file `pixel_annotated` example. These values
cannot affect the already fixed filename selection, arms, endpoints, thresholds,
or conditions, but the study is not fully blind to protected task-definition
metadata. Before authorized v5 extraction, the holdout claim was restricted to
protected NetCDF content, imagery, MASK arrays, model predictions, and model
performance. The post-access scope and one observed MASK count are disclosed in
`research/sen12-v5-execution-status.md`; no model output was observed before the
version-6 correction.

## Frozen data identity

- Dataset: Sen12Landslides harmonized Sentinel-2.
- Hugging Face revision:
  `40af2dd6b4e568edb6640d6e14dc67ebd01038a4`.
- Upstream code revision:
  `d26a25edc8e0b69550696cfb97bb5a983eaa2fde`.
- Official S12LS-LD harmonized Sentinel-2 membership: a deterministic sanitized
  union of `train`, `val`, and `test` rows in upstream
  `tasks/S12LS-LD/harmonized/s2/splits.json` at that code revision. The local
  artifact retains only filename and inventory, sorted by filename; SHA-256
  `9d538889e86cd2e1c4c61bb7bf201ecd86765c6428b9c87833f7341fdddbec0e`.
- Inputs: named `B02`, `B03`, `B04`, and `B08` variables at the first
  metadata-declared post-event index, ordered Blue--Green--Red--NIR.
- Values: divide harmonized DN by 10,000 and clip to `[0,1]`.
- Outcome: the static public binary `MASK`.
- Conservative independence unit: inventory/site, never patch or `ann_id`.

The supplied upstream random patch split is forbidden. The version-4
within-inventory split is used only for non-degeneracy and checkpoint selection;
it is not an independent unit and receives no confirmation credit. Prior
cross-project MASK access is treated as outcome access even though it occurred
through Sentinel-1 files.

## Frozen inventory allocation

- Development pool: `chimanimani`, `china`, and `dominicamaria`.
- Calibration split: a file is assigned to validation exactly when
  `int(SHA256("v4-mixed:<filename>"),16) mod 5 = 0`; all other development files
  train the model. This yields 464 training and 113 validation files, with
  validation counts 56, 12, and 45 respectively.
- Protected evaluation: `hiroshima`, `hokkaido`, `indonesia`, `itogon`,
  `newzealand`, `usa`, `thrissur`, `italy`, `kyrgyzstan1`, `kyrgyzstan2`.

The allocation is the set difference between the pinned S12LS-LD inventories
and all inventories with documented prior MASK access. The upstream random
train/validation/test assignment is discarded after taking its union. Within
each inventory, up to 256 official S12LS-LD files are selected by SHA-256
ordering of `sen12-s2-confirmation-v1:<filename>`. No mixed
annotated/unannotated task member is eligible.

## Frozen configurations

The same four-channel `ControlledS3Net` and `ControlledPixelLoss` code is used.
Seven arms cross:

1. zero-control, feature-only attention + base loss;
2. raw B04/B08 edge attention + base loss;
3. NDVI/NDVI-edge attention + base loss;
4. zero-control, feature-only attention + plain boundary loss;
5. raw-edge attention + plain boundary loss;
6. NDVI attention + plain boundary loss;
7. NDVI attention + equal-mass NDVI-modulated boundary loss.

Seeds are 42, 43, and 44. Training uses AdamW (`1.5e-3`, weight decay `1e-4`),
cosine decay to `1e-5`, batch size 16, 15 epochs, deterministic filename order
and data-loader seeds, no augmentation, and validation F1 at threshold 0.5 for
checkpoint selection. The protected threshold is fixed at 0.5.

## Pre-access fit gate

All 21 checkpoints, best-epoch validation predictions, validation histories,
and configuration hashes are frozen
before protected extraction. Protected access is forbidden unless:

- every arm/seed has validation F1 at least 0.20 and every arm's three-seed mean
  validation F1 is at least 0.25;
- each saved validation probability array is reproduced from its checkpoint
  under `allclose(atol=2*float32_eps, rtol=1e-6)` and yields an exactly
  identical threshold-0.5 classification array;
- all checkpoints and fit decisions have SHA-256 identities;
- the member index, official task membership, pinned Sen12 requirements, recorded
  runtime versions, model, loss, semantic-loader, preparation, execution, and
  test code have SHA-256 identities;
- an authorization file binds this protocol commit, code hashes, dataset
  revision, member index, inventory list, thresholds, fit decision,
  configurations, validation predictions, and checkpoints;
- every bound source artifact is byte-identical to a publicly resolvable
  protocol/code commit whose successful GitHub API retrieval and HTTP `Date`
  are recorded before the protected authorization; a public PushEvent is
  recorded as supplemental evidence when the eventually consistent feed
  exposes it;
- the authorization artifact is then published and its public byte identity is
  verified before protected extraction.

Failure stops the experiment without protected access.

## Protected estimands

For every arm and seed, compute pooled pixel F1 and the arithmetic mean of
per-patch symmetric one-pixel-Chebyshev-tolerance boundary F1 within each
protected inventory. Empty-prediction/empty-reference boundary F1 is one in the
primary analysis. Average seeds within inventory, then report the unweighted
mean over ten inventories. Percentile intervals resample the ten inventories
with 10,000 deterministic bootstrap draws. Effects for every seed and inventory
and absolute performance are retained in machine-readable output.
The interval is the two-sided 2.5th--97.5th percentile interval; using its upper
endpoint for conditions 1--2 is conservative relative to a one-sided 95% bound.

Primary contrasts are interpreted in percentage points; machine-readable output
stores F1 and contrasts as fractions, so `0.01` equals one percentage point:

1. NDVI-attention base minus raw-edge-attention base F1;
2. equal-mass NDVI-modulated minus plain NDVI-boundary F1;
3. the same loss contrast on boundary F1;
4. plain-boundary minus base F1 for zero, raw, and NDVI controls.

Also report every inventory and seed effect, absolute arm performance, patch
counts, positive-pixel prevalence, and empty-mask counts.

## Confirmation conditions

Cross-dataset recurrence is confirmed only if all hold:

1. the 95% inventory-bootstrap upper bound for NDVI-attention minus raw-edge
   attention F1 is below +1.0 point;
2. the 95% upper bound for NDVI-modulated minus plain-boundary F1 is below
   +1.0 point;
3. the generic plain-boundary mean F1 effect is positive for all three controls,
   its pooled 95% lower bound is above zero, and at least 8 of 10 inventories
   have positive control-averaged effects;
4. every arm's seed-averaged, inventory-macro absolute F1 is at least 0.10, so
   small contrasts among degenerate predictors cannot count as recurrence;
5. no registered arithmetic, hash, overlap, semantic, or execution check fails.

The overlap check converts pixel-center coordinates to pixel-edge footprints
and transforms each full bounding edge to EPSG:4326 from its declared CRS using
`transform_bounds` with 64 densification points. Antimeridian-crossing outputs
are represented as two longitude intervals. Zero positive-area
footprint-envelope overlap is required between different protected inventories.
A failure invalidates the ten-independent-unit analysis rather than silently
reducing or changing folds.

The boundary-F1 modulation contrast is descriptive unless its interval excludes
zero. Conditions are not changed after protected access.

## Predeclared sensitivities

- Repeat absolute performance and all protected contrasts on patches for which
  every listed `date_confidence` is exactly 1.0.
- Report per-inventory results without respondent/patch weighting.
- Report the result with empty/empty boundary F1 scored both one and zero if any
  protected empty mask is encountered.
- Report first-post image dates; all SCL counts including 255; valid-SCL
  coverage; SCL 8/9/10 cloud prevalence; and SCL 3/8/9/10 cloud-or-shadow
  prevalence descriptively. Cloud denominators use only official codes 0--11;
  255 is retained as unavailable and cannot exclude an inventory or patch.

## Claims prohibited regardless of outcome

- causal vegetation, trigger-time, operational, or geographic-population claims;
- treating patches or individual polygons as independent;
- claiming that a failed condition confirms the opposite mechanism;
- calling version 6 pre-access or preregistered; only the unchanged v5
  scientific design has verified pre-access chronology;
- changing folds, arms, thresholds, endpoints, or conditions after access.
