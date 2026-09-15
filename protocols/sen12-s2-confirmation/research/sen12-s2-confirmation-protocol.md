# Sen12Landslides Sentinel-2 confirmation protocol

## Status and purpose

This corrected version-3 protocol is frozen before any Sentinel-2 NetCDF from
the ten protected inventories is extracted or opened. It tests whether the HR-GLDD mechanism
conclusions recur with authoritative named bands and inventory-level evaluation.
It does not test HR-GLDD's physical band order.

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
row and supersedes versions 1--2.

During version-2 remediation, the upstream task file exposed inventory
membership counts and one protected-file `pixel_annotated` example. These values
cannot affect the already fixed filename selection, arms, endpoints, thresholds,
or conditions, but the study is not fully blind to protected task-definition
metadata. Its holdout claim is restricted to protected NetCDF content, imagery,
MASK arrays, model predictions, and model performance.

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

The supplied random patch split is forbidden. Prior cross-project MASK access is
treated as outcome access even though it occurred through Sentinel-1 files.

## Frozen inventory allocation

- Training: `chimanimani`, `dominicamaria`.
- Validation: `china`.
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

- every selected arm/seed has validation F1 at least 0.25;
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
4. no registered arithmetic, hash, overlap, semantic, or execution check fails.

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
- Report first-post image dates, all SCL class counts, SCL 8/9/10 cloud
  prevalence, and SCL 3/8/9/10 cloud-or-shadow prevalence descriptively;
  neither can exclude a protected inventory or patch.

## Claims prohibited regardless of outcome

- causal vegetation, trigger-time, operational, or geographic-population claims;
- treating patches or individual polygons as independent;
- claiming that a failed condition confirms the opposite mechanism;
- calling the analysis preregistered unless public commit chronology verifies;
- changing folds, arms, thresholds, endpoints, or conditions after access.
