# Sen12Landslides Sentinel-2 confirmation protocol

## Status and purpose

This protocol is frozen before any Sentinel-2 NetCDF from the ten protected
inventories is extracted or opened. It tests whether the HR-GLDD mechanism
conclusions recur with authoritative named bands and inventory-level evaluation.
It does not test HR-GLDD's physical band order.

## Frozen data identity

- Dataset: Sen12Landslides harmonized Sentinel-2.
- Hugging Face revision:
  `40af2dd6b4e568edb6640d6e14dc67ebd01038a4`.
- Upstream code revision:
  `d26a25edc8e0b69550696cfb97bb5a983eaa2fde`.
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

The allocation is the set difference between current S12LS-LD inventories and
all inventories with documented prior MASK access. Up to 256 files per
inventory are selected by SHA-256 ordering of the filename. The current
`annotated_only` S12LS-LD definition is used; definitions from the paper's older
mixed annotated/unannotated task are not imported.

## Frozen configurations

The same four-channel `ControlledS3Net` and `ControlledPixelLoss` code is used.
Seven arms cross:

1. zero attention + base loss;
2. raw B04/B08 edges + base loss;
3. NDVI/NDVI-edge attention + base loss;
4. zero attention + plain boundary loss;
5. raw-edge attention + plain boundary loss;
6. NDVI attention + plain boundary loss;
7. NDVI attention + equal-mass NDVI-modulated boundary loss.

Seeds are 42, 43, and 44. Training uses AdamW (`1.5e-3`, weight decay `1e-4`),
cosine decay to `1e-5`, batch size 16, 15 epochs, deterministic filename order
and data-loader seeds, no augmentation, and validation F1 at threshold 0.5 for
checkpoint selection. The protected threshold is fixed at 0.5.

## Pre-access fit gate

All 21 checkpoints, validation predictions, and configuration hashes are frozen
before protected extraction. Protected access is forbidden unless:

- every selected arm/seed has validation F1 at least 0.25;
- all checkpoints and fit decisions have SHA-256 identities;
- an authorization file binds this protocol commit, code hashes, dataset
  revision, inventory list, thresholds, and checkpoints;
- the protocol/code commit is publicly resolvable with a timestamp preceding
  the protected authorization.

Failure stops the experiment without protected access.

## Protected estimands

For every arm and seed, compute pooled F1 and one-pixel-tolerance boundary F1
within each protected inventory. Average seeds within inventory, then report the
unweighted mean over ten inventories. Percentile intervals resample inventories
with 10,000 deterministic bootstrap draws.

Primary contrasts, in points:

1. NDVI-input base minus raw-edge base F1;
2. equal-mass NDVI-modulated minus plain NDVI-boundary F1;
3. the same loss contrast on boundary F1;
4. plain-boundary minus base F1 for zero, raw, and NDVI controls.

Also report every inventory and seed effect, absolute arm performance, patch
counts, positive-pixel prevalence, and empty-mask counts.

## Confirmation conditions

Cross-dataset recurrence is confirmed only if all hold:

1. the 95% inventory-bootstrap upper bound for NDVI-input minus raw-edge F1 is
   below +1.0 point;
2. the 95% upper bound for NDVI-modulated minus plain-boundary F1 is below
   +1.0 point;
3. the generic plain-boundary mean F1 effect is positive for all three controls,
   its pooled 95% lower bound is above zero, and at least 8 of 10 inventories
   have positive control-averaged effects;
4. no registered arithmetic, hash, overlap, semantic, or execution check fails.

The boundary-F1 modulation contrast is descriptive unless its interval excludes
zero. Conditions are not changed after protected access.

## Predeclared sensitivities

- Repeat protected metrics on patches whose `date_confidence` is exactly 1.0.
- Report per-inventory results without respondent/patch weighting.
- Report the result with empty/empty boundary F1 scored both one and zero if any
  protected empty mask is encountered.
- Report first-post image dates and cloud/SCL prevalence descriptively; neither
  can exclude a protected inventory or patch.

## Claims prohibited regardless of outcome

- causal vegetation, trigger-time, operational, or geographic-population claims;
- treating patches or individual polygons as independent;
- claiming that a failed condition confirms the opposite mechanism;
- calling the analysis preregistered unless public commit chronology verifies;
- changing folds, arms, thresholds, endpoints, or conditions after access.
