# Preregistered LRD event-level boundary-weighting confirmation

This protocol, fold file, fetcher, fit/evaluation code, outcomes, and pass rule
must be committed before any validation or protected-event Sentinel-2 or mask
payload is fetched. The protocol commit is supplied to every stage and checked
against the working tree.

## Corpus and overlap

- Landslide Reference Data v3, Zenodo record 17007637, CC BY 4.0.
- Independence unit: released event ID (EID), never crop, polygon, or pixel.
- The corpus is not used by another live submission. Prior archived work used
  development EID masks/terrain only for killed terrain-headroom checks.
- No validation or protected EID image/mask has been fetched.
- GLaD4CD is prohibited by the programme-overlap gate and is not used.

## Fixed event folds

The outcome-independent EID allocation was originally frozen at
`research/dataset-metadata/lrd-prospective-confirmation/protected_event_folds.json`.

- Development (16): CA0001, CL0001, CN0001, CO0001, IE0001, IN0002, IN0004,
  IR0001, IS0001, IT0001, JP0001, KG0001, KG0002, KR0001, PH0003, PK0001.
- Validation (6): BR0001, CL0002, GT0001, IN0001, IN0003, NZ0001.
- Protected test (6): CA0002, CN0002, IR0002, PH0001, PH0004, US0001.

## Inputs and sampling

For each EID, use the first post-event (`POST1`) Sentinel-2 L2A B04, B03, and
B02 rasters as Red, Green, and Blue, plus the released binary mask. Reflectance
is clipped to `[0,10000]` and divided by 10000.

Rasters are padded on the bottom/right to a multiple of 128. Candidate
non-overlapping 128-pixel crop coordinates are ordered by SHA-256 of
`lrd-boundary-crops-v1:<EID>:<y>:<x>` using image dimensions only. Retain at
most 64 crops per EID, including empty-mask crops. No mask value controls crop
membership.

## Intervention

Train the same three-channel `ControlledS3Net(in_ch=3,gating_mode="none")`
under:

1. base loss: 0.5 BCE + 0.5 alpha-balanced focal loss;
2. plain boundary loss: the same per-pixel map weighted by `1 + 2*B_GT`.

Seeds: 42, 43, 44. AdamW learning rate 0.0015, weight decay 0.0001; cosine
decay to 0.00001; batch size 32; 15 epochs. Pooled validation F1 at threshold
0.5 selects the epoch. A validation-only threshold from
`{0.05,0.10,...,0.95}` maximizes pooled validation F1, with higher-threshold
tie breaking.

All six checkpoints and thresholds are frozen before protected data is
fetched. The fit stage writes a cryptographic protected-access authorization;
the fetcher refuses protected access without it.

## Outcomes and pass rule

Primary outcomes:

- event-macro F1;
- event-macro boundary F1 with one-pixel Chebyshev tolerance.

Secondary outcomes: event precision, recall, IoU, background FPR, per-event
effects, and paired seed effects.

Confirmation passes only if all four conditions hold:

1. mean event-macro boundary-F1 gain is at least +0.015;
2. every protected EID has a positive mean boundary-F1 effect;
3. mean event-macro F1 is non-inferior within -0.010;
4. at least two of three seeds have non-negative event-macro F1 effects.

After protected authorization, no code, split, checkpoint, threshold, metric,
or reporting rule may change. Any other outcome is reported as failed
confirmation without retuning.

## Scope

Six protected EIDs support a bounded cross-corpus confirmation. They do not
support population, sensor-invariant, operational, or global generalization.
