# Post-execution-documented CAS external check

Local notes carry a 2026-09-14T17:59:00Z timestamp, but this protocol was first
committed after the external run and its earlier version misstated the sampling
limits. It is therefore a corrected post-execution record, not independently
verifiable evidence of prospective specification or preregistration.

## Question

Does the positive, generic result from the corrected HR-GLDD control surface
replicate on a separate event-labelled RGB dataset: does ordinary boundary
weighting improve boundary delineation without materially reducing
segmentation F1?

This post-hoc analysis does **not** test NDVI specificity because CAS is RGB.
It tests only sensitivity of the generic boundary-weighting finding.

## Data and independence

- Dataset: CAS Landslide Dataset, Zenodo record 10294997, record licence
  CC BY-NC 4.0. Source-specific restrictions remain binding; no imagery is
  redistributed.
- Executed event allocation:
  `research/dataset-metadata/cas-boundary-confirmation/folds.json`.
- Independence unit: CAS subdataset region/event, never crop or tile.
- Development: Lombok, Moxitaidi-UAV-1m, Hokkaido, Wenchuan.
- Validation: Palu.
- Designated evaluation: Mengdong, Moxi-UAV-1m, Tiburon-Planet.

## Arms and equal budget

The same three-channel ControlledS3Net with zero-valued control maps is trained
under:

1. base loss: 0.5 BCE + 0.5 alpha-balanced focal loss;
2. plain boundary loss: the identical base map multiplied by
   `1 + 2 * B_GT`.

No NDVI, NIR, DEM, sensor metadata, or event identity enters either model.
Parameter count, initialisation, crops, augmentations, optimiser, schedule,
early-stopping rule, and seeds are identical. Seeds are 42, 43, and 44.

## Sampling and image semantics

Only RGB image files and binary masks are used. Every 512x512 CAS tile is
partitioned into fixed non-overlapping 128x128 crops. The executed limits are
80 source tiles per development event and 160 per validation/test event,
selected by salted SHA-256 filename order, which does not read image or mask
values. Empty-mask
crops remain included so false-positive behaviour is measurable.

## Outcomes

Primary outcomes:

- event-macro boundary F1 with one-pixel Chebyshev tolerance;
- event-macro binary F1.

Secondary outcomes:

- event-macro precision, recall, IoU, and background FPR;
- per-event results and worst-event F1;
- paired event effects for each seed.

Thresholds are selected only on the validation event from the fixed grid
`{0.05, 0.10, ..., 0.95}` by maximum pooled binary F1, with ties assigned to
the higher threshold. Protected labels never select a threshold, architecture,
crop, or stopping point.

## Pass/kill rule

The post-run record lists four quantitative conditions:

1. mean event-macro boundary-F1 gain is at least +0.015;
2. all three protected events have a positive mean boundary-F1 effect;
3. event-macro F1 is non-inferior within -0.010;
4. at least two of three seeds have non-negative event-macro F1 effect.

A fifth process condition in the original note required no post-access change
to code, thresholds, or reporting. Because the protocol was first committed
after execution, that process condition is not independently time-verifiable
and is recorded as such rather than passed. The failed-criteria classification
follows from three failures among the four quantitative conditions. It
cannot strengthen novelty or generalization claims.

## Deviations log

One Wenchuan image/mask triplet was opened after the initial hash allocation to
verify loader semantics (512x512 RGB image, binary mask values 0/1). No model
prediction or metric was inspected. Wenchuan was immediately removed from
protected testing and reassigned to development; the previously uninspected
Moxi-UAV-1m event was moved to protected testing before any model fit.

The first training attempt was stopped during development seed 42, epoch 5,
before any protected event was opened. Code review found that an imported
HR-GLDD confusion helper attached false-positive area using a fixed 3 m pixel
area, which is invalid for mixed-resolution CAS. The metric was removed and the
executed model/loss/data/threshold configuration was restarted from seed 42.

The second attempt was stopped during development seed 42, epoch 3, again
before protected access. The validation-F1 helper was made independent of the
HR-GLDD confusion routine so the executed and released CAS code are
byte-identical and contain no discarded 3 m area calculation. The retained
logs record no later change to the F1 equation or executed configuration; this
is not used as independently timestamped prospective evidence.

## Scope

Three designated regions support only a bounded post-hoc sensitivity result. They do not
support population-level, sensor-invariant, operational, or global
generalization claims. CAS licensing also excludes commercial-use framing.
