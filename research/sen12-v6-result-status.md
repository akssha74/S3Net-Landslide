# Sen12 Sentinel-2 v6 protected result

## Verdict

The registered cross-dataset recurrence claim is not confirmed. Four of five
conditions fail. No protected-result-driven redesign or rerun is permitted.

## Registered conditions

1. NDVI-attention minus raw-edge-attention F1: mean `+0.00700`, 95% inventory
   bootstrap interval `[+0.00282,+0.01162]`. The upper endpoint exceeds the
   registered `+0.01` margin: **fail**.
2. NDVI-modulated minus plain-boundary F1: mean `-0.000574`, interval
   `[-0.00412,+0.00261]`: **pass**.
3. Generic plain-boundary F1: control-averaged mean `+0.00608`, interval
   `[-0.000922,+0.01315]`; the lower endpoint is below zero and only five of
   ten inventories have strictly positive control-averaged effects: **fail**.
4. Absolute non-degeneracy: seven arm means range from `0.0121` to `0.0314`,
   below the registered `0.10` floor: **fail**.
5. Integrity: the registered conservative EPSG:4326 transformed-envelope check
   reports positive-area overlap between `kyrgyzstan1_s2_3307.nc` and
   `kyrgyzstan2_s2_6204.nc`: **fail**. Their same-CRS native pixel-edge
   rectangles share an edge but have zero positive-area overlap; neither
   duplicate content nor physical footprint overlap is established.

The NDVI-modulated minus plain-boundary boundary-F1 contrast is descriptive:
mean `-0.000426`, interval `[-0.00530,+0.00342]`.

## Interpretation

The authoritative-band protected study does not reproduce the HR-GLDD
mechanism conclusions. The model family also fails to transfer at useful
absolute accuracy under inventory holdout. Conditions 1--2 cannot be interpreted
as useful equivalence when the absolute predictions are degenerate, and the
overlap finding independently prevents a ten-unit confirmation claim.

The result does not satisfy the registered recurrence conditions and provides
no support for recurrence for this implementation. It is not evidence for the
opposite mechanism or a population-level geographic rejection. Version 6 is
post-access corrected only for descriptive SCL handling; that correction did
not alter any value above.
