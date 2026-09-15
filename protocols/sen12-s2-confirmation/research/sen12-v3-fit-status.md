# Sen12 Sentinel-2 v3 fit disposition

## Decision

Version 3 is stopped at the development fit gate. It is not eligible for
authorization or protected extraction.

## Evidence

- Public v3 protocol commit:
  `a37fd741e3268cbbc3f93fdc1c5f93d61fbe3941`.
- Development extraction: 577 pinned S12LS-LD files from `chimanimani`,
  `dominicamaria`, and `china`.
- Frozen v3 design: train on the first two inventories and validate on `china`.
- Observed validation F1 across seven arms and seeds 42--44:
  0.0216--0.0520.
- Required v3 floor: every run at least 0.25.
- Threshold-only development sensitivity: each run's best F1 remained at most
  0.0791, so calibration could not rescue the gate.
- Protected NetCDF, MASK, imagery, predictions, and performance remained
  unopened and unobserved.

## Development-only redesign evidence

After the v3 stop, a deterministic filename-hash split pooled all three
development inventories while retaining 20% of each inventory for calibration.
Under the unchanged seven arms, seeds, optimizer, loss, threshold, and 15
epochs, the 21-run development matrix produced:

- individual validation F1: 0.2123--0.3290;
- arm-level seed means: 0.2662--0.3166;
- all arm means at least 0.25;
- all individual runs at least 0.20.

This mixed split is not an independent-event validation and receives no
confirmation credit. It is used only as a non-degeneracy and checkpoint-selection
gate for version 4. The ten untouched inventories remain the only protected
confirmation units.

Version 4 separately requires every arm's protected seed-averaged,
inventory-macro absolute F1 to reach 0.10. Thus near-zero protected predictions
cannot satisfy recurrence merely because mechanism contrasts are also near zero.
