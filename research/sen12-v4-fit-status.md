# Sen12 Sentinel-2 v4 fit disposition

## Decision

Version 4 is stopped before authorization. It is not eligible for protected
extraction.

## Evidence

- Public v4 protocol commit:
  `f2f12a7989027ff9daca182be9b125546eb671df`.
- The publicly locked final fit completed all 21 arm/seed runs.
- Individual validation F1 ranged from 0.2123 to 0.3290, exceeding the 0.20
  floor.
- Arm-level three-seed means ranged from 0.2662 to 0.3166, exceeding the 0.25
  floor.
- Independent checkpoint re-inference preserved every threshold-0.5
  classification and every F1 value.
- Fifteen runs had maximum absolute differences of `1.19e-7`--`1.79e-7`.
  A later exact bit-level check found up to 24 local float32 ULPs at small
  probability values; threshold classifications remained identical.
- The frozen v4 numerical gate used an absolute `1e-7` cutoff and therefore
  failed closed before authorization.
- Protected NetCDF, MASK, imagery, predictions, and performance remained
  unopened and unobserved.

## Version-5 correction

Version 5 uses a float32-aware criterion:
`allclose(atol=2*float32_eps, rtol=1e-6)` plus exact
threshold-classification identity. This changes no arm, seed, training example,
validation example, hyperparameter, checkpoint-selection rule, protected
endpoint, confidence interval, or confirmation condition.
