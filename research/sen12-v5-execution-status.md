# Sen12 Sentinel-2 v5 protected execution disposition

## Authorized access

- Public v5 protocol commit:
  `a4fc49723d2305a9acdd580ad22b04cff1a0acf3`.
- Public protected-access authorization commit:
  `3f390776a37660ccab45d5f16ef6f21585313be2`.
- The authorization receipt was verified before extraction.
- Exactly 1,889 selected files across ten protected inventories were extracted.

## Evaluation stop

The first evaluation attempt stopped during strict dataset loading, before any
model checkpoint was loaded or any protected prediction was computed.
`indonesia_s2_165.nc` had SCL value 255 at every pixel of the selected post
image, while v5 accepted only official SCL classes 0--11.

Official Sentinel-2 SCL documentation defines only classes 0--11. The dataset
does not attach `_FillValue` or `missing_value` metadata to SCL. An
authorization-verified SCL-only diagnostic over all 1,889 protected files found
codes 1--11 and 255; exactly one file contained 255, and all 16,384 of that
file's post-image SCL pixels were 255. Its four optical bands were finite and
within the already frozen harmonized bounds.

The correction treats 255 conservatively as a dataset-specific unavailable-SCL
sentinel: retain it in SCL histograms, exclude it from descriptive cloud
denominators, and report unavailable counts. It does not exclude the file or
change imagery, MASK, model input, checkpoint, prediction, metric, contrast,
interval, condition, or sensitivity population.

## Post-access disclosure

During diagnosis, the selected file's SCL values and optical-band summary were
inspected. One MASK count (748 positive pixels) was also observed. No model
prediction, model performance, arm contrast, interval, or confirmation
condition was computed or observed before the correction was fixed.

Version 6 is therefore a post-access semantic execution correction, not a
pristine preregistration or a new protected confirmation. Its deterministic
scope is limited to descriptive SCL handling; all scientific analysis remains
the v5-authorized design.
