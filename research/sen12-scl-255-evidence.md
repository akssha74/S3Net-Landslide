# Sen12 SCL value-255 evidence

## External semantics

The Copernicus Sentinel-2 L2A specification defines SCL codes 0--11, with code
0 as no data. It does not define code 255:

- https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html
- https://custom-scripts.sentinel-hub.com/custom-scripts/sentinel-2/scene-classification/

The Sen12Landslides public documentation states that SCL is stored as an `int16`
time-series variable but does not define 255:

- https://github.com/PaulH97/Sen12Landslides
- https://huggingface.co/datasets/paulhoehn/Sen12Landslides

## Pinned-dataset observation

The authorization-verified diagnostic
`experiments/derived/results/sen12_v5_protected_scl_diagnostic.json` scanned
only the selected post-image SCL arrays. Across 1,889 protected files it found
codes 1--11 and 255. Exactly one file, `indonesia_s2_165.nc`, used 255, and all
16,384 SCL pixels at its selected post index were 255. The variable has neither
an `_FillValue` nor `missing_value` attribute. The corresponding four optical
bands are finite and within the frozen harmonized bounds.

## Conservative interpretation

Value 255 is not relabelled as official SCL code 0. It is retained explicitly in
the histogram and treated as a dataset-specific unavailable-SCL sentinel only
when calculating descriptive cloud denominators. The file remains in every
model and metric calculation. No scientific endpoint depends on SCL.
