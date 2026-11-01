# Validation of the initial source package

Checked on 2026-11-01 using Python 3.14.4 in the existing analysis environment.

- Held-out means, sample SDs, AP differences and mean paired relative changes
  recomputed from all 15 saved sampling summaries: passed.
- Two accelerated-AP tests, including repeated images, tied scores, ignored
  detections and absent ground truth: passed against the frozen accumulator.
- Eleven historical synthetic statistical checks: passed.
- Recovery/drift and probe-trajectory figures regenerated from saved CSVs: passed.
- Common token/private-key/private-Drive-link scan of release text: passed.

These checks do not rerun GPU training, verify every historical AP from raw
predictions, or recompute image-bootstrap confidence intervals from summary CSVs.
The preserved confidence intervals are reported results, not newly computed intervals.
