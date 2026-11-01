# Does Unfreezing the Backbone Break CD-ViTO?

Research code and result summaries for *A Controlled Re-examination for Cross-Domain
Few-Shot Object Detection*. This is an initial source release, not a claim of a
published or accepted paper. It extends [CD-ViTO](https://github.com/lovelyqian/CDFSOD-benchmark).

## Contents

| Directory | Contents |
| --- | --- |
| training/pc | Historical PC trainer, deterministic prototype extraction, support sampling and patch utilities |
| experiments/TN11/queue | Frozen T4 trainer and reference AP/statistics implementation |
| analysis/pc | Historical statistics implementation and synthetic statistical checks |
| protocols | Historical PR1/PR2 records and held-out protocol |
| results/development | Selected development per-run result summaries |
| results/heldout | All 15 paired held-out sampling summaries and reported confidence intervals |
| results/figure_data | Original numeric inputs for the included figure renderer |
| scripts, tests | Accelerated AP implementation, numeric checks and figure generation |

The E labels in the manuscript correspond to the TN identifiers in filenames.
This package contains selected result summaries; it is not a complete raw archive
of all E1--E15 artifacts. PROVENANCE.json records source hashes and packaging edits.
Historical Vietnamese filenames and argument names are retained.

## CPU verification

Use a virtual environment and install the analysis dependencies:

```sh
python -m pip install -r requirements-analysis.txt
python scripts/verify_results.py
python -m unittest discover -s tests -v
python analysis/pc/kiem_thu.py
python scripts/render_figures.py
python scripts/render_intervals.py
```

The verification script recomputes means and mean paired relative changes from
saved per-sampling AP summaries. It does not reconstruct image-bootstrap intervals
from summaries. AP tests compare accelerated accumulation to the frozen reference,
including repeated images, ties and ignored detections, using synthetic matches.
The statistical checks also use synthetic inputs. These are not GPU training tests.
The figure renderers preserve the saved data; the interval renderer uses a 4.8-inch
canvas and labels of at least 8 pt for the LNCS layout. Saved intervals are not
recomputed by plotting.

## Training prerequisites and execution

Training requires Linux/WSL, CUDA/PyTorch, the upstream CD-ViTO environment,
dataset registration, original annotations, pretrained weights and prototypes.
Follow the upstream repository's installation and data preparation instructions.
The historical scripts expect the upstream checkout at ~/CDFSOD-benchmark;
inspect the historical patch and va_*.py utilities before applying the required
model/data modifications to a separate checkout. Do not apply all patchers blindly:
their required subset depends on the experiment. A fresh end-to-end training
environment has not been validated as part of this source packaging.

Inspect available training arguments after preparing the upstream environment:

```sh
python training/pc/train_ext.py --help
python experiments/TN11/queue/trainer.py --help
```

protocols/tn11_protocol.json records held-out dataset schedules, sampling names,
configuration hashes, precision, effective batch and the locked recipe. It preserves
historical paths and references to external assets, which are not bundled. The T4
recipe uses FP16 with GradScaler, microbatch 4 and accumulation 4; the development
PC trainer is retained separately to avoid silently conflating numerical setups.
PR1/PR2 commit hashes are local Git records, not independently certified timestamps.

## Data availability and interpretation

Raw images, annotations, per-image predictions, checkpoints and private cloud
account credentials are excluded. Recomputing AP from images requires the matching
annotations and saved predictions; retraining requires the external inputs above.
No private cloud URLs or authentication tokens are needed for the CPU checks.
Recovery R is AP divided by the matched frozen reference. Mean paired relative
change is mean((final_AP-baseline_AP)/baseline_AP), not the ratio of group means.
Only UODD meets the registered held-out improvement rule; DeepFish and DIOR are
inconclusive. Development configurations must not be treated as the locked held-out
recipe or as evidence isolating the causal effect of backbone training.

## Licensing

Read LICENSE_SCOPE.md and THIRD_PARTY_NOTICES.md before reuse.
