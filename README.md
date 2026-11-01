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
| results/intervals | Saved paired intervals of the 24 E7/AdamW development cells |
| results/figure_data | Numeric inputs for Figs. 2 and 3 and per-sampling recovery ratios of the development configurations |
| figures | LaTeX source of the study-sequence diagram (Fig. 1) |
| scripts, tests | Accelerated AP implementation, numeric checks and figure renderers |

The E labels in the manuscript correspond to the TN identifiers in filenames.
This package contains selected result summaries; it is not a complete raw archive
of all E1--E15 artifacts. PROVENANCE.json records source hashes and packaging edits.
Historical Vietnamese filenames and argument names are retained. ap_nhanh.py and
thong_ke.py are byte-identical in analysis/pc and experiments/TN11/queue; both
historical locations are kept.

## CPU verification

Use a virtual environment and install the analysis dependencies:

```sh
python -m pip install -r requirements-analysis.txt
python scripts/verify_results.py
python -m unittest discover -s tests -v
python analysis/pc/kiem_thu.py
```

The verification script recomputes means and mean paired relative changes from
saved per-sampling AP summaries. It does not reconstruct image-bootstrap intervals
from summaries. AP tests compare accelerated accumulation to the frozen reference,
including repeated images, ties and ignored detections, using synthetic matches.
The statistical checks also use synthetic inputs. These are not GPU training tests.

## Figures

| Figure | Input | Generator |
| --- | --- | --- |
| Fig. 1 | none (diagram) | figures/fig1_workflow.tex |
| Fig. 2 | results/figure_data/data_fig3_lr_sweep.csv, data_fig4a_drift.csv | scripts/render_figures.py |
| Fig. 3 | results/figure_data/data_fig4b_trajectories.csv | scripts/render_figures.py |
| Fig. 4 | results/intervals/development_intervals.json | scripts/render_intervals.py |

CSV file names keep their historical figure numbers. The renderers plot saved values
without recomputing them and write to outputs/figures. Figs. 2-4 use a 4.8-inch
canvas with 8 pt labels for the LNCS layout; Fig. 1 requires a LaTeX distribution
with the standalone, newtx and TikZ packages:

```sh
python scripts/render_figures.py
python scripts/render_intervals.py
pdflatex -output-directory=outputs/figures figures/fig1_workflow.tex
```

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
