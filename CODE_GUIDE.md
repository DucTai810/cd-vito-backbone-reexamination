# Reading the historical code

The files in training/pc, experiments/TN11/queue, analysis/pc and protocols are the
code and records used in the experiments, copied without edits (PROVENANCE.json lists
each source hash). Their identifiers, comments and console messages are Vietnamese
without diacritics; the protocol records are Vietnamese. protocols/tn11_protocol.json
pins the SHA-256 of experiments/TN11/queue/trainer.py (trainer_sha256), so that file
must stay byte-identical. This guide gives the English meaning of the terms needed
to review them.

## Labels

| Label | Meaning |
| --- | --- |
| TN*n* | Experiment group E*n* of the manuscript |
| A | Frozen CD-ViTO detector |
| D, D(r) | Full backbone training at backbone/head learning-rate ratio r |
| C0 | Original setting: fp16 backbone, RPN head trained at the head learning rate |
| s0, s1, ... | Official 10-shot sampling and regenerated samplings; CSV and protocol fields call them splits |
| T, D, N, K | Cause groups in thong_ke.py: learning rate, backbone size (group S in the manuscript), shots, architecture |
| K1, K2, K3 | Branches of group K: RPN training, prototype refresh, precision |
| P3 | Fixed 200-image probe subsets used for probe-set AP trajectories |
| P4 | Per-update gradient log (p4.tsv): gradient norms per group and non-finite gradient counts |
| Q0-Q6 | Rule codes of the registered decision table for the cause groups |
| Muc, Bang, de cuong | Section, table, study proposal; these refer to the internal proposal, not the manuscript |

The thresholds and decision rules that comments cite as Bang 8 and Bang 9 are stated
in the manuscript under the first (PR1) and second (PR2) pre-registration.

## Decision labels in thong_ke.py

| Code value | Manuscript term |
| --- | --- |
| Duong / Am / Khong ro | positive / negative / unclear (cause groups) |
| Cai thien / Kem hon / Khong khac biet ro | improved / worse / inconclusive (one held-out domain) |
| Cai thien / Cai thien co dieu kien / Danh doi / Khong cai thien | improvement / conditional improvement / trade-off / no improvement (aggregate) |
| DAT / KHONG DAT | passed / failed (checks) |

## Files

| File | Purpose |
| --- | --- |
| analysis/pc/ap_nhanh.py | COCO-style AP that can resample test images for the bootstrap without rerunning COCOeval |
| analysis/pc/thong_ke.py | Paired two-stage bootstrap and the registered decision rules |
| analysis/pc/kiem_thu.py | Synthetic checks of thong_ke.py with known answers |
| training/pc/train_ext.py | Extended PC trainer; with all options off it matches the upstream trainer plus gradient accumulation |
| experiments/TN11/queue/trainer.py | The same trainer with FP16 GradScaler and checkpoint handling used on T4 |
| training/pc/danh_gia_bien_the.py | Evaluates weight variants without training: single-block resets (E9) and WiSE-FT interpolation |
| training/pc/sinh_lan_lay_mau.py | Generates samplings s1-s9 with the one-box-per-image rule of the official s0 |
| training/pc/kiem_quy_tac_s0.py | Infers that rule from the official s0 files |
| training/pc/tinh_proto_co_dinh.py | Computes class prototypes with a fixed random seed around the upstream tool |
| training/pc/tao_tap_p3.py | Creates and registers the P3 probe subsets |
| training/pc/va_builtin_lan_lay_mau.py | Patch: registers the generated samplings in detectron2 |
| training/pc/va_categories.py | Patch: adds class lists for the generated sampling names |
| training/pc/va_devit_mo_pttd.py | Patch: allows backbone training and selects its precision |
| training/pc/va_devit_mo_rpn.py | Patch: allows RPN-head training (C0) |
| training/pc/va_devit_chia_lo.py | Patch: runs box regression in chunks of proposals at inference to fit 16 GB |
| training/pc/va_devit_do_kich_thuoc.py | Patch: prints regression tensor sizes for diagnosis |
| training/pc/va_dinov2_sdpa.py | Patch: uses PyTorch SDPA when xformers is absent and checks equivalence |
| training/pc/kiem_*.py | Checks of registration, class lists, imports, chunked inference and train_ext.py |
| training/pc/cdfsod_benchmark_va.patch | Diff of the modified upstream files: dataset registration, a deformable-convolution CUDA kernel, devit.py, class lists and DINOv2 attention |

All va_*.py patches are idempotent: they detect an applied patch and skip it.

## Trainer options

| Option | Meaning |
| --- | --- |
| --accum N | Micro-batches accumulated per update |
| --mo-pttd {khong, toan_bo, kN} | Backbone training: none; all 24 blocks with patch embedding and final norm; last N blocks with final norm |
| --ti-le-lr-pttd r | Backbone/head learning-rate ratio r |
| --llrd h | Layer-wise learning-rate decay: block i is scaled by h^(23-i), patch embedding by h^24 |
| --toi-uu {sgd, adamw} | Optimizer |
| --l2sp a | L2-SP strength towards the initial backbone weights |
| --lora R | LoRA rank on query and value projections (0 = off) |
| --khoa-khoi i,j,... | Blocks kept fixed while training the whole backbone |
| --mo-rpn | Train the RPN head (C0) |
| --so-hoc {fp16, bf16, fp32} | Backbone precision during training; evaluation always uses fp16 |
| --tinh-lai-proto N | Refresh prototypes every N updates (K2; 0 = off) |
| --khong-checkpoint | Disable gradient checkpointing |

## Protocol records

protocols/PR1_as_committed.md and PR2_as_committed.md are the pre-registration texts
as committed. Paths in them refer to the research repository: phan_tich/ corresponds
to analysis/pc and ma_nguon/ to training/pc. The change log they mention
(SO_GHI_CHEP.md) is not part of this package.
