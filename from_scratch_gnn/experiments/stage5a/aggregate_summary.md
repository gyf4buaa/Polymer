# Stage 5A — Pure Capacity Scaling

Formal source commit: `72490c1a748ed9395025f4120c9eb04f28268695`.
The only scientific variable was GINE `hidden_dim`; all widths use four layers and the frozen G0 graph, readout, heads, and training protocol. C256 reuses historical G0 and adds zero formal runs.

## Overall OOF wMAE

| Width | Parameters | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 326,921 | 0.0225200631 | 0.0226009436 | 0.0226484083 | 0.0223392964 | 0.0224992420 | 0.0225215907 ± 0.0001184405 |
| 256 | 1,243,657 | 0.0227728722 | 0.0226486259 | 0.0228896667 | 0.0227330069 | 0.0228828622 | 0.0227854068 ± 0.0001024446 |
| 384 | 2,750,217 | 0.0228974576 | 0.0229105783 | 0.0231893684 | 0.0230076855 | 0.0228424972 | 0.0229695174 ± 0.0001365575 |
| 512 | 4,846,601 | 0.0230498732 | 0.0231986312 | 0.0231990076 | 0.0232354888 | 0.0232057849 | 0.0231777571 ± 0.0000730755 |

## Paired deltas vs C256

Negative values favor the candidate width.

- C128 − C256: `-0.0002528091, -0.0000476823, -0.0002412584, -0.0003937105, -0.0003836202`; mean `-0.0002638161 ± 0.0001401512`; lower in 5/5 seeds.
- C384 − C256: `+0.0001245854, +0.0002619524, +0.0002997017, +0.0002746785, -0.0000403650`; mean `+0.0001841106 ± 0.0001427898`; lower in 1/5 seeds.
- C512 − C256: `+0.0002770010, +0.0005500053, +0.0003093409, +0.0005024819, +0.0003229228`; mean `+0.0003923504 ± 0.0001244993`; lower in 0/5 seeds.

## Per-target OOF MAE

Mean ± sample SD across five seeds.

| Target | C128 | C256 | C384 | C512 |
|---|---:|---:|---:|---:|
| Tg | 51.268929 ± 0.901 | 51.796823 ± 0.92 | 52.912749 ± 0.806 | 52.052209 ± 0.662 |
| FFV | 0.0061714849 ± 0.000306 | 0.0058948628 ± 0.000128 | 0.0058454451 ± 0.000198 | 0.0059871038 ± 0.000159 |
| Tc | 0.02406085 ± 0.000336 | 0.024661791 ± 0.000366 | 0.024745396 ± 0.000416 | 0.025221458 ± 0.000547 |
| Density | 0.024474358 ± 0.000941 | 0.024601593 ± 0.000666 | 0.025413378 ± 0.00114 | 0.026327405 ± 0.000881 |
| Rg | 1.518029 ± 0.0514 | 1.5773881 ± 0.0216 | 1.5720716 ± 0.0328 | 1.5918834 ± 0.0256 |

## Paired per-target deltas vs C256

| Target | C128 mean ± SD | C384 mean ± SD | C512 mean ± SD |
|---|---:|---:|---:|
| Tg | -0.52789427 ± 1.68 | +1.1159258 ± 1.24 | +0.2553856 ± 0.721 |
| FFV | +0.00027662206 ± 0.000301 | -4.9417759e-05 ± 0.000296 | +9.224098e-05 ± 0.000213 |
| Tc | -0.00060094129 ± 0.000402 | +8.3605121e-05 ± 0.000723 | +0.00055966671 ± 0.00056 |
| Density | -0.00012723502 ± 0.00137 | +0.00081178573 ± 0.00151 | +0.0017258123 ± 0.0013 |
| Rg | -0.059359066 ± 0.035 | -0.0053165285 ± 0.0386 | +0.014495236 ± 0.0202 |

## Cost and training behavior

| Width | Mean fold epoch (s) | Mean seed runtime (s) | Peak PyTorch allocated / reserved (MiB) | Peak nvidia-smi VRAM (MiB) | Mean GPU util (%) | Mean best epoch | Mean best validation wMAE | Mean epochs completed (min–max) | Mean train loss at best epoch |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1.812 | 921.8 | 76.5 / 118.0 | 5332.0 | 80.2 | 60.9 | 0.02252192 | 100.9 (58–154) | 0.0450422 |
| 256 | 1.082 | 563.9 | 141.8 / 168.0 | 2261.0 | 36.5 | 63.8 | 0.02278572 | 103.8 (64–170) | 0.0420551 |
| 384 | 1.931 | 1049.3 | 216.5 / 256.0 | 5446.0 | 82.2 | 69.0 | 0.02296984 | 109.0 (61–167) | 0.040167 |
| 512 | 1.980 | 1097.4 | 292.8 / 410.0 | 5458.0 | 77.9 | 69.4 | 0.02317813 | 109.4 (71–198) | 0.0401523 |

Per-fold `best_epoch` and `epochs_completed` values are retained in `aggregate_summary.json` as the early-stop distribution.

## Execution

- Actual concurrency: 2 (maximum observed: 2)
- Formal wall time: 9632.0 s
- Summed process/training time: 15342.5 s
- Sampled GPU utilization mean / peak: 80.1% / 93.0%
- Peak nvidia-smi VRAM: 5458.0 MiB
- Host CPU peak: 87.6%
- Host RAM peak: 3332.6 MiB
- Host CPU/RAM peak coverage: The initial 12-job queue held its resource samples only in memory and lost them when SSH transport timed out; these host CPU/RAM peaks cover the resumed 3-job segment only. Per-fold GPU utilization and VRAM are present in all 15 run_metadata files.
- One C128-seed46 attempt was interrupted by an SSH timeout before completion and excluded; attempt 02 completed from the same frozen source/config. All 15 preregistered runs passed.

## Interpretation

Use the paired deltas and five-seed consistency as descriptive engineering evidence. Do not interpret these summaries as a formal significance test.
