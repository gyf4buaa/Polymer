# Stage 3C — readout / property-specific pooling ablation

Formal source commit: `a9ee75a1bf940f32f0966070caf2450f0c2fe1f4`.

R0 reuses the five historical Stage 3A.1 keep-dummy GINE runs; it was not retrained. R1 and R2 each add five frozen 5-fold OOF runs. The only scientific variable is readout.

## Overall OOF wMAE

| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |
|---|---:|---:|---:|---:|---:|---:|---:|
| R0 | 0.02277287222 | 0.02264862589 | 0.02288966669 | 0.02273300693 | 0.02288286218 | 0.02278540678 ± 0.0001024446 | 0.02264862589–0.02288966669 |
| R1 | 0.0226564311 | 0.02268401759 | 0.02269536571 | 0.02285440799 | 0.02293103224 | 0.02276425092 ± 0.000121194 | 0.0226564311–0.02293103224 |
| R2 | 0.02291787533 | 0.02299505325 | 0.02303736937 | 0.0229964671 | 0.02282943115 | 0.02295523924 ± 8.254391e-05 | 0.02282943115–0.02303736937 |

## Paired overall deltas (left minus right)

| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Left lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| R1-R0 | -0.0001164411232 | 3.539169882e-05 | -0.0001943009854 | 0.0001214010541 | 4.817005409e-05 | -2.115586031e-05 ± 0.0001297928 | 2/5 |
| R2-R0 | 0.0001450031154 | 0.0003464273605 | 0.0001477026774 | 0.0002634601647 | -5.343103464e-05 | 0.0001698324567 ± 0.000150752 | 1/5 |
| R2-R1 | 0.0002614442386 | 0.0003110356617 | 0.0003420036628 | 0.0001420591106 | -0.0001016010887 | 0.000190988317 ± 0.0001804049 | 1/5 |

## Per-target MAE and paired deltas

| Target | R0 mean ± SD | R1 mean ± SD | R2 mean ± SD | R1−R0 delta mean ± SD | R2−R0 delta mean ± SD | R2−R1 delta mean ± SD |
|---|---:|---:|---:|---:|---:|---:|
| Tg | 51.796823 ± 0.92003 | 52.273792 ± 0.41031 | 52.090827 ± 1.7987 | 0.4769687 ± 0.82864 | 0.29400383 ± 2.4951 | -0.18296487 ± 1.7548 |
| FFV | 0.0058948628 ± 0.00012834 | 0.0057920634 ± 6.1888e-05 | 0.0061286125 ± 0.00030001 | -0.00010279939 ± 0.00010732 | 0.00023374971 ± 0.00034773 | 0.0003365491 ± 0.00034791 |
| Tc | 0.024661791 ± 0.00036607 | 0.024716325 ± 0.00023249 | 0.024605827 ± 0.00046156 | 5.4534458e-05 ± 0.00059166 | -5.5963588e-05 ± 0.00072536 | -0.00011049805 ± 0.00044893 |
| Density | 0.024601593 ± 0.00066581 | 0.025292012 ± 0.00090683 | 0.025265553 ± 0.00020122 | 0.00069041949 ± 0.0015039 | 0.00066396015 ± 0.00078921 | -2.6459334e-05 ± 0.00077134 |
| Rg | 1.5773881 ± 0.02164 | 1.5508868 ± 0.020932 | 1.5661567 ± 0.0073823 | -0.026501304 ± 0.031267 | -0.011231415 ± 0.020568 | 0.015269889 ± 0.022488 |

## Parameter counts

| Model | Total trainable | Readout gate parameters | New vs R0 |
|---|---:|---:|---:|
| R0 | 1243657 | 0 | 0 |
| R1 | 1243913 | 256 | 256 |
| R2 | 1244937 | 1280 | 1280 |

## Attention diagnostics

Gate norms below are means over the five folds within each seed; entropy is normalized by log(node count) and averaged over OOF graphs.

### R1

- Gate `shared` L2 norm by seed: 42=0.523181, 43=0.550536, 44=0.497773, 45=0.437814, 46=0.488548; mean ± SD 0.49957 ± 0.042112.
- Tg normalized attention entropy: 0.957924 ± 0.0012534.
- FFV normalized attention entropy: 0.957924 ± 0.0012534.
- Tc normalized attention entropy: 0.957924 ± 0.0012534.
- Density normalized attention entropy: 0.957924 ± 0.0012534.
- Rg normalized attention entropy: 0.957924 ± 0.0012534.

### R2

- Gate `Density` L2 norm by seed: 42=0.59362, 43=0.560567, 44=0.516788, 45=0.689781, 46=0.524004; mean ± SD 0.576952 ± 0.070178.
- Gate `FFV` L2 norm by seed: 42=0.621682, 43=0.5885, 44=0.540197, 45=0.737546, 46=0.592821; mean ± SD 0.616149 ± 0.073901.
- Gate `Rg` L2 norm by seed: 42=0.659665, 43=0.622406, 44=0.560375, 45=0.796782, 46=0.532493; mean ± SD 0.634344 ± 0.1037.
- Gate `Tc` L2 norm by seed: 42=0.473652, 43=0.437935, 44=0.369181, 45=0.569156, 46=0.436191; mean ± SD 0.457223 ± 0.073083.
- Gate `Tg` L2 norm by seed: 42=0.659941, 43=0.635155, 44=0.548306, 45=0.807212, 46=0.581111; mean ± SD 0.646345 ± 0.10007.
- Tg normalized attention entropy: 0.933397 ± 0.0053291.
- FFV normalized attention entropy: 0.954235 ± 0.005145.
- Tc normalized attention entropy: 0.960298 ± 0.0053278.
- Density normalized attention entropy: 0.970343 ± 0.002441.
- Rg normalized attention entropy: 0.938469 ± 0.015657.

## Frozen setup and provenance

- Train SHA256: `1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1`
- Folds SHA256: `1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`
- Formal source commit: `a9ee75a1bf940f32f0966070caf2450f0c2fe1f4`
- R0: historical five seeds reused; new runs 0.
- R1/R2: 5/5 successful five-fold OOF runs; total new runs 10/10.
- Sum of run-metadata training durations: 6946.7 s (1.93 h).
- Formal queue wall time from RUN_START/RUN_SUCCESS records: 7476 s (2.08 h), including post-run diagnostics.
- CUDA device(s): NVIDIA GeForce RTX 4070.
- Peak PyTorch allocated/reserved VRAM: 144.7/168.0 MiB.
- Peak `nvidia-smi` memory: 4433.0 MiB.
- Initialization tests confirm R1 and R2 equal R0 within atol=rtol=1e-6; no R0 retraining was performed.
- No attention tensors were saved per atom; only fold gate norms and per-target OOF mean entropy are retained.

## Execution notes

- An initial R1 seed-42 attempt on `b149d46b934e33ab9cdcec19f91070f80cf0098f` completed five-fold training but failed while writing post-run attention diagnostics: `NameError: name 'rows' is not defined`. It was excluded from formal results.
- The complete ten-run queue was then executed on `a9ee75a1bf940f32f0966070caf2450f0c2fe1f4` with the frozen configurations. Failed-attempt training scores were not reused; the failed attempt and original logs remain outside the synced formal artifacts.
- Final-source R1/R2 CUDA fold-0 smokes passed on seed 43 for five epochs on the NVIDIA RTX 4070; graph-wise attention sums were one, activations and predictions were finite, gates updated, and smoke artifacts stayed isolated.

## Interpretation


R1 − R0 mean paired delta is -2.115586e-05 (R1 lower in 2/5 seeds). This small mixed-seed change does not establish a reliable benefit from learned shared node weighting.
R2 − R0 is 0.00016983246 (R2 lower in 1/5 seeds); R2 − R1 is 0.00019098832 (R2 lower in 1/5 seeds). Property-specific pooling does not improve overall OOF wMAE over either baseline.
Per-target changes are mixed: R1 improves mean FFV and Rg MAE while increasing Tg, Tc, and Density; R2 improves only Tc and Rg mean MAE while increasing Tg, FFV, and Density. No readout shows consistent benefit across multiple properties.

**Decision:** retain R0 (`global mean || global max`) as the next-stage readout and close the learned-pooling axis for this benchmark. These results do not start Stage 4.
