# Stage 4B — elemental physical priors

Formal source commit: `7df330921c156b0e2c88841f0947f0bb3369a52c`. G0 reuses its frozen five historical seeds; E completed five new five-fold OOF runs.

## Overall OOF wMAE

| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |
|---|---:|---:|---:|---:|---:|---:|---:|
| G0 | 0.0227728722 | 0.0226486259 | 0.0228896667 | 0.0227330069 | 0.0228828622 | 0.0227854068 ± 0.0001024446 | 0.0226486259–0.0228896667 |
| E | 0.0227670535 | 0.0227299322 | 0.0228735588 | 0.0227454390 | 0.0229688041 | 0.0228169575 ± 0.0001017838 | 0.0227299322–0.0229688041 |

## Paired E − G0 wMAE

| Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± sample SD | E lower |
|---:|---:|---:|---:|---:|---:|---:|
| -0.0000058187 | +0.0000813063 | -0.0000161079 | +0.0000124321 | +0.0000859419 | +0.0000315508 ± 0.0000486502 | 2/5 |

## Per-target MAE

| Target | G0 mean ± SD | E mean ± SD | E − G0 paired mean ± SD |
|---|---:|---:|---:|
| Tg | 51.79682332 ± 0.92003 | 52.70411506 ± 0.30379 | +0.9072917426 ± 1.0294 |
| FFV | 0.00589486282 ± 0.00012834 | 0.005900169329 ± 0.00017887 | +5.306508166e-06 ± 0.0002362 |
| Tc | 0.02466179092 ± 0.00036607 | 0.02445652278 ± 0.0003356 | -0.0002052681376 ± 0.00063277 |
| Density | 0.02460159259 ± 0.00066581 | 0.02498928755 ± 0.0013058 | +0.0003876949575 ± 0.0016963 |
| Rg | 1.577388115 ± 0.02164 | 1.554989481 ± 0.016295 | -0.02239863386 ± 0.027067 |

## Element audit and fixed normalization

- RDKit: `2026.03.2`; source: all 7,973 original SMILES.
- Observed atomic numbers: `0, 1, 5, 6, 7, 8, 9, 11, 14, 15, 16, 17, 20, 32, 34, 35, 48, 50, 52`.
- Dummy Z=0: 15,968 atoms; physical vector is exactly zero. Real atoms: 257,465 across 18 elements; all five values are finite for every observed real element.
- Observed real element counts: H 96, B 2, C 207,681, N 13,143, O 27,981, F 5,720, Na 6, Si 551, P 298, S 1,442, Cl 342, Ca 1, Ge 5, Se 6, Br 182, Cd 1, Sn 7, Te 1.

The scaler is the fixed Z=1–118 RDKit table's population mean/std (ddof=0), independent of benchmark rows, labels, and folds. Raw and normalized values for every observed element are in [`element_feature_manifest.json`](element_feature_manifest.json).

| Feature | Mean reference | Std reference |
|---|---:|---:|
| Atomic weight (u) | 146.403119 | 89.161979 |
| Covalent radius (Å) | 1.573475 | 0.419116 |
| van der Waals radius (Å) | 2.103814 | 0.260976 |
| Outer electrons | 5.559322 | 3.528437 |
| Period / row | 5.254237 | 1.611329 |

- Periodic-table reference SHA256: `17ce2ed2a6e9b73f0a6202c6bc5a91a9d9358205ab4f14babb39a49944160723`.
- Observed-element table SHA256: `3e240eaee91d9a4de4cef4df4436a5db1749bef881d0efc62fc4bad8ae7806af`.
- Manifest file SHA256: `0ddf47c3d6a01111e8fd549e4c0cd3e9ab7c6eec92d19a97206de022ad146580`.

## Execution and parameter counts

- New formal runs: 5; G0 retrains: 0.
- Concurrency: 2; formal wall: 2274.388223203001 s; summed fold training: 3926.5 s.
- GPU utilization mean/peak: 65.54605263157895% / 84.0%; `nvidia-smi` peak VRAM: 4780.0 MiB.
- PyTorch peak allocated/reserved VRAM: 142.212890625 / 168.0 MiB.
- Host CPU/RAM peak: 19.0% / 3710.1640625 MiB.
- Parameters: G0 1,243,657; E 1,244,937; added 1,280.

## Projection diagnostics

| Seed | Mean fold W_phys Frobenius norm | atomic_weight | covalent_radius | vdw_radius | outer_electrons | period |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 2.945142 | 1.253844 | 1.230397 | 1.249501 | 1.579998 | 1.230255 |
| 43 | 2.910085 | 1.225931 | 1.208096 | 1.228072 | 1.594179 | 1.204560 |
| 44 | 2.507591 | 1.057563 | 1.048963 | 1.072465 | 1.353103 | 1.041622 |
| 45 | 2.548473 | 1.083417 | 1.073483 | 1.096354 | 1.349568 | 1.067982 |
| 46 | 2.576528 | 1.091286 | 1.081494 | 1.103491 | 1.379523 | 1.074622 |

## Conclusion

Elemental physical priors show no stable OOF improvement; retain G0 and stop further simple 2D feature engineering for now.

Carry-forward candidate: **G0**. Stable improvement under the preregistered rule: **False**.
