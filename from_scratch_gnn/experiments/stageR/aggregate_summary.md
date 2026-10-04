# Stage R — Benchmark Reality Check

Frozen benchmark: `nopp2025_train_v1`, 7,973 samples; train SHA256 `1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1`; folds SHA256 `1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`.

All baselines use the same frozen five folds and `evaluate_oof`. LightGBM and kNN predictions are generated without validation labels entering model fit or preprocessing. C128 predictions are existing five-seed OOF artifacts; no GNN was trained in Stage R.

## OOF comparison

| Model | Overall OOF wMAE | Tg | FFV | Tc | Density | Rg |
|---|---:|---:|---:|---:|---:|---:|
| Median | 0.06098563 | 86.94653 | 0.02077689 | 0.07616204 | 0.1046539 | 3.884517 |
| Tanimoto kNN (k=5) | 0.03066069 | 53.70421 | 0.009314909 | 0.03601802 | 0.06092912 | 1.700242 |
| Clean LightGBM | 0.02352955 | 50.90582 | 0.006848803 | 0.02562413 | 0.02910599 | 1.513395 |
| C128 Own-GNN | 0.02252159 ± 0.00011844 | 51.26893 | 0.006171485 | 0.02406085 | 0.02447436 | 1.518029 |
| C256/G0 historical | 0.02278541 ± 0.00010244 | 51.79682 | 0.005894863 | 0.02466179 | 0.02460159 | 1.577388 |

A positive paired delta means the baseline has higher error than the five-seed C128 reference. The 95% intervals resample sample IDs and use the mean C128 per-sample absolute error across seeds; they do not include model-seed uncertainty.

| Baseline | Δ wMAE vs C128 | Paired sample-bootstrap 95% CI |
|---|---:|---:|
| Median | +0.03846404 | [+0.03611986, +0.04089902] |
| Tanimoto kNN (k=5) | +0.00813910 | [+0.00718886, +0.00909993] |
| Clean LightGBM | +0.00100796 | [+0.00043144, +0.00159247] |

## Label statistics

| Target | Count | Mean | Sample SD | Median | Min | Max |
|---|---:|---:|---:|---:|---:|---:|
| Tg | 511 | 96.4523 | 111.228 | 74.0402 | -148.03 | 472.25 |
| FFV | 7,030 | 0.367212 | 0.0296088 | 0.364264 | 0.226992 | 0.777097 |
| Tc | 737 | 0.256334 | 0.0895378 | 0.236 | 0.0465 | 0.524 |
| Density | 613 | 0.985484 | 0.146189 | 0.948193 | 0.748691 | 1.841 |
| Rg | 614 | 16.4198 | 4.60864 | 15.0522 | 9.72836 | 34.6729 |

### Label co-occurrence counts

| | Tg | FFV | Tc | Density | Rg |
|---|---:|---:|---:|---:|---:|
| Tg | 511 | 1 | 32 | 24 | 24 |
| FFV | 1 | 7,030 | 300 | 270 | 270 |
| Tc | 32 | 300 | 737 | 531 | 535 |
| Density | 24 | 270 | 531 | 613 | 610 |
| Rg | 24 | 270 | 535 | 610 | 614 |

## Frozen metric contribution (C128 mean across seeds)

| Target | n | Frozen weight | Weight × n / N | C128 MAE | wMAE contribution | Share |
|---|---:|---:|---:|---:|---:|---:|
| Tg | 511 | 0.0020523775 | 0.0001315396 | 51.26893 | 0.006743892 | 29.9% |
| FFV | 7,030 | 0.62392487 | 0.5501307 | 0.006171485 | 0.003395123 | 15.1% |
| Tc | 737 | 2.2199754 | 0.2052078 | 0.02406085 | 0.004937474 | 21.9% |
| Density | 613 | 1.0640942 | 0.08181233 | 0.02447436 | 0.002002304 | 8.9% |
| Rg | 614 | 0.046558118 | 0.003585436 | 1.518029 | 0.005442797 | 24.2% |

## Structural similarity diagnostics

Repeat-unit heavy atoms (atomic number > 1; dummy `*` excluded): n=7,973, mean=32.28, median=29, range=1–164; p10/p90=11/57.

Maximum Tanimoto uses Morgan radius 2, 2048 bits, against every fold-training SMILES. The split remains an ordinary random sample split; bins characterize higher-similarity interpolation and lower-similarity chemistry, not a leakage finding.

| Max-similarity bin | All validation samples | Tg | FFV | Tc | Density | Rg |
|---|---:|---:|---:|---:|---:|---:|
| <0.7 | 2,626 | 53.1277 (n=293) | 0.008622308 (n=2,178) | 0.02403139 (n=260) | 0.03645796 (n=238) | 1.534043 (n=239) |
| 0.7–0.9 | 2,849 | 49.77088 (n=128) | 0.00561638 (n=2,655) | 0.01883899 (n=147) | 0.02538542 (n=128) | 1.405683 (n=129) |
| >0.9 | 2,498 | 47.34816 (n=90) | 0.004412682 (n=2,197) | 0.02641016 (n=330) | 0.01245527 (n=247) | 1.561384 (n=246) |

### Largest C128 weighted-error samples

| sample_id | Fold | Max Tanimoto | Weighted contribution | Largest target errors |
|---|---:|---:|---:|---|
| 1599967213 | 3 | 0.275 | 0.0002092 | Rg 19.354, Density 0.23906 |
| 219039564 | 2 | 0.328 | 0.0001271 | Tg 110.52, Rg 3.5294 |
| 933916766 | 3 | 1.000 | 0.0001185 | Tg 37.057, Rg 11.306 |
| 1291515166 | 0 | 0.472 | 0.0001073 | Density 0.70929, Rg 0.36263 |
| 1582377269 | 4 | 1.000 | 0.0001007 | Rg 4.5181, Tc 0.26308 |
| 2986007 | 0 | 0.698 | 0.0000754 | Rg 5.4545, Tc 0.14401 |
| 190407869 | 2 | 0.455 | 0.0000745 | Rg 4.7425, Tc 0.15497 |
| 1525026548 | 1 | 0.978 | 0.0000727 | Rg 6.2176, Density 0.096972 |
| 1244604743 | 4 | 0.346 | 0.0000711 | Tg 56.095, Rg 4.535 |
| 1090817229 | 3 | 1.000 | 0.0000698 | Rg 6.2583, Tc 0.10267 |

## Morgan duplicate audit

There are 545 duplicate Morgan fingerprint groups across 1717 samples (group sizes: {'2': 327, '3': 98, '4': 45, '5': 21, '6': 16, '7': 19, '8': 8, '9': 2, '10': 1, '11': 2, '12': 3, '13': 1, '26': 1, '66': 1}). Canonical graph categories: 0 exact isomeric canonical groups; 30 stereochemical variants of one achiral graph; 515 groups with multiple achiral canonical graphs.

A matching fingerprint is not treated as a duplicate molecular graph. Canonicalization is atom-map-insensitive; polymer/repeat-unit equivalence remains **UNKNOWN**. No rows were removed or grouped for training.

- Group 1 (different_achiral_canonical_graphs, size 2):
  - `3128201` `*C(=O)c1ccc2c(c1)C(=O)N(c1ccc(C(=O)c3ccc(Cc4ccc(C(=O)c5ccc(N6C(=O)c7ccc(*)cc7C6=O)cc5)cc4)cc3)cc1)C2=O` → `*C(=O)c1ccc2c(c1)C(=O)N(c1ccc(C(=O)c3ccc(Cc4ccc(C(=O)c5ccc(N6C(=O)c7ccc(*)cc7C6=O)cc5)cc4)cc3)cc1)C2=O`
  - `11667291` `*C(=O)c1ccc2c(c1)C(=O)N(c1ccc(Cc3ccc(C(=O)c4ccc(N5C(=O)c6ccc(*)cc6C5=O)cc4)cc3)cc1)C2=O` → `*C(=O)c1ccc2c(c1)C(=O)N(c1ccc(Cc3ccc(C(=O)c4ccc(N5C(=O)c6ccc(*)cc6C5=O)cc4)cc3)cc1)C2=O`

- Group 14 (stereochemical_variants_same_achiral_graph, size 2):
  - `38242048` `*/C(F)=C(\F)C(F)(C(*)(F)F)C(F)(F)F` → `*/C(F)=C(\F)C(F)(C(*)(F)F)C(F)(F)F`
  - `1578702981` `*C(F)=C(F)C(F)(C(*)(F)F)C(F)(F)F` → `*C(F)=C(F)C(F)(C(*)(F)F)C(F)(F)F`

- Group 328 (different_achiral_canonical_graphs, size 3):
  - `2265305` `*Nc1ccc(-c2ccc(-c3ccc(N*)cc3)cc2)cc1` → `*Nc1ccc(-c2ccc(-c3ccc(N*)cc3)cc2)cc1`
  - `4600132` `*Nc1ccc(-c2ccc(N*)cc2)cc1` → `*Nc1ccc(-c2ccc(N*)cc2)cc1`
  - `1936130559` `*Nc1ccc(-c2ccc(-c3ccc(-c4ccc(N*)cc4)cc3)cc2)cc1` → `*Nc1ccc(-c2ccc(-c3ccc(-c4ccc(N*)cc4)cc3)cc2)cc1`

- Group 537 (different_achiral_canonical_graphs, size 10):
  - `166405771` `*CCCCCCNC(=O)C(CCCCCCCCCCCCCC)C(=O)N*` → `*CCCCCCNC(=O)C(CCCCCCCCCCCCCC)C(=O)N*`
  - `201596108` `*CCCCCCNC(=O)C(CCCCCCCCCCCCC)C(=O)N*` → `*CCCCCCNC(=O)C(CCCCCCCCCCCCC)C(=O)N*`
  - `286918876` `*CCCCCCCNC(=O)C(CCCCCCCCCCCC)C(=O)N*` → `*CCCCCCCNC(=O)C(CCCCCCCCCCCC)C(=O)N*`

- Group 544 (different_achiral_canonical_graphs, size 26):
  - `82977315` `*CCCCCCCCCCCCCCCCCCCCOC(=O)CCCCCCCC(=O)O*` → `*CCCCCCCCCCCCCCCCCCCCOC(=O)CCCCCCCC(=O)O*`
  - `116488904` `*CCCCCCCCCCCCOC(=O)CCCCCCCCCCCCC(=O)O*` → `*CCCCCCCCCCCCOC(=O)CCCCCCCCCCCCC(=O)O*`
  - `166991206` `*CCCCCCCCCCCCCCCCCCCCOC(=O)CCCCC(=O)O*` → `*CCCCCCCCCCCCCCCCCCCCOC(=O)CCCCC(=O)O*`

- Group 545 (different_achiral_canonical_graphs, size 66):
  - `6645418` `*CCCCCNC(=O)CCCCC(=O)N*` → `*CCCCCNC(=O)CCCCC(=O)N*`
  - `81137216` `*CCCCCCCCNC(=O)CCCCCCCCCCCCCCCCCCCCC(=O)N*` → `*CCCCCCCCNC(=O)CCCCCCCCCCCCCCCCCCCCC(=O)N*`
  - `234377391` `*CCCCCCCCCCNC(=O)CCCCCCCCCCCCCCC(=O)N*` → `*CCCCCCCCCCNC(=O)CCCCCCCCCCCCCCC(=O)N*`

The inspected differing-graph groups include regioisomers and different methylene/ether repeat lengths. The largest group (66 samples) consists of distinct polyamide-like repeat-unit graphs with varying chain lengths that nevertheless produce the same folded radius-2 bit set. The 30 stereochemical-only groups are consistent with this Morgan configuration having chirality disabled. These explain observed cases; the precise bit-collision versus shared-environment mechanism was not separately isolated.

## GNN / LightGBM error complementarity

The fixed 50:50 OOF blend scores **0.02112949** versus **0.02097316** for the five-seed averaged C128 predictions (Δ +0.00015634; paired sample-bootstrap 95% CI [-0.00014398, +0.00045882]). The blend weight was fixed at 0.5 and not tuned.

| Target | Abs-error correlation | Residual correlation | C128 seed-mean MAE | LightGBM MAE | 50:50 blend MAE | Blend Δ vs C128 |
|---|---:|---:|---:|---:|---:|---:|
| Tg | 0.720 | 0.850 | 48.6821 | 50.90582 | 47.94164 | -0.7404629 |
| FFV | 0.847 | 0.808 | 0.005436789 | 0.006848803 | 0.005655542 | +0.0002187532 |
| Tc | 0.855 | 0.885 | 0.02292792 | 0.02562413 | 0.02326855 | +0.0003406346 |
| Density | 0.718 | 0.697 | 0.02099468 | 0.02910599 | 0.02285489 | +0.001860211 |
| Rg | 0.814 | 0.852 | 1.438034 | 1.513395 | 1.413297 | -0.0247379 |

## Stage R conclusions

1. **Baseline gap:** C128 is far ahead of the fold-median and kNN baselines. Its advantage over the clean LightGBM is smaller but its paired sample-bootstrap interval remains above zero (LightGBM − C128 = +0.001008; 95% CI +0.000431 to +0.001592).
2. **Target profile:** C128 is better than LightGBM on FFV, Tc, and Density with paired intervals excluding zero. Tg and Rg are close; their intervals include zero, with LightGBM's point MAE slightly lower.
3. **Fusion:** C128 and LightGBM errors are strongly correlated (per-target absolute-error correlations 0.72–0.86). A fixed 50:50 blend does not improve on the C128 seed-averaged prediction; its paired interval crosses zero. There is no evidence here for a default GNN+tree blend, though target-specific fusion remains testable.
4. **Similarity:** Higher nearest-neighbor similarity aligns with lower FFV and Density errors, while Tc is best in the middle bin and Rg is nearly flat. C128's performance is not uniformly concentrated in high-similarity samples. This random split remains an IID-style benchmark, not a leakage test.
5. **Morgan duplicates:** 545 exact 2048-bit fingerprint groups cover 1,717 rows; none share an isomeric canonical molecular graph. Thirty groups are stereochemical variants of one achiral graph and 515 contain distinct achiral canonical graphs. Polymer equivalence remains UNKNOWN.
6. **Metric leverage:** Tg, Tc, and Rg together contribute 76.0% of C128's mean wMAE despite sparse labels; FFV contributes 15.1% and Density 8.9%. FFV overlaps with Tc, Density, and Rg on 270–300 samples, but with Tg on only one sample.
7. **Next evaluation:** Prioritize a separate similarity-disjoint benchmark before further architecture work. Keep multitask versus single-task and FFV transfer as controlled hypotheses; the current data do not settle them. Deprioritize an unqualified 50:50 GNN+tree blend.

## Baseline protocol and limits

- Median is calculated independently by target and fold from observed training labels only.
- kNN uses the top five target-labeled fold-training polymers and direct Tanimoto weights; if all five similarities are zero it falls back to the fold-training target median.
- LightGBM uses 2,510 SMILES-only features and fixed parameters: 400 trees, learning rate 0.04, 18 leaves, 10 minimum child samples, 0.5 column sample. Fold-specific median imputation is fit only on labeled training rows. There is no validation early stopping or tuning.
- The historical Kaggle script's private-label Tg shift, blend weights and all 3D conformer features were excluded. No released/private labels, supplementary data or pseudo-labels were used.
- C256/G0 is included from the Stage 5A aggregate as historical context; local sample-level C256 OOF files were unavailable, so its per-sample bootstrap and similarity strata are not reported.
- All C128 sample-level diagnostics use the five existing seed OOF files. No new GNN training or architecture search was run.

## Files

- `aggregate_summary.json` — all model metrics, weights, diagnostics and provenance.
- `baseline_comparison.csv` — comparison table and paired bootstrap intervals.
- `sample_diagnostics.csv` — per-sample labels, predictions, errors, fold, heavy atoms and max similarity.
- `similarity_strata.csv` — per-target MAE and counts by similarity bin.
- `morgan_duplicate_groups.csv` — all duplicate fingerprint groups and canonicalization evidence.
- `fusion_diagnostics.csv` — per-target OOF error correlations and fixed 50:50 blend results.
