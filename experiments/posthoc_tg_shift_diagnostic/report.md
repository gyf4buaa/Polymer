# Released-test alignment audit

> **POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE**

## Scope and correction

This update audits whether the released `public.csv` / `private.csv` can be mapped to the final Kaggle leaderboard scoring data. It does **not** perform a Tg sweep, search for a private-optimal offset, run inference, retrain, or change the clean C128 result definition. It compares only the already fixed clean predictions and the previously submitted `Tg +70` diagnostic.

**Correction:** the downloadable competition `test.csv` has only 3 preview rows. Its lack of overlap with the 3,502 released rows is expected and is **not evidence of a mismatch**. The preview replay is useful only as an inference implementation check: the formal 25-checkpoint replay matched the successful kernel output on those 3 rows. It cannot identify the hidden/released rows or their score split.

## Released dataset metadata and history

Kaggle's [public dataset page](https://www.kaggle.com/datasets/alexliu99/neurips-open-polymer-prediction-2025-test-data) and [metadata API](https://www.kaggle.com/api/v1/datasets/view/alexliu99/neurips-open-polymer-prediction-2025-test-data) call the files “Datasets used in the public and private leaderboards” and explicitly say `public.csv` is used for the public leaderboard while `private.csv` is used for the private leaderboard. The dataset is public (this visibility setting is separate from “private leaderboard”), owned by `alexliu99`, MIT-licensed, and has a single listed version: version 1, “Initial release”, Ready, created **2025-12-09 04:27:33 UTC**. The [file-list API](https://www.kaggle.com/api/v1/datasets/list/alexliu99/neurips-open-polymer-prediction-2025-test-data) dates the file records about one second later. The captured fields and source API URLs are in [dataset_metadata_audit.json](dataset_metadata_audit.json).

| File | Rows | Bytes | SHA-256 |
|---|---:|---:|---|
| `public.csv` | 295 | 19,513 | `0b5b59f9464fb30da82252cc9e4f56552ff641284c627f11aa93fa4d7031c514` |
| `private.csv` | 3,207 | 208,864 | `55cac160c1c139240968a96ba070b583e93902dd1901be2e54d722c4d590a952` |

The 295/3,502 (8.42%) and 3,207/3,502 (91.58%) split sizes are consistent with Kaggle's description that about 92% of test rows are used for private scoring ([leaderboard](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/leaderboard)). The competition discussion also records later test-data updates and rescoring ([organizer discussion](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/588643)). These support the claimed split meaning and make the released split sizes plausible; they do not independently prove that this version is exactly the final scored snapshot.

## Released target-label statistics

“Missingness” is the share of rows in that split with no observed value for the target. All statistics below are computed directly from the two hash-pinned files. Full precision is in [released_target_statistics.csv](released_target_statistics.csv).

| Split | Target | Rows | Observed | Missingness | Min | Max | Range | Mean | Median |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Public | Tg | 295 | 95 | 67.80% | -138.8021 | 359.0841 | 497.8862 | 102.8840 | 97.7639 |
| Public | FFV | 295 | 86 | 70.85% | 0.3150 | 0.6560 | 0.3410 | 0.3579 | 0.3525 |
| Public | Tc | 295 | 239 | 18.98% | 0.077306 | 0.6780 | 0.600694 | 0.256945 | 0.244302 |
| Public | Density | 295 | 244 | 17.29% | 0.746790 | 1.360334 | 0.613544 | 1.057359 | 1.057126 |
| Public | Rg | 295 | 54 | 81.69% | 11.208168 | 32.301133 | 21.092965 | 17.395581 | 17.435883 |
| Private | Tg | 3,207 | 166 | 94.82% | -86.6500 | 484.6900 | 571.3400 | 179.8207 | 169.6700 |
| Private | FFV | 3,207 | 137 | 95.73% | 0.3060 | 0.4540 | 0.1480 | 0.3561 | 0.3510 |
| Private | Tc | 3,207 | 1,165 | 63.67% | 0.024488 | 0.9540 | 0.929512 | 0.245331 | 0.239328 |
| Private | Density | 3,207 | 1,282 | 60.02% | 0.134543 | 1.732830 | 1.598287 | 1.086132 | 1.081129 |
| Private | Rg | 3,207 | 1,056 | 67.07% | 9.445485 | 35.529550 | 26.084065 | 18.379113 | 18.511292 |

The official public metadata/leaderboard does not expose the final scored target counts and min/max ranges for an independent comparison. The counts and ranges above are therefore plausible candidate scoring statistics only, conditional on this version being the final scoring truth. The released files omit Kaggle's numeric `id`; alignment here uses exact raw SMILES.

## Split and weight scenarios

The fixed clean C128 prediction file is `clean_c128_predictions.csv` (SHA-256 `f00b0b3bc83831ac9e1b6825bd868b40c8ddf22a919575a067bd11b73b8c1f43`). Scores use the pinned formal metric implementation. “Global” means target counts/ranges and weights derived over all 3,502 released rows; “split-specific” derives them separately within each assigned split. “Swapped” deliberately assigns the released private rows to public and released public rows to private. Online comparison columns are offline minus the corresponding known Kaggle score.

Known online scores: Clean Public/Private **0.06899 / 0.09524**; `Tg +70` Public/Private **0.06493 / 0.07758**. The full scenario values and signed differences are in [split_weight_scenarios.csv](split_weight_scenarios.csv).

| Assignment | Weights | Scored split | Clean | +70 | +70 − clean | Clean − online | +70 − online |
|---|---|---|---:|---:|---:|---:|---:|
| Normal | Global | Public | 0.112411 | 0.103587 | -0.008824 | +0.043421 | +0.038657 |
| Normal | Global | Private | 0.051971 | 0.047661 | -0.004310 | -0.043269 | -0.029919 |
| Normal | Split-specific | Public | 0.151137 | 0.142998 | -0.008139 | +0.082147 | +0.078068 |
| Normal | Split-specific | Private | 0.053859 | 0.048899 | -0.004960 | -0.041381 | -0.028681 |
| Swapped | Global | Public | 0.051971 | 0.047661 | -0.004310 | -0.017019 | -0.017269 |
| Swapped | Global | Private | 0.112411 | 0.103587 | -0.008824 | +0.017171 | +0.026007 |
| Swapped | Split-specific | Public | 0.053859 | 0.048899 | -0.004960 | -0.015131 | -0.016031 |
| Swapped | Split-specific | Private | 0.151137 | 0.142998 | -0.008139 | +0.055897 | +0.065418 |

None of the four assignment/weight combinations reproduces the four online scores. Under normal assignment, offline `+70` helps Public by about 0.0081–0.0088 but Private by only 0.0043–0.0050; online changes are -0.00406 Public and -0.01766 Private. Swapping the splits also fails and does not make the score pairs align. No other shifts were evaluated.

## SMILES canonicalization and formal graph semantics

The formal keep-dummy builder parses each SMILES directly with RDKit; it does not canonicalize or reserialize. To test whether an ordinary canonical round-trip could alter the C128 input, a deterministic random sample of 256 rows was combined with all 1,144 rows containing selected stereo/ring/charge/dummy markers (1,317 unique rows). Each was canonicalized with isomeric SMILES enabled, reparsed, rebuilt with the formal keep-dummy graph code, and compared by graph isomorphism over every formal categorical atom and bond feature.

- Raw text changed for 17 rows.
- All **1,317/1,317** feature-labeled graphs were isomorphic after round-trip.
- No parse failures or feature-labeled graph differences occurred.
- The sample included 1,305 two-distinct-endpoint graphs, 9 shared-endpoint graphs, and 3 four-dummy graphs.

This rules out ordinary RDKit canonicalization/re-serialization as a graph-changing explanation in the audited sample. It does not rule out other source-specific transformations outside this round-trip test. The formal graph builder's direct parsing behavior and the round-trip counts are recorded in [alignment_audit.json](alignment_audit.json).

## Alignment conclusion

### Causes ruled out or corrected

- The 3-row preview's lack of overlap is not mismatch evidence; it is expected for a preview and has been removed as a diagnostic inference.
- On the preview rows, the formal 25-checkpoint replay matches the successful kernel output, supporting the inference implementation but not validating the released labels.
- A random-plus-targeted sample of canonical SMILES round-trips preserved every formal keep-dummy atom/bond feature under graph isomorphism.
- Normal and swapped split assignments, with global and split-specific released-label weights, all fail to reproduce the four known scores.

### Remaining possibilities

- The released version may not be byte-for-byte the final leaderboard scoring snapshot, despite the owner's explicit description.
- The released structures cannot be linked to Kaggle numeric IDs, so exact row-to-scored-split identity cannot be independently verified.
- Kaggle's final per-target counts/ranges and effective score-weight scope are not publicly exposed in the reviewed metadata, so the exact final metric inputs cannot be reconstructed from public evidence.
- The documented test-data update/rescoring history leaves open a version or scoring-snapshot difference.

**Conclusion:** the dataset metadata supports the intended Public/Private meaning, and its split sizes fit the approximate official ratio, but these facts do not establish a trusted row-level mapping to the final leaderboard scoring data. The candidate normal/swapped assignments and weight scopes fail the clean and +70 sanity checks. **Released labels cannot be used to exactly reproduce leaderboard scoring.** No Tg sweep or private-optimal search was performed.
