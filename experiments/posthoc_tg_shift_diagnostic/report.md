# Post-hoc Tg shift diagnostic — metric gate failed

> **POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE**

## Stop condition

The clean and Tg +70 offline scores did not reproduce both real Kaggle score pairs within five-decimal display precision under any tested weight scope. The requested Tg shift sweep, split optima, target decomposition, and figures were therefore not computed.

## Fixed inference replay

- Formal source: `72490c1a748ed9395025f4120c9eb04f28268695`; main SHA context: `c9949c088d8822eac775a55ad601d278b747c9f9`.
- Checkpoints loaded and ensembled: **25/25**; each fold used its own checkpoint-embedded normalizer.
- Prediction file SHA-256: `f00b0b3bc83831ac9e1b6825bd868b40c8ddf22a919575a067bd11b73b8c1f43`.
- As an implementation cross-check, the same 25 checkpoints were replayed on the three rows in the downloadable Kaggle `test.csv` preview. Per-target predictions matched the successful kernel's three-row `submission.csv` to floating-point tolerance; the comparison details are in `kaggle_preview_replay.json`.

## Released-data identity limitation

- Dataset: `alexliu99/neurips-open-polymer-prediction-2025-test-data`; public/private files contain 295 / 3207 rows.
- Public SHA-256: `0b5b59f9464fb30da82252cc9e4f56552ff641284c627f11aa93fa4d7031c514`; private SHA-256: `55cac160c1c139240968a96ba070b583e93902dd1901be2e54d722c4d590a952`.
- Neither released truth file contains Kaggle's numeric `id`; this replay joins on exact raw SMILES. The downloadable competition preview has only three rows, and none of its structures matches the released set by raw or RDKit-canonical SMILES. It therefore does not establish an ID-to-SMILES mapping for the released 3,502 rows.

## Fidelity gate results

Known Kaggle values are Public/Private **0.06899 / 0.09524** for Clean and **0.06493 / 0.07758** for Tg +70.

| Weight scope | Public clean | Public +70 | Private clean | Private +70 | Pass |
|---|---|---|---|---|---|
| global_released_test_weights | 0.11241126 | 0.10358741 | 0.05197062 | 0.04766102 | False |
| split_specific_released_test_weights | 0.15113685 | 0.14299827 | 0.05385928 | 0.04889907 | False |
| official_training_snapshot_weights | 0.16745804 | 0.15975399 | 0.09061884 | 0.08685616 | False |

## Discrepancy diagnosis

The frozen model inference reproduces the successful kernel output on the available three-row Kaggle preview, so this does not point to a checkpoint, graph preprocessing, fold scaler, or ensemble-loading error. The official weighted-MAE formula was evaluated with global released-test weights, split-specific released-test weights, and the frozen official-train weights; none reproduces all four scored values. The remaining mismatch is at the scored-data / leaderboard-metric alignment boundary. The released files lack numeric Kaggle IDs, and none of the three structures in the downloadable preview matches the released set, even after RDKit canonicalization. The available artifacts therefore cannot distinguish a released truth snapshot mismatch from a difference in the leaderboard's effective scoring weights or scored rows.

This is the stopping point required by the predeclared gate. No private-score-informed shifts or curves were computed.
