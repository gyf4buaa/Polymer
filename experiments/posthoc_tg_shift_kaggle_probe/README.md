# POST-HOC / LEADERBOARD-PROBED / NOT VALID BLIND PERFORMANCE

> **POST-HOC / LEADERBOARD-PROBED / NOT VALID BLIND PERFORMANCE**

This is a small leaderboard diagnostic for the already-trained Clean Own-GNN C128 ensemble. It measures the online Kaggle score response to fixed Tg offsets. It is not a clean or blind performance estimate, and must not be presented as one.

## Frozen experiment

- Main checkout: `c9949c088d8822eac775a55ad601d278b747c9f9`
- Formal C128 source: `72490c1a748ed9395025f4120c9eb04f28268695`
- Model: Stage 5A C128 Own-GNN, keep-dummy graph, four-layer GINE, hidden dimension 128, mean|max pooling
- Ensemble: seeds 42–46, five folds per seed, arithmetic mean of all 25 fold models
- Kaggle asset dataset: `safaaa46/clean-own-gnn-c128-5seed-5fold-formal-20261004`
- The checkpoints, model, graph processing and ensemble were held fixed. Each probe changed only `submission["Tg"] += DELTA`.
- No retraining, external or supplementary labels, released/private labels, tree model, or changes to FFV/Tc/Density/Rg were used.
- Probe notebooks ran with Internet disabled on a Tesla T4. Their runtime reports assert 25/25 models loaded, row/ID order preserved, other four targets equal to the clean output, a constant Tg shift, and finite predictions.

## Kaggle results

Scores are from the read-only Kaggle competition submissions listing. Deltas are relative to the Clean C128 result. Lower is better.

| Tg shift | Submission ref | Kernel | Public | Δ Public | Private | Δ Private | Stage |
|---:|---:|---|---:|---:|---:|---:|---|
| 0 | `56831294` | `clean-own-gnn-c128-5seed-5fold` v5 | 0.06899 | +0.00000 | 0.09524 | +0.00000 | Clean reference |
| +40°C | `56840734` | `clean-own-gnn-c128-tg-40c-diagnostic` v2 | **0.06417** | **−0.00482** | 0.08232 | −0.01292 | Preregistered |
| +70°C | `56831959` | `clean-own-gnn-c128-posthoc-tg70-diagnostic` v4 | 0.06493 | −0.00406 | 0.07758 | −0.01766 | Known diagnostic reference |
| +100°C | `56840820` | `clean-own-gnn-c128-tg-100c-diagnostic` v1 | 0.06911 | +0.00012 | **0.07570** | **−0.01954** | Preregistered |
| +115°C | `56840967` | `clean-own-gnn-c128-tg-115c-diagnostic` v1 | 0.07223 | +0.00324 | 0.07597 | −0.01927 | Sole permitted follow-up |
| +130°C | `56840874` | `clean-own-gnn-c128-tg-130c-diagnostic` v1 | 0.07609 | +0.00710 | 0.07657 | −0.01867 | Preregistered |

Exact messages for the four new submissions were `Diagnostic: Clean C128 Tg +40C`, `Diagnostic: Clean C128 Tg +100C`, `Diagnostic: Clean C128 Tg +130C`, and `Diagnostic: Clean C128 Tg +115C`.

## Readout

- The lowest sampled Public score is at **+40°C**. The observed low region is around 0–70°C; Public has already worsened by +70°C and is clearly worse by +100°C.
- The lowest sampled Private score is at **+100°C**. Private improved from +70°C to +100°C, then worsened at +115°C and +130°C. The one follow-up at +115°C was selected because the preregistered +100°C and +130°C points bracketed that rise; it did not beat +100°C.
- The sampled Public and Private minima are separated: **+40°C vs +100°C**, a 60°C difference. This indicates the two leaderboard splits respond differently to this diagnostic offset. These sparse points do not establish a continuous mathematical optimum.
- No additional offset was submitted after the single +115°C follow-up. The experiment does not decompose the leaderboard score by target.

The initial +40°C notebook version 1 failed a strict post-CSV float round-trip assertion (maximum parse difference `8.3267e-17`) and was **not submitted**. The assertion was corrected; version 2 passed, and ref `56840734` is the only +40°C competition submission.

## Files

- [`tg_shift_kaggle_probe.csv`](tg_shift_kaggle_probe.csv): one row per reference/probe point, with scores and deltas.
- [`tg_shift_kaggle_probe.png`](tg_shift_kaggle_probe.png): Public and Private score curves against numeric Tg shift.
- [`probe_provenance.json`](probe_provenance.json): Kaggle refs/scores, notebook and run-report SHA256s, source SHAs, assertions and output hashes.
- `kernels/`: source notebooks and Kaggle `kernel-metadata.json` for the four new probe runs.
- `runs/`: successful runtime assertion reports for the four new probe runs.

The Code Competition runtime exposed a three-row preview in the retained run reports; the values in this table are the completed leaderboard scores returned by Kaggle. No released test labels were used to produce or select these leaderboard probes.
