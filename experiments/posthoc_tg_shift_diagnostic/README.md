# Post-hoc Tg shift diagnostic

> **POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE**

This separate experiment measures the effect of adding a constant to Tg predictions from the already-scored Clean Own-GNN C128 ensemble. It does not change the clean C128 score definition, Stage 5A artifacts, or Stage R results. No model is retrained, and no checkpoint, graph preprocessing, ensemble rule, or non-Tg prediction is changed.

## Fixed model and data

- Formal source: `72490c1a748ed9395025f4120c9eb04f28268695`.
- Model: keep-dummy graph, four-layer GINE, hidden width 128, mean|max pooling.
- Models: seeds 42–46, folds 0–4; each prediction is inverse-transformed with the normalizer embedded in its own formal fold checkpoint, then the 25 predictions are averaged arithmetically.
- Test labels: Kaggle dataset `alexliu99/neurips-open-polymer-prediction-2025-test-data`; only the released `public.csv` and `private.csv` are read. Those files have no numeric Kaggle `id` column. The analysis uses exact raw SMILES as the best available join key and writes that key into `clean_c128_predictions.csv`'s `id` column. This mapping is not verified against the hidden numeric IDs. The prediction table is diagnostic output, not a Kaggle submission file.
- Released label files are not committed. Their SHA-256 values and row counts are recorded in `provenance.json`.

## Metric fidelity gate

The runner first compares clean and Tg +70 scores against the scored Kaggle submissions:

- Clean ref `56831294`: Public `0.06899`, Private `0.09524`.
- Tg +70 ref `56831959`: Public `0.06493`, Private `0.07758`.

It tests the official weighted-MAE formula with global released-test weights, split-specific released-test weights, and the frozen official-train weights, and requires all four results to match Kaggle's five-decimal display precision. If no single scope matches all four, it writes `metric_fidelity_gate.json` and a gate-failure `report.md` / `provenance.json`, then stops before creating a shift sweep or figures. The current recorded run failed this gate; its report documents the discrepancy and no optimum or shift curve is claimed.

The [Kaggle competition evaluation definition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) gives `w_i = (1/r_i) * (K * sqrt(1/n_i) / sum_j sqrt(1/n_j))`; each split score is the per-sample mean of weighted absolute errors with blank labels masked. The selected scope and all tested alternatives are retained in the gate JSON.

## Reproduce

Use an environment with PyTorch, PyTorch Geometric, RDKit 2026.03.2, NumPy, and Matplotlib. Run the script from this repository with the **formal-source checkout**, released CSVs, and checkpoint cache supplied explicitly:

```bash
python experiments/posthoc_tg_shift_diagnostic/run_diagnostic.py \
  --formal-source <checkout-at-72490c1a748ed9395025f4120c9eb04f28268695> \
  --public-csv <released-test-data>/public.csv \
  --private-csv <released-test-data>/private.csv \
  --checkpoint-root <formal-c128-checkpoint-cache> \
  --checkpoint-manifest <formal-c128-checkpoint-cache>/checkpoint_manifest.json \
  --output-dir experiments/posthoc_tg_shift_diagnostic
```

The default run rebuilds graphs and performs inference once with the 25 fixed checkpoints. `--skip-inference` reuses an existing `clean_c128_predictions.csv` and `inference_provenance.json` for metric-only analysis. The script never trains or updates a model.

## Outputs

- `clean_c128_predictions.csv`: full released-test C128 predictions, identifier key, and public/private split.
- `inference_provenance.json`: source, checkpoint/scaler hashes, environment, row counts, and prediction hash.
- `metric_fidelity_gate.json`: clean/+70 offline score replay and gate result.
- If the gate passes: `shift_scores.csv`, `target_contributions.csv`, `residual_statistics.csv`, `diagnostic_summary.json`, and both requested figures are generated.
- On the current gate-failed run: `kaggle_preview_replay.json` records the three-row inference cross-check; `metric_fidelity_gate.json`, `provenance.json`, and `report.md` record why the sweep was stopped.
