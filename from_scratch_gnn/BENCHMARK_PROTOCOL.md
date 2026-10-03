# Benchmark Protocol

This file defines the comparison contract for the from-scratch GNN track.

Once the first formal internal baseline has been run, changes to this protocol should be treated as a benchmark-version change rather than silently modifying historical results.

## 1. Data identity

For every benchmark version, record:

- source training files;
- row count;
- target columns;
- file SHA256 hashes;
- preprocessing code commit.

Do not mix results produced from different data snapshots in the same comparison table without marking the benchmark version.

## 2. Fixed five-fold split

Create one persistent fold-assignment file containing at least:

```text
sample_id,fold
...,0
...,1
...
```

All formal models use exactly these assignments.

The initial benchmark should use the split strategy chosen to reproduce the competition-style development setting. A later structure-disjoint or similarity-grouped benchmark may be added as a separate benchmark version, but it must not replace the original folds silently.

## 3. OOF definition

OOF means that every training sample is predicted exactly once by a model that did not train on that sample.

For fold k:

```text
train = all folds except k
valid = fold k
```

After all five runs, concatenate the five validation predictions in original sample order.

Do not report refit-on-all-data predictions as OOF.

## 4. Main metric

The main local model-selection metric is competition-style weighted MAE.

The implementation must be centralized in one metric module and shared by all experiments.

In addition to the aggregate score, always report raw per-target MAE for:

- Tg;
- FFV;
- Tc;
- Density;
- Rg.

A formal result is incomplete if only the aggregate metric is reported.

## 5. Missing labels

Missing target labels are masked.

They must not be filled with artificial zeros or included in the loss/metric denominator.

For each target, the metric is computed only on samples with a valid ground-truth label.

## 6. Fold-local preprocessing

Anything that learns statistics from data must be fit on the training portion of the current fold only.

This includes, when applicable:

- target normalization;
- feature normalization;
- descriptor imputation;
- feature selection;
- dimensionality reduction;
- learned calibrators.

The fitted object may then transform that fold's validation samples.

## 7. Model initialization

For from-scratch experiments:

- no third-party pretrained competition weights;
- no validation information in initialization;
- no Private-label-derived adjustment.

Third-party architectures may be reimplemented/retrained as external references, but their pretrained/refit weights are not valid for the formal OOF comparison.

## 8. Early stopping and model selection

Within each fold:

1. train only on the training split;
2. evaluate on the held-out fold;
3. select the checkpoint using the predetermined validation criterion;
4. restore the best checkpoint;
5. generate the held-out predictions.

The early-stopping rule, patience, and monitored metric must be recorded in the experiment config.

## 9. Experiment identity

Every formal run must record:

```text
experiment_id
model_name
git_commit
benchmark_version
seed
fold
config
start/end state
best_epoch
best_validation_metric
checkpoint_path
```

At experiment completion, produce:

```text
oof_predictions.csv
metrics.json
fold_metrics.csv
config.yaml
training_history/
checkpoints/
```

Large checkpoints should not be committed directly to GitHub.

## 10. External reference comparability

The third-party GATv2 architecture is an external reference architecture, not the internal baseline or base model for Own-GNN. After Own-GNN v0 is frozen, later Own-GNN variants are judged primarily against Own-GNN v0 or the immediately preceding accepted Own-GNN variant.

For the formal OOF comparison it should be:

- initialized from scratch;
- trained on the same folds;
- evaluated by the same metric code;
- subject to the same leakage rules.

Architecture-specific hyperparameters may remain architecture-specific, but the data split and scoring contract are fixed.

## 11. Development versus ablation

During development, many exploratory experiments may be run.

The final ablation is performed only after the final model is chosen.

A final ablation removes one retained component at a time while keeping the rest of the final configuration fixed.

## 12. Multi-seed confirmation

Single-seed experiments are acceptable during fast iteration.

Before making a final claim, rerun the key comparison using multiple seeds and report:

```text
mean OOF wMAE ± standard deviation
```

At minimum, confirm:

- Own-GNN v0 internal baseline;
- Own-GNN final.

Multi-seed confirmation of the third-party external reference is optional unless a formal statistical comparison against that external model is specifically required.

## 13. Kaggle Private

Kaggle Private is an external hidden-test benchmark, not the local hyperparameter-tuning metric.

The intended order is:

```text
OOF development
→ ablation
→ multi-seed confirmation
→ freeze config + commit
→ test inference
→ Kaggle Private evaluation
```

Do not repeatedly modify the model in response to Private score without explicitly labeling that process as Private-set tuning.

## 14. Results registry

Every formal result should be appended to `results.csv`. Use `internal_baseline` for Own-GNN v0, `own_model` (or a more specific owned-model category) for later variants, and `external_reference` / `historical_reference` for contextual models that are not the development baseline.

Never overwrite an older row merely because a newer run performs better. The registry should preserve the experimental path, including failed or neutral changes when they are scientifically informative.


---

## Stage 0 frozen snapshot: nopp2025_train_v1

The initial local snapshot is frozen as follows:

- Official source: local data workspace data/competition_raw/train.csv; raw SHA-256 1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1.
- Source schema: id, SMILES, Tg, FFV, Tc, Density, Rg; canonical view: sample_id, SMILES, Tg, FFV, Tc, Density, Rg.
- Rows: 7,973. id is renamed to sample_id; rows and raw SMILES are retained without deduplication.
- Supplements and released public/private files are excluded. The raw training CSV is not committed.
- Audit result: 0 missing/duplicate IDs, 0 missing SMILES, 0 exact duplicate SMILES, 0 canonicalized duplicate SMILES, 0 RDKit-invalid SMILES (checked with RDKit 2026.03.2), 0 rows with all five targets missing. No anomaly is silently removed. See benchmark/diagnostics.json.

### Frozen split

All formal experiments use benchmark/folds.csv, SHA-256 1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a.

- Five folds, seed 20250604 (the existing modeling/src/config.py RANDOM_SEED), unstratified sample-level random split.
- To make assignments independent of row order and software versions, sort IDs by the SHA-256 digest of UTF-8 bytes for “polymer-stage0-sha256-rank-v1”, followed by a NUL byte, the decimal seed, another NUL byte, and the sample ID; assign sequential balanced chunks to folds 0–4.
- Sample counts by fold: 1,595; 1,595; 1,595; 1,594; 1,594.
- Per-fold valid label counts are stored in benchmark/data_manifest.json.
- The historical `modeling/src/cv.py` uses target-specific folds (quantile-stratified where possible). Those per-target folds do not satisfy this shared multi-target contract; historical OOF scores must be recomputed on the frozen map before formal comparison. No structure-similarity grouping or GroupKFold is used in this benchmark version.

### Frozen metric and training-snapshot weights

src/metrics.py::evaluate_oof is the only implementation of both weight calculation and score aggregation. The [official Kaggle evaluation rule](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025) is:

\[
w_i = \frac{1}{r_i}\cdot
\frac{K\sqrt{1/n_i}}{\sum_{j=1}^{K}\sqrt{1/n_j}},
\qquad
\mathrm{wMAE} = \frac{1}{N}\sum_{s=1}^{N}
\sum_{i:\, y_{s,i}\mathrm{\ observed}}w_i|\hat y_{s,i}-y_{s,i}|.
\]

Here, \(K=5\), \(n_i\) is the count of available labels, \(r_i=\max(y_i)-\min(y_i)\), and \(N\) is the number of samples being scored. Missing truths are masked from both the target MAE numerator and denominator; the aggregate follows the competition's per-sample outer mean. In particular, overall wMAE is not the simple mean of the five raw MAEs. For this snapshot, N = 7,973 and sum_j sqrt(1/n_j) = 0.1737459602374502.

Kaggle uses hidden-test counts and ranges when scoring a submission. The local development metric cannot read hidden/private labels, so this frozen version uses only the full official training CSV for \(n_i\) and \(r_i\). These values are fixed in benchmark/data_manifest.json; they are used for all full OOF comparisons and should be passed as target_weights when a fold-local validation metric is calculated.

| Target | \(n_i\) | \(r_i\) | \(w_i\) |
|---|---:|---:|---:|
| Tg | 511 | 620.2797376 | 0.00205237749659811 |
| FFV | 7,030 | 0.55010467 | 0.623924865972491 |
| Tc | 737 | 0.4775 | 2.21997543529438 |
| Density | 613 | 1.092307675 | 1.06409417613206 |
| Rg | 614 | 24.944550505 | 0.0465581184297421 |

The validator aligns OOF rows by sample_id, requires exact training-sample coverage, and writes overall OOF wMAE plus Tg/FFV/Tc/Density/Rg MAE to metrics.json. It does not train models or read Kaggle submissions. Implementation and usage details are in benchmark/README.md.
