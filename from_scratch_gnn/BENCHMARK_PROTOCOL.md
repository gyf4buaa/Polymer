# Benchmark Protocol

This file defines the comparison contract for the from-scratch GNN track.

Once the first formal baseline has been run, changes to this protocol should be treated as a benchmark-version change rather than silently modifying historical results.

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

Third-party architectures may be reimplemented/retrained as reference baselines, but their pretrained/refit weights are not valid for the formal OOF comparison.

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

## 10. Fair comparison with the third-party reference

The third-party GATv2 architecture is a reference architecture, not the base model for Own-GNN.

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

- the third-party reference;
- Own-GNN baseline;
- Own-GNN final.

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

Every formal result should be appended to `results.csv`.

Never overwrite an older row merely because a newer run performs better. The registry should preserve the experimental path, including failed or neutral changes when they are scientifically informative.
