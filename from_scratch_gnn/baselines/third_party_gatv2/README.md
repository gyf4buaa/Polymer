# Third-party GATv2 reference (Stage 1A)

This is a from-random-initialization OOF retraining of the public third-place
GATv2 architecture. It is evaluated on the Stage 0 `benchmark/folds.csv` and
scored with the authoritative `from_scratch_gnn/src/metrics.py` implementation
through `from_scratch_gnn/scripts/evaluate_oof.py`.

## Public reference and reproduced architecture

Source: [fresnellll/kaggle-NeurIPS-polymer-prediction-solution](https://github.com/fresnellll/kaggle-NeurIPS-polymer-prediction-solution),
commit `f385d220d348283792c9f3dc8ed4ab0619e6f7c4`; see its `src/prepare_data.py`
and `src/train.py`. The author’s MIT license and attribution apply to the
reimplemented reference architecture.

- Atom features: atomic number, degree, formal charge, radical electron count,
  RDKit hybridization enum, aromatic flag, and total hydrogen count.
- Edge features: bond order and conjugation flag. Bonds are directed both ways;
  when exactly two `*` atoms occur, a directed periodic edge joins them and is
  marked `[-1, 0]`.
- Backbone: six edge-aware PyG `GATv2Conv` layers, hidden width 384, eight
  concatenated heads (48 dimensions per head), attention dropout 0.2. Every
  layer is followed by `BatchNorm` and ELU; the five hidden layers use
  element-wise residual addition. Readout is global mean pooling.
- Morgan branch: radius 2, 1,024 bits. For each target, the top 50 bits by
  `SelectKBest(f_regression)` are selected and concatenated with the graph
  embedding.
- Five independent heads: `Linear(384 + k, 384) -> ReLU -> Dropout(0.2) ->
  Linear(384, 1)`.
- Training defaults from the public source: 600 maximum epochs, batch 64,
  AdamW at `1e-4`, early-stop patience 40, masked weighted MAE objective.
  Model initialization uses reproducible seeds 42 through 46 by fold; 42 is the
  source project’s seed, while frozen Stage 0 folds replace its original KFold.

The complete machine-readable run configuration is written to
`artifacts/config.json` for each run. The requested device is recorded in that
file; production runs on the RTX 4070 use CUDA.

The Mac CPU pilot artifacts are exploratory only and are retained for
provenance. They do not contribute checkpoints or predictions to the formal
result. The formal result starts from random initialization for all five frozen
folds, on CUDA, from the committed source revision.

## Necessary benchmark/leakage corrections

The public preparation script merges CSVs under its raw-data directory,
deduplicates by SMILES, selects Morgan bits before CV, and expands repeat units
before random splitting. The public trainer then splits those graph rows and
fits a linear calibrator on each validation fold before full-data refit. Those
steps do not satisfy this benchmark’s sample-keyed OOF contract.

This reference therefore:

1. Reads only Stage 0’s official `train.csv` snapshot and checks its SHA-256;
   it does not read supplements or released test labels.
2. Applies the frozen, shared fold assignment to original `sample_id`s before
   generating any repeat-unit augmentation. Augmented graphs and copied labels
   enter only their source sample’s training fold. Validation and OOF use the
   original sample graph once, without TTA.
3. Fits each task’s Morgan bit selector on observed labels from that fold’s
   training partition only. The selection indices are saved inside the
   corresponding best checkpoint.
4. Fits no validation calibrator, performs no full-data refit, fold ensemble,
   released-label shift, or Private-score adjustment.
5. Derives the optimization-loss weights from the current training partition.
   Fold validation and final OOF use the Stage 0 full-snapshot metric weights,
   as required by the frozen scoring contract.

No invalid original sample is dropped. Any failure to build an optional
augmented view is recorded in `artifacts/preprocessing_diagnostics.json`, and
the original training graph remains in use.

## Run

Set a path to the external official training CSV; the file itself is not stored
in Git. From the repository root:

```bash
python -m from_scratch_gnn.baselines.third_party_gatv2.train_oof \
  --train-csv /path/to/data/competition_raw/train.csv --smoke-test
python -m from_scratch_gnn.baselines.third_party_gatv2.train_oof \
  --train-csv /path/to/data/competition_raw/train.csv
```

The second command uses the full public-source training schedule. For a
production run, pass an explicit output directory and `--device cuda`. It
writes OOF predictions, Stage 0 validator metrics, fold metrics, configuration,
run metadata, and per-epoch histories below that directory. Best checkpoints
and smoke-test files are also written there. Runtime artifacts are Git-ignored.
A formal run appends one row to `from_scratch_gnn/results.csv`.

Parsed graph views are cached under `artifacts/cache/` with the training-file
SHA, RDKit version, and graph-schema version; this cache is Git-ignored. A
matching cache can be reused by linking it into a separate production output
directory.

The reference implementation is intentionally not a hyperparameter search.
See `artifacts/config.json`, `artifacts/run_metadata.json`, and
`artifacts/fold_metrics.csv` for the recorded run details.
