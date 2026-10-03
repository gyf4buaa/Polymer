# Stage 3B — Message-passing operator ablation

This package fixes the Stage 3A.1 Variant A graph representation (`keep_dummy`)
and replaces only the message-passing operator. It reuses the frozen Stage 0
folds, feature columns, input embeddings, outer residual/ReLU/dropout/LayerNorm
sequence, mean+max pooling, five property heads, loss, optimizer, and stopping
protocol. The historical GINE results are inputs to the aggregate and are
never retrained here.

## Fixed models

- GATv2: four heads × 64 channels, concatenated to width 256; edge-aware
  `GATv2Conv`, with generated self-loops disabled and attention dropout zero so
  the frozen 0.1 dropout remains outside the block.
- PNA: `mean`, `min`, `max`, `std` aggregators; `identity`, `amplification`,
  `attenuation` scalers; one tower; edge-aware `PNAConv`.
- Both use hidden width 256 and four layers. Their operator-internal
  projections consume the same 16-column encoded bond features. Parameter
  counts are reported, not forced to match.

PNA's degree histogram is computed once per run over every graph in the frozen
benchmark graph set. It uses directed in-degree from `edge_index[1]` and graph
topology only; no labels or targets enter it. `degree_histogram.json` records
the counts, graph schema and fingerprint, train-data hash, source commit, and
degree-statistics generator hash.

## Checks

```bash
python -m pytest -q from_scratch_gnn/models/operator_ablation/tests
python -m pytest -q from_scratch_gnn/tests/test_aggregate_operator_ablation.py
```

The tests cover the historical GINE forward path, fixed hidden width and graph
schema, edge-aware forwards, backward/loss reduction, checkpoint round trips,
operator config invariants, seed/artifact isolation, deterministic PNA degree
statistics, and paired aggregation.

Run the CPU tiny-overfit smoke against the existing, hash-verified keep-dummy
graph cache (the target values are synthetic and are not used for selection):

```bash
python -m from_scratch_gnn.models.operator_ablation.cpu_smoke --operator gatv2
python -m from_scratch_gnn.models.operator_ablation.cpu_smoke --operator pna
```

## Run locations

Each operator has its own config and artifact namespace:

```text
models/operator_ablation/gatv2/config.json
models/operator_ablation/gatv2/artifacts/seed_42/
models/operator_ablation/pna/config.json
models/operator_ablation/pna/artifacts/seed_42/
```

Smoke outputs belong under `artifacts/smoke/`; formal outputs never share a
smoke directory. The training adapter reuses the unmodified Stage 0 engine and
the keep-dummy graph builder. Its artifact config, source manifest, run
metadata, operator parameter counts, OOF predictions, metrics, fold metrics,
histories, and compressed graph diagnostics are kept per run. Checkpoints and
graph caches are runtime-only files.

The formal design is exactly seeds 42–46 for GATv2 and PNA. After all ten runs
complete, build the paired report with:

```bash
python -m from_scratch_gnn.scripts.aggregate_operator_ablation \
  --source-commit <formal-source-commit>
```

The aggregate validates hashes, OOF coverage, source commits, unique seed
artifacts, and the identical PNA degree histogram before calculating paired
overall and per-target differences against the stored Stage 3A.1 GINE scores.
