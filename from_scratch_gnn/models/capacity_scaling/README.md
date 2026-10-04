# Stage 5A — Pure Capacity Scaling

This package tests only G0's GINE `hidden_dim` at 128, 256, 384, and 512.
All widths use the accepted keep-dummy graph builder, the existing
`OwnGNNRepresentation` model, four GINE layers, mean+max pooling, five property
heads, and the frozen Own-GNN training engine. C256 is historical G0 and is
read by the aggregator without a new training job.

The config guard compares graph, model, benchmark, and training sections with
the G0 C256 config. Across widths, generated configs may differ only at
`model.hidden_dim` and `experiment_id`; seed changes only affect `seed`.
`parameter_audit.py` records the parameter-group totals and machine-readable
config diffs.

## Preflight

Run the full from-scratch GNN suite and CPU tiny-overfit for the three new
widths before recording the formal source commit:

```bash
python -m pytest -q from_scratch_gnn
python -m from_scratch_gnn.models.capacity_scaling.parameter_audit
python -m from_scratch_gnn.models.capacity_scaling.runner \
  --hidden-dim 128 --train-csv /path/to/train.csv --device cpu --tiny-overfit
```

Repeat the final command for widths 384 and 512. Tiny-overfit outputs are
execution checks only and do not select a width.

## CUDA runs

After committing the frozen source, run the C512+C384 two-slot smoke pilot:

```bash
python -m from_scratch_gnn.scripts.run_stage5a_queue --pilot \
  --train-csv /path/to/train.csv --concurrency 2
```

Then run the complete preregistered queue. Each job is one full width×seed
five-fold OOF process; the queue dynamically fills two slots and writes one
`registry_row.json` per run. No formal run appends to `results.csv`; the
aggregator updates that registry once after all 15 new jobs pass.

```bash
python -m from_scratch_gnn.scripts.run_stage5a_queue \
  --train-csv /path/to/train.csv --concurrency 2
python -m from_scratch_gnn.scripts.aggregate_stage5a
```

Formal outputs are isolated under `artifacts/formal/C{width}/seed_{seed}/`.
Checkpoints and graph caches are ignored by Git. The compact OOF, metrics,
fold summaries, histories, configs, provenance, and per-run registry rows are
preserved for aggregation and reporting.
