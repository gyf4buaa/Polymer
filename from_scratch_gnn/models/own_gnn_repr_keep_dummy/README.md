# Own-GNN Variant A — keep dummy atoms

This representation ablation preserves the complete RDKit graph for each
repeat-unit SMILES. Both `*` atoms and all original chemical bonds remain in
the graph. No endpoint closure edge or endpoint marker is added. Atomic number
zero uses the existing v0 atomic-number category 0, a valid learned embedding
bucket.

The model, GINEConv layers, hidden width, pooling, heads, loss, optimizer,
learning rate, weight decay, batch size, stopping rule, normalization, seed,
frozen five folds, and Stage 0 metric are shared with Own-GNN v0. The shared
runner calls the unchanged v0 training engine and writes artifacts under this
variant's `artifacts/` directory. Its independent schema is
`own_gnn_repr_keep_dummy_raw_graph_node7_edge16_endpoint1_v1`.

Graph diagnostics report dummy counts, atomic-number-zero nodes, topology,
preserved endpoint bonds, and graph coverage for every source sample. Invalid
SMILES raise an explicit error; no sample is silently dropped.

## Checks

```bash
python -m pytest -q from_scratch_gnn/models/own_gnn_repr_keep_dummy/tests
python -m from_scratch_gnn.models.own_gnn_repr_keep_dummy.train_oof \
  --train-csv /path/to/train.csv --device cpu --tiny-overfit
python -m from_scratch_gnn.models.own_gnn_repr_keep_dummy.train_oof \
  --train-csv /path/to/train.csv --device cuda --smoke-fold 0 --epochs 5
```

Formal outputs go to `artifacts/production_gpu_20261003/` and contain the
standard OOF predictions, metrics, fold summaries, configs, source manifest,
run metadata, graph diagnostics, histories, and checkpoints. This variant is
registered as `own_model`; its result never replaces the v0 registry row.

## Stage 3A result

CPU tiny-overfit passed: masked Huber loss fell from 0.34745914 to 0.00430032
in 120 steps, with an exact checkpoint round trip. The five-epoch CPU fold-0
smoke passed with best validation wMAE 0.02976372 at epoch 5 (27.95 seconds of
fold training). This is only an execution check, not an OOF result.

The full graph audit covered 7,973/7,973 samples, retained 15,968
atomic-number-zero dummy nodes, and reported zero graph fallbacks, dropped
rows, closure edges, or polymerization edges. The RTX 4070 was occupied at the
first availability check, so CUDA smoke and formal OOF training were not
started. See the Stage 3A status in the track README for the blocker and
cross-variant graph audit.
