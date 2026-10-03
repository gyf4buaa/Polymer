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

CUDA smoke passed on RTX 4070: fold 0, five epochs, best validation wMAE
0.03043654 at epoch 5, 6.85 seconds, 126.8 MiB peak PyTorch allocation, and a
saved checkpoint. This score is not used for model selection.

The formal frozen five-fold run passed Stage 0 OOF validation:

| Metric | Result |
|---|---:|
| OOF wMAE | 0.0227728722 |
| Tg MAE | 51.50385374 |
| FFV MAE | 0.00582648 |
| Tc MAE | 0.02412187 |
| Density MAE | 0.02529545 |
| Rg MAE | 1.61020157 |
| Runtime | 581.63 s |
| GPU utilization mean / max | 36.5% / 42% (39 samples) |
| Peak total GPU memory / PyTorch allocated | 2,261 / 141.8 MiB |

| Fold | Best epoch | Validation wMAE |
|---:|---:|---:|
| 0 | 58 | 0.02189877 |
| 1 | 78 | 0.02044939 |
| 2 | 63 | 0.02250620 |
| 3 | 54 | 0.02451637 |
| 4 | 50 | 0.02449581 |

The graph audit covered 7,973/7,973 samples, retained 15,968
atomic-number-zero dummy nodes, and reported zero graph fallbacks, dropped
rows, closure edges, or polymerization edges. No NaN or OOM occurred. The run
used source commit `364e69f37059daf1ffac3b09cadf41ce60afe6a1`, training SHA256
`1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1`, folds
SHA256 `1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`,
and config SHA256 `c4e34979ff01692b6b02016299dceeb6451c53953f8339ab55aaf33263079d13`.
Full checkpoints remain on the RTX worktree; compact formal result and
provenance artifacts are also copied under this variant's
`artifacts/production_gpu_20261003/` on the Mac. The aggregate score is only
0.00013415 below v0 and is a single-seed development signal, not a final-model
selection.
