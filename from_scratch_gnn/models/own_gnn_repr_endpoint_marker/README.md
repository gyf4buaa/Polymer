# Own-GNN Variant B — endpoint marker, no closure

For exactly two degree-one dummy atoms with distinct real neighbors, this
representation removes the two dummies, keeps every original real-atom
chemical bond, and sets `polymer_endpoint=1` on those two real atoms. It does
not connect the endpoints. If they were already adjacent, their ordinary
chemical bond remains once. The polymerization-edge feature is always zero.

No-dummy inputs are retained as-is with all endpoint bits zero. Ambiguous
dummy topologies (one, three, or more dummies; non-degree-one dummies; two
dummies sharing an endpoint; or dummy-to-dummy adjacency) retain every source
atom and chemical bond and record a deterministic reason. Invalid SMILES are
reported as graph-build errors and never silently dropped.

The same model, GINEConv layers, hidden width, pooling, heads, loss, optimizer,
learning rate, weight decay, batch size, stopping rule, normalization, seed,
frozen five folds, and Stage 0 metric are used as in Own-GNN v0. The model
receives the endpoint identity as a separate binary node feature. This
variant's graph schema is
`own_gnn_repr_endpoint_marker_no_closure_node7_edge16_endpoint1_v1`.

## Checks

```bash
python -m pytest -q from_scratch_gnn/models/own_gnn_repr_endpoint_marker/tests
python -m from_scratch_gnn.models.own_gnn_repr_endpoint_marker.train_oof \
  --train-csv /path/to/train.csv --device cpu --tiny-overfit
python -m from_scratch_gnn.models.own_gnn_repr_endpoint_marker.train_oof \
  --train-csv /path/to/train.csv --device cuda --smoke-fold 0 --epochs 5
```

Formal outputs go to `artifacts/production_gpu_20261003/` and contain the
standard OOF predictions, metrics, fold summaries, configs, source manifest,
run metadata, per-sample graph diagnostics, histories, and checkpoints. This
variant is registered as `own_model`; its result never replaces the v0 row.

## Stage 3A result

CPU tiny-overfit passed: masked Huber loss fell from 0.34953839 to 0.00224609
in 120 steps, with an exact checkpoint round trip. The five-epoch CPU fold-0
smoke passed with best validation wMAE 0.03110151 at epoch 5 (24.81 seconds of
fold training). This is only an execution check, not an OOF result.

CUDA smoke passed on RTX 4070: fold 0, five epochs, best validation wMAE
0.03065169 at epoch 5, 6.16 seconds, 122.7 MiB peak PyTorch allocation, and a
saved checkpoint. This score is not used for model selection.

The formal frozen five-fold run passed Stage 0 OOF validation:

| Metric | Result |
|---|---:|
| OOF wMAE | 0.0230024716 |
| Tg MAE | 54.05964086 |
| FFV MAE | 0.00583066 |
| Tc MAE | 0.02452482 |
| Density MAE | 0.02488108 |
| Rg MAE | 1.56622578 |
| Runtime | 635.45 s |
| GPU utilization mean / max | 37.4% / 44% (43 samples) |
| Peak total GPU memory / PyTorch allocated | 2,252 / 136.9 MiB |

| Fold | Best epoch | Validation wMAE |
|---:|---:|---:|
| 0 | 99 | 0.02191349 |
| 1 | 70 | 0.02067256 |
| 2 | 49 | 0.02329773 |
| 3 | 96 | 0.02436420 |
| 4 | 35 | 0.02476635 |

The graph audit covered 7,973/7,973 samples. It marked 7,940 valid graphs
(15,880 endpoint nodes), retained 88 dummy nodes in 33 lossless fallback
graphs, preserved all 1,244 existing chemical bonds between endpoint pairs,
and added zero closure/polymerization edges. No rows were dropped. No NaN or
OOM occurred. The run used source commit
`364e69f37059daf1ffac3b09cadf41ce60afe6a1`, training SHA256
`1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1`, folds
SHA256 `1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`,
and config SHA256 `42ad5aa3550ab1db4c3d13f76cdb68cbc7ffa61b77726cc19dcf9961c5636b88`.
Full checkpoints remain on the RTX worktree; compact formal result and
provenance artifacts are also copied under this variant's
`artifacts/production_gpu_20261003/` on the Mac. The score is only 0.00009545
above v0 and is not a statistically conclusive representation comparison.
