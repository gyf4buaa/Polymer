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

The full graph audit covered 7,973/7,973 samples. It marked 7,940 valid graphs
(15,880 endpoint nodes), retained 88 dummy nodes in 33 lossless fallback
graphs, preserved all 1,244 existing chemical bonds between endpoint pairs,
and added zero closure/polymerization edges. No rows were dropped. The RTX
4070 was occupied at the first availability check, so CUDA smoke and formal
OOF training were not started. See the Stage 3A status in the track README for
the blocker and cross-variant graph audit.
