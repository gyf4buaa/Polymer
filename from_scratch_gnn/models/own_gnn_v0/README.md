# Own-GNN v0

Own-GNN v0 is a clean, graph-only model trained from random initialization on
the frozen Stage 0 benchmark. It is independent of the third-party GATv2
reference implementation and does not use its weights, Morgan fingerprints,
global RDKit descriptors, physics features, or 3D inputs.

## Graph representation

For exactly two degree-one `*` atoms with distinct real neighbors, graph
construction removes both dummy atoms, keeps all ordinary atom/bond features,
and adds an undirected single polymerization edge between the real endpoints.
That edge has a dedicated `polymerization_edge=1` feature. Ordinary bonds have
`polymerization_edge=0`. If the endpoints already share a chemical bond, the
ordinary bond is retained and the special connection is represented as a
parallel marked edge.

If the input has a different number of dummy atoms, a dummy atom is not
degree-one, or both dummy atoms have the same neighbor, the deterministic
fallback retains the complete parsed graph, including its dummy atoms, and
adds no special edge. The sample ID and fallback reason are recorded in
`graph_diagnostics.json`. Invalid or empty SMILES raise an explicit error;
they are never silently removed.

Node inputs are categorical atomic number, degree, formal charge,
hybridization, aromaticity, total hydrogen count, and chirality. Edge inputs
are one-hot bond type, conjugation, ring membership, one-hot stereo, and the
polymerization marker. Unknown categories have reserved buckets.

## Model and training

- Atom embedding: one learned embedding table per node field; embeddings are
  summed to width 256.
- Message passing: four residual `GINEConv` blocks with edge attributes,
  LayerNorm, ReLU, and dropout 0.1.
- Readout: global mean pooling concatenated with global max pooling.
- Outputs: five independent MLP heads in Stage 0 target order: Tg, FFV, Tc,
  Density, Rg.
- Each fold fits target mean and population standard deviation on its training
  rows only. Missing labels stay masked. Normalized Huber losses are averaged
  equally across tasks represented in each batch.
- Optimizer: AdamW, learning rate 0.001, weight decay 0.00001, batch size 64,
  at most 600 epochs, early-stop patience 40. The held-out fold is scored with
  the fixed Stage 0 manifest weights and the authoritative metric module.
- Fold seeds are 42 through 46. No architecture or hyperparameter sweep is
  included in v0.

## Checks

From the repository root, with the benchmark data available at
`data/competition_raw/train.csv` (or passed explicitly):

```bash
python -m pytest -q from_scratch_gnn/models/own_gnn_v0/tests
python -m from_scratch_gnn.models.own_gnn_v0.train_oof \
  --train-csv data/competition_raw/train.csv --device cpu --tiny-overfit
python -m from_scratch_gnn.models.own_gnn_v0.train_oof \
  --train-csv data/competition_raw/train.csv --device cuda --smoke-fold 0 \
  --epochs 5 --batch-size 64
```

The tiny overfit uses up to 32 real benchmark rows to exercise gradients,
missing-label masking, and checkpoint round trips. The CUDA command exercises
the formal graph/model and only fold 0; it intentionally does not create an
incomplete OOF result. A formal five-fold run is:

```bash
python -m from_scratch_gnn.models.own_gnn_v0.train_oof \
  --train-csv data/competition_raw/train.csv --device cuda \
  --output-dir from_scratch_gnn/models/own_gnn_v0/artifacts/production_gpu
```

Formal training requires the frozen training and fold SHA256 values and a clean
committed source tree. It writes `oof_predictions.csv`, `metrics.json`,
`fold_metrics.csv`, `config.json`, `config.yaml`, `source_manifest.json`,
`run_metadata.json`, graph diagnostics, fold histories, and best checkpoints.
All run products and graph caches are Git-ignored under `artifacts/`.

## Formal result

The source implementation is ready for its first formal GPU run. The validated
five-fold OOF result will be recorded here after the Stage 0 validator accepts
all 7,973 sample IDs.
