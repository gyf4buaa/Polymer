# Own-GNN v0

Own-GNN v0 is the **internal baseline** for the owned-model development track: a clean, graph-only model trained from random initialization on the frozen Stage 0 benchmark. It is independent of the third-party GATv2 external reference implementation and does not use its weights, Morgan fingerprints, global RDKit descriptors, physics features, or 3D inputs. Subsequent Own-GNN variants should be compared primarily against this baseline; the third-party result is retained only as external context.

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

Formal run: `production_gpu_20261003`, source commit
`b3f5df52c81f8032f5f2a2390b437d5d3302b160`, RTX 4070 CUDA, fold seeds 42–46.
The source training CSV SHA256 is
`1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1`; the frozen
folds SHA256 is
`1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`.

Graph audit built 7,973/7,973 samples. The dummy-atom counts were 2 with one
dummy, 7,955 with two, 8 with three, and 8 with four. Endpoint closure was
applied to 7,940 graphs; 1,244 of those retain an ordinary endpoint bond
alongside the marked parallel polymer edge. The other 33 graphs use the
documented lossless fallback: 2 one-dummy graphs, 8 three-dummy graphs, 8
four-dummy graphs, and 15 shared-endpoint graphs. No sample was dropped.

| Fold | Best epoch | Validation wMAE |
|---:|---:|---:|
| 0 | 59 | 0.02221611 |
| 1 | 80 | 0.02075641 |
| 2 | 130 | 0.02326369 |
| 3 | 40 | 0.02463983 |
| 4 | 45 | 0.02366061 |

The independent Stage 0 validator accepted all 7,973 sample IDs, with zero
duplicates, omissions, or extra IDs. Full OOF metrics:

| Metric | Own-GNN v0 |
|---|---:|
| OOF wMAE | 0.0229070190 |
| Tg MAE | 53.73353017 |
| FFV MAE | 0.00583204 |
| Tc MAE | 0.02455388 |
| Density MAE | 0.02348644 |
| Rg MAE | 1.58151550 |

For external context, the frozen third-party GATv2 reference has wMAE 0.0240681831; Own-GNN v0 is 0.0011611640 lower (4.82% relative) on this single-seed run. This contextual comparison does not define the development objective: Own-GNN v0 itself is the internal baseline for subsequent variants, and this is not yet a multi-seed final claim.

The formal run took 590.7 seconds total (9m 51s), averaged 1.063 seconds per
executed epoch, and processed about 6,008 training samples per second. CUDA
utilization averaged 36.3% (maximum 42% over 40 samples). Peak PyTorch CUDA
allocation was 136.7 MiB (160 MiB reserved); peak `nvidia-smi` used memory was
1,764 MiB. No NaN or OOM occurred.

The verified files remain outside Git at
`/home/gyf/work/polymer/from_scratch_gnn/models/own_gnn_v0/artifacts/production_gpu_20261003/`:
OOF predictions, validator metrics, fold metrics, configs, source/run metadata,
graph diagnostics, five training histories, and five best checkpoints.
