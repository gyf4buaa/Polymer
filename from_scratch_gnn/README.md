# From-Scratch GNN Track

This directory is the clean-room experimental track for training a polymer property GNN from random initialization.

The goal is not to reproduce or fine-tune the existing third-party competition model. The third-party GATv2 solution is kept only as an **external reference**. **Own-GNN v0 is the internal baseline** for all subsequent model development. All later variants are judged first against Own-GNN v0 under the frozen OOF protocol; external references provide context rather than steering the development path.

## Scientific question

How far can a from-scratch polymer GNN go on the NeurIPS Open Polymer Prediction task, and which modeling choices actually improve generalization?

The development loop is:

```text
freeze benchmark
    ↓
establish external reference points
    ↓
Own-GNN v0: freeze the internal baseline
    ↓
controlled improvements relative to Own-GNN v0
    ↓
controlled feature/fusion improvements
    ↓
final ablation + multi-seed validation
    ↓
freeze model
    ↓
Kaggle hidden-test / Private evaluation
```

The **OOF benchmark is used for all model development and comparison**. Kaggle Private is reserved for the final external comparison.

---

## Stage 0 — Freeze the benchmark

Before changing the model:

1. Freeze the raw training-data version and record hashes.
2. Freeze one 5-fold split and save the fold assignment for every training sample.
3. Freeze one implementation of the competition-style weighted MAE.
4. Report both:
   - overall OOF weighted MAE;
   - per-target MAE for Tg, FFV, Tc, Density, and Rg.
5. Use identical folds and metric code for every model.
6. Fit normalization, feature selection, imputers, or other learned preprocessing **inside each training fold only**.
7. Each OOF prediction must come from a model that never trained on that sample.

Required artifacts for every formal experiment:

```text
oof_predictions.csv
metrics.json
fold_metrics.csv
config.yaml
training_history/
checkpoints/
```

The exact rules are in [BENCHMARK_PROTOCOL.md](BENCHMARK_PROTOCOL.md).

Stage 0 is frozen in [benchmark/README.md](benchmark/README.md). The tracked fold file and manifest are generated from the official training CSV, which remains outside this repository.

---

## Stage 1 — Establish external reference points

### 1A. Third-party GATv2 external reference

Retrain the public third-party GATv2 architecture **from random initialization** using the frozen folds and frozen metric.

Do not use its already-refit competition weights to calculate a training-set score.

This produces one fixed external reference line:

```text
Third-party architecture OOF wMAE = 0.02406818
```

The formal Stage 1A CUDA run completed from random initialization on all five
frozen folds. The Stage 0 validator covered all 7,973 training samples with no
duplicate, missing, or extra sample IDs. Per-fold best epochs were 180, 105,
69, 133, and 133; the corresponding validation wMAEs were 0.02211136,
0.02234731, 0.02578474, 0.02519220, and 0.02490654. The official OOF target
MAEs are Tg 54.32066, FFV 0.00719931, Tc 0.02426117, Density 0.02983760, and
Rg 1.54587552. Run configuration, source manifest, predictions, validator
metrics, fold metrics, histories, and checkpoints are under
`baselines/third_party_gatv2/artifacts/production_gpu_20261003/`; the Mac CPU
pilot remains exploratory provenance and did not contribute to this result.

Once established, this number should not move while Own-GNN is being developed. It is not the optimization baseline for later Own-GNN variants.

### 1B. Conventional descriptor reference

Train a simple non-GNN baseline such as:

```text
RDKit descriptors + Morgan fingerprint → LightGBM
```

Purpose: provide a simple non-GNN context point for how much graph learning adds. It is not the baseline for Own-GNN development.

### 1C. Historical own-GNN reference

If the old `own_gnn_competition_5fold_v1` experiment can be reproduced under the frozen benchmark, add it as another baseline.

If it cannot be reproduced exactly, keep its historical OOF result only as a historical reference, not as a formal benchmark result.

---

## Stage 2 — Freeze the internal baseline: Own-GNN v0

Build the simplest clean from-scratch GNN first.

### Input

Polymer repeat-unit SMILES.

### Graph construction

Start with ordinary atom and bond features.

For polymer endpoints, explicitly test a polymer-aware representation rather than silently inheriting the third-party dummy-atom convention. A candidate v0 representation is:

```text
[*]-A-...-B-[*]
       ↓
remove the two dummy atoms
       ↓
connect A ↔ B with a special polymerization edge
```

The special edge should be distinguishable from a normal chemical bond.

### Initial architecture

Keep v0 deliberately simple:

```text
atom / bond embedding
        ↓
4–6 edge-aware message-passing blocks
        ↓
global pooling
        ↓
shared representation
        ↓
5 property-specific heads
```

A standard operator such as GINE, MPNN, or another well-understood edge-aware layer is preferred over inventing a new convolution immediately.

### Training

For every fold:

1. train on four folds;
2. validate on the held-out fold;
3. calculate target statistics only on the training portion;
4. use a masked loss for missing targets;
5. early-stop on the frozen validation metric;
6. restore the best checkpoint;
7. write predictions for the held-out fold.

After five folds, concatenate all held-out predictions into one OOF file.

The first formal Own-GNN result completed on the frozen Stage 0 benchmark:

| Fold | Best epoch | Validation wMAE |
|---:|---:|---:|
| 0 | 59 | 0.02221611 |
| 1 | 80 | 0.02075641 |
| 2 | 130 | 0.02326369 |
| 3 | 40 | 0.02463983 |
| 4 | 45 | 0.02366061 |

The Stage 0 validator accepted all 7,973 OOF rows with no duplicate, missing,
or extra sample IDs. Overall OOF wMAE is **0.02290702**; target MAEs are Tg
53.73353, FFV 0.00583204, Tc 0.02455388, Density 0.02348644, and Rg 1.58152.
Compared with the frozen third-party GATv2 result (0.02406818), this is a
4.82% lower OOF wMAE for this single-seed run. The full metrics and histories
are in the ignored production artifact directory on the RTX node.

This is the first fully owned benchmark point and, from this stage onward, the **internal baseline**. Later Own-GNN variants should be accepted or rejected primarily by comparison with this result under the same frozen folds and metric. The third-party model remains an external context reference.

---

## Stage 3 — Controlled improvement from Own-GNN v0

Do not change many things at once. Each experiment should answer one hypothesis, and its primary comparison is against Own-GNN v0 or the immediately preceding accepted Own-GNN variant—not against the third-party reference.

Recommended order:

### 3A. Polymer graph representation

Compare, with everything else fixed:

- keep the `*` atoms;
- remove `*` and connect their neighboring atoms;
- add a special polymerization edge;
- encode endpoint identity without closing the graph.

Question: **Does a polymer-aware graph representation improve OOF generalization?**

### Stage 3A result

This was a representation-only ablation. Both variants reused the unchanged
Own-GNN v0 training engine, GINEConv backbone, four layers, hidden size 256,
dropout 0.1, mean+max pooling, five heads, masked Huber loss, AdamW settings,
target normalization, seed, frozen five folds, Stage 0 metric, and v0
node/bond chemical features. Variant A preserves every RDKit atom and bond,
including `*` (atomic-number-zero bucket 0). Variant B removes only two
degree-one dummy atoms with distinct real neighbors, marks those endpoints,
preserves original chemical bonds, and adds no closure edge. Ambiguous
topologies retain the full graph and record a deterministic fallback.

| Model | Representation | OOF wMAE | Tg | FFV | Tc | Density | Rg |
|---|---|---:|---:|---:|---:|---:|---:|
| Own-GNN v0 | remove `*` + marked closure edge | 0.0229070190 | 53.73353017 | 0.00583204 | 0.02455388 | 0.02348644 | 1.58151550 |
| Variant A | keep dummy atoms | 0.0227728722 | 51.50385374 | 0.00582648 | 0.02412187 | 0.02529545 | 1.61020157 |
| Variant B | remove dummy + endpoint marker, no closure | 0.0230024716 | 54.05964086 | 0.00583066 | 0.02452482 | 0.02488108 | 1.56622578 |

Both formal runs passed frozen Stage 0 OOF validation over all 7,973 samples.
Their source commit is `364e69f37059daf1ffac3b09cadf41ce60afe6a1`;
training-data SHA256 is
`1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1`, and
folds SHA256 is
`1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`.

| Fold | A best epoch | A validation wMAE | B best epoch | B validation wMAE |
|---:|---:|---:|---:|---:|
| 0 | 58 | 0.02189877 | 99 | 0.02191349 |
| 1 | 78 | 0.02044939 | 70 | 0.02067256 |
| 2 | 63 | 0.02250620 | 49 | 0.02329773 |
| 3 | 54 | 0.02451637 | 96 | 0.02436420 |
| 4 | 50 | 0.02449581 | 35 | 0.02476635 |

| Run | Runtime | CUDA device | Mean / max sampled GPU utilization | Peak total memory (`nvidia-smi`) | Peak PyTorch allocated VRAM |
|---|---:|---|---:|---:|---:|
| Variant A | 581.63 s | RTX 4070 | 36.5% / 42% (39 samples) | 2,261 MiB | 141.8 MiB |
| Variant B | 635.45 s | RTX 4070 | 37.4% / 44% (43 samples) | 2,252 MiB | 136.9 MiB |

CUDA smoke passed for both variants. Across formal runs there were no NaN,
OOM, parse failures, or dropped rows. The only runtime warning was PyG's
notice that the optional `torch-scatter` package is absent; it did not prevent
CUDA training or checkpoint creation.

| Graph audit | Variant A | Variant B |
|---|---:|---:|
| Graphs built | 7,973 / 7,973 | 7,973 / 7,973 |
| Valid two-endpoint graphs | 7,940 | 7,940 |
| Retained dummy nodes | 15,968 | 88 (fallback samples only) |
| Marked endpoint graphs / nodes | 0 / 0 | 7,940 / 15,880 |
| Existing ordinary bonds between endpoint atoms | 1,244 | 1,244 preserved |
| Fallbacks | 0 | 33 (2 one-dummy, 8 three-dummy, 8 four-dummy, 15 shared-endpoint) |
| Added closure / polymerization edges | 0 / 0 | 0 / 0 |

The absolute aggregate differences from v0 are small: A is lower by
0.00013415 wMAE and B is higher by 0.00009545. A improves Tg and Tc MAE but
has higher Density and Rg MAE; B improves Density and Rg slightly but has
higher Tg MAE. Thus the targets respond differently, and this single seed
does not establish a representation winner or prove that periodic closure is
generally beneficial. Keep v0 as the internal baseline; do not treat A's
small score gain as a final-model decision.

All 52 tests passed. Both variants also passed 120-step CPU tiny-overfit and
five-epoch CPU fold-0 checks before the RTX run. Formal results and the compact
OOF/provenance artifacts are under each variant's
`artifacts/production_gpu_20261003/`; checkpoint weights remain on the RTX
worktrees and are not committed. `results.csv` contains separate `own_model`
rows for A and B; the historical v0 row is unchanged.

Variant packages and independent graph schemas:

- `models/own_gnn_repr_keep_dummy/`
- `models/own_gnn_repr_endpoint_marker/`

### 3B. Message-passing operator

After graph construction is fixed, compare a small set such as:

- GINE;
- GATv2;
- GraphSAGE;
- a standard MPNN.

Question: **Does attention or a different edge update improve this task?**

### 3C. Depth and capacity

Compare a controlled range, for example:

- 3 layers;
- 4 layers;
- 6 layers;
- 8 layers.

Track both OOF score and train/validation behavior.

### 3D. Readout

Compare:

- mean pooling;
- sum pooling;
- max pooling;
- mean + max;
- learned/attention pooling.

### 3E. Multi-task head

Compare:

- one shared backbone + five independent heads;
- shared backbone + target-specific projection layers + five heads.

### 3F. Loss and task weighting

Compare only after the architecture is reasonably stable:

- MAE;
- Huber;
- weighted Huber;
- learned task weighting, if justified.

Every accepted change must improve the frozen OOF benchmark or provide a clear robustness/generalization advantage.

---

## Stage 4 — Add non-GNN information

Only after a strong graph-only model exists.

Add one information source at a time:

1. Morgan fingerprint;
2. RDKit global descriptors;
3. physics-inspired descriptors;
4. conformer / 3D descriptors.

Example fusion:

```text
graph embedding ───────┐
                       ├── fusion ── property heads
Morgan / descriptors ──┘
```

For every addition, compare against the immediately preceding model using the same folds.

The purpose is to measure **complementary information**, not simply accumulate features.

---

## Stage 5 — Final ablation

Development experiments and final ablations are not the same thing.

Once the final model is selected, remove one retained component at a time, for example:

- full model;
- minus polymerization-edge encoding;
- minus fingerprint branch;
- minus physics descriptors;
- minus 3D descriptors;
- minus enhanced pooling.

The final ablation table should make the source of each improvement explicit.

---

## Stage 6 — Multi-seed validation

During development, a single seed is acceptable for speed.

For the final comparison, rerun at least:

- third-party OOF reference;
- Own-GNN baseline;
- Own-GNN final;

with multiple seeds and report mean ± standard deviation.

This checks that the final gain is not a lucky initialization.

---

## Stage 7 — Freeze the final model

Before looking at Kaggle Private:

1. choose the final architecture using OOF only;
2. freeze the configuration;
3. record the git commit;
4. train the final 5-fold ensemble;
5. generate test predictions from the five fold models;
6. average the fold predictions;
7. create one frozen submission artifact.

Kaggle Private must **not** be used as an iterative hyperparameter-validation set.

---

## Stage 8 — Kaggle Private comparison

The final comparison has two levels.

### Internal development benchmark

Own-GNN v0 is the zero point for model development:

| Model | OOF wMAE | Tg | FFV | Tc | Density | Rg |
|---|---:|---:|---:|---:|---:|---:|
| **Own-GNN v0 internal baseline** | **0.02290702** | 53.73353 | 0.00583204 | 0.02455388 | 0.02348644 | 1.58152 |
| Own-GNN variants | TBD | | | | | |
| Own-GNN final | TBD | | | | | |

This table explains **whether each change improves our own model and why**.

### External context

External and historical models are reported separately so that they do not become the development objective:

| Reference | OOF wMAE | Role |
|---|---:|---|
| Third-party GATv2 retrained from scratch | 0.02406818 | External architecture reference |
| Conventional descriptor model | TBD | Non-GNN reference |
| Historical Own-GNN | TBD | Historical reference only unless reproduced on frozen folds |

### External hidden-test benchmark

After freezing:

| Model | Kaggle Private |
|---|---:|
| Existing third-party-backbone + Physics3D reference | 0.06830 |
| Own-GNN final | TBD |
| Own-GNN + owned physics/3D fusion | TBD |

This table answers the separate question:

**How well does the final independently trained model generalize to the hidden Kaggle test set?**

---

## Experiment discipline

Every formal experiment must be traceable to:

- one git commit;
- one config;
- one frozen fold definition;
- one seed;
- one set of checkpoints;
- one OOF prediction file;
- one metric summary.

The experiment registry is [results.csv](results.csv).

A model improvement is not considered established until its OOF result is reproducible under this protocol.
