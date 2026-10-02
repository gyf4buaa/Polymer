# From-Scratch GNN Track

This directory is the clean-room experimental track for training a polymer property GNN from random initialization.

The goal is not to reproduce or fine-tune the existing third-party competition model. The third-party GATv2 solution is kept only as a **reference baseline**. All models developed here should use our own training runs, checkpoints, OOF predictions, experiment logs, and ablations.

## Scientific question

How far can a from-scratch polymer GNN go on the NeurIPS Open Polymer Prediction task, and which modeling choices actually improve generalization?

The development loop is:

```text
freeze benchmark
    ↓
retrain third-party architecture as an OOF reference
    ↓
Own-GNN v0: simple graph-only baseline
    ↓
controlled model improvements
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

---

## Stage 1 — Establish the reference baselines

### 1A. Conventional descriptor baseline

Train a simple non-GNN baseline such as:

```text
RDKit descriptors + Morgan fingerprint → LightGBM
```

Purpose: determine whether graph learning actually improves over a strong low-cost molecular baseline.

### 1B. Third-party GATv2 reference

Retrain the public third-party GATv2 architecture **from random initialization** using the frozen folds and frozen metric.

Do not use its already-refit competition weights to calculate a training-set score.

This produces one fixed reference line:

```text
Third-party architecture OOF wMAE = ?
```

Once established, this number should not move while Own-GNN is being developed.

### 1C. Historical own-GNN reference

If the old `own_gnn_competition_5fold_v1` experiment can be reproduced under the frozen benchmark, add it as another baseline.

If it cannot be reproduced exactly, keep its historical OOF result only as a historical reference, not as a formal benchmark result.

---

## Stage 2 — Own-GNN v0

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

The first formal result is:

```text
Own-GNN v0
OOF wMAE = ?
Tg MAE = ?
FFV MAE = ?
Tc MAE = ?
Density MAE = ?
Rg MAE = ?
```

This is the first fully owned benchmark point.

---

## Stage 3 — Controlled GNN improvement

Do not change many things at once. Each experiment should answer one hypothesis.

Recommended order:

### 3A. Polymer graph representation

Compare, with everything else fixed:

- keep the `*` atoms;
- remove `*` and connect their neighboring atoms;
- add a special polymerization edge;
- encode endpoint identity without closing the graph.

Question: **Does a polymer-aware graph representation improve OOF generalization?**

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

### Development benchmark

Use the same frozen OOF protocol:

| Model | OOF wMAE | Tg | FFV | Tc | Density | Rg |
|---|---:|---:|---:|---:|---:|---:|
| Descriptor baseline | TBD | | | | | |
| Third-party GATv2 retrained from scratch | TBD | | | | | |
| Historical Own-GNN | TBD | | | | | |
| Own-GNN v0 | TBD | | | | | |
| Own-GNN final | TBD | | | | | |

This table explains **why the model improved**.

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
