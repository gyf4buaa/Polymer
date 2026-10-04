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

### Stage 3A.1 — paired-seed representation confirmation (complete)

This confirmation asks whether Variant A's small seed-42 advantage over the
frozen Own-GNN v0 baseline persists across the pre-registered paired seeds.
The only run-to-run variable was the base random seed. Own-GNN v0, Variant A,
and Variant B retained the exact Stage 3A architecture, graph representation,
chemical features, optimizer, loss, training settings, target normalization,
folds, and Stage 0 metric. The implementation derives fold initialization
seeds as `run_seed + fold_index`, identically for all three models.

Seed 42 is reused from Stage 3A and was not rerun. The twelve new full 5-fold
OOF runs are seeds 43–46 for each model. Every run was performed with the
Stage 3A.1 source commit `de70b3e0cfd51fc83920fa372f1d1fdab92a6845` on CUDA
using an NVIDIA RTX 4070. The frozen train and fold SHA256 values remain
`1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1` and
`1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`.

| Model | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± sample SD | Min–max |
|---|---:|---:|---:|---:|---:|---:|---:|
| Own-GNN v0 | 0.0229070190 | 0.0230364285 | 0.0230329353 | 0.0229046954 | 0.0231809810 | 0.0230124119 ± 0.0001141535 | 0.0229046954–0.0231809810 |
| Variant A — keep dummy | 0.0227728722 | 0.0226486259 | 0.0228896667 | 0.0227330069 | 0.0228828622 | 0.0227854068 ± 0.0001024446 | 0.0226486259–0.0228896667 |
| Variant B — endpoint marker, no closure | 0.0230024716 | 0.0232366175 | 0.0230425081 | 0.0232757492 | 0.0230131083 | 0.0231140909 ± 0.0001312694 | 0.0230024716–0.0232757492 |

Paired deltas use left minus right; a negative value favors the left-hand
representation.

| Comparison | Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± sample SD | Direction |
|---|---:|---:|---:|---:|---:|---:|---:|
| A − v0 | −0.0001341468 | −0.0003878026 | −0.0001432686 | −0.0001716884 | −0.0002981189 | −0.0002270051 ± 0.0001113636 | A lower in 5/5 |
| B − v0 | +0.0000954525 | +0.0002001890 | +0.0000095728 | +0.0003710538 | −0.0001678728 | +0.0001016791 ± 0.0002021432 | B higher in 4/5 |
| A − B | −0.0002295993 | −0.0005879916 | −0.0001528414 | −0.0005427422 | −0.0001302461 | −0.0003286841 ± 0.0002197597 | A lower in 5/5 |

| Target | Own-GNN v0 mean ± SD | Variant A mean ± SD | Variant B mean ± SD | A − v0 mean delta | B − v0 mean delta |
|---|---:|---:|---:|---:|---:|
| Tg | 53.5326 ± 0.606 | 51.7968 ± 0.920 | 53.6849 ± 0.774 | −1.73578 | +0.15229 |
| FFV | 0.00580818 ± 0.000211 | 0.00589486 ± 0.000128 | 0.00597187 ± 0.000344 | +0.00008668 | +0.00016369 |
| Tc | 0.0248217 ± 0.000199 | 0.0246618 ± 0.000366 | 0.0247491 ± 0.000367 | −0.00015986 | −0.00007254 |
| Density | 0.0247803 ± 0.00112 | 0.0246016 ± 0.000666 | 0.0252259 ± 0.000518 | −0.00017876 | +0.00044557 |
| Rg | 1.57709 ± 0.0220 | 1.57739 ± 0.0216 | 1.56873 ± 0.0223 | +0.00029658 | −0.00835883 |

The seed-level target MAEs and paired deltas are in
[`experiments/stage3a1/paired_summary.json`](experiments/stage3a1/paired_summary.json).
Per-fold best epochs, validation wMAE, runtime, GPU utilization, and memory
measurements are in
[`experiments/stage3a1/fold_run_summary.csv`](experiments/stage3a1/fold_run_summary.csv).
Each run's predictions, fold metrics, five histories, effective config,
gzip-compressed graph diagnostics, registry row, and provenance remain under its
`models/*/artifacts/paired_seed_{43,44,45,46}/` directory. File SHA256 values
are listed in [`experiments/stage3a1_run_artifact_manifest.json`](experiments/stage3a1_run_artifact_manifest.json).
Checkpoint weights and graph caches are excluded.

All twelve new OOF predictions passed Stage 0 validation on all 7,973 samples,
with no missing, extra, or duplicate IDs. The runs used 6,831 seconds of GPU
training time in total (113.9 minutes); the maximum recorded PyTorch allocated
VRAM was 135.7 MiB and maximum `nvidia-smi` memory use during training was
2,330 MiB. There were no NaN values, OOMs, or CUDA errors. The only repeated
runtime notice was PyG's optional `torch-scatter` acceleration warning. One
initial v0 seed-43 launch was stopped by the clean-source guard before epoch 1
because a temporary data symlink made the checkout dirty; the symlink was
removed and the same run completed with the frozen CSV supplied by absolute
path. No training run was duplicated. The two-epoch Variant A seed-43 CUDA
smoke passed before formal training and was not used for model selection.

Variant A's mean paired advantage is −0.00022701 wMAE, with negative deltas
for all five seeds; its paired-delta SD is 0.00011136. Its Tg and Tc MAEs are
lower in four of five seeds. The seed-42 Density and Rg trade-offs did not
repeat consistently: Variant A's Density delta changes direction across
seeds, and its mean Rg delta is close to zero relative to the seed spread.
This is a consistent signal in the five pre-registered seeds, but still a
small single-benchmark result rather than a general physical conclusion.
Variant B is higher than v0 in mean OOF wMAE and in four of five paired seeds.

**Decision:** carry Variant A forward as the fixed representation candidate
for later experiments. Its score is not a final-model selection. The planned
message-passing operator comparison is complete and reported in Stage 3B below;
that result does not establish a universally optimal representation or operator.

All 66 repository tests pass after the seed-plumbing, artifact-isolation, and
paired-summary checks.

### 3B. Message-passing operator (complete)

Stage 3B held the Stage 3A.1 Variant A keep-dummy graph, node/bond feature
schema, hidden width 256, four layers, residual/normalization/dropout, mean+max
readout, five property heads, loss, optimizer, target normalization, frozen
folds, and Stage 0 metric fixed. The sole scientific variable was the
message-passing operator. Historical GINE results were reused and **GINE was
not retrained**. GATv2 and PNA each completed the five pre-registered seeds
42–46 on CUDA using the same formal source commit,
`c57d016cc65c1b74e00b89e9ff1ba40acf59a0c9`.

| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |
|---|---:|---:|---:|---:|---:|---:|---:|
| GINE — historical Variant A | 0.0227728722 | 0.0226486259 | 0.0228896667 | 0.0227330069 | 0.0228828622 | 0.0227854068 ± 0.0001024446 | 0.0226486259–0.0228896667 |
| GATv2 — edge-aware | 0.0230836920 | 0.0233582343 | 0.0234626269 | 0.0236539060 | 0.0233196560 | 0.0233756230 ± 0.0002083684 | 0.0230836920–0.0236539060 |
| PNA — edge-aware | 0.0310363356 | 0.0328282117 | 0.0314809960 | 0.0321503291 | 0.0314973000 | 0.0317986345 ± 0.0006993773 | 0.0310363356–0.0328282117 |

Paired deltas use left minus right; negative favors the left model.

| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Left lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| GATv2 − GINE | +0.0003108197 | +0.0007096084 | +0.0005729602 | +0.0009208991 | +0.0004367939 | +0.0005902163 ± 0.0002374364 | 0/5 |
| PNA − GINE | +0.0082634634 | +0.0101795859 | +0.0085913293 | +0.0094173221 | +0.0086144378 | +0.0090132277 ± 0.0007782772 | 0/5 |
| GATv2 − PNA | −0.0079526437 | −0.0094699774 | −0.0080183691 | −0.0084964231 | −0.0081776440 | −0.0084230114 ± 0.0006218347 | 5/5 |

Per-target MAE is shown as mean ± sample SD across the five seeds. Paired
columns are the mean delta ± sample SD relative to GINE.

| Target | GINE | GATv2 | PNA | GATv2 − GINE | PNA − GINE |
|---|---:|---:|---:|---:|---:|
| Tg | 51.7968 ± 0.920 | 54.0599 ± 0.720 | 55.0562 ± 0.546 | +2.26304 ± 0.804 | +3.25940 ± 1.001 |
| FFV | 0.00589486 ± 0.000128 | 0.00594962 ± 0.000199 | 0.01084748 ± 0.000463 | +0.00005476 ± 0.000180 | +0.00495262 ± 0.000472 |
| Tc | 0.0246618 ± 0.000366 | 0.0250403 ± 0.000529 | 0.0324929 ± 0.001132 | +0.00037850 ± 0.000445 | +0.00783108 ± 0.001102 |
| Density | 0.0246016 ± 0.000666 | 0.0280656 ± 0.000977 | 0.0614923 ± 0.001865 | +0.00346405 ± 0.001156 | +0.03689076 ± 0.002295 |
| Rg | 1.57739 ± 0.0216 | 1.54987 ± 0.0185 | 1.92178 ± 0.0471 | −0.0275168 ± 0.03439 | +0.3443886 ± 0.06316 |

| Model | Fixed operator design | Trainable parameters | Message-passing parameters |
|---|---|---:|---:|
| GINE | Historical GINEConv | 1,243,657 | 543,748 |
| GATv2 | 4 × 64 heads, concatenated; edge-aware | 1,244,421 | 544,768 |
| PNA | mean/min/max/std; identity/amplification/attenuation; one tower; edge-aware | 5,176,581 | 4,476,928 |

All models kept four 256-wide message-passing layers. Parameter counts were
reported as measured and were not forced to match. The GATv2 and PNA operators
consume the same encoded 16-column bond features. Both new operators passed the
CPU tiny-overfit and two-epoch CUDA fold-0 smoke checks before formal training;
smoke outputs are labeled non-selection.

The ten successful formal runs used **12,360.8 seconds** total. All run
metadata records an NVIDIA RTX 4070; maximum PyTorch allocated/reserved VRAM
was 507.9/1,286.0 MiB and maximum `nvidia-smi` observed memory was 3,011 MiB.
Mean sampled GPU utilization was 35.2% for GATv2 and 61.7% for PNA. There were
no NaN values, OOMs, or CUDA errors. PyG emitted its optional `torch-scatter`
acceleration warning; it did not prevent training.

PNA's fixed degree histogram was built from topology for all **7,973** frozen
keep-dummy graphs, with no labels or targets. Directed in-degree counts for
degrees 0–6 were `[0, 48203, 140996, 77869, 6364, 0, 1]` over 273,433 nodes and
589,264 directed edges. The graph fingerprint is
`a3107fc9257796215375e2cf8dc387e9edbabcff7fc2d37b8dd8507886572315`; the
`degree_stats.py` generator SHA256 is
`cf55864c2c1aefbeca1d6df7042fab729dad83337ea0e5874089427549d881d4`.
Train SHA256 and folds SHA256 are unchanged from Stage 3A.1:
`1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1` and
`1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`.

The first GATv2 seed-46 process lost its SSH transport during fold 1 after
fold 0 had completed. The incomplete attempt was preserved on the RTX workspace
and excluded from the aggregate; seed 46 was rerun once with the same source,
config, folds, and training settings. The complete rerun is the only seed-46
result included here. This was a transport interruption, not a CUDA or model
error; the successful-run runtime above excludes time spent on the incomplete
attempt.

All 10 successful OOF artifacts passed Stage 0 validation on 7,973/7,973
samples with no duplicate, missing, or extra IDs. The full repository test
suite passed (84 tests), and `git diff --check` passed. The complete per-seed
effective-config hashes, paired values, graph fingerprint, runtime data, and
parameter metadata are in
[`experiments/stage3b/aggregate_summary.md`](experiments/stage3b/aggregate_summary.md)
and [`experiments/stage3b/aggregate_summary.json`](experiments/stage3b/aggregate_summary.json).
The interrupted-attempt note is in
[`experiments/stage3b/formal_execution_notes.json`](experiments/stage3b/formal_execution_notes.json);
formal lightweight artifacts are under
`models/operator_ablation/{gatv2,pna}/artifacts/seed_{42..46}/`. Checkpoints and
graph caches are excluded from version control. Ten Stage 3B seed-level rows
were appended to [`results.csv`](results.csv); historical GINE rows were left
unchanged.

**Decision:** operator choice has a larger measured effect here than the
Stage 3A.1 keep-dummy representation difference (A − v0 was
−0.0002270051 ± 0.0001113636). GATv2 is higher than GINE in all five paired
seeds (+0.0005902163 ± 0.0002374364); PNA is higher in all five
(+0.0090132277 ± 0.0007782772). GATv2 improves mean Rg MAE, but has higher mean
MAE on the other four targets. PNA has higher mean MAE on all five targets,
with the largest increases on Density, FFV, and Tc. Thus operator selection
can materially change performance under this frozen setup, but neither tested
replacement improves on GINE. **Keep keep-dummy GINE as the next-stage baseline.**
This is a result on one benchmark and one fixed architecture family, not a
claim that GINE is universally best for polymer properties. Stage 3B is
complete; Stage 3C had not started when Stage 3B closed.

### 3C. Readout and property-specific pooling (complete)

Stage 3C froze the Stage 3A.1 keep-dummy polymer representation and the Stage
3B GINE operator, including the four-layer, 256-wide trunk, five existing
property heads, optimizer, loss, training protocol, folds, and metric. The sole
scientific variable was graph readout:

- **R0:** historical shared global mean + global max; its five Stage 3A.1
  seeds were reused without retraining.
- **R1:** zero-initialized shared `Linear(256, 1, bias=False)` attentive mean
  plus unchanged global max.
- **R2:** five zero-initialized property-specific `Linear(256, 1, bias=False)`
  attentive means plus unchanged shared global max.

Zero initialization makes both attentive means equal uniform global mean at
initialization. Regression tests confirm same-seed common-parameter
initialization and R0/R1/R2 forward equivalence within `atol=rtol=1e-6`.
Formal execution added ten frozen five-fold OOF runs (five seeds each for R1
and R2), all on source commit
`a9ee75a1bf940f32f0966070caf2450f0c2fe1f4` and the unchanged frozen train/fold
hashes. All 10/10 runs completed; historical R0 contributed 0 new runs.

| Model | Mean OOF wMAE ± sample SD | Paired delta vs R0 | Lower seeds vs R0 |
|---|---:|---:|---:|
| R0 | 0.0227854068 ± 0.0001024446 | — | — |
| R1 | 0.0227642509 ± 0.0001211940 | −0.0000211559 ± 0.0001297928 | 2/5 |
| R2 | 0.0229552392 ± 0.0000825439 | +0.0001698325 ± 0.0001507520 | 1/5 |

R2 − R1 was `+0.0001909883 ± 0.0001804049`, with R2 lower in 1/5 paired
seeds. Per-target effects were mixed: shared attention lowered FFV and Rg mean
MAE but raised Tg, Tc, and Density; property-specific attention lowered Tc
and Rg mean MAE but raised Tg, FFV, and Density. The detailed seed-level,
per-target, parameter, entropy, gate-norm, and provenance tables are in
[`experiments/stage3c/aggregate_summary.md`](experiments/stage3c/aggregate_summary.md)
and [`experiments/stage3c/aggregate_summary.json`](experiments/stage3c/aggregate_summary.json).

**Decision:** retain the historical **R0 mean + max readout** for the next
stage. R1's small mean difference is inconsistent across paired seeds, and R2
does not improve overall OOF wMAE over either R0 or R1. The learned gates did
move away from zero, but the measured weighting did not produce a reliable
generalization gain. Close the readout axis for this benchmark; no additional
attention variants or Stage 4 features were started.

The first R1 seed-42 attempt on the implementation commit preceding the final
formal source completed training but failed in post-training attention
diagnostics because the row list had not been bound. It was excluded. The
alignment check was fixed and regression-tested, and all ten successful formal
runs were restarted on the single source commit recorded above. Details are in
[`experiments/stage3c/formal_execution_notes.json`](experiments/stage3c/formal_execution_notes.json).

CPU tiny-overfit checks passed for both learned readouts, with loss reduction,
nonzero trained gates, and checkpoint round-trip. Five-epoch fold-0 CUDA
smokes also passed for R1 and R2 on the final formal source commit and RTX
4070; all graph-wise attention sums were one within tolerance and artifacts
were isolated from formal runs.

### 3D. Depth and capacity

Compare a controlled range, for example:

- 3 layers;
- 4 layers;
- 6 layers;
- 8 layers.

Track both OOF score and train/validation behavior.

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

### Stage 4A result (complete)

Stage 4A tested fixed RDKit global descriptors (D) and a binary Morgan fingerprint (M) as separate, zero-initialized residual projections after the frozen 512-dimensional graph readout. The historical graph-only model G0 reused its five OOF seeds; each new variant completed five frozen five-fold OOF runs on the same seeds. No graph, fold, target scaling, or training settings changed.

| Model | OOF wMAE, mean ± sample SD | Paired delta vs G0 | Seeds with lower wMAE |
|---|---:|---:|---:|
| G0 | 0.02278541 ± 0.00010244 | — | — |
| D | 0.02303453 ± 0.00007864 | +0.00024912 ± 0.00013927 | 0/5 |
| M | 0.02587587 ± 0.00014020 | +0.00309046 ± 0.00022608 | 0/5 |

D had lower error than M in all five seeds, but both were worse than G0 in every seed. D improved Tg and Rg on average while worsening FFV, Tc, and Density; M worsened all five targets. The preregistered stability rule was a negative paired mean delta and improvement in at least four of five seeds, so neither variant qualifies. D+M was not trained. **G0 remained the model to carry forward into the separately preregistered Stage 4B test.**

D adds 10,240 trainable projection parameters (1,253,897 total); M adds 1,048,576 (2,292,233 total) to G0's 1,243,657. Any Morgan result therefore also reflects the substantially larger model capacity. The 20 descriptors covered all 7,973 original SMILES with no non-finite or constant columns. Feature matrix hashes, full per-seed and per-target tables, fold scalers, diagnostics, runtime, and execution notes are recorded in the [Stage 4A aggregate report](experiments/stage4a/aggregate_summary.md) and [feature manifest](experiments/stage4a/feature_manifest.json).

### Stage 4B result (complete)

Stage 4B added only five fixed RDKit node-level elemental properties through a zero-initialized `Linear(5, 256, bias=False)` residual. G0's five historical seeds were reused, and E completed five new frozen five-fold OOF runs at concurrency 2. Graph representation, dummy handling, GINE, readout, heads, optimizer, loss, folds, normalization, and metric remained fixed.

| Model | OOF wMAE, mean ± sample SD | Paired delta vs G0 | Seeds with lower wMAE |
|---|---:|---:|---:|
| G0 | 0.02278541 ± 0.00010244 | — | — |
| E | 0.02281696 ± 0.00010178 | +0.00003155 ± 0.00004865 | 2/5 |

E did not meet the preregistered stability rule (negative paired mean and lower error in at least four of five seeds). It improved Tc and Rg on average, while Tg and Density worsened and FFV was effectively unchanged; there was no stable overall gain. **G0 remains the carry-forward model, and Stage 4B recommends stopping further simple 2D feature engineering for now.**

The audit covered all 7,973 source SMILES: 15,968 dummy atoms and 257,465 real atoms across 18 observed real elements; every observed real element had finite values for all five properties. The fixed Z=1–118 RDKit reference means/stds and both table hashes are in the [element feature manifest](experiments/stage4b/element_feature_manifest.json). The [Stage 4B aggregate report](experiments/stage4b/aggregate_summary.md) contains per-seed, per-target, execution, parameter-count, and projection diagnostics; the [execution notes](experiments/stage4b/formal_execution_notes.json) retain resource samples and run status. Formal training used source commit `7df330921c156b0e2c88841f0947f0bb3369a52c`.

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
