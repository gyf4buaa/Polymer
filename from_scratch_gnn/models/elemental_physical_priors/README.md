# Stage 4B — elemental physical priors

Stage 4B tests whether five fixed RDKit periodic-table properties add stable value to the frozen keep-dummy Own-GNN G0 baseline. The categorical atom schema, dummy atoms, chemical bonds, GINE trunk, readout, task heads, loss, optimizer, folds, target normalization, and metric are unchanged.

The fixed node vector order is atomic weight, covalent radius, van der Waals radius, outer-electron count, and period. Values come only from the pinned RDKit `PeriodicTable` methods recorded in `features.py`. The reference scaler uses the population mean and standard deviation over all atomic numbers 1–118; it reads no benchmark rows, labels, or folds. Dummy atomic number zero receives exactly five zeros.

The model adds `Linear(5, 256, bias=False)` to the seven categorical embeddings at the initial node representation. The projection is constructed after all common G0 parameters and zero-initialized, so same-seed common initialization is preserved and E initially produces the same predictions as G0.

Before training, audit the original benchmark SMILES and write `experiments/stage4b/element_feature_manifest.json`:

```bash
python -m from_scratch_gnn.scripts.audit_stage4b_elements \
  --train-csv /path/to/train.csv \
  --output from_scratch_gnn/experiments/stage4b/element_feature_manifest.json
```

Each formal seed is one complete five-fold OOF process. Run seeds 42–46 on CUDA with the checked-in config, one process per seed, and at most two concurrent seeds. Each run writes a seed-local registry row; aggregate only after every seed completes:

```bash
python -m from_scratch_gnn.models.elemental_physical_priors.train_oof \
  --train-csv /home/gyf/work/polymer/data/competition_raw/train.csv --seed 42 --device cuda
python -m from_scratch_gnn.scripts.aggregate_stage4b
```

Formal outputs, graph caches, checkpoints, smoke outputs, and tiny-overfit outputs are seed/phase isolated. Checkpoints and caches are ignored by Git. G0's five frozen Stage 3A.1/Stage 4A results are reused without retraining.
