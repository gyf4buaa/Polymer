# Stage 4A — global information augmentation

Stage 4A asks whether information computed globally from each original polymer SMILES improves the frozen Stage 3 graph-only backbone. G0 is the historical keep-dummy, four-layer GINE, global mean plus global max model with five independent property heads. Its five Stage 3A.1 seeds are reused without retraining.

Only two new runs are registered:

- **D:** the fixed 20-column RDKit descriptor schema in `features.py`, standardized with each fold's training rows only.
- **M:** the raw binary Morgan fingerprint (radius 2, 2,048 bits, chirality enabled), with no standardization or target-based selection.

Both variants add one bias-free linear projection after the 512-dimensional historical graph readout and add that projection residually to the graph vector. The projection is constructed after the complete shared model and zero-initialized, so same-seed shared parameters remain identical and the initial D/M forward pass equals G0. The graph builder and frozen Stage 0 training engine remain unchanged.

The feature audit reads only the original benchmark SMILES. It never removes dummy atoms, builds a closure molecule, or reads targets. Formal training is blocked unless all 7,973 rows are covered, the descriptor matrix is finite, and the committed manifest hashes match the runtime feature matrices.

The preregistered stability rule for reporting is a negative paired mean wMAE delta with the augmented model lower in at least four of five seeds. This is a reporting rule only; every registered D/M seed is run regardless of interim metrics. If both variants meet it, a later Stage 4B may test D+M after this stage is fully reported. No D+M training or feature tuning is part of Stage 4A.

Run the deterministic feature audit with:

```bash
python -m from_scratch_gnn.scripts.audit_stage4a_features --train-csv /path/to/train.csv
```

On the RTX 4070, run one complete variant and seed per process, for example:

```bash
python -m from_scratch_gnn.models.global_information_augmentation.descriptor.train_oof \
  --train-csv /home/gyf/work/polymer/data/competition_raw/train.csv --seed 42 --device cuda
```

For each seed, launch D and M as a pair. Each process writes to its own variant/seed attempt directory and emits a per-run `registry_row.json`. After all ten runs finish, run `python -m from_scratch_gnn.scripts.aggregate_stage4a` once to validate artifacts, build the paired summary, and update `results.csv` atomically.

Raw feature matrices, graph caches, checkpoints, and Python caches are excluded from source/results commits. Fold scalers, OOF predictions, histories, compact metrics, source/feature provenance, runtime diagnostics, and per-run registry rows are retained.
