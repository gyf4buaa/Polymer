# Upstream-compatible GATv2 training reconstruction

This directory restores a training path compatible with the public GATv2 assets consumed by the archived inference script. It is a fallback/reproducibility path and is not intended to stand in for the separate historical self-trained GNN experiment.

## Provenance

The historical local training source has not yet been verified and imported into this repository. The reconstruction is based on two independently matching sources of evidence:

1. the current inference script, which fixes the model class, dimensions, target order, asset names and fold layout; and
2. the public MIT-licensed third-place solution by `fresnellll/kaggle-NeurIPS-polymer-prediction-solution`, whose `src/prepare_data.py` and `src/train.py` produce the same asset contract.

Reference upstream commit observed during recovery:

`f385d220d348283792c9f3dc8ed4ab0619e6f7c4`

This directory is therefore a **reconstruction/adaptation**, not a claim that these files are the historical local source.

## Recovered training contract

The reconstructed pipeline preserves the parts that can be evidenced from the archived inference path:

- targets: `Tg, FFV, Tc, Density, Rg`;
- graph node features: 7;
- graph edge features: 2;
- synthetic edge between the two polymer `*` connection atoms;
- Morgan fingerprint: radius 2, 1024 bits;
- task-specific top-50 Morgan bits selected with univariate `f_regression`;
- GATv2 hidden width: 384;
- GATv2 layers: 6;
- attention heads: 8;
- dropout: 0.2;
- 5-fold KFold training;
- validation-time linear calibration per target;
- final per-fold refit model saved as `final_refit_model.pth`.

The upstream recipe used AdamW at `1e-4`, up to 600 epochs and early-stopping patience 40. Those settings are retained as defaults.

## Expected raw data

Place competition / training CSV files in a directory such as:

```text
data/raw/
  Tg.csv
  FFV.csv
  Tc.csv
  Density.csv
  Rg.csv
```

Each target file is expected to contain SMILES in the first column and the target value in the second column. Additional CSV files are ignored unless their filename stem matches one of the five targets.

## Run

Create an environment using [requirements.txt](requirements.txt), then from the repository root:

```bash
python training/prepare_data.py --raw-dir data/raw --artifact-root training_artifacts
python training/train_gnn.py --artifact-root training_artifacts
```

The resulting layout is:

```text
training_artifacts/
├─ assets/
│  └─ best_fp_indices.pkl
├─ processed/
│  └─ all_augmented_master.pt
└─ models/
   ├─ fold_0/
   │  ├─ calibrators.pkl
   │  └─ final_refit_model.pth
   ├─ fold_1/
   ├─ fold_2/
   ├─ fold_3/
   └─ fold_4/
```

This layout is intentionally compatible with the archived inference script's asset discovery logic.

## What cannot be recovered from GitHub alone

The following should not be claimed as historically recovered unless another local/Kaggle artifact is found later:

- exact random states or package builds used in every historical experiment beyond the public upstream defaults;
- any private/local modifications that were never committed;
- exact historical train/validation splits if they differed from the upstream recipe;
- the exact training run that generated the archived third-party weights;
- byte-identical model files.

If an old Kaggle notebook, Kaggle Dataset, local zip, shell history, Downloads archive or Time Machine copy turns up later, compare it against this reconstruction before changing the provenance statement.
