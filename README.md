# Open Polymer Prediction — GNN + Physics-Inspired 3D Ensemble

Post-competition archive and reproducibility record for a NeurIPS Open Polymer Prediction 2025 solution family.

The repository combines a public GATv2 polymer model with an additional **physics-inspired descriptor / conformer branch**, target-wise LightGBM regressors, and target-specific blending. It is intended as a transparent research archive rather than a claim of an official winning submission.

## What is in this repository

```text
SMILES
├─ GATv2 branch
│  ├─ periodic polymer graph
│  ├─ 5-fold refit models
│  ├─ task-specific Morgan fingerprint features
│  └─ fold calibration
│
└─ Physics-inspired branch
   ├─ RDKit descriptors
   ├─ Morgan / MACCS / AtomPair / torsion fingerprints
   ├─ polymer geometry proxies
   └─ ETKDGv3 + MMFF/UFF monomer / dimer conformer descriptors
        ↓
   target-wise LightGBM
        ↓
target-specific GNN/LGBM blend
        ↓
Tg / FFV / Tc / Density / Rg
```

### Contribution boundary

The GATv2 architecture, its public refit model assets, and the associated training recipe originate from the public third-place solution by hongyu Guo and Sticky Day6027. Attribution and license information are recorded in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

The additional work archived here focuses on:

- physics-inspired polymer descriptors and geometry proxies;
- lightweight monomer / dimer 3D conformer descriptors;
- target-wise LightGBM models;
- target-specific GNN/LGBM blending;
- post-competition diagnostics and score provenance;
- packaging the combined pipeline for Kaggle Internet-Off inference.

The historical local GNN training source has not yet been verified and imported into this repository. A compatible training pipeline has therefore been reconstructed from the MIT-licensed upstream source and the current inference contract; see [training/README.md](training/README.md). It is explicitly marked as a reconstruction rather than the historical local source.

### Historical self-trained GNN route

Separate local audit records indicate that this project also had an independently run five-fold GATv2 experiment with its own training history, model weights and OOF evaluation. That historical route is **not yet imported into this repository** and should not be conflated with the public third-place GATv2 assets used by the archived v2 inference script.

Until the local source is recovered and reviewed, this repository keeps the two paths separate:

1. **archived v2 inference path** — public third-place GATv2 assets + the physics/conformer/LightGBM/blending work documented here;
2. **historical self-trained GNN path** — local experiment evidence exists, but source recovery is still pending.

The code under [training/](training/) is therefore an **upstream-compatible fallback/reconstruction**, not a reconstruction of the historical self-trained route.

## Current archived inference route

The primary script is:

- `kaggle_notebook_submission_gnn3_physics3d_blend.py`

Its target-wise blend weights are:

| Target | GNN | Physics / 3D LightGBM |
|---|---:|---:|
| Tg | 1.00 | 0.00 |
| FFV | 0.89 | 0.11 |
| Tc | 0.73 | 0.27 |
| Density | 0.63 | 0.37 |
| Rg | 0.62 | 0.38 |

The 3D conformer branch is used for `Tc` and `Rg`; the other targets use the base physics-inspired feature set.

## Score provenance

| Route | Public | Private | Status |
|---|---:|---:|---|
| GNN + Physics3D v2 | 0.06259 | 0.06830 | Recorded in a local revised report; no matching Kaggle submission reference was preserved |
| CodeBERTa adaptive | 0.06297 | 0.08104 | Structured Kaggle run record preserved, submission ref `53369928` |

The v2 code contains a `Tg` offset that was selected after private labels were released. Therefore **0.06830 is a post-competition diagnostic / report value and must not be presented as a blind-test leaderboard result**.

Detailed evidence and limitations are documented in [SCORE_PROVENANCE.md](SCORE_PROVENANCE.md).

## Physics-inspired / conformer feature branch

The LightGBM branch includes conventional cheminformatics features plus deliberately simple physical and geometric proxies, including:

- molecular weight, vdW-volume and surface-area proxies;
- polarity, aromaticity, rigidity, flexibility and branching descriptors;
- graph-distance and polymer connection-point descriptors;
- density- and free-volume-inspired ratios;
- monomer and two-repeat-chain conformers generated with RDKit ETKDGv3;
- MMFF/UFF conformer energies;
- radius of gyration, molecular span and pair-distance statistics;
- bounding-box anisotropy;
- PMI / NPR, asphericity, eccentricity, inertial-shape factor and spherocity.

These are best interpreted as **physics-inspired and conformer-based descriptors**, not as atomistic polymer simulations.

The released-label diagnostic suggests that the 3D branch is modest as a standalone model but contributes complementary residual information in the GNN ensemble, especially for `Rg` and `Tc`. See [evidence/physics3d_blend_released_diagnostic.md](evidence/physics3d_blend_released_diagnostic.md).

## Training recovery

The current inference contract matches the public GATv2 training pipeline at the level of:

- model class and dimensions;
- five-fold layout;
- `best_fp_indices.pkl`;
- `calibrators.pkl`;
- `final_refit_model.pth`;
- target order and graph feature dimensions.

A reconstructed, runnable training path is provided under [training/](training/). It is pinned conceptually to the public upstream implementation and does not claim to reproduce any lost local modifications that cannot be evidenced.

## Offline model assets

The original `gnn3_offline_assets.zip` is not committed because it is larger than GitHub's normal single-file limit and contains third-party model assets / wheels.

Recorded archive metadata:

- SHA-256: `4a01dbea2db931f45fd34634e542d90575aca5f9082b203f00600519d6ad419c`
- size: `105,938,993` bytes

The inference script searches attached Kaggle datasets under `/kaggle/input` and is designed for Internet-Off execution.

## Repository files

- `kaggle_notebook_submission_gnn3_physics3d_blend.py` — archived combined inference script.
- `training/` — reconstructed GATv2 data-preparation and training path.
- `SCORE_PROVENANCE.md` — score provenance and leaderboard caveats.
- `evidence/` — preserved diagnostic and structured score records.
- `THIRD_PARTY_NOTICES.md` — upstream attribution and license notice.

## Reproducibility note

Reproducing the exact historical v2 output requires the same competition data, third-party refit assets, selected fingerprint indices, calibrators, dependency versions and offline packaging. The reconstructed training code is intended to restore a transparent training path; it does not establish that newly trained weights are byte-identical to the historical archived assets.
