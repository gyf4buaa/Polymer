# Frozen benchmark (Stage 0)

This directory freezes the local development data, one five-fold assignment, and the scoring contract for every later baseline and Own-GNN experiment.

## Training data and canonical view

The formal training input is the official competition file data/competition_raw/train.csv from the local Polymer data workspace. It is outside this Git repository and is deliberately not committed. Old code confirms the entry point in modeling/src/config.py (TRAIN_PATH) and modeling/src/data.py (load_train()). The supplement CSVs are auxiliary historical experiments and are not part of this benchmark. Released public/private evaluation files are also excluded.

The raw SHA-256 is:

    1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1

The input has 7,973 samples and source columns id, SMILES, Tg, FFV, Tc, Density, Rg. The canonical loader in src/data.py renames id to sample_id and returns, in source order:

    sample_id, SMILES, Tg, FFV, Tc, Density, Rg

It does not drop or merge rows. Empty target cells become masked None values. folds.csv stores the sample ID, fold number, and original SMILES for audit; it does not contain labels or the raw training table.

| Target | Observed | Missing | Min | Max | Range |
|---|---:|---:|---:|---:|---:|
| Tg | 511 | 7,462 | -148.0297376 | 472.25 | 620.2797376 |
| FFV | 7,030 | 943 | 0.2269924 | 0.77709707 | 0.55010467 |
| Tc | 737 | 7,236 | 0.0465 | 0.524 | 0.4775 |
| Density | 613 | 7,360 | 0.748691234 | 1.840998909 | 1.092307675 |
| Rg | 614 | 7,359 | 9.7283551 | 34.672905605 | 24.944550505 |

diagnostics.json records the audit: 0 missing IDs, 0 duplicate IDs, 0 missing SMILES, 0 exact duplicate SMILES, 0 canonicalized duplicate SMILES, 0 invalid SMILES among 7,973 checked with RDKit 2026.03.2, and 0 rows missing all five labels. No anomaly is removed automatically. Exact-text and RDKit canonical-isomeric duplicate checks are diagnostics only; no rows are deduplicated or grouped for splitting.

data_manifest.json records the source hash, schema mapping, label counts/ranges, metric weights, fold seed, fold sizes, valid label counts per fold, and the hash of folds.csv.

## Fixed folds

All formal models use folds.csv. It has one row per source sample, folds numbered 0–4, and SHA-256:

    1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a

The split is an unstratified, sample-level random 5-fold split with seed 20250604, reusing RANDOM_SEED from the existing modeling/src/config.py. To keep the mapping stable across row ordering and Python/library versions, each ID is ranked by the SHA-256 digest of the UTF-8 byte string:

    polymer-stage0-sha256-rank-v1<NUL>20250604<NUL><sample_id>

The ranked IDs are divided into sequential, balanced chunks for folds 0 through 4. This is ordinary random K-fold development validation: it does not stratify sparse labels or group similar structures. The manifest stores each fold's sample count and valid count for every target. Historical modeling/src/cv.py creates separate, sometimes target-stratified folds per target; those historical OOF scores are not directly comparable and must be rerun on this shared map for formal comparisons.

| Fold | Samples |
|---:|---:|
| 0 | 1,595 |
| 1 | 1,595 |
| 2 | 1,595 |
| 3 | 1,594 |
| 4 | 1,594 |

## One authoritative metric

src/metrics.py::evaluate_oof is the sole implementation. Its weight and aggregation rule follows the [official Kaggle evaluation definition](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025):

For target i, let n_i be the number of available labels and r_i = max(y_i) - min(y_i). With K = 5,

    b_i = K * sqrt(1 / n_i) / sum_j sqrt(1 / n_j)
    w_i = b_i / r_i

For an evaluation set of N polymers, missing labels are masked and:

    wMAE = (1 / N) * sum_s sum_{i with observed y[s,i]} w_i * abs(pred[s,i] - y[s,i])

The five target MAEs are reported separately, each over only that target's observed labels. Equivalently, the overall contribution of target i is w_i * (n_i / N) * MAE_i. The overall score is not a simple arithmetic mean of the five raw MAEs. For this snapshot, N = 7,973 and sum_j sqrt(1/n_j) = 0.1737459602374502.

Kaggle's official hidden-test weights use hidden-test label counts and ranges. Those are unavailable for a leakage-free local benchmark, so this frozen local version estimates n_i and r_i from the official training CSV only. The resulting values are fixed in data_manifest.json and used for every full OOF comparison. This local OOF score is therefore not numerically interchangeable with Kaggle Private. The private labels and scores are never read by the loader or used to compute local metrics.

| Target | n_i | r_i | w_i | w_i n_i/N |
|---|---:|---:|---:|---:|
| Tg | 511 | 620.2797376 | 0.00205237749659811 | 0.000131539558605498 |
| FFV | 7,030 | 0.55010467 | 0.623924865972491 | 0.550130666974365 |
| Tc | 737 | 0.4775 | 2.21997543529438 | 0.205207813346539 |
| Density | 613 | 1.092307675 | 1.06409417613206 | 0.0818123328695536 |
| Rg | 614 | 24.944550505 | 0.0465581184297421 | 0.00358543643745913 |

When scoring a held-out fold for early stopping, pass target_weights from the frozen full-training data_manifest.json to evaluate_oof; do not recalculate weights from only that fold. A full OOF evaluation over all 7,973 rows derives the same weights directly from the canonical training labels.

## Commands

From the repository root, freeze/rebuild artifacts from the local source file:

    python from_scratch_gnn/scripts/freeze_benchmark.py \
      --train-csv /path/to/data/competition_raw/train.csv \
      --seed 20250604

Validate a keyed OOF CSV and write metrics.json beside it:

    python from_scratch_gnn/scripts/evaluate_oof.py \
      --train-csv /path/to/data/competition_raw/train.csv \
      --oof-csv /path/to/oof_predictions.csv

The validator requires exactly one prediction row per training sample_id, rejects duplicate, missing, or unknown IDs, aligns by ID, masks absent truth labels, and rejects absent/non-finite predictions where truth is observed. It then writes the overall wMAE, five target MAEs, target counts, weights, and validation summary to metrics.json.

Run the non-training self-tests:

    python -m unittest discover -s from_scratch_gnn/tests -t .

These utilities do not train a model or read submission artifacts.
