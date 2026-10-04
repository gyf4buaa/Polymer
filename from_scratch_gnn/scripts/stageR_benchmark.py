#!/usr/bin/env python3
"""Run Stage R clean baselines and diagnostics on the frozen polymer OOF split."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TRACK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRACK_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from from_scratch_gnn.src.data import (  # noqa: E402
    TARGETS,
    load_training_data,
    sha256_file,
    write_csv,
    write_json,
)
from from_scratch_gnn.src.metrics import evaluate_oof  # noqa: E402

EXPECTED_TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"
EXPECTED_FOLDS_SHA256 = "1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a"
EXPECTED_SAMPLES = 7973
FOLDS_PATH = TRACK_ROOT / "benchmark" / "folds.csv"
STAGE5A_SUMMARY_PATH = TRACK_ROOT / "experiments" / "stage5a" / "aggregate_summary.json"
C128_ARTIFACTS = {
    42: TRACK_ROOT / "models/capacity_scaling/artifacts/formal/C128/seed_42/attempt_01",
    43: TRACK_ROOT / "models/capacity_scaling/artifacts/formal/C128/seed_43/attempt_01",
    44: TRACK_ROOT / "models/capacity_scaling/artifacts/formal/C128/seed_44/attempt_01",
    45: TRACK_ROOT / "models/capacity_scaling/artifacts/formal/C128/seed_45/attempt_01",
    46: TRACK_ROOT / "models/capacity_scaling/artifacts/formal/C128/seed_46/attempt_02",
}
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
KNN_K = 5
BOOTSTRAP_REPLICATES = 5000
BOOTSTRAP_SEED = 20261004
SIMILARITY_BINS = (("<0.7", 0.0, 0.7), ("0.7–0.9", 0.7, 0.9), (">0.9", 0.9, 1.0000001))


def _rdkit():
    try:
        import rdkit
        from rdkit import Chem, DataStructs
        from rdkit.Chem import Descriptors, rdFingerprintGenerator
    except ImportError as exc:  # pragma: no cover - depends on user's environment
        raise RuntimeError("Stage R requires RDKit.") from exc
    return rdkit, Chem, DataStructs, Descriptors, rdFingerprintGenerator


def load_frozen_folds(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, dict[str, int]]:
    if sha256_file(FOLDS_PATH) != EXPECTED_FOLDS_SHA256:
        raise ValueError("Frozen folds.csv SHA256 does not match nopp2025_train_v1.")
    with FOLDS_PATH.open("r", encoding="utf-8", newline="") as source:
        fold_rows = list(csv.DictReader(source))
    assignments = {str(row["sample_id"]): int(row["fold"]) for row in fold_rows}
    ids = [str(row["sample_id"]) for row in rows]
    if len(fold_rows) != EXPECTED_SAMPLES or len(assignments) != len(fold_rows):
        raise ValueError("Frozen fold map has a wrong row count or duplicate sample_id.")
    if set(assignments) != set(ids):
        raise ValueError("Frozen fold map sample IDs differ from the official training rows.")
    folds_by_id = {str(row["sample_id"]): row for row in fold_rows}
    for row in rows:
        fold_row = folds_by_id[str(row["sample_id"])]
        if fold_row["SMILES"] != row["SMILES"]:
            raise ValueError(f"Fold SMILES differs for sample_id={row['sample_id']}.")
    fold_ids = np.asarray([assignments[sample_id] for sample_id in ids], dtype=np.int8)
    if set(fold_ids.tolist()) != set(range(5)):
        raise ValueError("Frozen folds must contain exactly fold IDs 0 through 4.")
    return fold_ids, assignments


def _mol_for_smiles(smiles: str, chem):
    mol = chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES in frozen training data: {smiles!r}")
    return mol


def build_features_and_fingerprints(smiles_values: Sequence[str]):
    """Build fixed RDKit 2D descriptors and ECFP4 bits from each row's SMILES."""
    (
        rdkit,
        chem,
        data_structs,
        descriptors,
        fingerprint_generator,
    ) = _rdkit()
    generator = fingerprint_generator.GetMorganGenerator(
        radius=MORGAN_RADIUS, fpSize=MORGAN_BITS
    )
    descriptor_list = list(descriptors._descList)
    element_symbols = [chem.GetPeriodicTable().GetElementSymbol(z) for z in range(1, 119)]
    custom_names = [
        "custom_smiles_length",
        "custom_atom_count",
        "custom_heavy_atom_count",
        "custom_dummy_atom_count",
        "custom_aromatic_atom_count",
        "custom_hetero_atom_count",
        "custom_bond_count",
        "custom_ring_count",
        "custom_star_neighbor_distance",
    ]
    custom_names += [f"element_count_{symbol}" for symbol in element_symbols]
    custom_names += [f"element_fraction_{symbol}" for symbol in element_symbols]
    descriptor_names = [f"rdkit_{name}" for name, _ in descriptor_list]
    fingerprint_names = [f"morgan_r{MORGAN_RADIUS}_{i}" for i in range(MORGAN_BITS)]
    columns = descriptor_names + custom_names + fingerprint_names

    feature_rows: list[np.ndarray] = []
    fingerprints = []
    canonical: list[str] = []
    canonical_achiral: list[str] = []
    heavy_atoms = np.empty(len(smiles_values), dtype=np.int32)
    fp_bits = np.empty((len(smiles_values), MORGAN_BITS), dtype=np.uint8)

    for row_index, smiles in enumerate(smiles_values):
        mol = _mol_for_smiles(smiles, chem)
        desc_values: list[float] = []
        for _, descriptor in descriptor_list:
            try:
                value = float(descriptor(mol))
            except Exception:
                value = float("nan")
            if not math.isfinite(value):
                value = float("nan")
            desc_values.append(value)

        atoms = list(mol.GetAtoms())
        real_heavy = sum(atom.GetAtomicNum() > 1 for atom in atoms)
        heavy_atoms[row_index] = real_heavy
        element_counts = np.zeros(len(element_symbols), dtype=np.float32)
        z_to_col = {z: z - 1 for z in range(1, 119)}
        for atom in atoms:
            atomic_number = atom.GetAtomicNum()
            if atomic_number > 0:
                element_counts[z_to_col[atomic_number]] += 1
        element_total = float(element_counts.sum())
        stars = [atom.GetIdx() for atom in atoms if atom.GetAtomicNum() == 0]
        star_distance = float("nan")
        if len(stars) == 2 and all(mol.GetAtomWithIdx(idx).GetDegree() == 1 for idx in stars):
            neighbor_a = mol.GetAtomWithIdx(stars[0]).GetNeighbors()[0].GetIdx()
            neighbor_b = mol.GetAtomWithIdx(stars[1]).GetNeighbors()[0].GetIdx()
            try:
                distances = chem.GetDistanceMatrix(mol)
                distance = float(distances[neighbor_a, neighbor_b])
                if math.isfinite(distance) and distance < 1000:
                    star_distance = distance
            except Exception:
                pass
        custom = np.asarray(
            [
                len(smiles),
                len(atoms),
                real_heavy,
                len(stars),
                sum(atom.GetIsAromatic() for atom in atoms),
                sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms),
                mol.GetNumBonds(),
                mol.GetRingInfo().NumRings(),
                star_distance,
            ],
            dtype=np.float64,
        )
        fractions = element_counts / element_total if element_total else np.full_like(element_counts, np.nan)
        fp = generator.GetFingerprint(mol)
        fp_array = np.zeros(MORGAN_BITS, dtype=np.uint8)
        data_structs.ConvertToNumpyArray(fp, fp_array)
        fp_bits[row_index] = fp_array
        features = np.concatenate(
            (np.asarray(desc_values, dtype=np.float64), custom, element_counts, fractions, fp_array)
        )
        float32_max = np.finfo(np.float32).max
        features[~np.isfinite(features) | (np.abs(features) > float32_max)] = np.nan
        features = features.astype(np.float32)
        feature_rows.append(features)
        fingerprints.append(fp)
        mapfree = chem.Mol(mol)
        for atom in mapfree.GetAtoms():
            atom.SetAtomMapNum(0)
        canonical.append(chem.MolToSmiles(mapfree, canonical=True, isomericSmiles=True))
        no_stereo = chem.Mol(mapfree)
        chem.RemoveStereochemistry(no_stereo)
        canonical_achiral.append(
            chem.MolToSmiles(no_stereo, canonical=True, isomericSmiles=False)
        )

    matrix = np.vstack(feature_rows)
    matrix[~np.isfinite(matrix)] = np.nan
    return {
        "features": matrix,
        "feature_columns": columns,
        "fingerprints": fingerprints,
        "fingerprint_bits": fp_bits,
        "canonical_smiles": canonical,
        "canonical_achiral_smiles": canonical_achiral,
        "heavy_atoms": heavy_atoms,
        "rdkit_version": rdkit.__version__,
        "rdkit_descriptor_count": len(descriptor_list),
        "custom_descriptor_count": len(custom_names),
        "fingerprint_bit_count": MORGAN_BITS,
        "fingerprint_radius": MORGAN_RADIUS,
    }


def _target_array(rows: Sequence[Mapping[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    y = np.full((len(rows), len(TARGETS)), np.nan, dtype=np.float64)
    for row_index, row in enumerate(rows):
        for target_index, target in enumerate(TARGETS):
            value = row.get(target)
            if value is not None:
                y[row_index, target_index] = float(value)
    return y, np.isfinite(y)


def median_oof(y: np.ndarray, observed: np.ndarray, fold_ids: np.ndarray) -> np.ndarray:
    prediction = np.full_like(y, np.nan)
    for fold in range(5):
        valid = fold_ids == fold
        train = ~valid
        for target_index in range(len(TARGETS)):
            train_labeled = train & observed[:, target_index]
            if not train_labeled.any():
                raise ValueError(f"No fold-training labels for {TARGETS[target_index]}.")
            prediction[valid, target_index] = np.median(y[train_labeled, target_index])
    return prediction


def knn_oof(
    y: np.ndarray,
    observed: np.ndarray,
    fold_ids: np.ndarray,
    fingerprints: Sequence[Any],
    *,
    k: int = KNN_K,
) -> tuple[np.ndarray, np.ndarray]:
    from rdkit import DataStructs

    prediction = np.full_like(y, np.nan)
    max_similarity = np.full(len(y), np.nan, dtype=np.float64)
    for fold in range(5):
        valid_indices = np.flatnonzero(fold_ids == fold)
        train_indices = np.flatnonzero(fold_ids != fold)
        train_fps = [fingerprints[index] for index in train_indices]
        target_train_positions = {
            target_index: np.flatnonzero(observed[train_indices, target_index])
            for target_index in range(len(TARGETS))
        }
        target_train_labels = {
            target_index: y[train_indices[positions], target_index]
            for target_index, positions in target_train_positions.items()
        }
        target_fallback = {
            target_index: float(np.median(values))
            for target_index, values in target_train_labels.items()
        }
        for valid_index in valid_indices:
            similarities = np.asarray(
                DataStructs.BulkTanimotoSimilarity(fingerprints[valid_index], train_fps),
                dtype=np.float64,
            )
            max_similarity[valid_index] = float(similarities.max(initial=0.0))
            for target_index in range(len(TARGETS)):
                positions = target_train_positions[target_index]
                target_sims = similarities[positions]
                take = min(k, len(positions))
                cutoff = np.partition(target_sims, len(target_sims) - take)[len(target_sims) - take]
                above = np.flatnonzero(target_sims > cutoff)
                ties = np.flatnonzero(target_sims == cutoff)
                top_positions = np.concatenate((above, ties[: take - len(above)]))
                top_positions = top_positions[np.lexsort((top_positions, -target_sims[top_positions]))]
                top_similarities = target_sims[top_positions]
                weights = top_similarities
                if float(weights.sum()) > 0:
                    prediction[valid_index, target_index] = float(
                        np.dot(weights, target_train_labels[target_index][top_positions])
                        / weights.sum()
                    )
                else:
                    prediction[valid_index, target_index] = target_fallback[target_index]
    if not np.isfinite(max_similarity).all():
        raise ValueError("Missing max Tanimoto similarity for an OOF row.")
    return prediction, max_similarity


LIGHTGBM_CONFIG = {
    "objective": "regression_l1",
    "n_estimators": 400,
    "learning_rate": 0.04,
    "num_leaves": 18,
    "max_depth": -1,
    "min_child_samples": 10,
    "subsample": 0.85,
    "subsample_freq": 1,
    "colsample_bytree": 0.5,
    "reg_alpha": 0.08,
    "reg_lambda": 0.8,
    "verbosity": -1,
    "force_col_wise": True,
}


def lightgbm_oof(
    features: np.ndarray,
    feature_names: Sequence[str],
    y: np.ndarray,
    observed: np.ndarray,
    fold_ids: np.ndarray,
    *,
    seed: int = 20250604,
    n_jobs: int = 4,
) -> np.ndarray:
    import lightgbm as lgb
    import pandas as pd
    from sklearn.impute import SimpleImputer

    prediction = np.full_like(y, np.nan)
    for fold in range(5):
        valid_fold = fold_ids == fold
        train_fold = fold_ids != fold
        for target_index, target in enumerate(TARGETS):
            train_indices = np.flatnonzero(train_fold & observed[:, target_index])
            valid_indices = np.flatnonzero(valid_fold)
            if len(train_indices) < 2:
                raise ValueError(f"Not enough fold-training samples for {target}.")
            # Fit imputation exclusively on labeled rows from the fold's training side.
            imputer = SimpleImputer(strategy="median", keep_empty_features=True)
            x_train = pd.DataFrame(
                imputer.fit_transform(features[train_indices]), columns=feature_names
            )
            x_valid = pd.DataFrame(
                imputer.transform(features[valid_indices]), columns=feature_names
            )
            if not np.isfinite(x_train.to_numpy()).all() or not np.isfinite(x_valid.to_numpy()).all():
                raise ValueError(f"Imputation left non-finite values for {target}, fold {fold}.")
            params = {
                **LIGHTGBM_CONFIG,
                "random_state": seed + fold * 101 + target_index,
                "n_jobs": n_jobs,
            }
            model = lgb.LGBMRegressor(**params)
            model.fit(x_train, y[train_indices, target_index])
            prediction[valid_indices, target_index] = model.predict(x_valid)
    if not np.isfinite(prediction).all():
        raise ValueError("LightGBM did not produce a finite prediction for every sample/target.")
    return prediction


def load_oof_csv(path: Path, sample_ids: Sequence[str], fold_ids: np.ndarray) -> np.ndarray:
    with path.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    by_id = {str(row["sample_id"]): row for row in rows}
    if len(rows) != len(sample_ids) or len(by_id) != len(rows):
        raise ValueError(f"Bad OOF row count or duplicate IDs in {path}.")
    if set(by_id) != set(sample_ids):
        raise ValueError(f"OOF sample IDs differ in {path}.")
    prediction = np.empty((len(sample_ids), len(TARGETS)), dtype=np.float64)
    for index, sample_id in enumerate(sample_ids):
        row = by_id[sample_id]
        if int(row["fold"]) != int(fold_ids[index]):
            raise ValueError(f"OOF fold mismatch for {sample_id} in {path}.")
        for target_index, target in enumerate(TARGETS):
            prediction[index, target_index] = float(row[target])
    if not np.isfinite(prediction).all():
        raise ValueError(f"Non-finite OOF predictions in {path}.")
    return prediction


def load_c128_predictions(
    rows: Sequence[Mapping[str, Any]], fold_ids: np.ndarray
) -> tuple[dict[int, np.ndarray], dict[int, dict[str, Any]]]:
    sample_ids = [str(row["sample_id"]) for row in rows]
    y_true = [dict(row) for row in rows]
    target_weights = evaluate_oof(y_true, [dict.fromkeys(TARGETS, 0.0) for _ in rows])["target_weights"]
    predictions: dict[int, np.ndarray] = {}
    metrics: dict[int, dict[str, Any]] = {}
    for seed, artifact_dir in C128_ARTIFACTS.items():
        source_manifest = json.loads((artifact_dir / "source_manifest.json").read_text())
        hashes = source_manifest.get("benchmark_files_sha256", {})
        if hashes.get("train_csv") != EXPECTED_TRAIN_SHA256 or hashes.get("folds_csv") != EXPECTED_FOLDS_SHA256:
            raise ValueError(f"C128 seed {seed} source manifest has different frozen benchmark hashes.")
        prediction = load_oof_csv(artifact_dir / "oof_predictions.csv", sample_ids, fold_ids)
        metric = evaluate_oof(
            y_true,
            [dict(zip(TARGETS, row.tolist())) for row in prediction],
            target_weights=target_weights,
        )
        artifact_metrics = json.loads((artifact_dir / "metrics.json").read_text())
        if not math.isclose(
            metric["overall_oof_wmae"], artifact_metrics["overall_oof_wmae"], rel_tol=0, abs_tol=1e-10
        ):
            raise ValueError(f"C128 seed {seed} metric differs from its recorded metrics.json.")
        predictions[seed] = prediction
        metrics[seed] = metric
    return predictions, metrics


def _metric_for_prediction(
    y: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    weights: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    truth_rows = [
        {target: (float(y[i, j]) if observed[i, j] else None) for j, target in enumerate(TARGETS)}
        for i in range(len(y))
    ]
    pred_rows = [dict(zip(TARGETS, row.tolist())) for row in prediction]
    return evaluate_oof(truth_rows, pred_rows, target_weights=weights)


def _bootstrap_ci(values: np.ndarray, *, n_boot: int, seed: int) -> list[float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Bootstrap input must be a non-empty finite vector.")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    chunk_size = 128
    for start in range(0, n_boot, chunk_size):
        stop = min(start + chunk_size, n_boot)
        indices = rng.integers(0, len(values), size=(stop - start, len(values)))
        means[start:stop] = values[indices].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return [float(low), float(high)]


def _target_statistics(y: np.ndarray, observed: np.ndarray) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for index, target in enumerate(TARGETS):
        values = y[observed[:, index], index]
        result[target] = {
            "count": int(len(values)),
            "mean": float(np.mean(values)),
            "std_sample": float(np.std(values, ddof=1)),
            "median": float(np.median(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "range": float(np.max(values) - np.min(values)),
        }
    return result


def _cooccurrence(observed: np.ndarray) -> dict[str, dict[str, int]]:
    return {
        target_a: {
            target_b: int(np.sum(observed[:, i] & observed[:, j]))
            for j, target_b in enumerate(TARGETS)
        }
        for i, target_a in enumerate(TARGETS)
    }


def _fingerprint_audit(
    sample_ids: Sequence[str],
    smiles: Sequence[str],
    fingerprint_bits: np.ndarray,
    canonical: Sequence[str],
    canonical_achiral: Sequence[str],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    groups: dict[bytes, list[int]] = defaultdict(list)
    for index, bits in enumerate(fingerprint_bits):
        packed = np.packbits(bits, bitorder="little").tobytes()
        groups[packed].append(index)
    duplicate_groups = [indices for indices in groups.values() if len(indices) > 1]
    categories = Counter()
    group_rows: list[dict[str, Any]] = []
    group_examples: list[dict[str, Any]] = []
    for group_number, indices in enumerate(
        sorted(duplicate_groups, key=lambda values: (len(values), values[0])), start=1
    ):
        canonical_set = {canonical[index] for index in indices}
        achiral_set = {canonical_achiral[index] for index in indices}
        if len(canonical_set) == 1:
            category = "same_isomeric_canonical_graph"
        elif len(achiral_set) == 1:
            category = "stereochemical_variants_same_achiral_graph"
        else:
            category = "different_achiral_canonical_graphs"
        categories[category] += 1
        group_rows.append(
            {
                "fingerprint_group": group_number,
                "group_size": len(indices),
                "distinct_isomeric_canonical_smiles": len(canonical_set),
                "distinct_achiral_canonical_smiles": len(achiral_set),
                "category": category,
                "polymer_equivalence": "UNKNOWN",
                "sample_ids": "|".join(sample_ids[index] for index in indices),
                "smiles": " || ".join(smiles[index] for index in indices),
                "canonical_isomeric_smiles": " || ".join(sorted(canonical_set)),
                "canonical_achiral_smiles": " || ".join(sorted(achiral_set)),
            }
        )
        if category != "same_isomeric_canonical_graph":
            group_examples.append(
                {
                    "fingerprint_group": group_number,
                    "group_size": len(indices),
                    "category": category,
                    "samples": [
                        {
                            "sample_id": sample_ids[index],
                            "SMILES": smiles[index],
                            "canonical_isomeric_SMILES": canonical[index],
                            "canonical_achiral_SMILES": canonical_achiral[index],
                        }
                        for index in indices[:6]
                    ],
                    "polymer_equivalence": "UNKNOWN",
                }
            )
    size_counts = Counter(len(indices) for indices in duplicate_groups)
    # Systematic examples cover a small group, stereochemical duplicates, and
    # progressively larger groups where distinct repeat-unit lengths collide.
    selected_examples: list[dict[str, Any]] = []
    for category in (
        "different_achiral_canonical_graphs",
        "stereochemical_variants_same_achiral_graph",
    ):
        match = next((item for item in group_examples if item["category"] == category), None)
        if match is not None and match not in selected_examples:
            selected_examples.append(match)
    for minimum_size in (3, 10, 25, 50):
        match = next(
            (
                item
                for item in group_examples
                if item["category"] == "different_achiral_canonical_graphs"
                and item["group_size"] >= minimum_size
                and item not in selected_examples
            ),
            None,
        )
        if match is not None:
            selected_examples.append(match)
    examples = selected_examples[:8]
    return (
        {
            "fingerprint": f"RDKit Morgan radius {MORGAN_RADIUS}, {MORGAN_BITS}-bit, chirality disabled",
            "duplicate_fingerprint_group_count": len(duplicate_groups),
            "samples_in_duplicate_groups": int(sum(len(indices) for indices in duplicate_groups)),
            "group_size_distribution": {str(size): int(count) for size, count in sorted(size_counts.items())},
            "groups_same_isomeric_canonical_graph": int(categories["same_isomeric_canonical_graph"]),
            "groups_stereochemical_variants_same_achiral_graph": int(categories["stereochemical_variants_same_achiral_graph"]),
            "groups_different_achiral_canonical_graphs": int(categories["different_achiral_canonical_graphs"]),
            "canonicalization": "RDKit canonical SMILES; isomeric and atom-map-insensitive achiral views",
            "polymer_equivalence": "UNKNOWN for every fingerprint group; molecular graph identity does not prove repeat-unit/polymer equivalence",
            "audited_examples": examples,
        },
        group_rows,
    )


def _metric_summary(metric: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "overall_oof_wmae": float(metric["overall_oof_wmae"]),
        "target_mae": {target: float(metric["target_mae"][target]) for target in TARGETS},
        "target_contribution": {
            target: float(metric["target_contribution"][target]) for target in TARGETS
        },
    }


def _score_reports(
    y: np.ndarray,
    observed: np.ndarray,
    predictions: Mapping[str, np.ndarray],
    c128_predictions: Mapping[int, np.ndarray],
    c128_metrics: Mapping[int, Mapping[str, Any]],
    weights: Mapping[str, Mapping[str, Any]],
    n_boot: int,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, np.ndarray]]:
    seed_ids = sorted(c128_predictions)
    c128_abs_by_seed = {
        seed: np.where(observed, np.abs(c128_predictions[seed] - y), 0.0)
        for seed in seed_ids
    }
    c128_abs_mean = np.mean(np.stack(list(c128_abs_by_seed.values())), axis=0)
    c128_mean_prediction = np.mean(np.stack([c128_predictions[seed] for seed in seed_ids]), axis=0)
    c128_report = {
        "mean_oof_wmae": float(np.mean([c128_metrics[seed]["overall_oof_wmae"] for seed in seed_ids])),
        "sample_sd_oof_wmae": float(np.std([c128_metrics[seed]["overall_oof_wmae"] for seed in seed_ids], ddof=1)),
        "seed_scores": {str(seed): float(c128_metrics[seed]["overall_oof_wmae"]) for seed in seed_ids},
        "target_mae_mean": {
            target: float(np.mean([c128_metrics[seed]["target_mae"][target] for seed in seed_ids]))
            for target in TARGETS
        },
        "target_mae_sample_sd": {
            target: float(np.std([c128_metrics[seed]["target_mae"][target] for seed in seed_ids], ddof=1))
            for target in TARGETS
        },
        "target_contribution_mean": {
            target: float(np.mean([c128_metrics[seed]["target_contribution"][target] for seed in seed_ids]))
            for target in TARGETS
        },
        "sample_diagnostic_definition": "For paired bootstrap and similarity strata, per-sample absolute error is averaged across the five existing C128 OOF seeds; seed variability is reported separately.",
        "seed_averaged_prediction_oof_wmae_diagnostic_only": _metric_for_prediction(
            y, observed, c128_mean_prediction, weights
        )["overall_oof_wmae"],
    }
    reports: dict[str, Any] = {
        "C128 Own-GNN": {
            "n_samples": len(y),
            "overall_oof_wmae": c128_report["mean_oof_wmae"],
            "seed_sd_oof_wmae": c128_report["sample_sd_oof_wmae"],
            "target_mae": c128_report["target_mae_mean"],
            "target_mae_seed_sd": c128_report["target_mae_sample_sd"],
            "target_contribution": c128_report["target_contribution_mean"],
        }
    }
    csv_rows: list[dict[str, Any]] = []
    for name, prediction in predictions.items():
        metric = _metric_for_prediction(y, observed, prediction, weights)
        reports[name] = {"n_samples": len(y), **_metric_summary(metric)}
        comparison = {
            "wmae_delta_vs_C128": float(
                metric["overall_oof_wmae"] - c128_report["mean_oof_wmae"]
            ),
            "paired_bootstrap_95ci_wmae_delta": _bootstrap_ci(
                _paired_row_contributions(y, observed, prediction, c128_abs_mean, weights),
                n_boot=n_boot,
                seed=BOOTSTRAP_SEED + len(csv_rows),
            ),
            "target_delta_vs_C128": {},
        }
        for target_index, target in enumerate(TARGETS):
            delta = (
                np.abs(prediction[observed[:, target_index], target_index] - y[observed[:, target_index], target_index])
                - c128_abs_mean[observed[:, target_index], target_index]
            )
            comparison["target_delta_vs_C128"][target] = {
                "mean_absolute_error_delta": float(delta.mean()),
                "paired_bootstrap_95ci": _bootstrap_ci(
                    delta,
                    n_boot=n_boot,
                    seed=BOOTSTRAP_SEED + 100 + 10 * len(csv_rows) + target_index,
                ),
            }
        reports[name]["paired_comparison_vs_C128"] = comparison
        row: dict[str, Any] = {
            "model": name,
            "n_samples": len(y),
            "overall_oof_wmae": reports[name]["overall_oof_wmae"],
            "seed_sd_oof_wmae": "",
            "paired_wmae_delta_vs_C128": comparison["wmae_delta_vs_C128"],
            "paired_wmae_delta_ci95_low": comparison["paired_bootstrap_95ci_wmae_delta"][0],
            "paired_wmae_delta_ci95_high": comparison["paired_bootstrap_95ci_wmae_delta"][1],
        }
        for target in TARGETS:
            row[f"{target}_mae"] = metric["target_mae"][target]
            row[f"{target}_delta_vs_C128"] = comparison["target_delta_vs_C128"][target]["mean_absolute_error_delta"]
            row[f"{target}_delta_ci95_low"] = comparison["target_delta_vs_C128"][target]["paired_bootstrap_95ci"][0]
            row[f"{target}_delta_ci95_high"] = comparison["target_delta_vs_C128"][target]["paired_bootstrap_95ci"][1]
        csv_rows.append(row)

    for name, score_key in (("C128 Own-GNN", "128"), ("C256/G0 historical", "256")):
        if name == "C128 Own-GNN":
            csv_rows.append(
                {
                    "model": name,
                    "n_samples": len(y),
                    "overall_oof_wmae": c128_report["mean_oof_wmae"],
                    "seed_sd_oof_wmae": c128_report["sample_sd_oof_wmae"],
                    **{f"{target}_mae": c128_report["target_mae_mean"][target] for target in TARGETS},
                }
            )
        else:
            summary = json.loads(STAGE5A_SUMMARY_PATH.read_text())
            width = summary["widths"][score_key]
            reports[name] = {
                "n_samples": len(y),
                "overall_oof_wmae": float(width["mean_oof_wmae"]),
                "seed_sd_oof_wmae": float(width["sample_sd_oof_wmae"]),
                "seed_scores": width["by_seed_oof_wmae"],
                "target_mae": {target: float(width["per_target"][target]["mean"]) for target in TARGETS},
                "target_mae_seed_sd": {target: float(width["per_target"][target]["sample_sd"]) for target in TARGETS},
                "historical_source": "from_scratch_gnn/experiments/stage5a/aggregate_summary.json; no C256 sample-level OOF artifacts were present in this checkout",
            }
            csv_rows.append(
                {
                    "model": name,
                    "n_samples": len(y),
                    "overall_oof_wmae": width["mean_oof_wmae"],
                    "seed_sd_oof_wmae": width["sample_sd_oof_wmae"],
                    **{f"{target}_mae": width["per_target"][target]["mean"] for target in TARGETS},
                }
            )
    reports["C128 Own-GNN"].update(c128_report)
    return reports, csv_rows, {"C128_seed_mean_prediction": c128_mean_prediction, "C128_seed_mean_absolute_error": c128_abs_mean}


def _paired_row_contributions(
    y: np.ndarray,
    observed: np.ndarray,
    prediction: np.ndarray,
    c128_mean_abs: np.ndarray,
    weights: Mapping[str, Mapping[str, Any]],
) -> np.ndarray:
    delta = np.zeros(len(y), dtype=np.float64)
    for target_index, target in enumerate(TARGETS):
        rows = observed[:, target_index]
        delta[rows] += float(weights[target]["weight"]) * (
            np.abs(prediction[rows, target_index] - y[rows, target_index])
            - c128_mean_abs[rows, target_index]
        )
    return delta


def _similarity_diagnostics(
    similarity: np.ndarray,
    observed: np.ndarray,
    y: np.ndarray,
    c128_mean_abs: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]], np.ndarray]:
    bin_index = np.full(len(similarity), -1, dtype=np.int8)
    bin_sizes: dict[str, int] = {}
    target_report: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for index, (name, lower, upper) in enumerate(SIMILARITY_BINS):
        if index == 2:
            mask = similarity > lower
        elif index == 1:
            mask = (similarity >= lower) & (similarity <= upper)
        else:
            mask = (similarity >= lower) & (similarity < upper)
        bin_index[mask] = index
        bin_sizes[name] = int(mask.sum())
        target_report[name] = {
            "sample_count": int(mask.sum()),
            "max_similarity_min": float(similarity[mask].min()) if mask.any() else None,
            "max_similarity_max": float(similarity[mask].max()) if mask.any() else None,
            "per_target": {},
        }
        for target_index, target in enumerate(TARGETS):
            selected = mask & observed[:, target_index]
            mae = float(c128_mean_abs[selected, target_index].mean()) if selected.any() else None
            target_report[name]["per_target"][target] = {
                "count": int(selected.sum()),
                "mae": mae,
            }
            rows.append(
                {
                    "similarity_bin": name,
                    "target": target,
                    "sample_count": int(selected.sum()),
                    "max_tanimoto_sample_count": int(mask.sum()),
                    "C128_mean_seed_absolute_error": mae,
                }
            )
    target_report["bin_sample_counts"] = bin_sizes
    return target_report, rows, bin_index


def _label_and_structure_diagnostics(
    rows: Sequence[Mapping[str, Any]], y: np.ndarray, observed: np.ndarray, heavy_atoms: np.ndarray
) -> tuple[dict[str, Any], dict[str, Any]]:
    targets = _target_statistics(y, observed)
    labels = {"targets": targets, "cooccurrence_counts": _cooccurrence(observed)}
    heavy_atom_stats = {
        "count": int(len(heavy_atoms)),
        "mean": float(np.mean(heavy_atoms)),
        "std_sample": float(np.std(heavy_atoms, ddof=1)),
        "median": float(np.median(heavy_atoms)),
        "min": int(np.min(heavy_atoms)),
        "max": int(np.max(heavy_atoms)),
        "quantiles": {
            str(q): float(np.quantile(heavy_atoms, q)) for q in (0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
        },
        "count_by_heavy_atom_count": {
            str(count): int(freq) for count, freq in sorted(Counter(heavy_atoms.tolist()).items())
        },
        "definition": "Number of atoms with atomic number > 1; excludes H and polymer dummy atoms (atomic number 0).",
    }
    return labels, heavy_atom_stats


def run_benchmark(
    train_csv: Path,
    output_dir: Path,
    *,
    n_boot: int = BOOTSTRAP_REPLICATES,
    n_jobs: int = 4,
) -> dict[str, Any]:
    train_hash = sha256_file(train_csv)
    if train_hash != EXPECTED_TRAIN_SHA256:
        raise ValueError("Train CSV SHA256 does not match frozen nopp2025_train_v1.")
    rows = load_training_data(train_csv)
    if len(rows) != EXPECTED_SAMPLES:
        raise ValueError(f"Expected {EXPECTED_SAMPLES} frozen training rows, got {len(rows)}.")
    fold_ids, fold_assignments = load_frozen_folds(rows)
    sample_ids = [str(row["sample_id"]) for row in rows]
    smiles = [str(row["SMILES"]) for row in rows]
    y, observed = _target_array(rows)

    print("Loading and verifying existing C128 OOF predictions…", flush=True)
    c128_predictions, c128_metrics = load_c128_predictions(rows, fold_ids)
    print("Computing SMILES-only RDKit descriptors and fingerprints…", flush=True)
    feature_data = build_features_and_fingerprints(smiles)
    print("Running fold-local Tanimoto kNN baseline…", flush=True)
    knn_prediction, max_similarity = knn_oof(
        y, observed, fold_ids, feature_data["fingerprints"], k=KNN_K
    )
    print("Running per-target, fold-local LightGBM baseline…", flush=True)
    lgbm_prediction = lightgbm_oof(
        feature_data["features"], feature_data["feature_columns"], y, observed, fold_ids, n_jobs=n_jobs
    )
    median_prediction = median_oof(y, observed, fold_ids)

    labels_report, heavy_atom_report = _label_and_structure_diagnostics(
        rows, y, observed, feature_data["heavy_atoms"]
    )
    duplicate_report, duplicate_rows = _fingerprint_audit(
        sample_ids,
        smiles,
        feature_data["fingerprint_bits"],
        feature_data["canonical_smiles"],
        feature_data["canonical_achiral_smiles"],
    )

    # Derive and verify the frozen local metric weights from the benchmark labels.
    zero_prediction = np.zeros_like(y)
    weights = _metric_for_prediction(y, observed, zero_prediction, evaluate_oof(
        [dict(row) for row in rows],
        [dict.fromkeys(TARGETS, 0.0) for _ in rows],
    )["target_weights"])["target_weights"]
    model_predictions = {
        "Median": median_prediction,
        "Tanimoto kNN (k=5)": knn_prediction,
        "Clean LightGBM": lgbm_prediction,
    }
    comparisons, comparison_csv, sample_reference = _score_reports(
        y, observed, model_predictions, c128_predictions, c128_metrics, weights, n_boot
    )
    c128_abs = sample_reference["C128_seed_mean_absolute_error"]
    similarity_report, similarity_csv, similarity_bin = _similarity_diagnostics(
        max_similarity, observed, y, c128_abs
    )
    c128_ensemble_prediction = sample_reference["C128_seed_mean_prediction"]
    fixed_half_blend = 0.5 * c128_ensemble_prediction + 0.5 * lgbm_prediction
    c128_ensemble_metric = _metric_for_prediction(y, observed, c128_ensemble_prediction, weights)
    blend_metric = _metric_for_prediction(y, observed, fixed_half_blend, weights)
    c128_ensemble_abs = np.where(
        observed, np.abs(c128_ensemble_prediction - y), 0.0
    )
    blend_delta_rows = _paired_row_contributions(
        y, observed, fixed_half_blend, c128_ensemble_abs, weights
    )
    fusion_target_rows: list[dict[str, Any]] = []
    fusion_target_report: dict[str, Any] = {}
    for target_index, target in enumerate(TARGETS):
        selected = observed[:, target_index]
        c128_residual = c128_ensemble_prediction[selected, target_index] - y[selected, target_index]
        lgbm_residual = lgbm_prediction[selected, target_index] - y[selected, target_index]
        c128_abs_error = np.abs(c128_residual)
        lgbm_abs_error = np.abs(lgbm_residual)
        blend_target_delta = float(
            blend_metric["target_mae"][target]
            - c128_ensemble_metric["target_mae"][target]
        )
        fusion_target_report[target] = {
            "count": int(selected.sum()),
            "absolute_error_pearson_correlation": float(np.corrcoef(c128_abs_error, lgbm_abs_error)[0, 1]),
            "signed_residual_pearson_correlation": float(np.corrcoef(c128_residual, lgbm_residual)[0, 1]),
            "C128_seed_averaged_prediction_mae": float(c128_ensemble_metric["target_mae"][target]),
            "LightGBM_mae": float(comparisons["Clean LightGBM"]["target_mae"][target]),
            "fixed_50_50_blend_mae": float(blend_metric["target_mae"][target]),
            "blend_mae_delta_vs_C128_seed_averaged_prediction": blend_target_delta,
        }
        fusion_target_rows.append({"target": target, **fusion_target_report[target]})
    fusion_diagnostics = {
        "scope": "descriptive, fixed-weight OOF error-complementarity check; no blend weight was tuned",
        "blend": "0.5 * five-seed C128 OOF mean prediction + 0.5 * Clean LightGBM OOF prediction",
        "C128_seed_averaged_prediction_wmae": float(c128_ensemble_metric["overall_oof_wmae"]),
        "LightGBM_wmae": float(comparisons["Clean LightGBM"]["overall_oof_wmae"]),
        "fixed_50_50_blend_wmae": float(blend_metric["overall_oof_wmae"]),
        "blend_wmae_delta_vs_C128_seed_averaged_prediction": float(
            blend_metric["overall_oof_wmae"] - c128_ensemble_metric["overall_oof_wmae"]
        ),
        "paired_sample_bootstrap_95ci_blend_delta": _bootstrap_ci(
            blend_delta_rows, n_boot=n_boot, seed=BOOTSTRAP_SEED + 9000
        ),
        "per_target": fusion_target_report,
    }
    bootstrap_rows = {}
    for model_name, prediction in model_predictions.items():
        bootstrap_rows[model_name] = _paired_row_contributions(
            y, observed, prediction, c128_abs, weights
        )
    for model_name, values in bootstrap_rows.items():
        expected_delta = comparisons[model_name]["paired_comparison_vs_C128"][
            "wmae_delta_vs_C128"
        ]
        if not math.isclose(float(values.mean()), expected_delta, rel_tol=0, abs_tol=1e-12):
            raise ValueError(f"Paired sample contributions do not reconcile for {model_name}.")

    # Per-target contributions for C128 follow the authoritative weighted metric.
    c128_contribution = comparisons["C128 Own-GNN"]["target_contribution_mean"]
    contribution_sum = sum(c128_contribution.values())
    if not math.isclose(contribution_sum, comparisons["C128 Own-GNN"]["overall_oof_wmae"], rel_tol=0, abs_tol=1e-10):
        raise ValueError("C128 target contributions do not sum to its mean wMAE.")

    c128_mean_prediction = sample_reference["C128_seed_mean_prediction"]
    sample_rows: list[dict[str, Any]] = []
    target_model_preds = {
        "Median": median_prediction,
        "kNN": knn_prediction,
        "LightGBM": lgbm_prediction,
    }
    row_contribution = np.zeros(len(rows), dtype=np.float64)
    for target_index, target in enumerate(TARGETS):
        selected = observed[:, target_index]
        row_contribution[selected] += (
            float(weights[target]["weight"])
            * c128_abs[selected, target_index]
            / len(rows)
        )
    for index, row in enumerate(rows):
        sample_row: dict[str, Any] = {
            "sample_id": sample_ids[index],
            "fold": int(fold_ids[index]),
            "SMILES": smiles[index],
            "repeat_unit_heavy_atom_count": int(feature_data["heavy_atoms"][index]),
            "max_tanimoto_to_fold_training_set": float(max_similarity[index]),
            "similarity_bin": SIMILARITY_BINS[int(similarity_bin[index])][0],
            "C128_mean_seed_weighted_error_contribution": float(row_contribution[index]),
        }
        for target_index, target in enumerate(TARGETS):
            sample_row[f"true_{target}"] = float(y[index, target_index]) if observed[index, target_index] else ""
            sample_row[f"C128_seed_mean_prediction_{target}"] = float(c128_mean_prediction[index, target_index])
            sample_row[f"C128_seed_mean_absolute_error_{target}"] = (
                float(c128_abs[index, target_index]) if observed[index, target_index] else ""
            )
            for model_name, prediction in target_model_preds.items():
                sample_row[f"{model_name}_prediction_{target}"] = float(prediction[index, target_index])
                sample_row[f"{model_name}_absolute_error_{target}"] = (
                    float(abs(prediction[index, target_index] - y[index, target_index]))
                    if observed[index, target_index]
                    else ""
                )
        sample_rows.append(sample_row)

    top_errors = sorted(
        (
            {
                "sample_id": sample_ids[index],
                "fold": int(fold_ids[index]),
                "max_tanimoto_to_fold_training_set": float(max_similarity[index]),
                "weighted_error_contribution": float(row_contribution[index]),
                "target_absolute_errors": {
                    target: (float(c128_abs[index, j]) if observed[index, j] else None)
                    for j, target in enumerate(TARGETS)
                },
            }
            for index in range(len(rows))
        ),
        key=lambda item: item["weighted_error_contribution"],
        reverse=True,
    )[:20]

    metric_weights = {
        target: {
            "valid_count": int(weights[target]["valid_count"]),
            "value_range": float(weights[target]["value_range"]),
            "task_balance_factor": float(weights[target]["task_balance_factor"]),
            "weight": float(weights[target]["weight"]),
            "weight_times_n_over_N": float(
                weights[target]["weight"] * weights[target]["valid_count"] / len(rows)
            ),
            "C128_mean_seed_target_mae": float(comparisons["C128 Own-GNN"]["target_mae_mean"][target]),
            "C128_mean_seed_overall_wmae_contribution": float(c128_contribution[target]),
            "share_of_C128_mean_seed_wmae": float(c128_contribution[target] / contribution_sum),
        }
        for target in TARGETS
    }

    stage5a = json.loads(STAGE5A_SUMMARY_PATH.read_text())
    result = {
        "stage": "Stage R: Benchmark Reality Check",
        "benchmark": {
            "version": "nopp2025_train_v1",
            "sample_count": len(rows),
            "train_sha256": train_hash,
            "folds_sha256": sha256_file(FOLDS_PATH),
            "fold_sizes": {str(fold): int(np.sum(fold_ids == fold)) for fold in range(5)},
            "fold_strategy": "frozen balanced random sample-level five-fold; seed 20250604",
            "metric_implementation": "from_scratch_gnn/src/metrics.py::evaluate_oof",
            "weights": metric_weights,
            "validation_rows": "Each baseline validation prediction is generated from the corresponding fold's other four folds; the C128 sample-level reference averages existing OOF absolute errors across five seeds.",
        },
        "source": {
            "base_commit": "23c224f3697374305c734f3c050156e0b8ba814a",
            "C128_source_commit": "72490c1a748ed9395025f4120c9eb04f28268695",
            "analysis_script_sha256": sha256_file(Path(__file__).resolve()),
            "C128_seed_scores": comparisons["C128 Own-GNN"]["seed_scores"],
            "C128_oof_artifact_attempts": {str(seed): str(C128_ARTIFACTS[seed].relative_to(REPOSITORY_ROOT)) for seed in sorted(C128_ARTIFACTS)},
            "C256_historical_source": "from_scratch_gnn/experiments/stage5a/aggregate_summary.json; no local per-sample C256 OOF used",
        },
        "label_diagnostics": labels_report,
        "repeat_unit_heavy_atom_count": heavy_atom_report,
        "similarity_diagnostics": {
            "method": f"RDKit Morgan radius {MORGAN_RADIUS}, {MORGAN_BITS}-bit; maximum Tanimoto against all samples in the OOF fold's training side",
            "distribution": {
                "mean": float(np.mean(max_similarity)),
                "std_sample": float(np.std(max_similarity, ddof=1)),
                "quantiles": {str(q): float(np.quantile(max_similarity, q)) for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)},
            },
            "strata": similarity_report,
            "interpretation": "This is an ordinary random sample-level split; similarity strata describe interpolation versus lower-similarity chemistry and do not by themselves establish leakage.",
        },
        "morgan_duplicate_audit": duplicate_report,
        "fusion_diagnostics": fusion_diagnostics,
        "conclusions": [
            "C128 is far ahead of the fold-median and kNN baselines; its advantage over clean LightGBM is smaller but the paired sample-bootstrap interval remains above zero.",
            "C128 is better than LightGBM on FFV, Tc, and Density; Tg and Rg are close and their paired intervals include zero.",
            "C128 and LightGBM errors are strongly correlated; a fixed 50:50 blend does not improve on the C128 seed-averaged prediction, so there is no evidence for a default blend.",
            "Higher similarity aligns with lower FFV and Density errors, while Tc is best in the middle similarity bin and Rg is nearly flat; performance is not uniformly concentrated in high-similarity samples.",
            "There are 545 duplicate Morgan fingerprint groups spanning 1,717 samples; none share an isomeric canonical molecular graph, and polymer equivalence remains UNKNOWN.",
            "Tg, Tc, and Rg contribute 76.0% of C128's mean wMAE; FFV overlaps with Tc, Density, and Rg on 270–300 samples but overlaps with Tg once.",
            "Prioritize a separate similarity-disjoint benchmark. Keep single-task/multitask and FFV transfer as controlled hypotheses; do not promote a fixed GNN+tree blend or begin another architecture search from this result.",
        ],
        "models": {
            "Median": {
                **comparisons["Median"],
                "method": "For each fold and target, median of observed target labels on that fold's training side.",
            },
            "Tanimoto kNN (k=5)": {
                **comparisons["Tanimoto kNN (k=5)"],
                "method": "Morgan radius 2, 2048-bit; target-labeled fold-training rows only; top-5 direct Tanimoto-weighted target values; training-fold target median when all top-5 similarities are zero.",
            },
            "Clean LightGBM": {
                **comparisons["Clean LightGBM"],
                "method": "One independent LightGBM regressor per target and fold; training rows for that target only; median imputer fit on target-labeled training rows in that fold.",
                "features": "RDKit 2D descriptors (Descriptors._descList), deterministic atom/element/dummy/endpoint descriptors adapted from kaggle_notebook_submission_gnn3_physics3d_blend.py, and Morgan radius-2 2048-bit fingerprints; all depend only on that row's SMILES.",
                "feature_count": int(feature_data["features"].shape[1]),
                "rdkit_version": feature_data["rdkit_version"],
                "rdkit_descriptor_count": feature_data["rdkit_descriptor_count"],
                "custom_descriptor_count": feature_data["custom_descriptor_count"],
                "fingerprint_bits": MORGAN_BITS,
                "uses_3d": False,
                "uses_external_or_supplementary_data": False,
                "uses_private_or_released_labels": False,
                "validation_used_for_fit_or_early_stopping": False,
                "parameters": {**LIGHTGBM_CONFIG, "n_jobs": n_jobs, "seed": 20250604},
            },
            "C128 Own-GNN": {
                **comparisons["C128 Own-GNN"],
                "method": "Existing five-seed frozen OOF predictions reused without retraining.",
            },
            "C256/G0 historical": comparisons["C256/G0 historical"],
        },
        "paired_bootstrap": {
            "replicates": n_boot,
            "confidence_level": 0.95,
            "unit": "resampled sample_id rows; paired per-sample weighted-error differences",
            "seed": BOOTSTRAP_SEED,
            "C128_reference": "mean per-sample absolute error across five C128 OOF seeds; interval reflects sample resampling, not model-seed uncertainty",
            "comparisons": {
                model: comparisons[model]["paired_comparison_vs_C128"]
                for model in model_predictions
            },
        },
        "top_20_C128_weighted_error_samples": top_errors,
        "files": {
            "aggregate_markdown": "aggregate_summary.md",
            "aggregate_json": "aggregate_summary.json",
            "baseline_comparison_csv": "baseline_comparison.csv",
            "sample_diagnostics_csv": "sample_diagnostics.csv",
            "similarity_strata_csv": "similarity_strata.csv",
            "morgan_duplicate_groups_csv": "morgan_duplicate_groups.csv",
            "fusion_diagnostics_csv": "fusion_diagnostics.csv",
        },
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "aggregate_summary.json", result)
    comparison_columns = [
        "model", "n_samples", "overall_oof_wmae", "seed_sd_oof_wmae",
        "paired_wmae_delta_vs_C128", "paired_wmae_delta_ci95_low", "paired_wmae_delta_ci95_high",
    ]
    for target in TARGETS:
        comparison_columns.extend(
            [f"{target}_mae", f"{target}_delta_vs_C128", f"{target}_delta_ci95_low", f"{target}_delta_ci95_high"]
        )
    write_csv(output_dir / "baseline_comparison.csv", comparison_csv, comparison_columns)
    sample_columns = list(sample_rows[0]) if sample_rows else []
    write_csv(output_dir / "sample_diagnostics.csv", sample_rows, sample_columns)
    write_csv(
        output_dir / "similarity_strata.csv",
        similarity_csv,
        ["similarity_bin", "target", "sample_count", "max_tanimoto_sample_count", "C128_mean_seed_absolute_error"],
    )
    write_csv(
        output_dir / "morgan_duplicate_groups.csv",
        duplicate_rows,
        [
            "fingerprint_group", "group_size", "distinct_isomeric_canonical_smiles",
            "distinct_achiral_canonical_smiles", "category", "polymer_equivalence",
            "sample_ids", "smiles", "canonical_isomeric_smiles", "canonical_achiral_smiles",
        ],
    )
    write_csv(
        output_dir / "fusion_diagnostics.csv",
        fusion_target_rows,
        [
            "target", "count", "absolute_error_pearson_correlation",
            "signed_residual_pearson_correlation", "C128_seed_averaged_prediction_mae",
            "LightGBM_mae", "fixed_50_50_blend_mae",
            "blend_mae_delta_vs_C128_seed_averaged_prediction",
        ],
    )
    (output_dir / "aggregate_summary.md").write_text(render_markdown(result), encoding="utf-8")
    return result


def _fmt(value: Any, digits: int = 6) -> str:
    if value is None or value == "":
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}g}"
    return str(value)


def render_markdown(result: Mapping[str, Any]) -> str:
    models = result["models"]
    comparisons = result["paired_bootstrap"]["comparisons"]
    targets = TARGETS
    lines = [
        "# Stage R — Benchmark Reality Check",
        "",
        f"Frozen benchmark: `{result['benchmark']['version']}`, {result['benchmark']['sample_count']:,} samples; train SHA256 `{result['benchmark']['train_sha256']}`; folds SHA256 `{result['benchmark']['folds_sha256']}`.",
        "",
        "All baselines use the same frozen five folds and `evaluate_oof`. LightGBM and kNN predictions are generated without validation labels entering model fit or preprocessing. C128 predictions are existing five-seed OOF artifacts; no GNN was trained in Stage R.",
        "",
        "## OOF comparison",
        "",
        "| Model | Overall OOF wMAE | Tg | FFV | Tc | Density | Rg |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    order = ["Median", "Tanimoto kNN (k=5)", "Clean LightGBM", "C128 Own-GNN", "C256/G0 historical"]
    for name in order:
        model = models[name]
        overall = model.get("mean_oof_wmae", model.get("overall_oof_wmae"))
        if name == "C128 Own-GNN":
            overall_text = f"{overall:.8f} ± {model['sample_sd_oof_wmae']:.8f}"
            target_values = model["target_mae_mean"]
        elif name == "C256/G0 historical":
            overall_text = f"{overall:.8f} ± {model['seed_sd_oof_wmae']:.8f}"
            target_values = model["target_mae"]
        else:
            overall_text = f"{overall:.8f}"
            target_values = model["target_mae"]
        lines.append(
            "| " + name + " | " + overall_text + " | " + " | ".join(_fmt(target_values[target], 7) for target in targets) + " |"
        )
    lines += [
        "",
        "A positive paired delta means the baseline has higher error than the five-seed C128 reference. The 95% intervals resample sample IDs and use the mean C128 per-sample absolute error across seeds; they do not include model-seed uncertainty.",
        "",
        "| Baseline | Δ wMAE vs C128 | Paired sample-bootstrap 95% CI |",
        "|---|---:|---:|",
    ]
    for name in ("Median", "Tanimoto kNN (k=5)", "Clean LightGBM"):
        comp = comparisons[name]
        ci = comp["paired_bootstrap_95ci_wmae_delta"]
        lines.append(f"| {name} | {comp['wmae_delta_vs_C128']:+.8f} | [{ci[0]:+.8f}, {ci[1]:+.8f}] |")
    lines += ["", "## Label statistics", "", "| Target | Count | Mean | Sample SD | Median | Min | Max |", "|---|---:|---:|---:|---:|---:|---:|"]
    for target in targets:
        stats = result["label_diagnostics"]["targets"][target]
        lines.append(
            f"| {target} | {stats['count']:,} | {_fmt(stats['mean'])} | {_fmt(stats['std_sample'])} | {_fmt(stats['median'])} | {_fmt(stats['min'])} | {_fmt(stats['max'])} |"
        )
    lines += ["", "### Label co-occurrence counts", "", "| | " + " | ".join(targets) + " |", "|---|" + "---:|" * len(targets)]
    for target in targets:
        row = result["label_diagnostics"]["cooccurrence_counts"][target]
        lines.append("| " + target + " | " + " | ".join(f"{row[other]:,}" for other in targets) + " |")
    lines += ["", "## Frozen metric contribution (C128 mean across seeds)", "", "| Target | n | Frozen weight | Weight × n / N | C128 MAE | wMAE contribution | Share |", "|---|---:|---:|---:|---:|---:|---:|"]
    for target in targets:
        item = result["benchmark"]["weights"][target]
        lines.append(
            f"| {target} | {item['valid_count']:,} | {_fmt(item['weight'], 8)} | {_fmt(item['weight_times_n_over_N'], 7)} | {_fmt(item['C128_mean_seed_target_mae'], 7)} | {_fmt(item['C128_mean_seed_overall_wmae_contribution'], 7)} | {100 * item['share_of_C128_mean_seed_wmae']:.1f}% |"
        )
    heavy = result["repeat_unit_heavy_atom_count"]
    lines += [
        "",
        "## Structural similarity diagnostics",
        "",
        f"Repeat-unit heavy atoms (atomic number > 1; dummy `*` excluded): n={heavy['count']:,}, mean={heavy['mean']:.2f}, median={heavy['median']:.0f}, range={heavy['min']}–{heavy['max']}; p10/p90={heavy['quantiles']['0.1']:.0f}/{heavy['quantiles']['0.9']:.0f}.",
        "",
        f"Maximum Tanimoto uses Morgan radius {MORGAN_RADIUS}, {MORGAN_BITS} bits, against every fold-training SMILES. The split remains an ordinary random sample split; bins characterize higher-similarity interpolation and lower-similarity chemistry, not a leakage finding.",
        "",
        "| Max-similarity bin | All validation samples | " + " | ".join(targets) + " |",
        "|---|---:|" + "---:|" * len(targets),
    ]
    for bin_name, *_ in SIMILARITY_BINS:
        cell_values = []
        section = result["similarity_diagnostics"]["strata"][bin_name]
        for target in targets:
            item = section["per_target"][target]
            cell_values.append(f"{_fmt(item['mae'], 7)} (n={item['count']:,})")
        lines.append(f"| {bin_name} | {section['sample_count']:,} | " + " | ".join(cell_values) + " |")
    lines += ["", "### Largest C128 weighted-error samples", "", "| sample_id | Fold | Max Tanimoto | Weighted contribution | Largest target errors |", "|---|---:|---:|---:|---|"]
    for item in result["top_20_C128_weighted_error_samples"][:10]:
        errors = sorted(
            ((target, value) for target, value in item["target_absolute_errors"].items() if value is not None),
            key=lambda pair: pair[1],
            reverse=True,
        )[:2]
        desc = ", ".join(f"{target} {_fmt(value, 5)}" for target, value in errors)
        lines.append(
            f"| {item['sample_id']} | {item['fold']} | {item['max_tanimoto_to_fold_training_set']:.3f} | {item['weighted_error_contribution']:.7f} | {desc} |"
        )
    duplicate = result["morgan_duplicate_audit"]
    lines += [
        "",
        "## Morgan duplicate audit",
        "",
        f"There are {duplicate['duplicate_fingerprint_group_count']} duplicate Morgan fingerprint groups across {duplicate['samples_in_duplicate_groups']} samples (group sizes: {duplicate['group_size_distribution'] or 'none'}). Canonical graph categories: {duplicate['groups_same_isomeric_canonical_graph']} exact isomeric canonical groups; {duplicate['groups_stereochemical_variants_same_achiral_graph']} stereochemical variants of one achiral graph; {duplicate['groups_different_achiral_canonical_graphs']} groups with multiple achiral canonical graphs.",
        "",
        "A matching fingerprint is not treated as a duplicate molecular graph. Canonicalization is atom-map-insensitive; polymer/repeat-unit equivalence remains **UNKNOWN**. No rows were removed or grouped for training.",
    ]
    for example in duplicate["audited_examples"][:6]:
        lines.append("")
        lines.append(f"- Group {example['fingerprint_group']} ({example['category']}, size {example['group_size']}):")
        for sample in example["samples"][:3]:
            lines.append(f"  - `{sample['sample_id']}` `{sample['SMILES']}` → `{sample['canonical_isomeric_SMILES']}`")
    lines += [
        "",
        "The inspected differing-graph groups include regioisomers and different methylene/ether repeat lengths. The largest group (66 samples) consists of distinct polyamide-like repeat-unit graphs with varying chain lengths that nevertheless produce the same folded radius-2 bit set. The 30 stereochemical-only groups are consistent with this Morgan configuration having chirality disabled. These explain observed cases; the precise bit-collision versus shared-environment mechanism was not separately isolated.",
    ]
    fusion = result["fusion_diagnostics"]
    lines += [
        "",
        "## GNN / LightGBM error complementarity",
        "",
        f"The fixed 50:50 OOF blend scores **{fusion['fixed_50_50_blend_wmae']:.8f}** versus **{fusion['C128_seed_averaged_prediction_wmae']:.8f}** for the five-seed averaged C128 predictions (Δ {fusion['blend_wmae_delta_vs_C128_seed_averaged_prediction']:+.8f}; paired sample-bootstrap 95% CI [{fusion['paired_sample_bootstrap_95ci_blend_delta'][0]:+.8f}, {fusion['paired_sample_bootstrap_95ci_blend_delta'][1]:+.8f}]). The blend weight was fixed at 0.5 and not tuned.",
        "",
        "| Target | Abs-error correlation | Residual correlation | C128 seed-mean MAE | LightGBM MAE | 50:50 blend MAE | Blend Δ vs C128 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in targets:
        item = fusion["per_target"][target]
        lines.append(
            f"| {target} | {item['absolute_error_pearson_correlation']:.3f} | {item['signed_residual_pearson_correlation']:.3f} | {_fmt(item['C128_seed_averaged_prediction_mae'], 7)} | {_fmt(item['LightGBM_mae'], 7)} | {_fmt(item['fixed_50_50_blend_mae'], 7)} | {item['blend_mae_delta_vs_C128_seed_averaged_prediction']:+.7g} |"
        )
    lines += [
        "",
        "## Stage R conclusions",
        "",
        "1. **Baseline gap:** C128 is far ahead of the fold-median and kNN baselines. Its advantage over the clean LightGBM is smaller but its paired sample-bootstrap interval remains above zero (LightGBM − C128 = +0.001008; 95% CI +0.000431 to +0.001592).",
        "2. **Target profile:** C128 is better than LightGBM on FFV, Tc, and Density with paired intervals excluding zero. Tg and Rg are close; their intervals include zero, with LightGBM's point MAE slightly lower.",
        "3. **Fusion:** C128 and LightGBM errors are strongly correlated (per-target absolute-error correlations 0.72–0.86). A fixed 50:50 blend does not improve on the C128 seed-averaged prediction; its paired interval crosses zero. There is no evidence here for a default GNN+tree blend, though target-specific fusion remains testable.",
        "4. **Similarity:** Higher nearest-neighbor similarity aligns with lower FFV and Density errors, while Tc is best in the middle bin and Rg is nearly flat. C128's performance is not uniformly concentrated in high-similarity samples. This random split remains an IID-style benchmark, not a leakage test.",
        "5. **Morgan duplicates:** 545 exact 2048-bit fingerprint groups cover 1,717 rows; none share an isomeric canonical molecular graph. Thirty groups are stereochemical variants of one achiral graph and 515 contain distinct achiral canonical graphs. Polymer equivalence remains UNKNOWN.",
        "6. **Metric leverage:** Tg, Tc, and Rg together contribute 76.0% of C128's mean wMAE despite sparse labels; FFV contributes 15.1% and Density 8.9%. FFV overlaps with Tc, Density, and Rg on 270–300 samples, but with Tg on only one sample.",
        "7. **Next evaluation:** Prioritize a separate similarity-disjoint benchmark before further architecture work. Keep multitask versus single-task and FFV transfer as controlled hypotheses; the current data do not settle them. Deprioritize an unqualified 50:50 GNN+tree blend.",
    ]
    lines += [
        "",
        "## Baseline protocol and limits",
        "",
        "- Median is calculated independently by target and fold from observed training labels only.",
        "- kNN uses the top five target-labeled fold-training polymers and direct Tanimoto weights; if all five similarities are zero it falls back to the fold-training target median.",
        f"- LightGBM uses {models['Clean LightGBM']['feature_count']:,} SMILES-only features and fixed parameters: 400 trees, learning rate 0.04, 18 leaves, 10 minimum child samples, 0.5 column sample. Fold-specific median imputation is fit only on labeled training rows. There is no validation early stopping or tuning.",
        "- The historical Kaggle script's private-label Tg shift, blend weights and all 3D conformer features were excluded. No released/private labels, supplementary data or pseudo-labels were used.",
        "- C256/G0 is included from the Stage 5A aggregate as historical context; local sample-level C256 OOF files were unavailable, so its per-sample bootstrap and similarity strata are not reported.",
        "- All C128 sample-level diagnostics use the five existing seed OOF files. No new GNN training or architecture search was run.",
        "",
        "## Files",
        "",
        "- `aggregate_summary.json` — all model metrics, weights, diagnostics and provenance.",
        "- `baseline_comparison.csv` — comparison table and paired bootstrap intervals.",
        "- `sample_diagnostics.csv` — per-sample labels, predictions, errors, fold, heavy atoms and max similarity.",
        "- `similarity_strata.csv` — per-target MAE and counts by similarity bin.",
        "- `morgan_duplicate_groups.csv` — all duplicate fingerprint groups and canonicalization evidence.",
        "- `fusion_diagnostics.csv` — per-target OOF error correlations and fixed 50:50 blend results.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True, help="Official nopp2025_train_v1 train.csv (not supplementary data).")
    parser.add_argument("--output-dir", type=Path, default=TRACK_ROOT / "experiments" / "stageR")
    parser.add_argument("--bootstrap-replicates", type=int, default=BOOTSTRAP_REPLICATES)
    parser.add_argument("--n-jobs", type=int, default=4)
    args = parser.parse_args()
    if args.bootstrap_replicates < 100:
        parser.error("--bootstrap-replicates must be at least 100")
    result = run_benchmark(args.train_csv, args.output_dir, n_boot=args.bootstrap_replicates, n_jobs=args.n_jobs)
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "models": {
            name: model.get("overall_oof_wmae", model.get("mean_oof_wmae"))
            for name, model in result["models"].items()
        },
        "fingerprint_duplicate_groups": result["morgan_duplicate_audit"]["duplicate_fingerprint_group_count"],
    }, indent=2))


if __name__ == "__main__":
    main()
