"""Deterministic, target-independent Stage 4A feature extraction."""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ...src.data import sha256_file, write_json

DESCRIPTOR_NAMES = (
    "MolWt",
    "MolLogP",
    "MolMR",
    "TPSA",
    "LabuteASA",
    "HeavyAtomCount",
    "NumHeteroatoms",
    "NumHDonors",
    "NumHAcceptors",
    "NumRotatableBonds",
    "RingCount",
    "NumAromaticRings",
    "NumAliphaticRings",
    "NumSaturatedRings",
    "NumAromaticHeterocycles",
    "NumAliphaticHeterocycles",
    "NumSaturatedHeterocycles",
    "NumAromaticCarbocycles",
    "NumAliphaticCarbocycles",
    "FractionCSP3",
)
DESCRIPTOR_DIM = len(DESCRIPTOR_NAMES)
MORGAN_RADIUS = 2
MORGAN_BITS = 2048
MORGAN_CHIRALITY = True


@dataclass(frozen=True)
class FeatureMatrices:
    descriptors: np.ndarray
    morgan: np.ndarray
    manifest: dict[str, Any]


def _canonical_matrix_sha256(matrix: np.ndarray, dtype: str) -> str:
    canonical = np.ascontiguousarray(matrix, dtype=np.dtype(dtype))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _descriptor_functions():
    from rdkit.Chem import Descriptors, Lipinski, rdMolDescriptors

    return (
        Descriptors.MolWt,
        Descriptors.MolLogP,
        Descriptors.MolMR,
        Descriptors.TPSA,
        Descriptors.LabuteASA,
        Descriptors.HeavyAtomCount,
        Lipinski.NumHeteroatoms,
        Lipinski.NumHDonors,
        Lipinski.NumHAcceptors,
        Lipinski.NumRotatableBonds,
        rdMolDescriptors.CalcNumRings,
        rdMolDescriptors.CalcNumAromaticRings,
        rdMolDescriptors.CalcNumAliphaticRings,
        rdMolDescriptors.CalcNumSaturatedRings,
        rdMolDescriptors.CalcNumAromaticHeterocycles,
        rdMolDescriptors.CalcNumAliphaticHeterocycles,
        rdMolDescriptors.CalcNumSaturatedHeterocycles,
        rdMolDescriptors.CalcNumAromaticCarbocycles,
        rdMolDescriptors.CalcNumAliphaticCarbocycles,
        rdMolDescriptors.CalcFractionCSP3,
    )


def generate_feature_matrices(smiles_values: Sequence[str]) -> FeatureMatrices:
    """Extract fixed features from each original SMILES without editing molecules.

    Only the SMILES sequence is accepted, so this API cannot read labels or use
    target-driven feature selection. Dummy atoms are retained exactly as parsed.
    """
    import rdkit
    from rdkit import Chem, DataStructs
    from rdkit.Chem import rdFingerprintGenerator

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=MORGAN_RADIUS,
        fpSize=MORGAN_BITS,
        includeChirality=MORGAN_CHIRALITY,
    )
    descriptor_functions = _descriptor_functions()
    descriptor_matrix = np.empty((len(smiles_values), DESCRIPTOR_DIM), dtype="<f8")
    fingerprint_matrix = np.zeros((len(smiles_values), MORGAN_BITS), dtype=np.uint8)
    failures: list[dict[str, Any]] = []
    for row_index, smiles in enumerate(smiles_values):
        try:
            molecule = Chem.MolFromSmiles(str(smiles))
            if molecule is None:
                raise ValueError("RDKit could not parse the original benchmark SMILES")
            # Do not remove dummy atoms, sanitize into a closure, or rewrite the molecule.
            descriptor_matrix[row_index] = [
                float(function(molecule)) for function in descriptor_functions
            ]
            fingerprint = generator.GetFingerprint(molecule)
            DataStructs.ConvertToNumpyArray(fingerprint, fingerprint_matrix[row_index])
        except Exception as exc:
            failures.append(
                {"row_index": row_index, "error_type": type(exc).__name__, "error": str(exc)}
            )
    if failures:
        first = failures[:10]
        raise FeatureGenerationError(
            f"Could not compute Stage 4A features for {len(failures)}/{len(smiles_values)} "
            f"original SMILES; first failures: {first}"
        )
    if descriptor_matrix.shape != (len(smiles_values), 20):
        raise RuntimeError("Descriptor matrix has an unexpected shape")
    if fingerprint_matrix.shape != (len(smiles_values), 2048):
        raise RuntimeError("Morgan matrix has an unexpected shape")

    nonfinite_by_descriptor = {
        name: int((~np.isfinite(descriptor_matrix[:, column])).sum())
        for column, name in enumerate(DESCRIPTOR_NAMES)
    }
    nonfinite_count = int((~np.isfinite(descriptor_matrix)).sum())
    stats: dict[str, dict[str, float | bool]] = {}
    constant_columns: list[str] = []
    for column, name in enumerate(DESCRIPTOR_NAMES):
        values = descriptor_matrix[:, column]
        is_constant = bool(values.size and np.all(values == values[0]))
        if is_constant:
            constant_columns.append(name)
        stats[name] = {
            "min": float(np.min(values)) if values.size else 0.0,
            "max": float(np.max(values)) if values.size else 0.0,
            "mean": float(np.mean(values)) if values.size else 0.0,
            "std_population": float(np.std(values, ddof=0)) if values.size else 0.0,
            "constant": is_constant,
        }
    unique_fingerprints, counts = np.unique(fingerprint_matrix, axis=0, return_counts=True)
    duplicate_rows = int(len(fingerprint_matrix) - len(unique_fingerprints))
    duplicate_groups = int(np.sum(counts > 1))
    bits_on = fingerprint_matrix.sum(axis=1, dtype=np.int64)
    manifest = {
        "schema_version": "stage4a_global_features_v1",
        "source": {
            "sample_count": int(len(smiles_values)),
            "smiles_source": "raw benchmark SMILES parsed directly by RDKit; original dummy atoms retained",
            "labels_read": False,
        },
        "rdkit_version": str(rdkit.__version__),
        "descriptors": {
            "names_in_order": list(DESCRIPTOR_NAMES),
            "shape": list(descriptor_matrix.shape),
            "dtype": "float64 little-endian C-order",
            "raw_matrix_sha256": _canonical_matrix_sha256(descriptor_matrix, "<f8"),
            "nonfinite_count": nonfinite_count,
            "nonfinite_count_by_descriptor": nonfinite_by_descriptor,
            "constant_columns": constant_columns,
            "statistics": stats,
        },
        "morgan": {
            "algorithm": "Morgan bit fingerprint via rdkit.Chem.rdFingerprintGenerator.GetMorganGenerator",
            "radius": MORGAN_RADIUS,
            "bit_count": MORGAN_BITS,
            "chirality": MORGAN_CHIRALITY,
            "representation": "binary bit fingerprint",
            "shape": list(fingerprint_matrix.shape),
            "dtype": "uint8 C-order",
            "raw_matrix_sha256": _canonical_matrix_sha256(fingerprint_matrix, "|u1"),
            "average_bits_on": float(np.mean(bits_on)) if bits_on.size else 0.0,
            "min_bits_on": int(np.min(bits_on)) if bits_on.size else 0,
            "max_bits_on": int(np.max(bits_on)) if bits_on.size else 0,
            "duplicate_fingerprint_rows": duplicate_rows,
            "duplicate_fingerprint_groups": duplicate_groups,
            "target_based_selection": False,
            "released_label_adjustment": False,
            "augmentation": False,
        },
    }
    return FeatureMatrices(descriptor_matrix, fingerprint_matrix, manifest)


class FeatureGenerationError(RuntimeError):
    """A fixed descriptor/fingerprint failed on one or more original molecules."""


def build_feature_manifest(
    *, smiles_values: Sequence[str], source_data_path: str | Path
) -> FeatureMatrices:
    features = generate_feature_matrices(smiles_values)
    source_hash = sha256_file(source_data_path)
    features.manifest["source"]["data_sha256"] = source_hash
    features.manifest["descriptors"]["source_data_sha256"] = source_hash
    features.manifest["morgan"]["source_data_sha256"] = source_hash
    return features


def save_feature_cache(cache_path: str | Path, features: FeatureMatrices) -> None:
    path = Path(cache_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, descriptors=features.descriptors, morgan=features.morgan)


def load_feature_cache(cache_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(cache_path, allow_pickle=False) as saved:
        descriptors = np.asarray(saved["descriptors"], dtype="<f8")
        morgan = np.asarray(saved["morgan"], dtype=np.uint8)
    return descriptors, morgan


def fit_descriptor_scaler(
    descriptor_matrix: np.ndarray,
    train_indices: Sequence[int],
    *,
    source_data_sha256: str,
) -> dict[str, Any]:
    if descriptor_matrix.ndim != 2 or descriptor_matrix.shape[1] != DESCRIPTOR_DIM:
        raise ValueError("Descriptor matrix must have shape [N, 20]")
    if not train_indices:
        raise ValueError("Cannot fit descriptor scaler without training rows")
    train_values = descriptor_matrix[np.asarray(train_indices, dtype=np.int64)]
    mean = np.mean(train_values, axis=0, dtype=np.float64)
    std = np.std(train_values, axis=0, ddof=0, dtype=np.float64)
    zero_variance = (~np.isfinite(std)) | (std == 0.0)
    scale = std.copy()
    scale[zero_variance] = 1.0
    return {
        "schema_version": "stage4a_descriptor_scaler_v1",
        "feature_names": list(DESCRIPTOR_NAMES),
        "fit_row_count": int(len(train_indices)),
        "train_mean": [float(value) for value in mean],
        "train_std": [float(value) for value in std],
        "scale_used": [float(value) for value in scale],
        "zero_variance_columns": [
            name for name, zero in zip(DESCRIPTOR_NAMES, zero_variance) if zero
        ],
        "zero_variance_handling": "use scale=1 and keep the column",
        "source_data_sha256": source_data_sha256,
    }


def transform_descriptors(
    descriptor_matrix: np.ndarray, scaler: Mapping[str, Any]
) -> np.ndarray:
    mean = np.asarray(scaler["train_mean"], dtype=np.float64)
    scale = np.asarray(scaler["scale_used"], dtype=np.float64)
    if descriptor_matrix.ndim != 2 or descriptor_matrix.shape[1] != DESCRIPTOR_DIM:
        raise ValueError("Descriptor matrix must have shape [N, 20]")
    if mean.shape != (DESCRIPTOR_DIM,) or scale.shape != (DESCRIPTOR_DIM,):
        raise ValueError("Scaler mean/std must preserve the fixed 20-column schema")
    transformed = (descriptor_matrix - mean) / scale
    if not np.isfinite(transformed).all():
        raise ValueError("Standardized descriptor matrix contains NaN or inf")
    return np.asarray(transformed, dtype=np.float32)


def write_feature_manifest(path: str | Path, manifest: Mapping[str, Any]) -> None:
    write_json(Path(path), dict(manifest))
