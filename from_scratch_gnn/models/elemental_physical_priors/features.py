"""Fixed RDKit periodic-table priors and benchmark element audit for Stage 4B."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from typing import Any, Sequence

import numpy as np

FEATURE_NAMES = (
    "atomic_weight",
    "covalent_radius",
    "vdw_radius",
    "outer_electrons",
    "period",
)
FEATURE_APIS = (
    "PeriodicTable.GetAtomicWeight",
    "PeriodicTable.GetRcovalent",
    "PeriodicTable.GetRvdw",
    "PeriodicTable.GetNOuterElecs",
    "PeriodicTable.GetRow",
)
FEATURE_UNITS = (
    "atomic mass units; RDKit PeriodicTable.GetAtomicWeight",
    "angstrom; RDKit PeriodicTable.GetRcovalent",
    "angstrom; RDKit PeriodicTable.GetRvdw",
    "electron count; RDKit PeriodicTable.GetNOuterElecs",
    "periodic-table row number (1-based); RDKit PeriodicTable.GetRow",
)
REFERENCE_ATOMIC_NUMBERS = tuple(range(1, 119))
PHYSICAL_FEATURE_DIM = len(FEATURE_NAMES)
REFERENCE_SCHEMA = "rdkit_periodic_table_z1_118_v1"
FEATURE_SCHEMA = "stage4b_elemental_physical_priors_v1"


class ElementFeatureAuditError(ValueError):
    """Raised when RDKit cannot provide a complete finite fixed feature table."""


def _canonical_json_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _canonical_reference_sha256(table: np.ndarray) -> str:
    values = np.ascontiguousarray(table, dtype="<f8")
    header = (REFERENCE_SCHEMA + "\0" + "atomic_number," + ",".join(FEATURE_NAMES) + "\0")
    return hashlib.sha256(header.encode("utf-8") + values.tobytes(order="C")).hexdigest()


def build_periodic_table_reference() -> dict[str, Any]:
    """Return RDKit Z=1..118 values and fixed population normalization constants.

    This function accepts no benchmark data, labels, folds, or paths by design.
    """
    import rdkit
    from rdkit import Chem

    table_api = Chem.GetPeriodicTable()
    methods = (
        table_api.GetAtomicWeight,
        table_api.GetRcovalent,
        table_api.GetRvdw,
        table_api.GetNOuterElecs,
        table_api.GetRow,
    )
    table = np.empty((len(REFERENCE_ATOMIC_NUMBERS), 1 + PHYSICAL_FEATURE_DIM), dtype="<f8")
    invalid: list[dict[str, Any]] = []
    undefined: list[dict[str, Any]] = []
    for row_index, atomic_number in enumerate(REFERENCE_ATOMIC_NUMBERS):
        values: list[float] = []
        for feature_index, method in enumerate(methods):
            try:
                raw = method(int(atomic_number))
                value = float(raw)
            except Exception as exc:  # pragma: no cover - guards unsupported RDKit builds
                invalid.append(
                    {
                        "atomic_number": atomic_number,
                        "feature": FEATURE_NAMES[feature_index],
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                value = float("nan")
            if not math.isfinite(value):
                invalid.append(
                    {
                        "atomic_number": atomic_number,
                        "feature": FEATURE_NAMES[feature_index],
                        "value": repr(value),
                    }
                )
            # A zero radius/weight/row is the RDKit table's undefined sentinel for
            # these five strictly positive elemental quantities.
            if value == 0.0:
                undefined.append(
                    {
                        "atomic_number": atomic_number,
                        "feature": FEATURE_NAMES[feature_index],
                        "value": value,
                    }
                )
            values.append(value)
        table[row_index, 0] = atomic_number
        table[row_index, 1:] = values
    if invalid or undefined:
        raise ElementFeatureAuditError(
            "RDKit periodic-table reference is incomplete: "
            + json.dumps(
                {"nonfinite_or_lookup_errors": invalid, "undefined_zero_sentinels": undefined},
                ensure_ascii=False,
                sort_keys=True,
            )
        )

    raw_features = table[:, 1:]
    means = np.mean(raw_features, axis=0, dtype=np.float64)
    stds = np.std(raw_features, axis=0, ddof=0, dtype=np.float64)
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ElementFeatureAuditError("Periodic-table reference normalization is undefined")
    rdkit_version = str(rdkit.__version__)
    rows = [
        {
            "atomic_number": int(row[0]),
            "symbol": str(table_api.GetElementSymbol(int(row[0]))),
            "values": {name: float(value) for name, value in zip(FEATURE_NAMES, row[1:])},
        }
        for row in table
    ]
    return {
        "schema": REFERENCE_SCHEMA,
        "rdkit_version": rdkit_version,
        "feature_names_in_order": list(FEATURE_NAMES),
        "feature_apis_in_order": list(FEATURE_APIS),
        "feature_units_and_semantics_in_order": list(FEATURE_UNITS),
        "atomic_number_range_inclusive": [1, 118],
        "reference_population": "all integer atomic numbers Z=1..118, one row per element",
        "ddof": 0,
        "mean_ref": {name: float(value) for name, value in zip(FEATURE_NAMES, means)},
        "std_ref": {name: float(value) for name, value in zip(FEATURE_NAMES, stds)},
        "table_sha256": _canonical_reference_sha256(table),
        "table_rows": rows,
        "_numeric_table": table,
        "_means": means,
        "_stds": stds,
    }


def physical_lookup(reference: dict[str, Any] | None = None) -> np.ndarray:
    """Return float32 normalized vectors indexed by atomic number, with Z=0 zeroed."""
    reference = reference or build_periodic_table_reference()
    raw_table = np.asarray(reference["_numeric_table"], dtype=np.float64)
    means = np.asarray(reference["_means"], dtype=np.float64)
    stds = np.asarray(reference["_stds"], dtype=np.float64)
    lookup = np.zeros((119, PHYSICAL_FEATURE_DIM), dtype=np.float32)
    lookup[1:119] = ((raw_table[:, 1:] - means) / stds).astype(np.float32)
    if not np.isfinite(lookup).all() or not np.array_equal(lookup[0], np.zeros(PHYSICAL_FEATURE_DIM, dtype=np.float32)):
        raise ElementFeatureAuditError("Normalized elemental feature lookup is non-finite or invalid")
    return lookup


def normalize_atomic_numbers(
    atomic_numbers: Sequence[int] | np.ndarray,
    reference: dict[str, Any] | None = None,
) -> np.ndarray:
    """Normalize atom atomic numbers; the RDKit dummy bucket Z=0 maps to exact zeros."""
    numbers = np.asarray(atomic_numbers, dtype=np.int64)
    if np.any(numbers < 0) or np.any(numbers > 118):
        raise ElementFeatureAuditError("Atomic numbers must be in the inclusive range 0..118")
    return physical_lookup(reference)[numbers]


def _public_reference(reference: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in reference.items() if not key.startswith("_")}


def build_element_feature_manifest(
    smiles_values: Sequence[str],
    *,
    source_train_sha256: str,
) -> dict[str, Any]:
    """Build a deterministic audit from original SMILES without accessing labels/folds."""
    import rdkit
    from rdkit import Chem

    reference = build_periodic_table_reference()
    if str(rdkit.__version__) != reference["rdkit_version"]:
        raise ElementFeatureAuditError("RDKit version changed while building the feature audit")
    lookup = physical_lookup(reference)
    atom_counts: Counter[int] = Counter()
    invalid_smiles: list[dict[str, Any]] = []
    for row_index, smiles in enumerate(smiles_values):
        molecule = Chem.MolFromSmiles(str(smiles))
        if molecule is None:
            invalid_smiles.append({"row_index": row_index, "smiles": str(smiles)})
            continue
        for atom in molecule.GetAtoms():
            atomic_number = int(atom.GetAtomicNum())
            if atomic_number < 0 or atomic_number > 118:
                raise ElementFeatureAuditError(
                    f"Observed unsupported atomic number {atomic_number} at SMILES row {row_index}"
                )
            atom_counts[atomic_number] += 1
    if invalid_smiles:
        raise ElementFeatureAuditError(
            f"RDKit could not parse {len(invalid_smiles)} benchmark SMILES; "
            f"first examples: {invalid_smiles[:10]}"
        )

    table_api = Chem.GetPeriodicTable()
    observed_rows: list[dict[str, Any]] = []
    for atomic_number in sorted(atom_counts):
        if atomic_number == 0:
            observed_rows.append(
                {
                    "atomic_number": 0,
                    "symbol": "*",
                    "atom_count": int(atom_counts[0]),
                    "raw_values": {name: None for name in FEATURE_NAMES},
                    "normalized_values": {name: 0.0 for name in FEATURE_NAMES},
                    "finite_all_five": True,
                    "policy": "dummy atom physical vector fixed to all zeros; no periodic-table properties assigned",
                }
            )
            continue
        raw = reference["table_rows"][atomic_number - 1]["values"]
        normalized = lookup[atomic_number]
        finite = all(math.isfinite(float(value)) for value in normalized)
        if not finite:
            raise ElementFeatureAuditError(
                f"Observed element Z={atomic_number} has a non-finite normalized feature"
            )
        observed_rows.append(
            {
                "atomic_number": atomic_number,
                "symbol": str(table_api.GetElementSymbol(atomic_number)),
                "atom_count": int(atom_counts[atomic_number]),
                "raw_values": {name: float(raw[name]) for name in FEATURE_NAMES},
                "normalized_values": {
                    name: float(value) for name, value in zip(FEATURE_NAMES, normalized)
                },
                "finite_all_five": True,
            }
        )
    observed_hash = _canonical_json_sha256(observed_rows)
    result = {
        "schema": FEATURE_SCHEMA,
        "source": {
            "sample_count": int(len(smiles_values)),
            "smiles_source": "original benchmark SMILES parsed directly with RDKit; no graph rewrite",
            "labels_read_by_feature_extractor": False,
            "folds_read_by_feature_extractor": False,
            "source_train_sha256": str(source_train_sha256),
        },
        "rdkit_version": str(rdkit.__version__),
        "features": {
            "names_in_order": list(FEATURE_NAMES),
            "apis_in_order": list(FEATURE_APIS),
            "units_and_semantics_in_order": list(FEATURE_UNITS),
            "dimension": PHYSICAL_FEATURE_DIM,
            "normalization": "(raw - mean_ref) / std_ref; fixed RDKit Z=1..118 table, population std ddof=0; dummy Z=0 exact zeros",
            "reference": _public_reference(reference),
        },
        "coverage": {
            "observed_atomic_numbers": sorted(int(value) for value in atom_counts),
            "dummy_atom_count": int(atom_counts.get(0, 0)),
            "real_atom_count": int(sum(count for z, count in atom_counts.items() if z > 0)),
            "observed_real_element_count": int(sum(z > 0 for z in atom_counts)),
            "observed_element_table": observed_rows,
            "observed_element_table_sha256": observed_hash,
            "all_observed_real_elements_finite_all_five": all(
                row["finite_all_five"] for row in observed_rows if row["atomic_number"] > 0
            ),
        },
    }
    result["reference_table_sha256"] = reference["table_sha256"]
    return result


def manifest_json_bytes(manifest: dict[str, Any]) -> bytes:
    """Stable pretty JSON representation used for a checked-in audit artifact."""
    return (json.dumps(manifest, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode("utf-8")
