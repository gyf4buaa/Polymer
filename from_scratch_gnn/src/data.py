"""Small, dependency-free loader and audit helpers for the frozen benchmark."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")
RAW_ID_COLUMN = "id"
CANONICAL_COLUMNS = ("sample_id", "SMILES", *TARGETS)
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_label(value: str | None, *, row_number: int, target: str) -> float | None:
    text = "" if value is None else value.strip()
    if text.lower() in MISSING_TOKENS:
        return None
    try:
        parsed = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric label in CSV row {row_number}, target {target}: {value!r}"
        ) from exc
    if not math.isfinite(parsed):
        raise ValueError(
            f"Non-finite label in CSV row {row_number}, target {target}: {value!r}"
        )
    return parsed


def load_training_data(path: str | Path) -> list[dict[str, Any]]:
    """Load official train.csv into the canonical sample_id/SMILES/target schema.

    The source file is not copied into this repository. id is renamed to
    sample_id; source row order, SMILES text, and every record are retained.
    """
    source_path = Path(path)
    with source_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("Training CSV contains duplicate column names.")
        required = {RAW_ID_COLUMN, "SMILES", *TARGETS}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"Training CSV is missing required columns: {missing}")

        rows: list[dict[str, Any]] = []
        for row_number, raw in enumerate(reader, start=2):
            row: dict[str, Any] = {
                "sample_id": "" if raw.get(RAW_ID_COLUMN) is None else raw[RAW_ID_COLUMN],
                "SMILES": "" if raw.get("SMILES") is None else raw["SMILES"],
            }
            for target in TARGETS:
                row[target] = _parse_label(
                    raw.get(target), row_number=row_number, target=target
                )
            rows.append(row)
    return rows


def _rdkit_modules():
    try:
        import rdkit
        from rdkit import Chem
    except ImportError:
        return None, None
    return Chem, getattr(rdkit, "__version__", None)


def validate_smiles(smiles: str) -> bool | None:
    """Return RDKit validity, or None when RDKit is not installed."""
    chem, _ = _rdkit_modules()
    if chem is None:
        return None
    if not smiles.strip():
        return False
    return chem.MolFromSmiles(smiles) is not None


def diagnose_training_data(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Return diagnostics without removing, merging, or rewriting any sample."""
    id_rows: dict[str, list[int]] = defaultdict(list)
    smiles_rows: dict[str, list[int]] = defaultdict(list)
    canonical_smiles_rows: dict[str, list[int]] = defaultdict(list)
    missing_ids: list[int] = []
    missing_smiles: list[int] = []
    all_labels_missing: list[int] = []
    target_stats: dict[str, dict[str, Any]] = {}

    chem, rdkit_version = _rdkit_modules()
    invalid_smiles: list[dict[str, Any]] = []

    for row_index, row in enumerate(rows):
        sample_id = str(row.get("sample_id", ""))
        smiles = str(row.get("SMILES", ""))
        if not sample_id.strip():
            missing_ids.append(row_index)
        else:
            id_rows[sample_id].append(row_index)
        if not smiles.strip():
            missing_smiles.append(row_index)
        else:
            smiles_rows[smiles].append(row_index)
            if chem is not None:
                try:
                    molecule = chem.MolFromSmiles(smiles)
                except Exception as exc:
                    molecule = None
                    error = str(exc)
                else:
                    error = None
                if molecule is None:
                    invalid = {"sample_id": sample_id, "row_index": row_index, "SMILES": smiles}
                    if error is not None:
                        invalid["error"] = error
                    invalid_smiles.append(invalid)
                else:
                    canonical = chem.MolToSmiles(
                        molecule, canonical=True, isomericSmiles=True
                    )
                    canonical_smiles_rows[canonical].append(row_index)

    for target in TARGETS:
        values = [float(row[target]) for row in rows if row.get(target) is not None]
        target_stats[target] = {
            "valid_count": len(values),
            "missing_count": len(rows) - len(values),
            "min": min(values) if values else None,
            "max": max(values) if values else None,
            "range": max(values) - min(values) if values else None,
        }

    for row_index, row in enumerate(rows):
        if all(row.get(target) is None for target in TARGETS):
            all_labels_missing.append(row_index)

    duplicate_ids = [
        {"sample_id": sample_id, "row_indices": indices, "count": len(indices)}
        for sample_id, indices in sorted(id_rows.items())
        if len(indices) > 1
    ]
    duplicate_smiles = [
        {"SMILES": smiles, "row_indices": indices, "count": len(indices)}
        for smiles, indices in sorted(smiles_rows.items())
        if len(indices) > 1
    ]
    duplicate_canonical_smiles = [
        {"canonical_SMILES": smiles, "row_indices": indices, "count": len(indices)}
        for smiles, indices in sorted(canonical_smiles_rows.items())
        if len(indices) > 1
    ]

    return {
        "sample_count": len(rows),
        "missing_sample_id_count": len(missing_ids),
        "missing_sample_id_row_indices": missing_ids,
        "duplicate_sample_id_group_count": len(duplicate_ids),
        "duplicate_sample_id_groups": duplicate_ids,
        "missing_smiles_count": len(missing_smiles),
        "missing_smiles_row_indices": missing_smiles,
        "duplicate_smiles_group_count": len(duplicate_smiles),
        "duplicate_smiles_groups": duplicate_smiles,
        "smiles_validation": {
            "method": "RDKit Chem.MolFromSmiles",
            "available": chem is not None,
            "rdkit_version": rdkit_version,
            "invalid_count": len(invalid_smiles) if chem is not None else None,
            "invalid_smiles": invalid_smiles if chem is not None else None,
        },
        "canonical_smiles_audit": {
            "method": "RDKit canonical isomeric SMILES; diagnostic only",
            "available": chem is not None,
            "duplicate_group_count": (
                len(duplicate_canonical_smiles) if chem is not None else None
            ),
            "duplicate_groups": (
                duplicate_canonical_smiles if chem is not None else None
            ),
        },
        "all_targets_missing_count": len(all_labels_missing),
        "all_targets_missing_row_indices": all_labels_missing,
        "targets": target_stats,
    }


def build_fold_assignments(
    sample_ids: Iterable[str], *, n_splits: int = 5, seed: int = 42
) -> dict[str, int]:
    """Make balanced random folds by SHA-256 ranking IDs with a recorded seed.

    This is a sample-level, unstratified split. Its output is independent of
    input row order and runtime/library versions.
    """
    ids = [str(sample_id) for sample_id in sample_ids]
    if n_splits < 2:
        raise ValueError("n_splits must be at least 2.")
    if len(ids) < n_splits:
        raise ValueError("There must be at least one sample per fold.")
    if any(not sample_id.strip() for sample_id in ids):
        raise ValueError("Cannot assign folds to a missing sample_id.")
    if len(set(ids)) != len(ids):
        raise ValueError("Cannot assign unique folds while sample_id values repeat.")

    seed_prefix = f"polymer-stage0-sha256-rank-v1\0{seed}\0".encode("utf-8")
    ranked_ids = sorted(
        ids,
        key=lambda sample_id: (
            hashlib.sha256(seed_prefix + sample_id.encode("utf-8")).digest(),
            sample_id,
        ),
    )

    assignment: dict[str, int] = {}
    start = 0
    base_size, extra = divmod(len(ranked_ids), n_splits)
    for fold in range(n_splits):
        size = base_size + (1 if fold < extra else 0)
        for sample_id in ranked_ids[start : start + size]:
            assignment[sample_id] = fold
        start += size
    return assignment


def fold_diagnostics(
    rows: Sequence[Mapping[str, Any]], assignments: Mapping[str, int], *, n_splits: int
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for fold in range(n_splits):
        fold_rows = [row for row in rows if assignments[str(row["sample_id"])] == fold]
        result[str(fold)] = {
            "sample_count": len(fold_rows),
            "valid_target_counts": {
                target: sum(row.get(target) is not None for row in fold_rows)
                for target in TARGETS
            },
        }
    return result


def write_csv(path: str | Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output, fieldnames=list(columns), extrasaction="ignore", lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column) for column in columns})


def write_json(path: str | Path, value: Mapping[str, Any]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
