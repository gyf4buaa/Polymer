#!/usr/bin/env python3
"""Freeze the source manifest, diagnostics, and sample-level five-fold map."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TRACK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRACK_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from from_scratch_gnn.src.data import (  # noqa: E402
    CANONICAL_COLUMNS,
    TARGETS,
    build_fold_assignments,
    diagnose_training_data,
    fold_diagnostics,
    load_training_data,
    sha256_file,
    write_csv,
    write_json,
)
from from_scratch_gnn.src.metrics import competition_weights  # noqa: E402

N_SPLITS = 5
FOLD_COLUMNS = ("sample_id", "fold", "SMILES")


def _has_unusable_ids(diagnostics: dict[str, Any]) -> bool:
    return bool(
        diagnostics["missing_sample_id_count"]
        or diagnostics["duplicate_sample_id_group_count"]
    )


def freeze(
    train_csv: Path,
    output_dir: Path,
    *,
    seed: int = 20250604,
    source_label: str = "data/competition_raw/train.csv",
) -> dict[str, Any]:
    rows = load_training_data(train_csv)
    diagnostics = diagnose_training_data(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "diagnostics.json", diagnostics)

    # Duplicate or missing keys make an ID-keyed fold file ambiguous. Keep the
    # diagnostic report and stop instead of dropping or merging any rows.
    if _has_unusable_ids(diagnostics):
        raise ValueError(
            "Cannot freeze ID-keyed folds: sample_id is missing or duplicated. "
            "See diagnostics.json; no samples were removed."
        )

    assignments = build_fold_assignments(
        (str(row["sample_id"]) for row in rows), n_splits=N_SPLITS, seed=seed
    )
    fold_rows = [
        {"sample_id": row["sample_id"], "fold": assignments[str(row["sample_id"])],
         "SMILES": row["SMILES"]}
        for row in rows
    ]
    folds_path = output_dir / "folds.csv"
    write_csv(folds_path, fold_rows, FOLD_COLUMNS)

    folds_by_fold = fold_diagnostics(rows, assignments, n_splits=N_SPLITS)
    fold_sizes = Counter(assignments.values())
    if sum(fold_sizes.values()) != len(rows):
        raise RuntimeError("Fold assignment does not cover every training row exactly once.")
    for target in TARGETS:
        fold_label_count = sum(
            folds_by_fold[str(fold)]["valid_target_counts"][target]
            for fold in range(N_SPLITS)
        )
        if fold_label_count != diagnostics["targets"][target]["valid_count"]:
            raise RuntimeError(
                f"Fold label count mismatch for {target}: {fold_label_count}."
            )
    global_weights = competition_weights(rows)
    manifest = {
        "benchmark_version": "nopp2025_train_v1",
        "created_by": "from_scratch_gnn/scripts/freeze_benchmark.py",
        "source": {
            "description": "Official competition training file; no supplement or released test data",
            "path_label": source_label,
            "file_name": train_csv.name,
            "sha256": sha256_file(train_csv),
            "committed_to_repository": False,
            "source_columns": ["id", "SMILES", *TARGETS],
            "canonical_view_columns": list(CANONICAL_COLUMNS),
            "id_mapping": {"source": "id", "canonical": "sample_id"},
        },
        "sample_count": len(rows),
        "target_statistics_and_weights": {
            target: {
                **diagnostics["targets"][target],
                "task_balance_factor": global_weights[target]["task_balance_factor"],
                "weight": global_weights[target]["weight"],
            }
            for target in TARGETS
        },
        "metric": {
            "implementation": "from_scratch_gnn/src/metrics.py::evaluate_oof",
            "weight_formula": (
                "weight_i = (1 / range_i) * "
                "(K * sqrt(1 / n_i) / sum_j sqrt(1 / n_j))"
            ),
            "local_statistics_source": "official training labels only",
            "private_labels_used": False,
        },
        "folds": {
            "file": "folds.csv",
            "sha256": sha256_file(folds_path),
            "n_splits": N_SPLITS,
            "seed": seed,
            "strategy": "unstratified sample-level random balanced 5-fold",
            "assignment_algorithm": (
                "SHA-256 rank of UTF-8 string "
                "'polymer-stage0-sha256-rank-v1\\0{seed}\\0{sample_id}', "
                "then sequential balanced fold chunks"
            ),
            "row_order": "source train.csv order; assignment is independent of row order",
            "fold_sizes": {str(fold): fold_sizes.get(fold, 0) for fold in range(N_SPLITS)},
            "fold_target_valid_counts": folds_by_fold,
        },
        "diagnostics_file": "diagnostics.json",
    }
    write_json(output_dir / "data_manifest.json", manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv", type=Path, required=True,
        help="Path to the official competition train.csv (kept outside Git).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=TRACK_ROOT / "benchmark",
        help="Directory for folds.csv, data_manifest.json, and diagnostics.json.",
    )
    parser.add_argument("--seed", type=int, default=20250604)
    parser.add_argument(
        "--source-label",
        default="data/competition_raw/train.csv",
        help="Portable provenance label recorded in data_manifest.json.",
    )
    args = parser.parse_args()
    manifest = freeze(
        args.train_csv, args.output_dir, seed=args.seed, source_label=args.source_label
    )
    print(json.dumps({
        "sample_count": manifest["sample_count"],
        "fold_sizes": manifest["folds"]["fold_sizes"],
        "folds_sha256": manifest["folds"]["sha256"],
        "data_manifest": str(args.output_dir / "data_manifest.json"),
        "diagnostics": str(args.output_dir / "diagnostics.json"),
    }, indent=2))


if __name__ == "__main__":
    main()
