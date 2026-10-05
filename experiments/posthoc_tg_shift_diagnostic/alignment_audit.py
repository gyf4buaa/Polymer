#!/usr/bin/env python3
"""Audit released-test identity, scoring scenarios, labels, and graph stability.

This is a bounded POST-HOC audit. It does not run inference, retrain, or search
for Tg shifts. It only scores the fixed clean predictions and the already
submitted +70 diagnostic under explicit split/weight assumptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path
from statistics import fmean, median
from typing import Any, Mapping, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")
EXPECTED_SCORES = {
    "public_clean": 0.06899,
    "private_clean": 0.09524,
    "public_tg_plus_70": 0.06493,
    "private_tg_plus_70": 0.07758,
}
DATASET_REF = "alexliu99/neurips-open-polymer-prediction-2025-test-data"
DATASET_VIEW = f"https://www.kaggle.com/api/v1/datasets/view/{DATASET_REF}"
DATASET_FILES = f"https://www.kaggle.com/api/v1/datasets/list/{DATASET_REF}"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def statistics_by_target(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    stats: dict[str, Any] = {"rows": len(rows), "targets": {}}
    for target in TARGETS:
        values = [float(row[target]) for row in rows if row[target] is not None]
        missing = len(rows) - len(values)
        if values:
            minimum, maximum = min(values), max(values)
            stats["targets"][target] = {
                "rows": len(rows),
                "observed_label_count": len(values),
                "missing_count": missing,
                "missingness": missing / len(rows) if rows else None,
                "min": minimum,
                "max": maximum,
                "range": maximum - minimum,
                "mean": fmean(values),
                "median": median(values),
            }
        else:
            stats["targets"][target] = {
                "rows": len(rows), "observed_label_count": 0,
                "missing_count": missing, "missingness": 1.0 if rows else None,
                "min": None, "max": None, "range": None, "mean": None, "median": None,
            }
    return stats


def target_weights(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    counts = {}
    ranges = {}
    for target in TARGETS:
        values = [float(row[target]) for row in rows if row[target] is not None]
        if not values:
            raise ValueError(f"No observed released labels for {target}")
        counts[target] = len(values)
        ranges[target] = max(values) - min(values)
        if ranges[target] <= 0:
            raise ValueError(f"Zero released range for {target}")
    scale = sum((1.0 / counts[t]) ** 0.5 for t in TARGETS)
    return {
        t: {
            "weight": (5.0 * (1.0 / counts[t]) ** 0.5 / scale) / ranges[t],
            "valid_count": counts[t],
            "value_range": ranges[t],
        }
        for t in TARGETS
    }


def score_scenarios(
    rows: Sequence[Mapping[str, Any]],
    prediction_by_smiles: Mapping[str, Mapping[str, Any]],
    formal_source: Path,
) -> list[dict[str, Any]]:
    if str(formal_source) not in sys.path:
        sys.path.insert(0, str(formal_source))
    from from_scratch_gnn.src.metrics import evaluate_oof

    normal = [dict(row) for row in rows]
    swapped = [dict(row, split=("private" if row["split"] == "public" else "public")) for row in rows]
    output = []
    for assignment, assigned_rows in (("normal", normal), ("public_private_swapped", swapped)):
        groups = {
            split: [row for row in assigned_rows if row["split"] == split]
            for split in ("public", "private")
        }
        global_weights = target_weights(assigned_rows)
        for weight_scope in ("global", "split_specific"):
            weights_by_split = {
                split: global_weights if weight_scope == "global" else target_weights(group)
                for split, group in groups.items()
            }
            for split, group in groups.items():
                truth, prediction = [], []
                for row in group:
                    pred = prediction_by_smiles[str(row["SMILES"])]
                    truth.append({t: row[t] for t in TARGETS})
                    prediction.append({t: float(pred[t]) for t in TARGETS})
                clean = float(evaluate_oof(truth, prediction, target_weights=weights_by_split[split])["overall_oof_wmae"])
                plus70_prediction = [dict(pred, Tg=float(pred["Tg"]) + 70.0) for pred in prediction]
                plus70 = float(evaluate_oof(truth, plus70_prediction, target_weights=weights_by_split[split])["overall_oof_wmae"])
                output.append({
                    "assignment": assignment,
                    "weight_scope": weight_scope,
                    "scored_split": split,
                    "rows": len(group),
                    "clean_score": clean,
                    "plus70_score": plus70,
                    "plus70_minus_clean": plus70 - clean,
                    "expected_online_clean": EXPECTED_SCORES[f"{split}_clean"],
                    "clean_minus_online": clean - EXPECTED_SCORES[f"{split}_clean"],
                    "expected_online_plus70": EXPECTED_SCORES[f"{split}_tg_plus_70"],
                    "plus70_minus_online": plus70 - EXPECTED_SCORES[f"{split}_tg_plus_70"],
                    "weight_values": weights_by_split[split],
                })
    return output


def graph_attrs(data: Any) -> Any:
    import networkx as nx

    graph = nx.Graph()
    x = data.x.detach().cpu().numpy()
    for i, values in enumerate(x):
        graph.add_node(i, features=tuple(int(v) for v in values))
    edge_index = data.edge_index.detach().cpu().numpy()
    edge_attr = data.edge_attr.detach().cpu().numpy()
    for e in range(edge_index.shape[1]):
        left, right = int(edge_index[0, e]), int(edge_index[1, e])
        if left < right:
            attrs = tuple(float(v) for v in edge_attr[e])
            graph.add_edge(left, right, features=attrs)
    return graph


def audit_graph_roundtrips(
    rows: Sequence[Mapping[str, Any]], formal_source: Path, sample_size: int, seed: int
) -> dict[str, Any]:
    import networkx as nx
    from rdkit import Chem

    if str(formal_source) not in sys.path:
        sys.path.insert(0, str(formal_source))
    from from_scratch_gnn.models.own_gnn_repr_keep_dummy.graph import build_polymer_graph

    rng = random.Random(seed)
    sample = rng.sample(list(rows), min(sample_size, len(rows)))
    # Add high-information cases even when they were not selected randomly.
    markers = set("@%/\\+-")
    special = [row for row in rows if any(ch in str(row["SMILES"]) for ch in markers)]
    selected_by_smiles = {str(row["SMILES"]): row for row in sample + special}
    selected = list(selected_by_smiles.values())
    node_match = nx.algorithms.isomorphism.categorical_node_match("features", None)
    edge_match = nx.algorithms.isomorphism.categorical_edge_match("features", None)
    count_equal = 0
    count_different = 0
    parse_failures = []
    topology_counts = Counter()
    atom_order_changed = 0
    examples = []
    for index, row in enumerate(selected):
        raw = str(row["SMILES"])
        mol = Chem.MolFromSmiles(raw)
        if mol is None:
            parse_failures.append({"index": index, "reason": "raw_parse_failed"})
            continue
        canonical = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        rebuilt_mol = Chem.MolFromSmiles(canonical)
        if rebuilt_mol is None:
            parse_failures.append({"index": index, "reason": "canonical_reparse_failed"})
            continue
        atom_order_changed += int(raw != canonical)
        raw_graph, raw_info = build_polymer_graph(raw, sample_id=f"audit-{index}-raw")
        can_graph, can_info = build_polymer_graph(canonical, sample_id=f"audit-{index}-canonical")
        topology_counts[raw_info.topology] += 1
        matcher = nx.algorithms.isomorphism.GraphMatcher(
            graph_attrs(raw_graph), graph_attrs(can_graph), node_match=node_match, edge_match=edge_match
        )
        if matcher.is_isomorphic():
            count_equal += 1
        else:
            count_different += 1
            if len(examples) < 10:
                examples.append({
                    "sample_index": index,
                    "original_atoms": raw_info.graph_node_count,
                    "canonical_atoms": can_info.graph_node_count,
                    "original_directed_edges": raw_info.directed_edge_count,
                    "canonical_directed_edges": can_info.directed_edge_count,
                    "dummy_atoms": raw_info.dummy_atom_count,
                    "original_topology": raw_info.topology,
                    "canonical_topology": can_info.topology,
                    "raw_smiles_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "canonical_smiles_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                })
    return {
        "method": "RDKit canonical isomeric SMILES serialization then parse; build both with formal keep-dummy graph builder; categorical graph isomorphism compares every formal atom and bond feature",
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "random_seed": seed,
        "random_sample_size_requested": sample_size,
        "special_representation_rows_in_full_released_set": len(special),
        "unique_rows_checked": len(selected),
        "raw_strings_changed_by_canonicalization": atom_order_changed,
        "feature_labeled_graph_isomorphism_equal": count_equal,
        "feature_labeled_graph_isomorphism_different": count_different,
        "parse_failures": parse_failures,
        "formal_topology_counts": dict(topology_counts),
        "different_graph_examples": examples,
        "inference_run": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal-source", type=Path, default=Path("/private/tmp/Polymer-C128-formal-72490c1a"))
    parser.add_argument("--public-csv", type=Path, default=Path("/Users/gyf/Library/Caches/Polymer-stage5a-c128/posthoc_tg_shift_diagnostic/public.csv"))
    parser.add_argument("--private-csv", type=Path, default=Path("/Users/gyf/Library/Caches/Polymer-stage5a-c128/posthoc_tg_shift_diagnostic/private.csv"))
    parser.add_argument("--prediction-csv", type=Path, default=HERE / "clean_c128_predictions.csv")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    parser.add_argument("--graph-sample-size", type=int, default=256)
    args = parser.parse_args()

    sys.path.insert(0, str(HERE))
    import run_diagnostic as diag

    if diag.formal_git_sha(args.formal_source) != diag.EXPECTED_SOURCE:
        raise SystemExit("formal source checkout is not the pinned C128 source")
    released, data_provenance = diag.load_released_splits(args.public_csv, args.private_csv)
    predictions, predictions_by_smiles = diag.load_predictions(args.prediction_csv)
    if len(predictions) != len(released):
        raise ValueError("prediction/test row count mismatch")
    released_smiles = {str(row["SMILES"]) for row in released}
    if released_smiles != set(predictions_by_smiles):
        raise ValueError("exact raw SMILES sets differ between released files and predictions")

    by_split = {
        split: [row for row in released if row["split"] == split]
        for split in ("public", "private")
    }
    stats = {split: statistics_by_target(rows) for split, rows in by_split.items()}
    stats["combined"] = statistics_by_target(released)
    weight_scenarios = score_scenarios(released, predictions_by_smiles, args.formal_source)
    graph_result = audit_graph_roundtrips(released, args.formal_source, args.graph_sample_size, seed=20261005)

    official_sources = {
        "dataset_metadata_api": DATASET_VIEW,
        "dataset_file_list_api": DATASET_FILES,
        "competition_leaderboard": "https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/leaderboard",
        "competition_data_update_discussion": "https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/discussion/588643",
    }
    dataset_metadata = {
        "dataset_ref": DATASET_REF,
        "dataset_id": 8954694,
        "owner": "Alex Liu (alexliu99)",
        "title": "NeurIPS - Open Polymer Prediction 2025 Test Data",
        "subtitle": "Datasets used in the public and private leaderboards",
        "description_claim": "public.csv is used for the public leaderboard; private.csv is used for the private leaderboard.",
        "dataset_visibility": "public",
        "license": "MIT",
        "current_version_number": 1,
        "version_history": [{"version": 1, "note": "Initial release", "status": "Ready", "created_utc": "2025-12-09T04:27:33.087Z"}],
        "last_updated_utc": "2025-12-09T04:27:33.087Z",
        "files": {
            "public.csv": {"created_utc": "2025-12-09T04:27:34.105Z", "bytes": 19513, "sha256": diag.sha256_file(args.public_csv)},
            "private.csv": {"created_utc": "2025-12-09T04:27:34.024Z", "bytes": 208864, "sha256": diag.sha256_file(args.private_csv)},
        },
        "sources": official_sources,
        "scope_note": "The owner explicitly labels the files as the public/private leaderboard datasets. This is provenance metadata, not an independent proof that the files equal the final scoring snapshot or that their rows map to Kaggle numeric IDs.",
    }
    n_public, n_private = len(by_split["public"]), len(by_split["private"])
    alignment = {
        "classification": "POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE",
        "released_row_count": len(released),
        "rows_public": n_public,
        "rows_private": n_private,
        "public_fraction": n_public / len(released),
        "private_fraction": n_private / len(released),
        "official_final_exact_target_counts_ranges": "not independently exposed by Kaggle's public metadata or leaderboard page; released-file statistics are candidates only if version 1 matches final scoring truth",
        "split_size_comparison": "295/3502 = 8.42% public and 3207/3502 = 91.58% private; consistent with Kaggle's approximate statement that about 92% is used for private scoring, but not identity proof",
        "numeric_kaggle_ids_present": False,
        "joined_by": "exact raw SMILES",
        "preview_rows": 3,
        "preview_nonoverlap_is_mismatch_evidence": False,
        "preview_note": "The downloadable 3-row competition test.csv is a preview. Its non-overlap with the released full test set is expected and provides no evidence of a released-test mismatch.",
        "metadata": dataset_metadata,
        "released_test_provenance": data_provenance,
        "target_statistics": stats,
        "known_online_scores": EXPECTED_SCORES,
        "split_weight_scenarios": weight_scenarios,
        "graph_serialization_roundtrip": graph_result,
        "conclusion": "The released dataset claims public/private leaderboard meaning, and row counts fit the approximate split ratio. However, it lacks Kaggle IDs, the final scored label counts/ranges are not publicly exposed for independent confirmation, and none of the tested mapping/weight scenarios reproduces the four known scores. Released labels cannot be used to exactly reproduce leaderboard scoring.",
        "no_tg_sweep": True,
        "inference_run": False,
        "prediction_csv_sha256": diag.sha256_file(args.prediction_csv),
    }
    write_json(args.output_dir / "dataset_metadata_audit.json", dataset_metadata)
    write_json(args.output_dir / "alignment_audit.json", alignment)

    csv_rows = []
    for record in weight_scenarios:
        csv_rows.append({k: v for k, v in record.items() if k != "weight_values"})
    diag.write_csv(
        args.output_dir / "split_weight_scenarios.csv",
        csv_rows,
        ("assignment", "weight_scope", "scored_split", "rows", "clean_score", "plus70_score", "plus70_minus_clean", "expected_online_clean", "clean_minus_online", "expected_online_plus70", "plus70_minus_online"),
    )
    stats_rows = []
    for split, summary in stats.items():
        for target, values in summary["targets"].items():
            stats_rows.append({"split": split, "target": target, **values})
    diag.write_csv(
        args.output_dir / "released_target_statistics.csv",
        stats_rows,
        ("split", "target", "rows", "observed_label_count", "missing_count", "missingness", "min", "max", "range", "mean", "median"),
    )
    print(json.dumps({
        "alignment_audit": str((args.output_dir / "alignment_audit.json").resolve()),
        "scenario_count": len(weight_scenarios),
        "graph_roundtrip": graph_result,
        "conclusion": alignment["conclusion"],
    }, indent=2))


if __name__ == "__main__":
    main()
