#!/usr/bin/env python3
"""Replay the formal C128 ensemble on released test SMILES, then diagnose Tg shifts.

The first scoring gate is mandatory: if no documented metric scope reproduces
both Kaggle clean and +70 scores to display precision, this script writes only
the inference output and gate report, then exits before creating any sweep.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")
EXPECTED_SOURCE = "72490c1a748ed9395025f4120c9eb04f28268695"
RELEASE_DATASET = "alexliu99/neurips-open-polymer-prediction-2025-test-data"
EXPECTED_SCORES = {
    "public_clean": 0.06899,
    "private_clean": 0.09524,
    "public_tg_plus_70": 0.06493,
    "private_tg_plus_70": 0.07758,
}
COARSE_SHIFTS = (-40, -20, 0, 20, 40, 60, 70, 80, 100, 120)
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(columns), extrasaction="raise", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = list(reader.fieldnames or [])
        rows = list(reader)
    if not columns:
        raise ValueError(f"CSV has no header: {path}")
    return columns, rows


def parse_truth(raw: str | None, *, source: Path, row_no: int, target: str) -> float | None:
    text = "" if raw is None else raw.strip()
    if text.lower() in MISSING_TOKENS:
        return None
    value = float(text)
    if not math.isfinite(value):
        raise ValueError(f"Non-finite {target} at {source}:{row_no}")
    return value


def load_released_splits(public_csv: Path, private_csv: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    combined: list[dict[str, Any]] = []
    per_file: dict[str, Any] = {}
    global_ids: set[str] = set()
    expected_columns = {"SMILES", *TARGETS}
    for split, path in (("public", public_csv), ("private", private_csv)):
        columns, raw_rows = read_csv(path)
        if set(columns) not in (expected_columns, expected_columns | {"id"}):
            raise ValueError(f"Unexpected {split} schema in {path}: {columns}")
        if len(columns) != len(set(columns)):
            raise ValueError(f"Duplicate columns in {path}: {columns}")
        has_kaggle_id = "id" in columns
        ids: set[str] = set()
        split_rows: list[dict[str, Any]] = []
        label_counts = Counter()
        for row_no, raw in enumerate(raw_rows, start=2):
            smiles = (raw.get("SMILES") or "").strip()
            if not smiles:
                raise ValueError(f"Blank SMILES at {path}:{row_no}")
            # The post-competition released files omit Kaggle's numeric ID.
            # The organizers use the exact released SMILES as the stable key.
            sample_id = (raw.get("id") or "").strip() if has_kaggle_id else smiles
            if not sample_id:
                raise ValueError(f"Blank ID at {path}:{row_no}")
            if sample_id in ids or sample_id in global_ids:
                raise ValueError(f"Duplicate ID/SMILES across released splits: {sample_id[:80]}")
            ids.add(sample_id)
            truth = {}
            for target in TARGETS:
                value = parse_truth(raw.get(target), source=path, row_no=row_no, target=target)
                truth[target] = value
                label_counts[target] += value is not None
            split_rows.append({"id": sample_id, "SMILES": smiles, "split": split, **truth})
        global_ids.update(ids)
        combined.extend(split_rows)
        per_file[split] = {
            "path": str(path),
            "sha256": sha256_file(path),
            "rows": len(split_rows),
            "columns": columns,
            "has_numeric_kaggle_id": has_kaggle_id,
            "id_key": "id" if has_kaggle_id else "exact raw SMILES",
            "unique_ids": len(ids),
            "label_counts": dict(label_counts),
        }
    if len(global_ids) != len(combined):
        raise ValueError("Released split IDs are not unique across the combined dataset")
    return combined, {
        "dataset": RELEASE_DATASET,
        "files": per_file,
        "rows_total": len(combined),
        "unique_id_count": len(global_ids),
        "id_mapping_note": (
            "Released public.csv/private.csv have no numeric Kaggle id column. "
            "The analysis uses exact raw SMILES as the released polymer identifier; "
            "the 3-row Kaggle preview is not a full ID mapping."
        ),
    }


def formal_git_sha(formal_source: Path) -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=formal_source, text=True
    ).strip()


def run_inference(
    *,
    formal_source: Path,
    checkpoint_root: Path,
    checkpoint_manifest: Path,
    rows: Sequence[Mapping[str, Any]],
    released_data_provenance: Mapping[str, Any],
    output_csv: Path,
    inference_manifest_path: Path,
    batch_size: int,
    num_threads: int,
) -> dict[str, Any]:
    source_sha = formal_git_sha(formal_source)
    if source_sha != EXPECTED_SOURCE:
        raise ValueError(f"Formal source mismatch: expected {EXPECTED_SOURCE}, got {source_sha}")
    if str(formal_source) not in sys.path:
        sys.path.insert(0, str(formal_source))

    import rdkit
    import torch
    import torch_geometric
    from torch_geometric.loader import DataLoader

    from from_scratch_gnn.models.own_gnn_repr_keep_dummy import graph as keep_dummy_graph
    from from_scratch_gnn.models.polymer_representation_ablation.model import OwnGNNRepresentation

    if rdkit.__version__ != "2026.03.2":
        raise ValueError(f"Formal graph replay expects RDKit 2026.03.2, got {rdkit.__version__}")
    if torch_geometric.__version__ != "2.7.0":
        raise ValueError(f"Formal graph replay expects PyG 2.7.0, got {torch_geometric.__version__}")
    torch.set_num_threads(max(1, int(num_threads)))
    device = torch.device("cpu")

    manifest = json.loads(checkpoint_manifest.read_text(encoding="utf-8"))
    entries = manifest.get("checkpoints", [])
    expected_pairs = {(seed, fold) for seed in range(42, 47) for fold in range(5)}
    found_pairs = {(int(entry["seed"]), int(entry["fold"])) for entry in entries}
    if len(entries) != 25 or found_pairs != expected_pairs or manifest.get("checkpoint_count") != 25:
        raise ValueError("Checkpoint manifest does not contain exactly seeds 42–46 × folds 0–4")

    started = time.monotonic()
    graphs = []
    graph_failures = []
    graph_info_counts = Counter()
    for index, row in enumerate(rows):
        if index and index % 500 == 0:
            print(f"Built {index}/{len(rows)} formal keep-dummy test graphs", flush=True)
        try:
            graph, info = keep_dummy_graph.build_polymer_graph(
                str(row["SMILES"]), sample_id=str(row["id"])
            )
        except Exception as exc:
            graph_failures.append({"id": str(row["id"]), "error": str(exc)})
            continue
        graphs.append(graph)
        graph_info_counts[info.topology] += 1
    if graph_failures or len(graphs) != len(rows):
        raise RuntimeError(f"Graph construction failed for {len(graph_failures)} of {len(rows)} rows")
    print(f"Built {len(graphs)}/{len(rows)} graphs with formal keep-dummy preprocessing", flush=True)

    loader = DataLoader(graphs, batch_size=batch_size, shuffle=False, num_workers=0)
    prediction_sum = np.zeros((len(rows), len(TARGETS)), dtype=np.float64)
    checkpoint_records = []
    models_loaded = 0
    for model_index, entry in enumerate(sorted(entries, key=lambda item: (int(item["seed"]), int(item["fold"]))), start=1):
        seed, fold = int(entry["seed"]), int(entry["fold"])
        relative = Path(entry["relative_path"])
        checkpoint_path = checkpoint_root / relative
        if not checkpoint_path.is_file():
            fallback = Path(entry.get("local_path", ""))
            if fallback.is_file():
                checkpoint_path = fallback
            else:
                raise FileNotFoundError(f"Missing checkpoint for seed={seed} fold={fold}: {checkpoint_path}")
        file_hash = sha256_file(checkpoint_path)
        if file_hash != entry["sha256"]:
            raise ValueError(f"Checkpoint SHA mismatch seed={seed} fold={fold}")
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if checkpoint.get("git_commit") != EXPECTED_SOURCE:
            raise ValueError(f"Checkpoint source SHA mismatch seed={seed} fold={fold}")
        if int(checkpoint.get("fold", -1)) != fold:
            raise ValueError(f"Checkpoint fold metadata mismatch seed={seed} fold={fold}")
        if int(checkpoint.get("seed", -1)) != int(entry["fold_initialization_seed"]):
            raise ValueError(f"Checkpoint seed metadata mismatch seed={seed} fold={fold}")
        model_config = checkpoint.get("model_config", {})
        if model_config != {"hidden_dim": 128, "num_layers": 4, "dropout": 0.1}:
            raise ValueError(f"Unexpected C128 model config seed={seed} fold={fold}: {model_config}")
        normalizer = checkpoint.get("normalizer")
        if not isinstance(normalizer, dict) or set(normalizer) != set(TARGETS):
            raise ValueError(f"Missing fold normalizer seed={seed} fold={fold}")
        means = np.asarray([float(normalizer[target]["mean"]) for target in TARGETS], dtype=np.float64)
        stds = np.asarray([float(normalizer[target]["std"]) for target in TARGETS], dtype=np.float64)
        if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
            raise ValueError(f"Invalid fold scaler seed={seed} fold={fold}")

        model = OwnGNNRepresentation(hidden_dim=128, num_layers=4, dropout=0.1).to(device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        normalized_batches = []
        with torch.inference_mode():
            for batch in loader:
                normalized_batches.append(model(batch.to(device)).cpu().numpy())
        normalized_predictions = np.concatenate(normalized_batches, axis=0).astype(np.float64, copy=False)
        if normalized_predictions.shape != prediction_sum.shape:
            raise ValueError(f"Unexpected prediction shape from seed={seed} fold={fold}")
        prediction_sum += normalized_predictions * stds.reshape(1, -1) + means.reshape(1, -1)
        models_loaded += 1
        checkpoint_records.append({
            "seed": seed,
            "fold": fold,
            "fold_initialization_seed": int(entry["fold_initialization_seed"]),
            "attempt": entry["attempt"],
            "path": str(checkpoint_path.resolve()),
            "sha256": file_hash,
            "size_bytes": checkpoint_path.stat().st_size,
            "scaler_source": "normalizer embedded in this formal fold checkpoint",
            "scaler_sha256": canonical_sha256(normalizer),
        })
        print(f"Loaded/inferred model {models_loaded}/25: seed={seed} fold={fold}", flush=True)
        del model, checkpoint

    if models_loaded != 25:
        raise RuntimeError(f"Expected 25 contributing checkpoints, got {models_loaded}")
    predictions = prediction_sum / 25.0
    if not np.isfinite(predictions).all():
        raise ValueError("Ensemble predictions contain NaN or infinity")
    output_rows = []
    for index, row in enumerate(rows):
        record = {"id": row["id"], "SMILES": row["SMILES"]}
        record.update({target: float(predictions[index, task]) for task, target in enumerate(TARGETS)})
        record["split"] = row["split"]
        output_rows.append(record)
    columns = ("id", "SMILES", *TARGETS, "split")
    write_csv(output_csv, output_rows, columns)
    elapsed = time.monotonic() - started
    source_files = {
        str(path): {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
        for path in (checkpoint_manifest,)
    }
    result = {
        "classification": "POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE",
        "formal_source_sha": source_sha,
        "model": "Stage 5A C128 Own-GNN; keep-dummy graph; 4-layer GINE; mean|max pooling",
        "seeds": [42, 43, 44, 45, 46],
        "folds_per_seed": 5,
        "models_expected": 25,
        "models_loaded_and_ensembled": models_loaded,
        "ensemble_rule": "arithmetic mean of 25 inverse-transformed fold predictions",
        "scaler_rule": "each checkpoint's own embedded fold normalizer was used before averaging",
        "graph_preprocessing": "formal source keep_dummy_graph.build_polymer_graph; no graph changes",
        "rdkit_version": rdkit.__version__,
        "torch_version": torch.__version__,
        "torch_geometric_version": torch_geometric.__version__,
        "device": str(device),
        "batch_size": batch_size,
        "graphs_built": len(graphs),
        "graph_topology_counts": dict(graph_info_counts),
        "graph_failures": graph_failures,
        "inference_seconds": elapsed,
        "released_test": {
            "rows": len(rows),
            "numeric_kaggle_ids_available": False,
            "analysis_id": "exact raw SMILES key from released data",
            "source": dict(released_data_provenance),
        },
        "checkpoint_manifest": source_files,
        "checkpoints": checkpoint_records,
        "prediction_output": {
            "path": str(output_csv.resolve()),
            "rows": len(output_rows),
            "columns": list(columns),
            "sha256": sha256_file(output_csv),
            "size_bytes": output_csv.stat().st_size,
        },
    }
    write_json(inference_manifest_path, result)
    return result


def compute_test_weights(rows: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float | int]]:
    counts = {}
    ranges = {}
    for target in TARGETS:
        values = [float(row[target]) for row in rows if row[target] is not None]
        if not values:
            raise ValueError(f"No observed values for {target}")
        counts[target] = len(values)
        ranges[target] = max(values) - min(values)
        if ranges[target] <= 0:
            raise ValueError(f"Zero range for {target}")
    normalizer = sum(math.sqrt(1.0 / counts[target]) for target in TARGETS)
    return {
        target: {
            "weight": (len(TARGETS) * math.sqrt(1.0 / counts[target]) / normalizer) / ranges[target],
            "valid_count": counts[target],
            "value_range": ranges[target],
        }
        for target in TARGETS
    }


def score_rows(
    truth_rows: Sequence[Mapping[str, Any]],
    prediction_rows: Sequence[Mapping[str, Any]],
    weights: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    # Reuse the formal source's single scoring implementation, passing the
    # explicitly documented weight scope being tested by the fidelity gate.
    from from_scratch_gnn.src.metrics import evaluate_oof
    score = evaluate_oof(truth_rows, prediction_rows, target_weights=weights)
    if not math.isfinite(float(score["overall_oof_wmae"])):
        raise ValueError("Metric returned a non-finite wMAE")
    return score


def load_predictions(path: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    columns, raw_rows = read_csv(path)
    required = {"id", "SMILES", "split", *TARGETS}
    if set(columns) != required or len(columns) != len(required):
        raise ValueError(f"Unexpected prediction schema: {columns}")
    rows = []
    by_smiles = {}
    ids = set()
    for row_no, raw in enumerate(raw_rows, start=2):
        smiles = raw["SMILES"]
        sample_id = raw["id"]
        if not smiles or not sample_id or sample_id in ids or smiles in by_smiles:
            raise ValueError(f"Blank or duplicate prediction ID/SMILES at row {row_no}")
        ids.add(sample_id)
        row = {"id": sample_id, "SMILES": smiles, "split": raw["split"]}
        for target in TARGETS:
            value = float(raw[target])
            if not math.isfinite(value):
                raise ValueError(f"Non-finite prediction {target} at row {row_no}")
            row[target] = value
        rows.append(row)
        by_smiles[smiles] = row
    return rows, by_smiles


def prepare_aligned(
    released_rows: Sequence[Mapping[str, Any]],
    prediction_by_smiles: Mapping[str, Mapping[str, Any]],
) -> dict[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]]:
    released_smiles = {str(row["SMILES"]) for row in released_rows}
    predicted_smiles = set(prediction_by_smiles)
    if released_smiles != predicted_smiles:
        raise ValueError(
            f"Released/prediction SMILES differ: missing={len(released_smiles-predicted_smiles)} "
            f"extra={len(predicted_smiles-released_smiles)}"
        )
    result = {}
    for split in ("public", "private"):
        truth = []
        prediction = []
        for row in released_rows:
            if row["split"] != split:
                continue
            pred = prediction_by_smiles[str(row["SMILES"])]
            if pred["split"] != split:
                raise ValueError(f"Split mismatch for released SMILES in {split}")
            truth.append({target: row[target] for target in TARGETS})
            prediction.append({target: pred[target] for target in TARGETS})
        result[split] = (truth, prediction)
    return result


def get_metric_candidates(
    *,
    aligned: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    released_rows: Sequence[Mapping[str, Any]],
    formal_source: Path,
) -> dict[str, Any]:
    global_test_weights = compute_test_weights(released_rows)
    manifest_path = formal_source / "from_scratch_gnn/benchmark/data_manifest.json"
    benchmark = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_weights = benchmark["target_statistics_and_weights"]
    candidates = {
        "global_released_test_weights": {split: global_test_weights for split in aligned},
        "split_specific_released_test_weights": {
            split: compute_test_weights([row for row in released_rows if row["split"] == split])
            for split in aligned
        },
        "official_training_snapshot_weights": {split: train_weights for split in aligned},
    }
    scenarios = {}
    for scope, weights_by_split in candidates.items():
        scores = {}
        for split, (truth, prediction) in aligned.items():
            clean = score_rows(truth, prediction, weights_by_split[split])
            shifted = [dict(row) for row in prediction]
            for row in shifted:
                row["Tg"] += 70.0
            plus70 = score_rows(truth, shifted, weights_by_split[split])
            scores[f"{split}_clean"] = float(clean["overall_oof_wmae"])
            scores[f"{split}_tg_plus_70"] = float(plus70["overall_oof_wmae"])
        errors = {key: abs(value - EXPECTED_SCORES[key]) for key, value in scores.items()}
        scenarios[scope] = {
            "scores": scores,
            "absolute_error_vs_kaggle_display": errors,
            "all_four_within_display_precision": all(error <= 0.0000051 for error in errors.values()),
            "weights": {
                split: {target: dict(values) for target, values in weights_by_split[split].items()}
                for split in weights_by_split
            },
        }
    return {
        "expected_kaggle_scores": EXPECTED_SCORES,
        "display_tolerance": 0.0000051,
        "candidate_scopes": scenarios,
        "official_metric_formula": (
            "w_i=(1/r_i)*(K*sqrt(1/n_i)/sum_j sqrt(1/n_j)); "
            "wMAE=(1/N_split)*sum_rows sum_observed_targets(w_i*abs(error))"
        ),
        "training_manifest_sha256": sha256_file(manifest_path),
    }


def choose_metric_scope(gate: Mapping[str, Any]) -> str | None:
    passed = [
        scope for scope, details in gate["candidate_scopes"].items()
        if details["all_four_within_display_precision"]
    ]
    if not passed:
        return None
    priority = (
        "global_released_test_weights",
        "split_specific_released_test_weights",
        "official_training_snapshot_weights",
    )
    return next(scope for scope in priority if scope in passed)


def selected_weights(gate: Mapping[str, Any], scope: str, split: str) -> dict[str, dict[str, Any]]:
    return gate["candidate_scopes"][scope]["weights"][split]


def metric_for_delta(
    aligned: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    gate: Mapping[str, Any],
    scope: str,
    split: str,
    delta: float,
) -> dict[str, Any]:
    truth, base_prediction = aligned[split]
    prediction = [dict(row) for row in base_prediction]
    for row in prediction:
        row["Tg"] += float(delta)
    return score_rows(truth, prediction, selected_weights(gate, scope, split))


def contribution_records(
    aligned: Mapping[str, tuple[list[dict[str, Any]], list[dict[str, Any]]]],
    gate: Mapping[str, Any],
    scope: str,
    scenarios: Sequence[tuple[str, float]],
) -> list[dict[str, Any]]:
    records = []
    for split in ("public", "private"):
        truth, base_prediction = aligned[split]
        weights = selected_weights(gate, scope, split)
        row_count = len(truth)
        for scenario, delta in scenarios:
            prediction = [dict(row) for row in base_prediction]
            for row in prediction:
                row["Tg"] += float(delta)
            result = score_rows(truth, prediction, weights)
            for target in TARGETS:
                records.append({
                    "scenario": scenario,
                    "delta_celsius": float(delta),
                    "split": split,
                    "target": target,
                    "contribution": float(result["target_contribution"][target]),
                    "target_mae": result["target_mae"][target],
                    "observed_labels": result["target_counts"][target],
                    "split_rows_denominator": row_count,
                    "metric_weight": float(weights[target]["weight"]),
                })
            records.append({
                "scenario": scenario,
                "delta_celsius": float(delta),
                "split": split,
                "target": "Total",
                "contribution": float(result["overall_oof_wmae"]),
                "target_mae": None,
                "observed_labels": row_count,
                "split_rows_denominator": row_count,
                "metric_weight": None,
            })
    return records


def make_figures(
    *,
    score_rows_for_curve: Sequence[Mapping[str, Any]],
    residuals: Mapping[str, np.ndarray],
    optima: Mapping[str, float],
    figure_dir: Path,
) -> tuple[Path, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure_dir.mkdir(parents=True, exist_ok=True)
    score_figure = figure_dir / "tg_shift_score_curves.png"
    residual_figure = figure_dir / "tg_residual_distributions.png"

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    colors = {"public": "#1769aa", "private": "#d24b36"}
    for split in ("public", "private"):
        records = [row for row in score_rows_for_curve if row["split"] == split]
        records.sort(key=lambda row: row["delta_celsius"])
        axis.plot(
            [row["delta_celsius"] for row in records],
            [row["wmae"] for row in records],
            color=colors[split], linewidth=2.0, marker="o", markersize=3.2,
            label=f"{split.title()} leaderboard split",
        )
    axis.axvline(0, color="#555555", linestyle="--", linewidth=1.1, label="Clean Δ=0")
    axis.axvline(70, color="#7b3294", linestyle="--", linewidth=1.1, label="Diagnostic Δ=70")
    axis.axvline(optima["public"], color=colors["public"], linestyle=":", linewidth=1.4,
                 label=f"Public optimum {optima['public']:.2f}°C")
    axis.axvline(optima["private"], color=colors["private"], linestyle=":", linewidth=1.4,
                 label=f"Private optimum {optima['private']:.2f}°C")
    axis.set_xlabel("Tg additive shift (°C)")
    axis.set_ylabel("Kaggle weighted MAE (lower is better)")
    axis.set_title("Post-hoc Tg shift diagnostic on released test labels")
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, ncol=2, fontsize=9)
    fig.savefig(score_figure, dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(10.5, 6.2), constrained_layout=True)
    for split in ("public", "private"):
        values = np.sort(residuals[split])
        ecdf = np.arange(1, len(values) + 1) / len(values)
        axis.step(values, ecdf, where="post", color=colors[split], linewidth=2.0,
                  label=f"{split.title()} (n={len(values)})")
        axis.axvline(optima[split], color=colors[split], linestyle=":", linewidth=1.4,
                     label=f"{split.title()} median {optima[split]:.2f}°C")
    axis.axvline(70, color="#7b3294", linestyle="--", linewidth=1.1, label="+70°C diagnostic")
    axis.axhline(0.5, color="#777777", linestyle="--", linewidth=0.8)
    axis.set_xlabel("Tg residual: Tg_true − Tg_pred (°C)")
    axis.set_ylabel("Empirical cumulative fraction")
    axis.set_title("Released-label Tg residual distributions")
    axis.set_ylim(0, 1.02)
    axis.grid(alpha=0.22)
    axis.legend(frameon=False, fontsize=9)
    fig.savefig(residual_figure, dpi=190)
    plt.close(fig)
    return score_figure, residual_figure


def format_table(headers: Sequence[str], rows: Sequence[Sequence[Any]]) -> str:
    def cell(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.8f}"
        return str(value)
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    out.extend("| " + " | ".join(cell(v) for v in row) + " |" for row in rows)
    return "\n".join(out)


def run_analysis(
    *,
    formal_source: Path,
    public_csv: Path,
    private_csv: Path,
    prediction_csv: Path,
    inference_manifest_path: Path,
    output_dir: Path,
    id_mapping_note: str,
) -> dict[str, Any]:
    if formal_git_sha(formal_source) != EXPECTED_SOURCE:
        raise ValueError("Analysis must import metric code from the exact formal source checkout")
    if str(formal_source) not in sys.path:
        sys.path.insert(0, str(formal_source))
    released_rows, data_provenance = load_released_splits(public_csv, private_csv)
    predictions, prediction_by_smiles = load_predictions(prediction_csv)
    aligned = prepare_aligned(released_rows, prediction_by_smiles)
    if len(predictions) != len(released_rows):
        raise ValueError("Prediction and released truth row counts differ")

    # Validate the one-time replay gate against both real Kaggle score pairs.
    gate = get_metric_candidates(aligned=aligned, released_rows=released_rows, formal_source=formal_source)
    scope = choose_metric_scope(gate)
    gate["passed"] = scope is not None
    gate["selected_metric_scope"] = scope
    gate["scored_submission_refs"] = {
        "clean": {"ref": "56831294", "public": 0.06899, "private": 0.09524},
        "tg_plus_70": {"ref": "56831959", "public": 0.06493, "private": 0.07758},
    }
    write_json(output_dir / "metric_fidelity_gate.json", gate)
    if scope is None:
        inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
        gate_rows = []
        for candidate_scope, details in gate["candidate_scopes"].items():
            scores = details["scores"]
            gate_rows.append((
                candidate_scope,
                scores["public_clean"],
                scores["public_tg_plus_70"],
                scores["private_clean"],
                scores["private_tg_plus_70"],
                details["all_four_within_display_precision"],
            ))
        report_lines = [
            "# Post-hoc Tg shift diagnostic — metric gate failed",
            "",
            "> **POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE**",
            "",
            "## Stop condition",
            "",
            "The clean and Tg +70 offline scores did not reproduce both real Kaggle score pairs within five-decimal display precision under any tested weight scope. The requested Tg shift sweep, split optima, target decomposition, and figures were therefore not computed.",
            "",
            "## Fixed inference replay",
            "",
            f"- Formal source: `{EXPECTED_SOURCE}`; main SHA context: `c9949c088d8822eac775a55ad601d278b747c9f9`.",
            f"- Checkpoints loaded and ensembled: **{inference_manifest['models_loaded_and_ensembled']}/25**; each fold used its own checkpoint-embedded normalizer.",
            f"- Prediction file SHA-256: `{sha256_file(prediction_csv)}`.",
            "- As an implementation cross-check, the same 25 checkpoints were replayed on the three rows in the downloadable Kaggle `test.csv` preview. Per-target predictions matched the successful kernel's three-row `submission.csv` to floating-point tolerance; the comparison details are in `kaggle_preview_replay.json`.",
            "",
            "## Released-data identity limitation",
            "",
            f"- Dataset: `{data_provenance['dataset']}`; public/private files contain {data_provenance['files']['public']['rows']} / {data_provenance['files']['private']['rows']} rows.",
            f"- Public SHA-256: `{data_provenance['files']['public']['sha256']}`; private SHA-256: `{data_provenance['files']['private']['sha256']}`.",
            "- Neither released truth file contains Kaggle's numeric `id`; this replay joins on exact raw SMILES. The downloadable competition preview has only three rows, and none of its structures matches the released set by raw or RDKit-canonical SMILES. It therefore does not establish an ID-to-SMILES mapping for the released 3,502 rows.",
            "",
            "## Fidelity gate results",
            "",
            "Known Kaggle values are Public/Private **0.06899 / 0.09524** for Clean and **0.06493 / 0.07758** for Tg +70.",
            "",
            format_table(
                ("Weight scope", "Public clean", "Public +70", "Private clean", "Private +70", "Pass"),
                gate_rows,
            ),
            "",
            "## Discrepancy diagnosis",
            "",
            "The frozen model inference reproduces the successful kernel output on the available three-row Kaggle preview, so this does not point to a checkpoint, graph preprocessing, fold scaler, or ensemble-loading error. The official weighted-MAE formula was evaluated with global released-test weights, split-specific released-test weights, and the frozen official-train weights; none reproduces all four scored values. The remaining mismatch is at the scored-data / leaderboard-metric alignment boundary. The released files lack numeric Kaggle IDs, and none of the three structures in the downloadable preview matches the released set, even after RDKit canonicalization. The available artifacts therefore cannot distinguish a released truth snapshot mismatch from a difference in the leaderboard's effective scoring weights or scored rows.",
            "",
            "This is the stopping point required by the predeclared gate. No private-score-informed shifts or curves were computed.",
            "",
        ]
        report_path = output_dir / "report.md"
        report_path.write_text("\n".join(report_lines), encoding="utf-8")
        provenance = {
            "classification": "POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE",
            "status": "stopped_at_metric_fidelity_gate",
            "formal_source_sha": EXPECTED_SOURCE,
            "main_source_sha": "c9949c088d8822eac775a55ad601d278b747c9f9",
            "released_data": data_provenance,
            "checkpoint_manifest": inference_manifest["checkpoint_manifest"],
            "checkpoints": inference_manifest["checkpoints"],
            "prediction": {
                "path": prediction_csv.name,
                "sha256": sha256_file(prediction_csv),
                "size_bytes": prediction_csv.stat().st_size,
            },
            "inference_manifest_sha256": sha256_file(inference_manifest_path),
            "metric_fidelity_gate": {
                "path": "metric_fidelity_gate.json",
                "sha256": sha256_file(output_dir / "metric_fidelity_gate.json"),
                "passed": False,
            },
            "kaggle_preview_replay": {
                "path": "kaggle_preview_replay.json",
                "sha256": sha256_file(output_dir / "kaggle_preview_replay.json"),
            } if (output_dir / "kaggle_preview_replay.json").is_file() else None,
            "report": {"path": report_path.name, "sha256": sha256_file(report_path)},
            "shift_sweep_performed": False,
            "figures_generated": False,
        }
        write_json(output_dir / "provenance.json", provenance)
        print("Metric fidelity gate failed; wrote gate report and stopped before shift sweep.", flush=True)
        return {"gate_passed": False, "gate": gate}

    # The absolute-error optimum is the median residual on observed Tg rows.
    residuals: dict[str, np.ndarray] = {}
    optima: dict[str, float] = {}
    residual_stats = []
    for split in ("public", "private"):
        truth, prediction = aligned[split]
        values = np.asarray([
            float(y["Tg"]) - float(p["Tg"])
            for y, p in zip(truth, prediction)
            if y["Tg"] is not None
        ], dtype=np.float64)
        if len(values) == 0 or not np.isfinite(values).all():
            raise ValueError(f"Invalid Tg residual vector for {split}")
        residuals[split] = values
        median = float(np.median(values))
        optima[split] = median
        quantiles = np.quantile(values, [0.10, 0.25, 0.50, 0.75, 0.90], method="linear")
        residual_stats.append({
            "split": split,
            "observed_tg_count": len(values),
            "mean": float(np.mean(values)),
            "median": median,
            "std_population": float(np.std(values, ddof=0)),
            "p10": float(quantiles[0]),
            "p25": float(quantiles[1]),
            "p50": float(quantiles[2]),
            "p75": float(quantiles[3]),
            "p90": float(quantiles[4]),
        })
    write_csv(
        output_dir / "residual_statistics.csv", residual_stats,
        ("split", "observed_tg_count", "mean", "median", "std_population", "p10", "p25", "p50", "p75", "p90"),
    )

    delta_kinds: dict[float, set[str]] = {}
    for delta in COARSE_SHIFTS:
        delta_kinds.setdefault(float(delta), set()).add("coarse")
    for split in ("public", "private"):
        center = optima[split]
        for delta in np.arange(center - 20.0, center + 20.0001, 2.0):
            delta_kinds.setdefault(round(float(delta), 8), set()).add(f"fine_{split}")
        delta_kinds.setdefault(round(center, 8), set()).add(f"exact_median_{split}")

    shift_rows = []
    curve_rows = []
    for delta in sorted(delta_kinds):
        for split in ("public", "private"):
            score = metric_for_delta(aligned, gate, scope, split, delta)
            record = {
                "split": split,
                "delta_celsius": float(delta),
                "grid_kind": ";".join(sorted(delta_kinds[delta])),
                "wmae": float(score["overall_oof_wmae"]),
            }
            for target in TARGETS:
                record[f"{target}_contribution"] = float(score["target_contribution"][target])
            shift_rows.append(record)
            curve_rows.append(record)
    shift_columns = (
        "split", "delta_celsius", "grid_kind", "wmae",
        *(f"{target}_contribution" for target in TARGETS),
    )
    write_csv(output_dir / "shift_scores.csv", shift_rows, shift_columns)

    scenarios = [
        ("clean", 0.0),
        ("Tg_plus_70", 70.0),
        ("public_optimal_shift", optima["public"]),
        ("private_optimal_shift", optima["private"]),
    ]
    contributions = contribution_records(aligned, gate, scope, scenarios)
    write_csv(
        output_dir / "target_contributions.csv", contributions,
        ("scenario", "delta_celsius", "split", "target", "contribution", "target_mae", "observed_labels", "split_rows_denominator", "metric_weight"),
    )

    scenario_scores = {}
    for name, delta in scenarios:
        scenario_scores[name] = {
            split: float(metric_for_delta(aligned, gate, scope, split, delta)["overall_oof_wmae"])
            for split in ("public", "private")
        }
    clean_tg_private = next(
        row["contribution"] for row in contributions
        if row["scenario"] == "clean" and row["split"] == "private" and row["target"] == "Tg"
    )
    plus70_tg_private = next(
        row["contribution"] for row in contributions
        if row["scenario"] == "Tg_plus_70" and row["split"] == "private" and row["target"] == "Tg"
    )
    private_gap = scenario_scores["clean"]["private"] - scenario_scores["Tg_plus_70"]["private"]
    tg_gap = clean_tg_private - plus70_tg_private
    if abs(private_gap - tg_gap) > 1e-10:
        raise AssertionError("Private clean→+70 score change is not fully explained by Tg contribution")

    opt_grid_diagnostics = {}
    for split in ("public", "private"):
        in_fine = [r for r in shift_rows if r["split"] == split and "fine_" + split in r["grid_kind"]]
        best_grid = min(in_fine, key=lambda row: row["wmae"])
        opt_grid_diagnostics[split] = {
            "exact_median_shift": optima[split],
            "best_2c_grid_shift": best_grid["delta_celsius"],
            "difference_celsius": abs(best_grid["delta_celsius"] - optima[split]),
            "best_2c_grid_score": best_grid["wmae"],
            "exact_median_score": scenario_scores["public_optimal_shift" if split == "public" else "private_optimal_shift"][split],
        }

    public_opt_private_score = scenario_scores["public_optimal_shift"]["private"]
    private_optimum_score = scenario_scores["private_optimal_shift"]["private"]
    optima_gap = abs(optima["private"] - optima["public"])
    summary = {
        "classification": "POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE",
        "formal_source_sha": EXPECTED_SOURCE,
        "main_source_sha": "c9949c088d8822eac775a55ad601d278b747c9f9",
        "metric_fidelity_gate": {"passed": True, "selected_scope": scope},
        "known_kaggle_scores": EXPECTED_SCORES,
        "offline_scores": scenario_scores,
        "public_optimal_shift_celsius": optima["public"],
        "private_optimal_shift_celsius": optima["private"],
        "public_private_optimum_gap_celsius": optima_gap,
        "distance_plus70_to_private_optimum_celsius": abs(70.0 - optima["private"]),
        "public_optimum_applied_to_private_score": public_opt_private_score,
        "best_private_score_under_single_constant_tg_shift": private_optimum_score,
        "private_clean_score_reduction_from_plus70": private_gap,
        "private_tg_contribution_reduction_from_plus70": tg_gap,
        "private_clean_tg_contribution": clean_tg_private,
        "private_clean_tg_share_of_total": clean_tg_private / scenario_scores["clean"]["private"],
        "private_tg_explains_clean_to_plus70_improvement_fraction": tg_gap / private_gap if private_gap else None,
        "residual_statistics": residual_stats,
        "median_vs_fine_grid": opt_grid_diagnostics,
        "released_data": data_provenance,
        "id_mapping_note": id_mapping_note,
        "submission_refs": {"clean": "56831294", "Tg_plus_70": "56831959"},
    }
    write_json(output_dir / "diagnostic_summary.json", summary)

    score_figure, residual_figure = make_figures(
        score_rows_for_curve=curve_rows, residuals=residuals, optima=optima,
        figure_dir=output_dir / "figures",
    )
    prediction_sha = sha256_file(prediction_csv)
    inference_manifest = json.loads(inference_manifest_path.read_text(encoding="utf-8"))
    report_lines = [
        "# Post-hoc Tg shift diagnostic",
        "",
        "> **POST-HOC / USES RELEASED TEST LABELS / NOT VALID BLIND PERFORMANCE**",
        "",
        "This report replays the frozen Stage 5A C128 checkpoints on the released Public/Private structures. It does not redefine the clean C128 score or alter Stage R results.",
        "",
        "## Replay and metric gate",
        "",
        f"- Formal source: `{EXPECTED_SOURCE}`; main source context: `{summary['main_source_sha']}`.",
        f"- Models: {inference_manifest['models_loaded_and_ensembled']}/25; per-fold embedded normalizers; arithmetic mean after inverse transforms.",
        f"- Scoring scope that passed the four-score fidelity gate: `{scope}`.",
        f"- Released public/private rows: {data_provenance['files']['public']['rows']} / {data_provenance['files']['private']['rows']}; predictions were joined by exact raw SMILES.",
        f"- Numeric Kaggle IDs are absent from the released truth files. In `clean_c128_predictions.csv`, `id` is therefore the exact SMILES key; this table is diagnostic output, not a submission file.",
        f"- Public CSV SHA-256: `{data_provenance['files']['public']['sha256']}`.",
        f"- Private CSV SHA-256: `{data_provenance['files']['private']['sha256']}`.",
        f"- Clean prediction CSV SHA-256: `{prediction_sha}`.",
        "",
        "## Scores and exact shifts",
        "",
        format_table(
            ("Scenario", "Applied Δ (°C)", "Public wMAE", "Private wMAE"),
            [
                ("Clean", 0.0, scenario_scores["clean"]["public"], scenario_scores["clean"]["private"]),
                ("Tg +70", 70.0, scenario_scores["Tg_plus_70"]["public"], scenario_scores["Tg_plus_70"]["private"]),
                ("Public optimum", optima["public"], scenario_scores["public_optimal_shift"]["public"], scenario_scores["public_optimal_shift"]["private"]),
                ("Private optimum", optima["private"], scenario_scores["private_optimal_shift"]["public"], scenario_scores["private_optimal_shift"]["private"]),
            ],
        ),
        "",
        "## Tg residual statistics (truth − clean prediction)",
        "",
        format_table(
            ("Split", "n", "Mean", "Median / optimum", "Std (population)", "P10", "P25", "P50", "P75", "P90"),
            [(r["split"], r["observed_tg_count"], r["mean"], r["median"], r["std_population"], r["p10"], r["p25"], r["p50"], r["p75"], r["p90"]) for r in residual_stats],
        ),
        "",
        "## Target contribution decomposition",
        "",
        "Each value is one target's contribution to the split score; Total is the sum of the five contributions.",
        "",
        format_table(
            ("Scenario", "Target", "Public contribution", "Private contribution"),
            [
                (scenario, target,
                 next(r["contribution"] for r in contributions if r["scenario"] == scenario and r["split"] == "public" and r["target"] == target),
                 next(r["contribution"] for r in contributions if r["scenario"] == scenario and r["split"] == "private" and r["target"] == target))
                for scenario, _delta in scenarios for target in (*TARGETS, "Total")
            ],
        ),
        "",
        "## Answers",
        "",
        f"1. Public optimum: **{optima['public']:.4f}°C**; Private optimum: **{optima['private']:.4f}°C** (difference **{optima_gap:.4f}°C**).",
        f"2. +70°C is **{abs(70.0-optima['private']):.4f}°C** from the Private optimum.",
        f"3. Clean → +70 improvement is **{scenario_scores['clean']['public']-scenario_scores['Tg_plus_70']['public']:.8f} Public** and **{private_gap:.8f} Private**. Since the other four predictions are held fixed, the entire score delta is Tg; its Private target contribution falls by `{tg_gap:.8f}`.",
        f"4. The Private clean score's Tg contribution is `{clean_tg_private:.8f}` ({clean_tg_private/scenario_scores['clean']['private']:.1%} of the clean total).",
        f"5. A single constant correction at the Private residual median reaches **{private_optimum_score:.8f} Private** offline; applying the Public-optimal shift to Private gives **{public_opt_private_score:.8f}**.",
        "6. These are post-hoc diagnostics using released test labels and are not blind performance estimates.",
        "",
        "## Figures and files",
        "",
        "- `figures/tg_shift_score_curves.png` — Public/Private score against Tg shift.",
        "- `figures/tg_residual_distributions.png` — Tg residual empirical distributions.",
        "- `shift_scores.csv`, `target_contributions.csv`, `residual_statistics.csv`, `diagnostic_summary.json`, and `provenance.json` contain the tabular results and hashes.",
        "",
    ]
    (output_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")

    provenance = {
        "classification": summary["classification"],
        "formal_source_sha": EXPECTED_SOURCE,
        "main_source_sha": summary["main_source_sha"],
        "metric_scope": scope,
        "released_data": data_provenance,
        "checkpoint_manifest": inference_manifest["checkpoint_manifest"],
        "checkpoints": inference_manifest["checkpoints"],
        "prediction": {"path": prediction_csv.name, "sha256": prediction_sha, "size_bytes": prediction_csv.stat().st_size},
        "inference_manifest_sha256": sha256_file(inference_manifest_path),
        "outputs": {
            path.name: {"sha256": sha256_file(path), "size_bytes": path.stat().st_size}
            for path in (
                output_dir / "metric_fidelity_gate.json",
                output_dir / "shift_scores.csv",
                output_dir / "target_contributions.csv",
                output_dir / "residual_statistics.csv",
                output_dir / "diagnostic_summary.json",
                output_dir / "report.md",
                score_figure,
                residual_figure,
            )
        },
    }
    write_json(output_dir / "provenance.json", provenance)
    print(json.dumps({
        "gate_passed": True,
        "metric_scope": scope,
        "public_optimum": optima["public"],
        "private_optimum": optima["private"],
        "best_private_score": private_optimum_score,
        "prediction_sha256": prediction_sha,
        "output_dir": str(output_dir.resolve()),
    }, indent=2), flush=True)
    return {"gate_passed": True, "summary": summary}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--formal-source", type=Path, required=True)
    parser.add_argument("--public-csv", type=Path, required=True)
    parser.add_argument("--private-csv", type=Path, required=True)
    parser.add_argument("--checkpoint-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--skip-inference", action="store_true", help="Reuse an existing clean prediction CSV")
    parser.add_argument("--id-mapping-note", default="Released truth files omit numeric Kaggle IDs; exact SMILES is the unique released-data key.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    formal_source = args.formal_source.resolve()
    public_csv = args.public_csv.resolve()
    private_csv = args.private_csv.resolve()
    checkpoint_root = args.checkpoint_root.resolve()
    checkpoint_manifest = args.checkpoint_manifest.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_csv = output_dir / "clean_c128_predictions.csv"
    inference_manifest_path = output_dir / "inference_provenance.json"

    rows, data_provenance = load_released_splits(public_csv, private_csv)
    print(json.dumps({"released_rows": data_provenance["rows_total"], "splits": data_provenance["files"]}, indent=2), flush=True)
    if not args.skip_inference:
        inference = run_inference(
            formal_source=formal_source,
            checkpoint_root=checkpoint_root,
            checkpoint_manifest=checkpoint_manifest,
            rows=rows,
            released_data_provenance=data_provenance,
            output_csv=prediction_csv,
            inference_manifest_path=inference_manifest_path,
            batch_size=args.batch_size,
            num_threads=args.num_threads,
        )
        print(f"Inference complete: {inference['models_loaded_and_ensembled']}/25; SHA256={inference['prediction_output']['sha256']}", flush=True)
    elif not prediction_csv.is_file() or not inference_manifest_path.is_file():
        raise FileNotFoundError("--skip-inference requires existing clean_c128_predictions.csv and inference_provenance.json")

    analysis = run_analysis(
        formal_source=formal_source,
        public_csv=public_csv,
        private_csv=private_csv,
        prediction_csv=prediction_csv,
        inference_manifest_path=inference_manifest_path,
        output_dir=output_dir,
        id_mapping_note=args.id_mapping_note,
    )
    return 0 if analysis["gate_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
