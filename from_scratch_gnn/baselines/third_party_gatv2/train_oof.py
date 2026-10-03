"""Train the public GATv2 architecture from random initialization on frozen folds."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

TRACK_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = TRACK_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from from_scratch_gnn.src.data import (  # noqa: E402
    TARGETS,
    load_training_data,
    sha256_file,
    write_json,
)
from from_scratch_gnn.src.metrics import competition_weights, evaluate_oof  # noqa: E402
from from_scratch_gnn.scripts.evaluate_oof import validate_and_score  # noqa: E402
from from_scratch_gnn.baselines.third_party_gatv2.model import (  # noqa: E402
    MORGAN_BITS,
    SELECTED_MORGAN_BITS,
    PolymerGNNV12Res,
    augment_repeat_units,
    fit_fold_fp_indices,
    process_smiles,
)

FROZEN_SEED = 20250604
PUBLIC_INIT_SEED = 42
DEFAULT_CONFIG: dict[str, Any] = {
    "experiment_id": "third_party_gatv2_frozen_oof_v1",
    "model_name": "public_third_place_gatv2_from_scratch",
    "benchmark_version": "nopp2025_train_v1",
    "benchmark_folds": "from_scratch_gnn/benchmark/folds.csv",
    "benchmark_fold_sha256": "1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a",
    "source_train_sha256": "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1",
    "fold_seed": FROZEN_SEED,
    "initialization_seed_base": PUBLIC_INIT_SEED,
    "initialization_seed_rule": "42 + fold; independent of frozen fold construction",
    "n_splits": 5,
    "targets": list(TARGETS),
    "architecture": {
        "node_features": [
            "atomic_number",
            "degree",
            "formal_charge",
            "radical_electrons",
            "hybridization_enum",
            "is_aromatic",
            "total_hydrogens",
        ],
        "edge_features": ["bond_order_double", "is_conjugated"],
        "periodic_star_edge_features": [-1.0, 0.0],
        "layers": 6,
        "hidden_dim": 384,
        "attention_heads": 8,
        "attention_dropout": 0.2,
        "residual": "elementwise_add_on_each_hidden_layer",
        "normalization": "PyG BatchNorm after every GATv2Conv",
        "activation": "ELU after every BatchNorm",
        "pooling": "global_mean_pool",
        "morgan": {
            "radius": 2,
            "bits": MORGAN_BITS,
            "selection": "fold-train SelectKBest(f_regression), up to 50 bits per target",
        },
        "task_heads": "five independent Linear(384+k,384)-ReLU-Dropout(0.2)-Linear(384,1)",
    },
    "preprocessing": {
        "repeat_unit_augmentation": "public three-repeat-unit augmentation; train fold only",
        "validation_input": "original frozen-fold SMILES only; no TTA or calibration",
        "target_normalization": "none (matches public source)",
        "feature_selection_fit": "current fold's train samples with an observed target only",
    },
    "training": {
        "optimizer": "AdamW",
        "learning_rate": 1e-4,
        "epochs_max": 600,
        "batch_size": 64,
        "early_stopping": "patience=40, minimum improvement=0, restore best checkpoint",
        "selection_metric": "Stage 0 evaluate_oof with frozen training-snapshot target weights",
        "loss": "masked weighted MAE; fold-train statistics, divide by batch graph count",
        "loss_weights": "competition_weights from observed labels in current train fold only",
        "shuffle_train": True,
    },
    "postprocessing": {
        "validation_linear_calibrator": False,
        "released_label_shift": False,
        "private_labels_or_scores": False,
        "fold_ensemble": False,
        "test_time_augmentation": False,
    },
    "public_source": {
        "repository": "https://github.com/fresnellll/kaggle-NeurIPS-polymer-prediction-solution",
        "commit": "f385d220d348283792c9f3dc8ed4ab0619e6f7c4",
        "training_script": "src/train.py",
        "preprocessing_script": "src/prepare_data.py",
    },
    "artifacts": {
        "checkpoints": "artifacts/checkpoints/fold_<k>/best.pt (ignored by Git)",
        "histories": "artifacts/training_history/fold_<k>.csv",
        "oof_predictions": "artifacts/oof_predictions.csv",
        "metrics": "artifacts/metrics.json",
        "fold_metrics": "artifacts/fold_metrics.csv",
    },
}


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable.")
    return device


def _load_frozen_inputs(
    train_csv: Path, folds_csv: Path
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[int]]:
    manifest_path = TRACK_ROOT / "benchmark" / "data_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if sha256_file(train_csv) != DEFAULT_CONFIG["source_train_sha256"]:
        raise ValueError("Training CSV SHA-256 differs from the frozen Stage 0 snapshot.")
    if sha256_file(folds_csv) != DEFAULT_CONFIG["benchmark_fold_sha256"]:
        raise ValueError("folds.csv SHA-256 differs from the frozen Stage 0 assignment.")

    rows = load_training_data(train_csv)
    with folds_csv.open("r", encoding="utf-8-sig", newline="") as source:
        fold_rows = list(csv.DictReader(source))
    if len(rows) != len(fold_rows):
        raise ValueError(f"Training/fold row counts differ: {len(rows)} != {len(fold_rows)}")
    fold_by_id: dict[str, int] = {}
    smiles_by_id: dict[str, str] = {}
    for fold_row in fold_rows:
        sample_id = str(fold_row["sample_id"])
        if sample_id in fold_by_id:
            raise ValueError(f"Duplicate sample_id in frozen folds: {sample_id}")
        fold_by_id[sample_id] = int(fold_row["fold"])
        smiles_by_id[sample_id] = fold_row["SMILES"]
    ids = [str(row["sample_id"]) for row in rows]
    if set(ids) != set(fold_by_id):
        raise ValueError("Training IDs do not exactly match frozen fold IDs.")
    for row in rows:
        sample_id = str(row["sample_id"])
        if row["SMILES"] != smiles_by_id[sample_id]:
            raise ValueError(f"SMILES mismatch against folds.csv for sample {sample_id}")
    assignments = [fold_by_id[sample_id] for sample_id in ids]
    weights = {
        target: {"weight": manifest["target_statistics_and_weights"][target]["weight"]}
        for target in TARGETS
    }
    return rows, weights, assignments


def _make_labeled_graph(graph: Data, labels: Mapping[str, Any]) -> Data:
    item = Data(
        x=graph.x,
        edge_index=graph.edge_index,
        edge_attr=graph.edge_attr,
        morgan_fp=graph.morgan_fp,
        y=torch.tensor(
            [float(labels[target]) if labels[target] is not None else float("nan") for target in TARGETS],
            dtype=torch.float32,
        ).unsqueeze(0),
    )
    return item


def _loss_weight_map(rows: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> dict[str, Any]:
    return competition_weights([rows[int(index)] for index in indices])


class MaskedWeightedMAELoss(torch.nn.Module):
    """Training objective; score reporting remains centralized in Stage 0 metrics."""

    def __init__(self, weights: Mapping[str, Mapping[str, Any]], device: torch.device):
        super().__init__()
        self.register_buffer(
            "weights",
            torch.tensor([float(weights[target]["weight"]) for target in TARGETS], device=device),
        )

    def forward(self, predictions: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
        observed = torch.isfinite(truth)
        if not observed.any():
            return predictions.sum() * 0.0
        errors = torch.abs(predictions - torch.nan_to_num(truth, nan=0.0))
        weighted = errors * self.weights.unsqueeze(0)
        # Keep the public objective's per-graph denominator and mask all missing labels.
        return weighted.masked_select(observed).sum() / predictions.shape[0]


def _score_validation(
    model: PolymerGNNV12Res,
    loader: DataLoader,
    device: torch.device,
    truth_rows: Sequence[Mapping[str, Any]],
    validation_indices: Sequence[int],
    validation_weights: Mapping[str, Any],
) -> tuple[float, dict[str, float | None], np.ndarray]:
    model.eval()
    chunks: list[torch.Tensor] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            chunks.append(model(batch).detach())
    predictions = torch.cat(chunks, dim=0).cpu().numpy()
    aligned_truth = [truth_rows[int(index)] for index in validation_indices]
    prediction_records = [
        {target: float(pred[target_index]) for target_index, target in enumerate(TARGETS)}
        for pred in predictions
    ]
    result = evaluate_oof(aligned_truth, prediction_records, target_weights=validation_weights)
    return float(result["overall_oof_wmae"]), result["target_mae"], predictions


def _write_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def train_fold(
    fold: int,
    rows: Sequence[Mapping[str, Any]],
    fold_ids: Sequence[int],
    graphs: Sequence[Data],
    augmented_graphs: Sequence[Data | None],
    all_fingerprints: np.ndarray,
    validation_weights: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    config: Mapping[str, Any],
    *,
    max_epochs: int,
    patience: int,
    batch_size: int,
    log_every: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any], np.ndarray]:
    train_indices = [index for index, assigned_fold in enumerate(fold_ids) if assigned_fold != fold]
    valid_indices = [index for index, assigned_fold in enumerate(fold_ids) if assigned_fold == fold]
    selected_bits = fit_fold_fp_indices(all_fingerprints, rows, train_indices)

    train_graphs: list[Data] = []
    for index in train_indices:
        train_graphs.append(_make_labeled_graph(graphs[index], rows[index]))
        if augmented_graphs[index] is not None:
            train_graphs.append(_make_labeled_graph(augmented_graphs[index], rows[index]))
    validation_graphs = [_make_labeled_graph(graphs[index], rows[index]) for index in valid_indices]
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(validation_graphs, batch_size=batch_size * 2, shuffle=False, num_workers=0)

    seed = PUBLIC_INIT_SEED + fold
    _set_seed(seed)
    model = PolymerGNNV12Res(
        selected_bits,
        hidden_dim=int(config["architecture"]["hidden_dim"]),
        num_layers=int(config["architecture"]["layers"]),
        heads=int(config["architecture"]["attention_heads"]),
        dropout=float(config["architecture"]["attention_dropout"]),
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(config["training"]["learning_rate"])
    )
    fold_loss_weights = _loss_weight_map(rows, train_indices)
    loss_fn = MaskedWeightedMAELoss(fold_loss_weights, device)

    history: list[dict[str, Any]] = []
    best_score = math.inf
    best_epoch = 0
    patience_used = 0
    checkpoint_path = output_dir / "checkpoints" / f"fold_{fold}" / "best.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "training_history" / f"fold_{fold}.csv"
    fold_start = time.monotonic()

    for epoch in range(1, max_epochs + 1):
        model.train()
        loss_sum = torch.zeros((), dtype=torch.float32, device=device)
        graph_count = 0
        for batch_index, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = loss_fn(prediction, batch.y)
            # Inputs/labels are prevalidated and missing labels are masked by
            # the loss. Avoid a scalar device-to-host sync on every batch.
            loss.backward()
            optimizer.step()
            batch_graphs = int(batch.num_graphs)
            loss_sum = loss_sum + loss.detach() * batch_graphs
            graph_count += batch_graphs
            if (batch_index + 1) % 50 == 0 or batch_index + 1 == len(train_loader):
                print(
                    f"fold={fold} epoch={epoch:03d} "
                    f"train_batch={batch_index + 1}/{len(train_loader)}",
                    flush=True,
                )
        train_loss = float((loss_sum / max(graph_count, 1)).detach().cpu())
        if not math.isfinite(train_loss):
            raise FloatingPointError(f"Non-finite training loss in fold {fold}, epoch {epoch}.")
        validation_score, target_mae, _ = _score_validation(
            model,
            valid_loader,
            device,
            rows,
            valid_indices,
            validation_weights,
        )
        record: dict[str, Any] = {
            "fold": fold,
            "epoch": epoch,
            "train_loss": train_loss,
            "validation_wmae": validation_score,
            **{f"{target}_mae": target_mae[target] for target in TARGETS},
        }
        history.append(record)
        _write_csv(history_path, list(record), history)

        if validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch
            patience_used = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "selected_morgan_bits": selected_bits,
                    "fold": fold,
                    "best_epoch": best_epoch,
                    "validation_wmae": best_score,
                    "seed": seed,
                    "config": dict(config),
                    "git_commit": subprocess.check_output(
                        ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
                    ).strip(),
                },
                checkpoint_path,
            )
        else:
            patience_used += 1

        if epoch == 1 or epoch % log_every == 0 or patience_used == 0:
            print(
                f"fold={fold} epoch={epoch:03d} train_loss={train_loss:.6f} "
                f"val_wMAE={validation_score:.6f} best={best_score:.6f}@{best_epoch} "
                f"patience={patience_used}/{patience}",
                flush=True,
            )
        if patience_used >= patience:
            break

    if best_epoch == 0:
        raise RuntimeError(f"Fold {fold} failed to produce a finite validation checkpoint.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    best_model = PolymerGNNV12Res(
        checkpoint["selected_morgan_bits"],
        hidden_dim=int(config["architecture"]["hidden_dim"]),
        num_layers=int(config["architecture"]["layers"]),
        heads=int(config["architecture"]["attention_heads"]),
        dropout=float(config["architecture"]["attention_dropout"]),
    ).to(device)
    best_model.load_state_dict(checkpoint["state_dict"])
    best_validation_score, best_target_mae, valid_predictions = _score_validation(
        best_model,
        valid_loader,
        device,
        rows,
        valid_indices,
        validation_weights,
    )
    summary = {
        "fold": fold,
        "sample_count": len(valid_indices),
        "train_original_sample_count": len(train_indices),
        "train_graph_count_with_augmentation": len(train_graphs),
        "best_epoch": best_epoch,
        "best_validation_wmae": best_validation_score,
        **{f"{target}_mae": best_target_mae[target] for target in TARGETS},
        "seed": seed,
        "duration_seconds": time.monotonic() - fold_start,
        "checkpoint_path": str(checkpoint_path),
        "selected_morgan_bits": selected_bits,
        "fold_train_loss_weights": fold_loss_weights,
    }
    return history, summary, valid_predictions


def _build_graphs(
    rows: Sequence[Mapping[str, Any]],
    cache_path: Path,
    source_sha256: str,
) -> tuple[list[Data], list[Data | None], list[dict[str, Any]]]:
    import rdkit

    cache_meta = {
        "cache_version": 1,
        "source_train_sha256": source_sha256,
        "rdkit_version": rdkit.__version__,
        "graph_schema": "public_gatv2_node7_edge2_morgan1024_repeat3_v1",
    }
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if saved.get("metadata") == cache_meta and len(saved.get("graphs", [])) == len(rows):
            print(f"Reusing local preprocessing cache: {cache_path}", flush=True)
            return saved["graphs"], saved["augmented_graphs"], saved["issues"]
        print("Ignoring stale preprocessing cache and rebuilding graphs.", flush=True)

    base_graphs: list[Data] = []
    augmented_graphs: list[Data | None] = []
    issues: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        if index and index % 1000 == 0:
            print(f"preprocessing graph {index}/{len(rows)}…", flush=True)
        smiles = str(row["SMILES"])
        base_graphs.append(process_smiles(smiles))
        augmented = augment_repeat_units(smiles, repeats=3)
        if augmented == smiles:
            augmented_graphs.append(None)
            continue
        try:
            augmented_graphs.append(process_smiles(augmented))
        except Exception as exc:
            augmented_graphs.append(None)
            issues.append(
                {
                    "sample_id": row["sample_id"],
                    "row_index": index,
                    "issue": "repeat_augmentation_graph_failed; original sample retained",
                    "error": str(exc),
                }
            )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": cache_meta,
            "graphs": base_graphs,
            "augmented_graphs": augmented_graphs,
            "issues": issues,
        },
        cache_path,
    )
    return base_graphs, augmented_graphs, issues


def run_formal(args: argparse.Namespace) -> dict[str, Any]:
    run_start = time.monotonic()
    run_started_at = _timestamp()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows, validation_weights, fold_ids = _load_frozen_inputs(args.train_csv, args.folds_csv)
    fold_sizes = {fold: fold_ids.count(fold) for fold in range(5)}
    if (
        set(fold_ids) != set(range(5))
        or any(size <= 0 for size in fold_sizes.values())
        or sum(fold_sizes.values()) != len(rows)
    ):
        raise ValueError("Frozen folds do not cover exactly folds 0 through 4.")

    runtime_config = json.loads(json.dumps(DEFAULT_CONFIG))
    runtime_config["training"]["epochs_max"] = args.epochs
    runtime_config["training"]["batch_size"] = args.batch_size
    runtime_config["training"]["early_stopping"] = (
        f"patience={args.patience}, minimum improvement=0, restore best checkpoint"
    )
    device = _device(args.device)
    runtime_config["runtime"] = {
        "device": str(device),
        "device_name": (
            torch.cuda.get_device_name(device) if device.type == "cuda" else str(device)
        ),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": __import__("torch_geometric").__version__,
        "rdkit": __import__("rdkit").__version__,
        "pandas": __import__("pandas").__version__,
        "fold_sizes": {str(k): v for k, v in fold_sizes.items()},
        "started_at_utc": run_started_at,
    }
    write_json(output_dir / "config.json", runtime_config)
    print("Building original and train-only repeat-unit graphs…", flush=True)
    cache_path = output_dir / "cache" / "graph_views_v1.pt"
    graphs, augmented, preprocessing_issues = _build_graphs(
        rows, cache_path, runtime_config["source_train_sha256"]
    )
    fingerprints = np.stack([graph.morgan_fp.squeeze(0).numpy() for graph in graphs])
    runtime_config["runtime"]["samples_with_repeat_augmentation"] = sum(
        item is not None for item in augmented
    )
    runtime_config["runtime"]["repeat_augmentation_failures"] = len(preprocessing_issues)
    write_json(output_dir / "config.json", runtime_config)
    write_json(output_dir / "preprocessing_diagnostics.json", {
        "source_sample_count": len(rows),
        "original_graph_count": len(graphs),
        "repeat_augmented_sample_count": runtime_config["runtime"]["samples_with_repeat_augmentation"],
        "repeat_augmentation_failure_count": len(preprocessing_issues),
        "repeat_augmentation_failures": preprocessing_issues,
        "source_rows_dropped": 0,
    })

    out_of_fold: list[dict[str, Any] | None] = [None] * len(rows)
    fold_summaries: list[dict[str, Any]] = []
    for fold in range(5):
        print(f"Starting frozen fold {fold}/4 on {args.device}…", flush=True)
        _, summary, predictions = train_fold(
            fold,
            rows,
            fold_ids,
            graphs,
            augmented,
            fingerprints,
            validation_weights,
            output_dir,
            device,
            runtime_config,
            max_epochs=args.epochs,
            patience=args.patience,
            batch_size=args.batch_size,
            log_every=args.log_every,
        )
        validation_indices = [i for i, fold_id in enumerate(fold_ids) if fold_id == fold]
        for local_index, sample_index in enumerate(validation_indices):
            out_of_fold[sample_index] = {
                "sample_id": rows[sample_index]["sample_id"],
                "fold": fold,
                **{
                    target: float(predictions[local_index, target_index])
                    for target_index, target in enumerate(TARGETS)
                },
            }
        # Per-fold truth/OOF scores are already produced by the Stage 0 metric.
        fold_summaries.append(summary)
        print(
            f"Finished fold {fold}: best_epoch={summary['best_epoch']} "
            f"val_wMAE={summary['best_validation_wmae']:.6f} "
            f"elapsed={summary['duration_seconds'] / 60:.1f} min",
            flush=True,
        )

    if any(row is None for row in out_of_fold):
        raise RuntimeError("Not every training sample received exactly one OOF prediction.")
    prediction_rows = [row for row in out_of_fold if row is not None]
    oof_path = output_dir / "oof_predictions.csv"
    _write_csv(oof_path, ["sample_id", "fold", *TARGETS], prediction_rows)
    _write_csv(
        output_dir / "fold_metrics.csv",
        [
            "fold", "sample_count", "train_original_sample_count",
            "train_graph_count_with_augmentation", "best_epoch", "best_validation_wmae",
            *[f"{target}_mae" for target in TARGETS], "seed", "duration_seconds",
            "checkpoint_path",
        ],
        [
            {key: summary[key] for key in [
                "fold", "sample_count", "train_original_sample_count",
                "train_graph_count_with_augmentation", "best_epoch", "best_validation_wmae",
                *[f"{target}_mae" for target in TARGETS], "seed", "duration_seconds",
                "checkpoint_path",
            ]}
            for summary in fold_summaries
        ],
    )
    # Use the Stage 0 validator as the final authority (coverage, alignment, metrics).
    official_result = validate_and_score(args.train_csv, oof_path, output_dir / "metrics.json")
    run_metadata = {
        "experiment_id": runtime_config["experiment_id"],
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True
        ).strip(),
        "benchmark_version": runtime_config["benchmark_version"],
        "benchmark_fold_sha256": sha256_file(args.folds_csv),
        "source_train_sha256": sha256_file(args.train_csv),
        "fold_metrics": fold_summaries,
        "overall_oof_wmae": official_result["overall_oof_wmae"],
        "target_mae": official_result["target_mae"],
        "started_at_utc": runtime_config["runtime"]["started_at_utc"],
        "completed_at_utc": _timestamp(),
        "duration_seconds": time.monotonic() - run_start,
        "hardware": runtime_config["runtime"],
    }
    write_json(output_dir / "run_metadata.json", run_metadata)

    # Preserve existing experiment rows; add one formal line for this immutable run.
    results_path = TRACK_ROOT / "results.csv"
    new_result = {
        "experiment_id": runtime_config["experiment_id"],
        "model_name": "public third-place GATv2, from scratch",
        "category": "third_party_architecture_reference",
        "benchmark_version": runtime_config["benchmark_version"],
        "git_commit": run_metadata["git_commit"],
        "seed": PUBLIC_INIT_SEED,
        "oof_wmae": official_result["overall_oof_wmae"],
        **{f"{target.lower()}_mae": official_result["target_mae"][target] for target in TARGETS},
        "kaggle_private": "",
        "status": "formal_oof_reference",
        "notes": "Frozen Stage 0 folds; train-fold-only Morgan selection/augmentation; no calibrator or released-label adjustment.",
        "folds_sha256": sha256_file(args.folds_csv),
        "train_data_sha256": sha256_file(args.train_csv),
        "run_metadata": "baselines/third_party_gatv2/artifacts/run_metadata.json",
    }
    if results_path.exists():
        with results_path.open("r", encoding="utf-8-sig", newline="") as source:
            reader = csv.DictReader(source)
            existing = list(reader)
            fields = list(reader.fieldnames or [])
    else:
        existing, fields = [], []
    for field in new_result:
        if field not in fields:
            fields.append(field)
    existing.append(new_result)
    _write_csv(results_path, fields, existing)
    print(
        f"OOF complete: wMAE={official_result['overall_oof_wmae']:.8f}; "
        + ", ".join(f"{target} MAE={official_result['target_mae'][target]:.8f}" for target in TARGETS),
        flush=True,
    )
    return run_metadata


def run_smoke_test(args: argparse.Namespace) -> dict[str, Any]:
    """Exercise graph, masked loss, backward, tiny-set fit, and checkpoint round trip."""
    rows, _, _ = _load_frozen_inputs(args.train_csv, args.folds_csv)
    # Use real SMILES with a deliberately small mixed-missingness label matrix.
    smoke_smiles = ["[*]CC[*]", "[*]CO[*]", "[*]CCO[*]", "[*]CNC[*]"]
    graphs = [process_smiles(smiles) for smiles in smoke_smiles]
    dataset = []
    for index, graph in enumerate(graphs):
        item = Data(
            x=graph.x,
            edge_index=graph.edge_index,
            edge_attr=graph.edge_attr,
            morgan_fp=graph.morgan_fp,
            y=torch.tensor(
                [float(index + j + 1) if (index + j) % 4 else float("nan") for j in range(5)],
                dtype=torch.float32,
            ).unsqueeze(0),
        )
        dataset.append(item)
    loader = DataLoader(dataset, batch_size=len(dataset), shuffle=False)
    device = _device(args.device)
    _set_seed(PUBLIC_INIT_SEED)
    bits = {target: list(range(50)) for target in TARGETS}
    model = PolymerGNNV12Res(bits).to(device)
    smoke_truth = [
        {target: (index + task_idx + 1 if (index + task_idx) % 4 else None)
         for task_idx, target in enumerate(TARGETS)}
        for index in range(len(dataset))
    ]
    targets_for_loss = competition_weights(smoke_truth)
    loss_fn = MaskedWeightedMAELoss(targets_for_loss, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    batch = next(iter(loader)).to(device)
    mask = torch.isfinite(batch.y)
    if not mask.any() or mask.all():
        raise AssertionError("Smoke batch must contain both observed and missing labels.")
    initial_loss = float(loss_fn(model(batch), batch.y).detach().cpu())
    for _ in range(20):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(batch), batch.y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(batch).detach().cpu()
    checkpoint_path = args.output_dir.resolve() / "smoke_test" / "smoke.pt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "selected_morgan_bits": bits}, checkpoint_path)
    restored = PolymerGNNV12Res(bits).to(device)
    restored.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=False)["state_dict"]
    )
    restored.eval()
    with torch.no_grad():
        actual = restored(batch).detach().cpu()
    max_checkpoint_delta = float(torch.max(torch.abs(expected - actual)))
    final_loss = float(loss_fn(actual.to(device), batch.y).detach().cpu())
    result = {
        "status": "passed",
        "device": str(device),
        "graph_count": len(dataset),
        "node_count": int(batch.x.size(0)),
        "edge_count": int(batch.edge_index.size(1)),
        "observed_label_count": int(mask.sum().item()),
        "missing_label_count": int((~mask).sum().item()),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "checkpoint_max_prediction_delta": max_checkpoint_delta,
        "checkpoint": str(checkpoint_path),
        "checkpoints_loaded_only_from_smoke_run": True,
    }
    if not final_loss < initial_loss:
        raise AssertionError(f"Smoke loss did not fall: {initial_loss} -> {final_loss}")
    if max_checkpoint_delta > 1e-6:
        raise AssertionError(f"Checkpoint predictions changed by {max_checkpoint_delta}")
    write_json(checkpoint_path.parent / "smoke_test.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv", type=Path,
        default=Path(os.environ.get("POLYMER_TRAIN_CSV", "data/competition_raw/train.csv")),
        help="Path to the Stage 0 official train.csv (must match its recorded SHA-256).",
    )
    parser.add_argument(
        "--folds-csv", type=Path, default=TRACK_ROOT / "benchmark" / "folds.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path(__file__).resolve().parent / "artifacts",
    )
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "mps", "cuda"])
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=10)
    args = parser.parse_args()
    args.train_csv = args.train_csv.resolve()
    args.folds_csv = args.folds_csv.resolve()
    if args.epochs < 1 or args.patience < 1 or args.batch_size < 1:
        parser.error("epochs, patience, and batch-size must be positive")
    return args


def main() -> None:
    args = parse_args()
    if args.smoke_test:
        run_smoke_test(args)
    else:
        run_formal(args)


if __name__ == "__main__":
    main()
