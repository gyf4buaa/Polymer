"""Train Own-GNN v0 on the frozen Stage 0 five-fold benchmark."""
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
import threading
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

TRACK_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = TRACK_ROOT.parent
MODEL_ROOT = Path(__file__).resolve().parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from from_scratch_gnn.scripts.evaluate_oof import validate_and_score  # noqa: E402
from from_scratch_gnn.src.data import (  # noqa: E402
    TARGETS,
    load_training_data,
    sha256_file,
    write_csv,
    write_json,
)
from from_scratch_gnn.src.metrics import evaluate_oof  # noqa: E402
from from_scratch_gnn.models.own_gnn_v0.graph import (  # noqa: E402
    GRAPH_SCHEMA,
    GraphBuildError,
    build_polymer_graph,
)
from from_scratch_gnn.models.own_gnn_v0.model import (  # noqa: E402
    OwnGNNv0,
    masked_huber_loss,
)


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested, but PyTorch CUDA is unavailable.")
        return torch.device("cuda")
    if requested == "mps":
        if not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested, but PyTorch MPS is unavailable.")
        return torch.device("mps")
    if requested == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _require_clean_source() -> None:
    status = _git_output("status", "--porcelain", "--untracked-files=all")
    # The official training CSV lives outside Git under data/ by design.
    # Allow that input directory while requiring all executable source to be
    # committed and clean before a formal run.
    unexpected = [
        line for line in status.splitlines()
        if not line.startswith("?? data/")
    ]
    if unexpected:
        unexpected_text = "\n".join(unexpected)
        raise RuntimeError(
            "Formal training requires committed source and a clean working tree; "
            f"git status reported:\n{unexpected_text}"
        )


def _load_config() -> dict[str, Any]:
    return json.loads((MODEL_ROOT / "config.json").read_text(encoding="utf-8"))


def _effective_config(args: argparse.Namespace) -> dict[str, Any]:
    """Return the checked-in config with only an explicit seed override applied."""
    config = json.loads(json.dumps(_load_config()))
    if args.seed is not None:
        config["seed"] = int(args.seed)
    return config


def _default_output_dir(model_root: Path, seed: int, configured_seed: int = 42) -> Path:
    """Keep the legacy default for seed 42 and isolate each added seed."""
    artifact_name = "production" if seed == configured_seed else f"paired_seed_{seed}"
    return model_root / "artifacts" / artifact_name


def _config_sha256(config: Mapping[str, Any]) -> str:
    serialized = json.dumps(
        config, indent=2, ensure_ascii=False, allow_nan=False
    ) + "\n"
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _refuse_cross_seed_artifact_reuse(output_dir: Path, seed: int) -> None:
    config_path = output_dir / "config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        existing_seed = existing.get("seed")
        if existing_seed is not None and int(existing_seed) != seed:
            raise RuntimeError(
                f"Artifact directory {output_dir} already belongs to seed "
                f"{existing_seed}; refusing to overwrite it with seed {seed}."
            )


def _load_frozen_inputs(
    train_csv: Path, folds_csv: Path
) -> tuple[list[dict[str, Any]], list[int], dict[str, Any]]:
    manifest = json.loads(
        (TRACK_ROOT / "benchmark" / "data_manifest.json").read_text(encoding="utf-8")
    )
    actual_train_sha = sha256_file(train_csv)
    expected_train_sha = manifest["source"]["sha256"]
    if actual_train_sha != expected_train_sha:
        raise ValueError(
            f"Training-data SHA256 mismatch: expected {expected_train_sha}, "
            f"got {actual_train_sha}"
        )
    actual_folds_sha = sha256_file(folds_csv)
    expected_folds_sha = manifest["folds"]["sha256"]
    if actual_folds_sha != expected_folds_sha:
        raise ValueError(
            f"Frozen-fold SHA256 mismatch: expected {expected_folds_sha}, "
            f"got {actual_folds_sha}"
        )

    rows = load_training_data(train_csv)
    if len(rows) != int(manifest["sample_count"]):
        raise ValueError(
            f"Expected {manifest['sample_count']} training rows, got {len(rows)}"
        )
    folds_by_id: dict[str, int] = {}
    smiles_by_id: dict[str, str] = {}
    with folds_csv.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"sample_id", "fold", "SMILES"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Fold file must contain {sorted(required)}")
        for row_number, raw in enumerate(reader, start=2):
            sample_id = str(raw["sample_id"])
            if sample_id in folds_by_id:
                raise ValueError(f"Duplicate sample_id in folds.csv: {sample_id}")
            fold = int(raw["fold"])
            if fold not in range(5):
                raise ValueError(f"Invalid fold {fold} at folds.csv row {row_number}")
            folds_by_id[sample_id] = fold
            smiles_by_id[sample_id] = str(raw["SMILES"])

    row_ids = [str(row["sample_id"]) for row in rows]
    if len(set(row_ids)) != len(row_ids) or set(folds_by_id) != set(row_ids):
        raise ValueError("folds.csv sample IDs do not exactly cover training data")
    for row in rows:
        sample_id = str(row["sample_id"])
        if smiles_by_id[sample_id] != str(row["SMILES"]):
            raise ValueError(f"Frozen fold SMILES differs from train.csv for {sample_id}")
    fold_ids = [folds_by_id[sample_id] for sample_id in row_ids]
    expected_sizes = {str(key): int(value) for key, value in manifest["folds"]["fold_sizes"].items()}
    actual_sizes = {str(fold): fold_ids.count(fold) for fold in range(5)}
    if actual_sizes != expected_sizes:
        raise ValueError(f"Frozen-fold counts differ from manifest: {actual_sizes}")
    return rows, fold_ids, manifest


def _build_graphs(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_path: Path,
    source_sha256: str,
    diagnostics_path: Path,
) -> tuple[list[Data], dict[str, Any]]:
    import rdkit

    cache_metadata = {
        "cache_schema": GRAPH_SCHEMA,
        "source_train_sha256": source_sha256,
        "rdkit_version": rdkit.__version__,
    }
    if cache_path.exists():
        saved = torch.load(cache_path, map_location="cpu", weights_only=False)
        if (
            saved.get("metadata") == cache_metadata
            and len(saved.get("graphs", [])) == len(rows)
            and len(saved.get("graph_diagnostics", [])) == len(rows)
        ):
            print(f"Reusing graph cache: {cache_path}", flush=True)
            write_json(diagnostics_path, saved["diagnostics"])
            return saved["graphs"], saved["diagnostics"]

    graphs: list[Data] = []
    graph_diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    star_count = Counter()
    fallback_reasons = Counter()
    closed_count = 0
    already_bonded_count = 0
    for index, row in enumerate(rows):
        if index and index % 1000 == 0:
            print(f"building polymer graphs {index}/{len(rows)}", flush=True)
        sample_id = str(row["sample_id"])
        try:
            graph, info = build_polymer_graph(str(row["SMILES"]), sample_id=sample_id)
        except GraphBuildError as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            continue
        graphs.append(graph)
        row_info = info.as_dict()
        graph_diagnostics.append(row_info)
        star_count[str(info.dummy_atom_count)] += 1
        closed_count += int(info.endpoint_closure_applied)
        already_bonded_count += int(info.endpoint_neighbors_already_bonded)
        if info.fallback_reason is not None:
            fallback_reasons[info.fallback_reason] += 1

    diagnostics = {
        "graph_schema": GRAPH_SCHEMA,
        "source_sha256": source_sha256,
        "source_sample_count": len(rows),
        "graph_count": len(graphs),
        "source_rows_dropped": 0,
        "rdkit_parse_failures": failures,
        "dummy_atom_count_distribution": dict(sorted(star_count.items(), key=lambda item: int(item[0]))),
        "endpoint_closure_count": closed_count,
        "endpoint_neighbors_already_bonded_count": already_bonded_count,
        "fallback_count": sum(fallback_reasons.values()),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "fallback_samples": [
            {
                "sample_id": info["sample_id"],
                "dummy_atom_count": info["dummy_atom_count"],
                "reason": info["fallback_reason"],
            }
            for info in graph_diagnostics
            if info["fallback_reason"] is not None
        ],
    }
    write_json(diagnostics_path, diagnostics)
    if failures:
        raise RuntimeError(
            f"Could not build graphs for {len(failures)} source samples. "
            "All source rows are retained; see graph_diagnostics.json."
        )
    if len(graphs) != len(rows):
        raise RuntimeError("Graph preparation did not preserve all source rows")

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "metadata": cache_metadata,
            "graphs": graphs,
            "graph_diagnostics": graph_diagnostics,
            "diagnostics": diagnostics,
        },
        cache_path,
    )
    return graphs, diagnostics


def _target_normalizer(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> dict[str, dict[str, float]]:
    result: dict[str, dict[str, float]] = {}
    for target in TARGETS:
        values = [float(rows[index][target]) for index in indices if rows[index][target] is not None]
        if not values:
            raise ValueError(f"Training split contains no observed {target} labels")
        mean = float(np.mean(values))
        std = float(np.std(values, ddof=0))
        if not math.isfinite(std) or std < 1e-12:
            std = 1.0
        result[target] = {"mean": mean, "std": std}
    return result


def _normalized_targets(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int], stats: Mapping[str, Mapping[str, float]]
) -> list[list[float]]:
    result: list[list[float]] = []
    for index in indices:
        values = []
        for target in TARGETS:
            raw = rows[index][target]
            if raw is None:
                values.append(float("nan"))
            else:
                values.append((float(raw) - stats[target]["mean"]) / stats[target]["std"])
        result.append(values)
    return result


def _make_dataset(
    graphs: Sequence[Data],
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    stats: Mapping[str, Mapping[str, float]],
) -> list[Data]:
    normalized = _normalized_targets(rows, indices, stats)
    dataset: list[Data] = []
    for index, target_values in zip(indices, normalized):
        item = graphs[index].clone()
        item.y = torch.tensor(target_values, dtype=torch.float32).unsqueeze(0)
        item.sample_id = str(rows[index]["sample_id"])
        dataset.append(item)
    return dataset


class _GpuSampler:
    """Low-overhead nvidia-smi sampling for recorded utilization and memory."""

    def __init__(self, interval_seconds: float = 15.0) -> None:
        self.interval_seconds = interval_seconds
        self.values: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self) -> None:
        try:
            output = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.used",
                    "--format=csv,noheader,nounits",
                ],
                text=True,
                stderr=subprocess.DEVNULL,
                timeout=5,
            ).splitlines()[0]
            utilization, memory = (float(value.strip()) for value in output.split(",", 1))
            self.values.append({"utilization_percent": utilization, "memory_used_mb": memory})
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            return

    def _run(self) -> None:
        self._sample()
        while not self._stop.wait(self.interval_seconds):
            self._sample()

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=6)

    def summary(self, start_index: int = 0) -> dict[str, float | int | None]:
        samples = self.values[start_index:]
        utilizations = [item["utilization_percent"] for item in samples]
        memories = [item["memory_used_mb"] for item in samples]
        return {
            "gpu_utilization_sample_count": len(samples),
            "gpu_utilization_mean_percent": float(np.mean(utilizations)) if utilizations else None,
            "gpu_utilization_max_percent": float(max(utilizations)) if utilizations else None,
            "nvidia_smi_peak_memory_used_mb": float(max(memories)) if memories else None,
        }


def _evaluate_model(
    model: OwnGNNv0,
    loader: DataLoader,
    rows: Sequence[Mapping[str, Any]],
    indices: Sequence[int],
    stats: Mapping[str, Mapping[str, float]],
    weights: Mapping[str, Any],
    device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    model.eval()
    normalized_predictions: list[list[float]] = []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            outputs = model(batch).detach().cpu().numpy()
            normalized_predictions.extend(outputs.tolist())
    predictions = np.asarray(normalized_predictions, dtype=np.float64)
    for task_index, target in enumerate(TARGETS):
        predictions[:, task_index] = (
            predictions[:, task_index] * stats[target]["std"] + stats[target]["mean"]
        )
    truth = [
        {target: rows[index][target] for target in TARGETS}
        for index in indices
    ]
    predicted = [
        {target: float(predictions[row_index, task_index]) for task_index, target in enumerate(TARGETS)}
        for row_index in range(len(indices))
    ]
    metrics = evaluate_oof(truth, predicted, target_weights=weights)
    return metrics, predictions


def _model_config(config: Mapping[str, Any]) -> dict[str, Any]:
    model_config = config["model"]
    if not isinstance(model_config, dict):
        raise ValueError("config.json model section must be an object")
    return {
        key: model_config[key]
        for key in ("hidden_dim", "num_layers", "dropout")
    }


def _train_fold(
    *,
    fold: int,
    rows: Sequence[Mapping[str, Any]],
    fold_ids: Sequence[int],
    graphs: Sequence[Data],
    manifest: Mapping[str, Any],
    output_dir: Path,
    device: torch.device,
    config: Mapping[str, Any],
    batch_size: int,
    max_epochs: int,
    patience: int,
    log_every: int,
    gpu_sampler: _GpuSampler | None,
) -> tuple[dict[str, Any], list[dict[str, float]], np.ndarray]:
    training = config["training"]
    if not isinstance(training, dict):
        raise ValueError("config.json training section must be an object")
    train_indices = [index for index, fold_id in enumerate(fold_ids) if fold_id != fold]
    valid_indices = [index for index, fold_id in enumerate(fold_ids) if fold_id == fold]
    stats = _target_normalizer(rows, train_indices)
    train_dataset = _make_dataset(graphs, rows, train_indices, stats)
    valid_dataset = _make_dataset(graphs, rows, valid_indices, stats)
    generator = torch.Generator()
    seed = int(config["seed"]) + fold
    _set_seed(seed)
    generator.manual_seed(seed)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, generator=generator
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)

    model = OwnGNNv0(**_model_config(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    model_root = _model_config(config)
    delta = float(training.get("huber_delta", 1.0))
    fixed_weights = manifest["target_statistics_and_weights"]
    checkpoint_dir = output_dir / "checkpoints"
    history_dir = output_dir / "training_history"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    history_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / f"fold_{fold}_best.pt"
    history_rows: list[dict[str, float]] = []
    best_score = math.inf
    best_epoch = 0
    best_metrics: dict[str, Any] | None = None
    epochs_without_improvement = 0
    fold_started = time.monotonic()
    utilization_start_index = len(gpu_sampler.values) if gpu_sampler else 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, max_epochs + 1):
        epoch_started = time.monotonic()
        model.train()
        losses = []
        train_started = time.monotonic()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            prediction = model(batch)
            loss = masked_huber_loss(prediction, batch.y, delta=delta)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Non-finite training loss in fold {fold}, epoch {epoch}")
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        train_seconds = time.monotonic() - train_started
        validation_metrics, _ = _evaluate_model(
            model, valid_loader, rows, valid_indices, stats, fixed_weights, device
        )
        epoch_seconds = time.monotonic() - epoch_started
        validation_score = float(validation_metrics["overall_oof_wmae"])
        if not math.isfinite(validation_score):
            raise RuntimeError(f"Non-finite validation wMAE in fold {fold}, epoch {epoch}")
        row = {
            "epoch": float(epoch),
            "train_loss": float(np.mean(losses)) if losses else float("nan"),
            "validation_wmae": validation_score,
            "validation_Tg_mae": float(validation_metrics["target_mae"]["Tg"] or 0.0),
            "validation_FFV_mae": float(validation_metrics["target_mae"]["FFV"] or 0.0),
            "validation_Tc_mae": float(validation_metrics["target_mae"]["Tc"] or 0.0),
            "validation_Density_mae": float(validation_metrics["target_mae"]["Density"] or 0.0),
            "validation_Rg_mae": float(validation_metrics["target_mae"]["Rg"] or 0.0),
            "epoch_seconds": epoch_seconds,
            "train_samples_per_second": len(train_indices) / train_seconds if train_seconds else 0.0,
        }
        history_rows.append(row)
        if validation_score < best_score:
            best_score = validation_score
            best_epoch = epoch
            best_metrics = validation_metrics
            epochs_without_improvement = 0
            torch.save(
                {
                    "state_dict": model.state_dict(),
                    "normalizer": stats,
                    "model_config": model_root,
                    "experiment_id": config["experiment_id"],
                    "benchmark_version": config["benchmark_version"],
                    "git_commit": _git_output("rev-parse", "HEAD"),
                    "fold": fold,
                    "seed": seed,
                    "epoch": epoch,
                    "best_validation_wmae": best_score,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1

        if epoch == 1 or epoch % log_every == 0:
            print(
                f"fold={fold} epoch={epoch}/{max_epochs} "
                f"train_loss={row['train_loss']:.5f} val_wMAE={validation_score:.6f} "
                f"best={best_score:.6f}@{best_epoch} "
                f"epoch_seconds={epoch_seconds:.1f}",
                flush=True,
            )
        if epochs_without_improvement >= patience:
            break

    if best_epoch == 0 or best_metrics is None:
        raise RuntimeError(f"Fold {fold} did not produce a valid checkpoint")
    _write_history(history_dir / f"fold_{fold}.csv", history_rows)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    final_metrics, validation_predictions = _evaluate_model(
        model, valid_loader, rows, valid_indices, stats, fixed_weights, device
    )
    fold_seconds = time.monotonic() - fold_started
    epochs_completed = len(history_rows)
    peak_allocated_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2))
        if device.type == "cuda"
        else None
    )
    peak_reserved_mb = (
        float(torch.cuda.max_memory_reserved(device) / (1024**2))
        if device.type == "cuda"
        else None
    )
    gpu_metrics = (
        gpu_sampler.summary(utilization_start_index)
        if gpu_sampler is not None
        else {
            "gpu_utilization_sample_count": 0,
            "gpu_utilization_mean_percent": None,
            "gpu_utilization_max_percent": None,
            "nvidia_smi_peak_memory_used_mb": None,
        }
    )
    summary: dict[str, Any] = {
        "fold": fold,
        "sample_count": len(valid_indices),
        "train_sample_count": len(train_indices),
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "best_validation_wmae": float(final_metrics["overall_oof_wmae"]),
        **{
            f"{target}_mae": float(final_metrics["target_mae"][target] or 0.0)
            for target in TARGETS
        },
        "seed": seed,
        "duration_seconds": fold_seconds,
        "seconds_per_epoch": fold_seconds / epochs_completed,
        "training_samples_per_second": len(train_indices) * epochs_completed / fold_seconds,
        "peak_vram_allocated_mb": peak_allocated_mb,
        "peak_vram_reserved_mb": peak_reserved_mb,
        **gpu_metrics,
        "checkpoint_path": str(checkpoint_path),
        "normalizer": stats,
    }
    write_json(output_dir / f"fold_{fold}_summary.json", summary)
    print(
        f"completed fold {fold}: best_epoch={best_epoch} "
        f"val_wMAE={summary['best_validation_wmae']:.6f} "
        f"runtime={fold_seconds / 60:.1f} min",
        flush=True,
    )
    prediction_rows = [
        {
            "sample_id": str(rows[index]["sample_id"]),
            "fold": fold,
            **{
                target: float(validation_predictions[row_index, target_index])
                for target_index, target in enumerate(TARGETS)
            },
        }
        for row_index, index in enumerate(valid_indices)
    ]
    return summary, prediction_rows, validation_predictions


def _write_history(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "epoch",
        "train_loss",
        "validation_wmae",
        "validation_Tg_mae",
        "validation_FFV_mae",
        "validation_Tc_mae",
        "validation_Density_mae",
        "validation_Rg_mae",
        "epoch_seconds",
        "train_samples_per_second",
    ]
    write_csv(path, rows, columns)


def _runtime_info(device: torch.device, fold_sizes: Mapping[str, int]) -> dict[str, Any]:
    import rdkit
    import torch_geometric

    return {
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else str(device),
        "processor": platform.processor() or platform.machine(),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torch_geometric": torch_geometric.__version__,
        "rdkit": rdkit.__version__,
        "numpy": np.__version__,
        "fold_sizes": dict(fold_sizes),
        "started_at_utc": _timestamp(),
    }


def _write_config_files(output_dir: Path, config: Mapping[str, Any]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    (output_dir / "config.json").write_text(serialized, encoding="utf-8")
    # JSON is valid YAML 1.2; writing the same document keeps the recorded
    # formal config dependency-free and byte-for-byte aligned with config.json.
    (output_dir / "config.yaml").write_text(serialized, encoding="utf-8")


def _source_manifest(
    commit: str,
    train_csv: Path,
    folds_csv: Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_config = config if config is not None else _load_config()
    tracked = _git_output("ls-files", "from_scratch_gnn/models/own_gnn_v0").splitlines()
    files = {}
    for relative in tracked:
        path = REPOSITORY_ROOT / relative
        if path.is_file():
            files[relative] = sha256_file(path)
    return {
        "git_commit": commit,
        "branch": _git_output("branch", "--show-current"),
        "seed": int(effective_config["seed"]),
        "representation": effective_config.get("graph", {}).get(
            "representation", "endpoint_closure"
        ),
        "graph_schema": effective_config.get("graph", {}).get("schema", GRAPH_SCHEMA),
        "source_config_sha256": sha256_file(MODEL_ROOT / "config.json"),
        "effective_config_sha256": _config_sha256(effective_config),
        "source_files_sha256": files,
        "benchmark_files_sha256": {
            "train_csv": sha256_file(train_csv),
            "folds_csv": sha256_file(folds_csv),
            "manifest_json": sha256_file(TRACK_ROOT / "benchmark" / "data_manifest.json"),
        },
    }


def _write_fold_metrics(path: Path, summaries: Sequence[Mapping[str, Any]]) -> None:
    columns = [
        "fold",
        "sample_count",
        "train_sample_count",
        "best_epoch",
        "epochs_completed",
        "best_validation_wmae",
        *[f"{target}_mae" for target in TARGETS],
        "seed",
        "duration_seconds",
        "seconds_per_epoch",
        "training_samples_per_second",
        "peak_vram_allocated_mb",
        "peak_vram_reserved_mb",
        "gpu_utilization_sample_count",
        "gpu_utilization_mean_percent",
        "gpu_utilization_max_percent",
        "nvidia_smi_peak_memory_used_mb",
        "checkpoint_path",
    ]
    write_csv(path, summaries, columns)


def _append_result(
    result_path: Path,
    *,
    config: Mapping[str, Any],
    commit: str,
    metrics: Mapping[str, Any],
    folds_sha256: str,
    train_sha256: str,
    run_metadata_path: Path,
) -> None:
    with result_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    new_row = {field: "" for field in fields}
    seed = int(config["seed"])
    values = {
        "experiment_id": (
            "own_gnn_v0_frozen_oof_v1"
            if seed == 42
            else f"own_gnn_v0_frozen_oof_v1_seed_{seed}"
        ),
        "model_name": "Own-GNN v0",
        "category": "own_model" if seed == 42 else "internal_baseline",
        "benchmark_version": config["benchmark_version"],
        "git_commit": commit,
        "seed": config["seed"],
        "oof_wmae": metrics["overall_oof_wmae"],
        **{f"{target.lower()}_mae": metrics["target_mae"][target] for target in TARGETS},
        "status": "formal_oof_model",
        "notes": "Random initialization; polymer endpoint closure; categorical atom/bond graph only; no Morgan, global RDKit descriptors, physics features, 3D, calibration, or released-label adjustment.",
        "folds_sha256": folds_sha256,
        "train_data_sha256": train_sha256,
        "run_metadata": str(run_metadata_path.relative_to(TRACK_ROOT)),
    }
    for key, value in values.items():
        if key in new_row:
            new_row[key] = value
    experiment_id = str(values["experiment_id"])
    if seed == 42:
        # Preserve the original seed-42 writer behavior and historical row.
        existing.append(new_row)
    else:
        matches = [
            index for index, row in enumerate(existing)
            if row.get("experiment_id") == experiment_id
        ]
        if len(matches) > 1:
            raise RuntimeError(f"results.csv contains duplicate experiment_id {experiment_id}")
        if matches:
            existing[matches[0]] = new_row
        else:
            existing.append(new_row)
    temp_path = result_path.with_suffix(result_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
    os.replace(temp_path, result_path)


def _run_formal(args: argparse.Namespace) -> dict[str, Any]:
    _require_clean_source()
    run_started = time.monotonic()
    config = _effective_config(args)
    config["training"]["epochs_max"] = args.epochs or int(config["training"]["epochs_max"])
    config["training"]["patience"] = args.patience or int(config["training"]["patience"])
    config["training"]["batch_size"] = args.batch_size or int(config["training"]["batch_size"])

    device = _device(args.device)
    train_csv = args.train_csv.resolve()
    folds_csv = args.folds_csv.resolve()
    rows, fold_ids, manifest = _load_frozen_inputs(train_csv, folds_csv)
    fold_sizes = {str(fold): fold_ids.count(fold) for fold in range(5)}
    config["runtime"] = _runtime_info(device, fold_sizes)
    output_dir = args.output_dir.resolve()
    _refuse_cross_seed_artifact_reuse(output_dir, int(config["seed"]))
    _write_config_files(output_dir, config)
    print("Building or loading all frozen training graphs…", flush=True)
    graphs, graph_diagnostics = _build_graphs(
        rows,
        cache_path=output_dir.parent / "cache" / "polymer_graphs_v1.pt",
        source_sha256=sha256_file(train_csv),
        diagnostics_path=output_dir / "graph_diagnostics.json",
    )
    run_started_at = config["runtime"]["started_at_utc"]
    gpu_sampler = _GpuSampler() if device.type == "cuda" else None
    if gpu_sampler is not None:
        gpu_sampler.start()

    oof_by_index: list[dict[str, Any] | None] = [None] * len(rows)
    summaries = []
    try:
        for fold in range(5):
            summary, prediction_rows, _ = _train_fold(
                fold=fold,
                rows=rows,
                fold_ids=fold_ids,
                graphs=graphs,
                manifest=manifest,
                output_dir=output_dir,
                device=device,
                config=config,
                batch_size=int(config["training"]["batch_size"]),
                max_epochs=int(config["training"]["epochs_max"]),
                patience=int(config["training"]["patience"]),
                log_every=args.log_every,
                gpu_sampler=gpu_sampler,
            )
            summaries.append(summary)
            row_by_id = {str(row["sample_id"]): index for index, row in enumerate(rows)}
            for prediction in prediction_rows:
                index = row_by_id[prediction["sample_id"]]
                oof_by_index[index] = prediction
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()

    if any(item is None for item in oof_by_index):
        raise RuntimeError("Not every Stage 0 sample received exactly one OOF prediction")
    oof_rows = [item for item in oof_by_index if item is not None]
    oof_path = output_dir / "oof_predictions.csv"
    write_csv(oof_path, oof_rows, ("sample_id", "fold", *TARGETS))
    official_metrics = validate_and_score(train_csv, oof_path, output_dir / "metrics.json")
    _write_fold_metrics(output_dir / "fold_metrics.csv", summaries)

    commit = _git_output("rev-parse", "HEAD")
    metadata = {
        "experiment_id": config["experiment_id"],
        "seed": int(config["seed"]),
        "model_name": config["model_name"],
        "git_commit": commit,
        "branch": _git_output("branch", "--show-current"),
        "benchmark_version": config["benchmark_version"],
        "benchmark_fold_sha256": sha256_file(folds_csv),
        "source_train_sha256": sha256_file(train_csv),
        "fold_metrics": summaries,
        "overall_oof_wmae": official_metrics["overall_oof_wmae"],
        "target_mae": official_metrics["target_mae"],
        "oof_validation": official_metrics["validation"],
        "graph_diagnostics_summary": {
            key: graph_diagnostics[key]
            for key in (
                "source_sample_count",
                "graph_count",
                "source_rows_dropped",
                "dummy_atom_count_distribution",
                "endpoint_closure_count",
                "endpoint_neighbors_already_bonded_count",
                "fallback_count",
                "fallback_reasons",
            )
        },
        "runtime": config["runtime"],
        "effective_config_sha256": sha256_file(output_dir / "config.json"),
        "source_config_sha256": sha256_file(MODEL_ROOT / "config.json"),
        "started_at_utc": run_started_at,
        "completed_at_utc": _timestamp(),
        "duration_seconds": time.monotonic() - run_started,
        "gpu_sampling": gpu_sampler.summary() if gpu_sampler is not None else None,
    }
    write_json(output_dir / "run_metadata.json", metadata)
    write_json(
        output_dir / "source_manifest.json",
        _source_manifest(commit, train_csv, folds_csv, config=config),
    )
    _append_result(
        TRACK_ROOT / "results.csv",
        config=config,
        commit=commit,
        metrics=official_metrics,
        folds_sha256=sha256_file(folds_csv),
        train_sha256=sha256_file(train_csv),
        run_metadata_path=output_dir / "run_metadata.json",
    )
    print(
        f"Stage 0 OOF validated: wMAE={official_metrics['overall_oof_wmae']:.8f}; "
        + ", ".join(
            f"{target} MAE={official_metrics['target_mae'][target]:.8f}"
            for target in TARGETS
        ),
        flush=True,
    )
    return metadata


def _tiny_sample_indices(rows: Sequence[Mapping[str, Any]], limit: int = 32) -> list[int]:
    """Greedily choose a small real-data sample that covers observed tasks."""
    remaining = set(range(len(rows)))
    selected: list[int] = []
    covered: set[str] = set()
    while remaining and len(selected) < limit:
        best_index = max(
            remaining,
            key=lambda index: (
                sum(rows[index][target] is not None and target not in covered for target in TARGETS),
                sum(rows[index][target] is not None for target in TARGETS),
                -index,
            ),
        )
        selected.append(best_index)
        remaining.remove(best_index)
        covered.update(target for target in TARGETS if rows[best_index][target] is not None)
        if covered == set(TARGETS) and len(selected) >= 16:
            break
    if len(selected) < 16:
        raise ValueError("Could not select at least 16 labeled smoke-test samples")
    return selected


def _run_tiny_overfit(args: argparse.Namespace) -> dict[str, Any]:
    config = _effective_config(args)
    device = _device(args.device)
    rows = load_training_data(args.train_csv.resolve())
    if sha256_file(args.train_csv.resolve()) != json.loads(
        (TRACK_ROOT / "benchmark" / "data_manifest.json").read_text(encoding="utf-8")
    )["source"]["sha256"]:
        raise ValueError("Tiny overfit input does not match the frozen training SHA256")
    indices = _tiny_sample_indices(rows, limit=32)
    selected_rows = [rows[index] for index in indices]
    selected_ids = {str(row["sample_id"]) for row in selected_rows}
    stats = _target_normalizer(rows, indices)
    import rdkit

    graphs = []
    graph_info = []
    for row in selected_rows:
        graph, info = build_polymer_graph(str(row["SMILES"]), sample_id=str(row["sample_id"]))
        graphs.append(graph)
        graph_info.append(info.as_dict())
    dataset = _make_dataset(graphs, selected_rows, list(range(len(selected_rows))), stats)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False))).to(device)

    _set_seed(int(config["seed"]))
    model = OwnGNNv0(**_model_config(config)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    model.eval()
    with torch.no_grad():
        initial = float(masked_huber_loss(model(batch), batch.y).cpu())
    model.train()
    steps = args.tiny_steps
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_huber_loss(model(batch), batch.y)
        if not torch.isfinite(loss):
            raise RuntimeError("Tiny overfit produced a non-finite loss")
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        predictions = model(batch).detach().cpu()
        final = float(masked_huber_loss(predictions, batch.y.cpu()).cpu())

    output_dir = args.output_dir.resolve() / "tiny_overfit"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "tiny_overfit.pt"
    torch.save({"state_dict": model.state_dict(), "normalizer": stats}, checkpoint_path)
    restored = OwnGNNv0(**_model_config(config)).to(device)
    restored.load_state_dict(
        torch.load(checkpoint_path, map_location=device, weights_only=True)["state_dict"]
    )
    restored.eval()
    with torch.no_grad():
        restored_prediction = restored(batch).detach().cpu()
    checkpoint_delta = float(torch.max(torch.abs(predictions - restored_prediction)))
    observed_labels = int(torch.isfinite(batch.y).sum().item())
    fallback_count = sum(info["fallback_reason"] is not None for info in graph_info)
    result = {
        "status": "passed",
        "device": str(device),
        "sample_count": len(selected_rows),
        "sample_ids": [str(row["sample_id"]) for row in selected_rows],
        "observed_label_count": observed_labels,
        "missing_label_count": int(batch.y.numel() - observed_labels),
        "graph_fallback_count": fallback_count,
        "steps": steps,
        "initial_masked_huber_loss": initial,
        "final_masked_huber_loss": final,
        "relative_loss_reduction": 1.0 - final / initial if initial else None,
        "checkpoint_max_prediction_delta": checkpoint_delta,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_round_trip": checkpoint_delta <= 1e-6,
        "source_train_sha256": sha256_file(args.train_csv.resolve()),
        "selected_ids_are_from_frozen_training_data": selected_ids.issubset(
            {str(row["sample_id"]) for row in rows}
        ),
    }
    if not final < initial * 0.75:
        raise AssertionError(
            f"Tiny-set loss did not fall by at least 25%: {initial:.6f} -> {final:.6f}"
        )
    if checkpoint_delta > 1e-6:
        raise AssertionError(f"Checkpoint round-trip changed predictions by {checkpoint_delta}")
    write_json(output_dir / "tiny_overfit.json", result)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)
    return result


def _run_smoke_fold(args: argparse.Namespace) -> dict[str, Any]:
    config = _effective_config(args)
    device = _device(args.device)
    rows, fold_ids, manifest = _load_frozen_inputs(args.train_csv.resolve(), args.folds_csv.resolve())
    fold = int(args.smoke_fold)
    if fold not in range(5):
        raise ValueError("--smoke-fold must be between 0 and 4")
    output_dir = args.output_dir.resolve() / f"gpu_smoke_fold_{fold}"
    fold_config = json.loads(json.dumps(config))
    fold_config["training"]["epochs_max"] = args.epochs or 5
    fold_config["training"]["patience"] = max(1, args.patience or 5)
    fold_config["training"]["batch_size"] = args.batch_size or 64
    fold_sizes = {str(index): fold_ids.count(index) for index in range(5)}
    fold_config["runtime"] = _runtime_info(device, fold_sizes)
    fold_config["runtime"]["smoke_fold_only"] = True
    _refuse_cross_seed_artifact_reuse(output_dir, int(fold_config["seed"]))
    _write_config_files(output_dir, fold_config)
    graphs, graph_diagnostics = _build_graphs(
        rows,
        cache_path=args.output_dir.resolve().parent / "cache" / "polymer_graphs_v1.pt",
        source_sha256=sha256_file(args.train_csv.resolve()),
        diagnostics_path=output_dir / "graph_diagnostics.json",
    )
    gpu_sampler = _GpuSampler() if device.type == "cuda" else None
    if gpu_sampler is not None:
        gpu_sampler.start()
    start = time.monotonic()
    try:
        summary, _, _ = _train_fold(
            fold=fold,
            rows=rows,
            fold_ids=fold_ids,
            graphs=graphs,
            manifest=manifest,
            output_dir=output_dir,
            device=device,
            config=fold_config,
            batch_size=int(fold_config["training"]["batch_size"]),
            max_epochs=int(fold_config["training"]["epochs_max"]),
            patience=int(fold_config["training"]["patience"]),
            log_every=args.log_every,
            gpu_sampler=gpu_sampler,
        )
    finally:
        if gpu_sampler is not None:
            gpu_sampler.stop()
    metadata = {
        "status": "passed",
        "git_commit": _git_output("rev-parse", "HEAD"),
        "seed": int(fold_config["seed"]),
        "effective_config_sha256": sha256_file(output_dir / "config.json"),
        "device": str(device),
        "fold": fold,
        "requested_epochs": int(fold_config["training"]["epochs_max"]),
        "batch_size": int(fold_config["training"]["batch_size"]),
        "duration_seconds": time.monotonic() - start,
        "fold_summary": summary,
        "graph_diagnostics_summary": {
            key: graph_diagnostics[key]
            for key in (
                "source_sample_count",
                "endpoint_closure_count",
                "endpoint_neighbors_already_bonded_count",
                "fallback_count",
                "fallback_reasons",
            )
        },
        "gpu_sampling": gpu_sampler.summary() if gpu_sampler else None,
    }
    write_json(output_dir / "smoke_fold_metadata.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False), flush=True)
    return metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--train-csv",
        type=Path,
        default=Path(os.environ.get("POLYMER_TRAIN_CSV", "data/competition_raw/train.csv")),
        help="Path to Stage 0 train.csv (the recorded SHA256 is enforced).",
    )
    parser.add_argument(
        "--folds-csv", type=Path, default=TRACK_ROOT / "benchmark" / "folds.csv"
    )
    parser.add_argument(
        "--output-dir", type=Path,
        help="Artifact directory (defaults to a seed-specific path).",
    )
    parser.add_argument(
        "--seed", type=int,
        help="Override only the configured random seed; default preserves seed 42.",
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "mps", "cuda"))
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--patience", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--tiny-overfit", action="store_true")
    parser.add_argument("--tiny-steps", type=int, default=120)
    parser.add_argument("--smoke-fold", type=int, choices=range(5))
    args = parser.parse_args()
    if args.epochs is not None and args.epochs < 1:
        parser.error("--epochs must be positive")
    if args.patience is not None and args.patience < 1:
        parser.error("--patience must be positive")
    if args.batch_size is not None and args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.seed is not None and args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.log_every < 1 or args.tiny_steps < 1:
        parser.error("--log-every and --tiny-steps must be positive")
    if args.tiny_overfit and args.smoke_fold is not None:
        parser.error("Choose either --tiny-overfit or --smoke-fold")
    args.train_csv = args.train_csv.resolve()
    args.folds_csv = args.folds_csv.resolve()
    if args.output_dir is None:
        configured_seed = int(_load_config()["seed"])
        effective_seed = configured_seed if args.seed is None else args.seed
        args.output_dir = _default_output_dir(MODEL_ROOT, effective_seed, configured_seed)
    args.output_dir = args.output_dir.resolve()
    return args


def main() -> None:
    args = parse_args()
    if args.tiny_overfit:
        _run_tiny_overfit(args)
    elif args.smoke_fold is not None:
        _run_smoke_fold(args)
    else:
        _run_formal(args)


if __name__ == "__main__":
    main()
