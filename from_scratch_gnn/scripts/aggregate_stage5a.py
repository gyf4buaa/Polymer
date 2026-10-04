"""Aggregate Stage 5A outputs and update the registry in one process."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..models.capacity_scaling.protocol import (
    EXPERIMENT_ROOT,
    FOLDS_SHA256,
    MODEL_ROOT,
    NEW_WIDTHS,
    SEEDS,
    TRACK_ROOT,
    TRAIN_SHA256,
    WIDTHS,
    historical_c256_artifact_dir,
)
from ..src.data import write_json

TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _mean_sd(values: Iterable[float]) -> tuple[float, float]:
    numbers = [float(value) for value in values]
    if not numbers:
        raise ValueError("Cannot summarize an empty sequence")
    return statistics.fmean(numbers), statistics.stdev(numbers) if len(numbers) > 1 else 0.0


def _find_successful_formal_dir(width: int, seed: int) -> Path:
    seed_root = MODEL_ROOT / "artifacts" / "formal" / f"C{width}" / f"seed_{seed}"
    valid = []
    if seed_root.exists():
        for candidate in sorted(seed_root.glob("attempt_*")):
            required = ("metrics.json", "run_metadata.json", "registry_row.json", "config.json")
            if all((candidate / filename).is_file() for filename in required):
                valid.append(candidate)
    if len(valid) != 1:
        raise FileNotFoundError(
            f"Expected exactly one complete formal artifact for C{width}/seed_{seed}; "
            f"found {len(valid)} under {seed_root}"
        )
    return valid[0]


def _validate_metric_artifact(directory: Path, *, seed: int) -> dict[str, Any]:
    metrics = _read_json(directory / "metrics.json")
    metadata = _read_json(directory / "run_metadata.json")
    config = _read_json(directory / "config.json")
    if metrics.get("n_samples") != 7973:
        raise ValueError(f"Unexpected sample count in {directory}")
    if metrics.get("validation", {}).get("truth_source_sha256") != TRAIN_SHA256:
        raise ValueError(f"Training hash mismatch in {directory}")
    if metadata.get("benchmark_fold_sha256") != FOLDS_SHA256:
        raise ValueError(f"Fold hash mismatch in {directory}")
    if metadata.get("source_train_sha256") != TRAIN_SHA256:
        raise ValueError(f"Training hash mismatch in run metadata {directory}")
    if int(config.get("seed", -1)) != seed:
        raise ValueError(f"Seed mismatch in {directory}")
    return {"metrics": metrics, "metadata": metadata, "config": config}


def load_historical_c256() -> dict[int, dict[str, Any]]:
    """Read existing G0 results; this function never launches a C256 job."""
    result: dict[int, dict[str, Any]] = {}
    for seed in SEEDS:
        directory = historical_c256_artifact_dir(seed)
        values = _validate_metric_artifact(directory, seed=seed)
        config = values["config"]
        if int(config["model"]["hidden_dim"]) != 256 or int(config["model"]["num_layers"]) != 4:
            raise ValueError(f"Historical C256 architecture mismatch for seed {seed}")
        if config.get("graph", {}).get("representation") != "keep_dummy":
            raise ValueError(f"Historical C256 graph representation mismatch for seed {seed}")
        result[seed] = {
            **values,
            "artifact_dir": directory,
            "historical_reuse": True,
        }
    return result


def _best_epoch_train_loss(directory: Path, fold: int, best_epoch: int) -> float | None:
    history_path = directory / "training_history" / f"fold_{fold}.csv"
    if not history_path.is_file():
        return None
    with history_path.open("r", encoding="utf-8", newline="") as source:
        for row in csv.DictReader(source):
            if int(float(row["epoch"])) == best_epoch:
                return float(row["train_loss"])
    return None


def _width_behavior(width_runs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    folds = [fold for run in width_runs.values() for fold in run["metadata"]["fold_metrics"]]
    losses = []
    early_stop = []
    for seed, run in width_runs.items():
        directory = run["artifact_dir"]
        for fold in run["metadata"]["fold_metrics"]:
            loss = _best_epoch_train_loss(
                directory, int(fold["fold"]), int(fold["best_epoch"])
            )
            if loss is not None:
                losses.append(loss)
            early_stop.append(
                {
                    "seed": seed,
                    "fold": int(fold["fold"]),
                    "best_epoch": int(fold["best_epoch"]),
                    "epochs_completed": int(fold["epochs_completed"]),
                }
            )
    mean_best_epoch, sd_best_epoch = _mean_sd(fold["best_epoch"] for fold in folds)
    mean_best_validation, sd_best_validation = _mean_sd(
        fold["best_validation_wmae"] for fold in folds
    )
    mean_stop, sd_stop = _mean_sd(item["epochs_completed"] for item in early_stop)
    return {
        "mean_best_epoch": mean_best_epoch,
        "sd_best_epoch": sd_best_epoch,
        "mean_best_validation_wmae": mean_best_validation,
        "sd_best_validation_wmae": sd_best_validation,
        "mean_training_loss_at_best_epoch": statistics.fmean(losses) if losses else None,
        "training_loss_at_best_epoch_fold_count": len(losses),
        "early_stop_epochs": early_stop,
        "mean_epochs_completed": mean_stop,
        "sd_epochs_completed": sd_stop,
    }


def _performance_summary(width_runs: Mapping[int, Mapping[str, Any]]) -> dict[str, Any]:
    fold_rows = [fold for run in width_runs.values() for fold in run["metadata"]["fold_metrics"]]
    seed_runtime = [float(run["metadata"]["duration_seconds"]) for run in width_runs.values()]
    gpu_util = [
        float(fold["gpu_utilization_mean_percent"])
        for fold in fold_rows
        if fold.get("gpu_utilization_mean_percent") is not None
    ]
    peak_allocated = [
        float(fold["peak_vram_allocated_mb"])
        for fold in fold_rows
        if fold.get("peak_vram_allocated_mb") is not None
    ]
    peak_reserved = [
        float(fold["peak_vram_reserved_mb"])
        for fold in fold_rows
        if fold.get("peak_vram_reserved_mb") is not None
    ]
    nvidia_smi = [
        float(fold["nvidia_smi_peak_memory_used_mb"])
        for fold in fold_rows
        if fold.get("nvidia_smi_peak_memory_used_mb") is not None
    ]
    return {
        "mean_fold_epoch_seconds": statistics.fmean(
            float(fold["seconds_per_epoch"]) for fold in fold_rows
        ),
        "mean_seed_runtime_seconds": statistics.fmean(seed_runtime),
        "sample_sd_seed_runtime_seconds": statistics.stdev(seed_runtime) if len(seed_runtime) > 1 else 0.0,
        "summed_seed_runtime_seconds": sum(seed_runtime),
        "peak_pytorch_allocated_mb": max(peak_allocated) if peak_allocated else None,
        "peak_pytorch_reserved_mb": max(peak_reserved) if peak_reserved else None,
        "peak_nvidia_smi_vram_mb": max(nvidia_smi) if nvidia_smi else None,
        "mean_gpu_utilization_percent": statistics.fmean(gpu_util) if gpu_util else None,
    }


def _execution_for_width(
    width: int,
    width_runs: Mapping[int, Mapping[str, Any]],
    queue_notes: Mapping[str, Any],
) -> dict[str, Any]:
    run_map = {item["output_dir"]: item for item in queue_notes.get("runs", [])}
    records = []
    for seed, run in width_runs.items():
        record = run_map.get(str(run["artifact_dir"]))
        if record is None:
            # Queue paths may be recorded with the equivalent non-resolved form.
            record = next(
                (
                    item for item in queue_notes.get("runs", [])
                    if item.get("job", {}).get("hidden_dim") == width
                    and item.get("job", {}).get("seed") == seed
                    and item.get("status") == "passed"
                ),
                None,
            )
        if record:
            records.append(record)
    return {
        "queue_records": records,
        "source_commit": queue_notes.get("source_commit"),
        "concurrency": queue_notes.get("concurrency"),
        "formal_wall_seconds": queue_notes.get("formal_wall_seconds"),
        "host_cpu_peak_percent": queue_notes.get("resource_monitor", {}).get("peak_host_cpu_percent"),
        "host_ram_peak_mb": queue_notes.get("resource_monitor", {}).get("peak_host_ram_used_mb"),
        "queue_gpu_utilization_mean_percent": queue_notes.get("resource_monitor", {}).get("mean_gpu_utilization_percent"),
        "queue_nvidia_smi_peak_vram_mb": queue_notes.get("resource_monitor", {}).get("peak_nvidia_smi_memory_used_mb"),
    }


def _summarize_width(
    width: int,
    runs: Mapping[int, Mapping[str, Any]],
    parameter_count: int,
    queue_notes: Mapping[str, Any],
) -> dict[str, Any]:
    overall = {seed: float(run["metrics"]["overall_oof_wmae"]) for seed, run in runs.items()}
    mean, sd = _mean_sd(overall.values())
    target_summary: dict[str, Any] = {}
    for target in TARGETS:
        values = {
            seed: float(run["metrics"]["target_mae"][target])
            for seed, run in runs.items()
        }
        target_mean, target_sd = _mean_sd(values.values())
        target_summary[target] = {
            "by_seed": values,
            "mean": target_mean,
            "sample_sd": target_sd,
        }
    performance = _performance_summary(runs)
    return {
        "hidden_dim": width,
        "parameter_count": parameter_count,
        "by_seed_oof_wmae": overall,
        "mean_oof_wmae": mean,
        "sample_sd_oof_wmae": sd,
        "per_target": target_summary,
        "performance": performance,
        "training_behavior": _width_behavior(runs),
        "execution": _execution_for_width(width, runs, queue_notes),
    }


def _paired_summary(
    width_runs: Mapping[int, Mapping[int, Mapping[str, Any]]],
    reference_runs: Mapping[int, Mapping[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for width in (128, 384, 512):
        width_values = width_runs[width]
        overall = {
            seed: float(width_values[seed]["metrics"]["overall_oof_wmae"])
            - float(reference_runs[seed]["metrics"]["overall_oof_wmae"])
            for seed in SEEDS
        }
        mean, sd = _mean_sd(overall.values())
        per_target = {}
        for target in TARGETS:
            paired = {
                seed: float(width_values[seed]["metrics"]["target_mae"][target])
                - float(reference_runs[seed]["metrics"]["target_mae"][target])
                for seed in SEEDS
            }
            target_mean, target_sd = _mean_sd(paired.values())
            per_target[target] = {
                "by_seed": paired,
                "mean": target_mean,
                "sample_sd": target_sd,
                "lower_seed_count": sum(value < 0 for value in paired.values()),
            }
        result[str(width)] = {
            "reference_width": 256,
            "by_seed": overall,
            "paired_mean": mean,
            "paired_sample_sd": sd,
            "lower_seed_count": sum(value < 0 for value in overall.values()),
            "per_target": per_target,
        }
    return result


def _write_capacity_curve(path: Path, widths: Mapping[int, Mapping[str, Any]]) -> None:
    columns = (
        "hidden_dim",
        "parameter_count",
        "mean_oof_wmae",
        "sd_oof_wmae",
        "mean_runtime",
        "peak_vram",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, lineterminator="\n")
        writer.writeheader()
        for width in WIDTHS:
            item = widths[width]
            writer.writerow(
                {
                    "hidden_dim": width,
                    "parameter_count": item["parameter_count"],
                    "mean_oof_wmae": item["mean_oof_wmae"],
                    "sd_oof_wmae": item["sample_sd_oof_wmae"],
                    "mean_runtime": item["performance"]["mean_seed_runtime_seconds"],
                    "peak_vram": item["performance"]["peak_nvidia_smi_vram_mb"],
                }
            )


def _update_results_registry(result_path: Path, rows: list[Mapping[str, Any]]) -> None:
    with result_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    existing_ids = [row.get("experiment_id", "") for row in existing]
    if len(existing_ids) != len(set(existing_ids)):
        raise ValueError("results.csv already contains duplicate experiment_id values")
    additions: list[dict[str, Any]] = []
    for row in rows:
        experiment_id = str(row["experiment_id"])
        if experiment_id in existing_ids or any(
            str(item["experiment_id"]) == experiment_id for item in additions
        ):
            raise ValueError(f"Refusing to duplicate registry row {experiment_id}")
        new_row = {field: "" for field in fields}
        values = {
            "experiment_id": experiment_id,
            "model_name": row["model_name"],
            "category": row["category"],
            "benchmark_version": "nopp2025_train_v1",
            "git_commit": row["source_commit"],
            "seed": row["seed"],
            "oof_wmae": row["overall_oof_wmae"],
            **{f"{target.lower()}_mae": row["target_mae"][target] for target in TARGETS},
            "status": row["status"],
            "notes": "Stage 5A pure hidden-width scaling; C256 historical G0 reused; no other scientific variable changed.",
            "folds_sha256": row["folds_sha256"],
            "train_data_sha256": row["train_sha256"],
            "run_metadata": row["run_metadata"],
        }
        for key, value in values.items():
            if key in new_row:
                new_row[key] = value
        additions.append(new_row)
    temporary = result_path.with_suffix(result_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([*existing, *additions])
    os.replace(temporary, result_path)


def _markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 5A — Pure Capacity Scaling",
        "",
        f"Formal source commit: `{summary['formal_source_commit']}`.",
        "The only scientific variable was GINE `hidden_dim`; all widths use four layers and the frozen G0 graph, readout, heads, and training protocol. C256 reuses historical G0 and adds zero formal runs.",
        "",
        "## Overall OOF wMAE",
        "",
        "| Width | Parameters | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for width in WIDTHS:
        item = summary["widths"][str(width)]
        vals = [item["by_seed_oof_wmae"][seed] for seed in SEEDS]
        lines.append(
            f"| {width} | {item['parameter_count']:,} | "
            + " | ".join(f"{value:.10f}" for value in vals)
            + f" | {item['mean_oof_wmae']:.10f} ± {item['sample_sd_oof_wmae']:.10f} |"
        )
    lines.extend(["", "## Paired deltas vs C256", "", "Negative values favor the candidate width.", ""])
    for width in (128, 384, 512):
        item = summary["paired"][str(width)]
        values = [item["by_seed"][seed] for seed in SEEDS]
        lines.append(
            f"- C{width} − C256: `" + ", ".join(f"{value:+.10f}" for value in values)
            + f"`; mean `{item['paired_mean']:+.10f} ± {item['paired_sample_sd']:.10f}`; lower in {item['lower_seed_count']}/5 seeds."
        )
    lines.extend(["", "## Per-target OOF MAE", "", "Mean ± sample SD across five seeds.", ""])
    lines.append("| Target | C128 | C256 | C384 | C512 |")
    lines.append("|---|---:|---:|---:|---:|")
    for target in TARGETS:
        cells = []
        for width in WIDTHS:
            item = summary["widths"][str(width)]["per_target"][target]
            cells.append(f"{item['mean']:.8g} ± {item['sample_sd']:.3g}")
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Paired per-target deltas vs C256", ""])
    lines.append("| Target | C128 mean ± SD | C384 mean ± SD | C512 mean ± SD |")
    lines.append("|---|---:|---:|---:|")
    for target in TARGETS:
        cells = []
        for width in (128, 384, 512):
            item = summary["paired"][str(width)]["per_target"][target]
            cells.append(f"{item['mean']:+.8g} ± {item['sample_sd']:.3g}")
        lines.append(f"| {target} | " + " | ".join(cells) + " |")
    lines.extend(["", "## Cost and training behavior", "", "| Width | Mean fold epoch (s) | Mean seed runtime (s) | Peak PyTorch allocated / reserved (MiB) | Peak nvidia-smi VRAM (MiB) | Mean GPU util (%) | Mean best epoch | Mean train loss at best epoch |", "|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for width in WIDTHS:
        item = summary["widths"][str(width)]
        perf = item["performance"]
        behavior = item["training_behavior"]
        lines.append(
            f"| {width} | {perf['mean_fold_epoch_seconds']:.3f} | {perf['mean_seed_runtime_seconds']:.1f} | "
            f"{perf['peak_pytorch_allocated_mb']:.1f} / {perf['peak_pytorch_reserved_mb']:.1f} | "
            f"{perf['peak_nvidia_smi_vram_mb']:.1f} | {perf['mean_gpu_utilization_percent']:.1f} | "
            f"{behavior['mean_best_epoch']:.1f} | {behavior['mean_training_loss_at_best_epoch']:.6g} |"
        )
    lines.extend(["", "## Execution", "", f"- Actual concurrency: {summary['execution']['concurrency']}", f"- Formal wall time: {summary['execution']['formal_wall_seconds']:.1f} s", f"- Summed process wall time: {summary['execution']['summed_process_wall_seconds']:.1f} s", f"- Host CPU peak: {summary['execution']['host_cpu_peak_percent']:.1f}%", f"- Host RAM peak: {summary['execution']['host_ram_peak_mb']:.1f} MiB", "", "## Interpretation", "", "Use the paired deltas and five-seed consistency as descriptive engineering evidence. Do not interpret these summaries as a formal significance test.", ""])
    return "\n".join(lines)


def aggregate(
    *,
    queue_notes_path: Path,
    registry_path: Path | None = None,
    output_root: Path = EXPERIMENT_ROOT,
    update_registry: bool = True,
) -> dict[str, Any]:
    queue_notes = _read_json(queue_notes_path)
    if queue_notes.get("status") != "passed" or queue_notes.get("mode") != "formal":
        raise ValueError("Stage 5A aggregation requires a complete successful formal queue")
    if queue_notes.get("planned_job_count") != 15 or queue_notes.get("completed_job_count") != 15:
        raise ValueError("Stage 5A formal queue must contain all 15 preregistered jobs")
    parameter_audit_path = EXPERIMENT_ROOT / "parameter_audit.json"
    if not parameter_audit_path.is_file():
        raise FileNotFoundError(
            "Stage 5A parameter audit is missing; run the frozen source audit first: "
            f"{parameter_audit_path}"
        )
    parameter_audit = _read_json(parameter_audit_path)
    reference_runs = load_historical_c256()
    new_runs: dict[int, dict[int, dict[str, Any]]] = {}
    new_registry_rows = []
    for width in NEW_WIDTHS:
        new_runs[width] = {}
        for seed in SEEDS:
            directory = _find_successful_formal_dir(width, seed)
            values = _validate_metric_artifact(directory, seed=seed)
            if int(values["config"]["model"]["hidden_dim"]) != width:
                raise ValueError(f"Width mismatch in {directory}")
            if values["metadata"].get("git_commit") != queue_notes.get("source_commit"):
                raise ValueError(f"Formal source commit mismatch in {directory}")
            new_runs[width][seed] = {
                **values,
                "artifact_dir": directory,
                "historical_reuse": False,
            }
            row = _read_json(directory / "registry_row.json")
            new_registry_rows.append(row)

    all_runs: dict[int, dict[int, dict[str, Any]]] = {
        128: new_runs[128],
        256: reference_runs,
        384: new_runs[384],
        512: new_runs[512],
    }
    widths_summary = {
        str(width): _summarize_width(
            width,
            all_runs[width],
            parameter_audit["parameter_counts"][str(width)]["total_trainable_parameters"],
            queue_notes,
        )
        for width in WIDTHS
    }
    paired = _paired_summary(new_runs, reference_runs)
    notes = queue_notes.get("resource_monitor", {})
    summary = {
        "stage": "5A",
        "formal_source_commit": queue_notes["source_commit"],
        "benchmark_version": "nopp2025_train_v1",
        "train_sha256": TRAIN_SHA256,
        "folds_sha256": FOLDS_SHA256,
        "widths": widths_summary,
        "paired": paired,
        "parameter_audit": parameter_audit,
        "execution": {
            "concurrency": queue_notes["concurrency"],
            "max_observed_active_jobs": queue_notes["max_observed_active_jobs"],
            "formal_wall_seconds": queue_notes["formal_wall_seconds"],
            "summed_process_wall_seconds": queue_notes["summed_process_wall_seconds"],
            "sum_training_seconds": sum(
                float(run["metadata"]["duration_seconds"])
                for runs in new_runs.values()
                for run in runs.values()
            ),
            "host_cpu_peak_percent": notes.get("peak_host_cpu_percent"),
            "host_ram_peak_mb": notes.get("peak_host_ram_used_mb"),
            "mean_gpu_utilization_percent": notes.get("mean_gpu_utilization_percent"),
            "peak_gpu_utilization_percent": notes.get("peak_gpu_utilization_percent"),
            "peak_nvidia_smi_vram_mb": notes.get("peak_nvidia_smi_memory_used_mb"),
            "failed_or_restarted_attempts": [
                record for record in queue_notes.get("runs", [])
                if record.get("status") == "failed"
            ],
            "queue_runs": queue_notes.get("runs", []),
        },
        "historical_reuse": {
            "C256_new_formal_runs": 0,
            "seeds": list(SEEDS),
            "artifact_directories": {
                str(seed): str(reference_runs[seed]["artifact_dir"].relative_to(TRACK_ROOT))
                for seed in SEEDS
            },
        },
        "new_formal_runs": 15,
        "no_other_scientific_variable_changed": True,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    write_json(output_root / "aggregate_summary.json", summary)
    (output_root / "aggregate_summary.md").write_text(_markdown(summary), encoding="utf-8")
    _write_capacity_curve(output_root / "capacity_curve.csv", widths_summary)
    if update_registry:
        _update_results_registry(
            registry_path or TRACK_ROOT / "results.csv", new_registry_rows
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--queue-notes",
        type=Path,
        default=EXPERIMENT_ROOT / "formal_execution_notes.json",
    )
    parser.add_argument("--registry", type=Path, default=TRACK_ROOT / "results.csv")
    parser.add_argument("--output-root", type=Path, default=EXPERIMENT_ROOT)
    parser.add_argument("--no-registry-update", action="store_true")
    args = parser.parse_args()
    summary = aggregate(
        queue_notes_path=args.queue_notes.resolve(),
        registry_path=args.registry.resolve(),
        output_root=args.output_root.resolve(),
        update_registry=not args.no_registry_update,
    )
    print(
        json.dumps(
            {
                "stage": summary["stage"],
                "formal_source_commit": summary["formal_source_commit"],
                "new_formal_runs": summary["new_formal_runs"],
                "historical_c256_runs": summary["historical_reuse"]["C256_new_formal_runs"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
