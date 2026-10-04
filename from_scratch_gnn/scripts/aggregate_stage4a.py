"""Validate and aggregate Stage 4A runs, then update results.csv once."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any, Mapping

from ..models.global_information_augmentation.runner import (
    FORMAL_SEEDS,
    TRACK_ROOT,
    VARIANTS,
)
from ..src.data import sha256_file, write_json
from ..src.metrics import TARGETS

EXPECTED_TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"
EXPECTED_FOLDS_SHA256 = "1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a"
EXPERIMENT_ROOT = TRACK_ROOT / "experiments" / "stage4a"
HISTORICAL_G0_PATH = TRACK_ROOT / "experiments" / "stage3c" / "aggregate_summary.json"
FEATURE_MANIFEST_PATH = EXPERIMENT_ROOT / "feature_manifest.json"
EXECUTION_NOTES_PATH = EXPERIMENT_ROOT / "formal_execution_notes.json"


def _stats(values: Mapping[str, float]) -> dict[str, Any]:
    ordered = [float(values[str(seed)]) for seed in FORMAL_SEEDS]
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Stage 4A values must be finite for every frozen seed")
    return {
        "by_seed": {str(seed): float(values[str(seed)]) for seed in FORMAL_SEEDS},
        "mean": statistics.fmean(ordered),
        "sample_std": statistics.stdev(ordered),
        "min": min(ordered),
        "max": max(ordered),
    }


def _paired(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, Any]:
    deltas = {
        str(seed): float(left[str(seed)]) - float(right[str(seed)])
        for seed in FORMAL_SEEDS
    }
    summary = _stats(deltas)
    summary["left_lower_seed_count"] = sum(value < 0 for value in deltas.values())
    summary["tie_seed_count"] = sum(value == 0 for value in deltas.values())
    return summary


def _successful_runs(variant: str) -> dict[str, dict[str, Any]]:
    variant_root = TRACK_ROOT / "models" / "global_information_augmentation" / variant
    root = variant_root / "artifacts" / "formal"
    found: dict[str, list[dict[str, Any]]] = {str(seed): [] for seed in FORMAL_SEEDS}
    if not root.is_dir():
        raise FileNotFoundError(f"Formal artifact directory is missing: {root}")
    expected_variant = VARIANTS[variant]
    for seed_root in sorted(root.glob("seed_*")):
        for run_dir in sorted(seed_root.glob("attempt_*")):
            required = (
                "run_metadata.json",
                "metrics.json",
                "source_manifest.json",
                "registry_row.json",
                "global_information_metadata.json",
                "global_information_diagnostics.json",
            )
            if not all((run_dir / name).is_file() for name in required):
                continue
            metadata = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            source = json.loads((run_dir / "source_manifest.json").read_text(encoding="utf-8"))
            registry = json.loads((run_dir / "registry_row.json").read_text(encoding="utf-8"))
            info = json.loads(
                (run_dir / "global_information_metadata.json").read_text(encoding="utf-8")
            )
            diagnostics = json.loads(
                (run_dir / "global_information_diagnostics.json").read_text(encoding="utf-8")
            )
            seed = str(int(metadata["seed"]))
            if seed not in found:
                raise ValueError(f"Unexpected formal seed in {run_dir}: {seed}")
            if metadata.get("variant") != expected_variant or source.get("variant") != expected_variant:
                raise ValueError(f"Formal variant mismatch in {run_dir}")
            if metadata.get("stage") != "4A":
                raise ValueError(f"Formal run is not recorded as Stage 4A: {run_dir}")
            if registry.get("seed") != int(seed) or registry.get("variant") != expected_variant:
                raise ValueError(f"Registry row does not match formal run: {run_dir}")
            if registry.get("experiment_id") != _expected_experiment_id(variant, int(seed)):
                raise ValueError(f"Registry experiment_id is not unique/seed-specific: {run_dir}")
            validation = metrics["validation"]
            if (
                validation["training_sample_count"] != 7973
                or validation["prediction_row_count"] != 7973
                or validation["duplicate_prediction_sample_id_count"] != 0
                or validation["missing_prediction_sample_id_count"] != 0
                or validation["extra_prediction_sample_id_count"] != 0
            ):
                raise ValueError(f"Frozen OOF row validation failed in {run_dir}")
            hashes = source["benchmark_files_sha256"]
            if hashes["train_csv"] != EXPECTED_TRAIN_SHA256:
                raise ValueError(f"Training SHA256 mismatch in {run_dir}")
            if hashes["folds_csv"] != EXPECTED_FOLDS_SHA256:
                raise ValueError(f"Fold SHA256 mismatch in {run_dir}")
            if source["feature_manifest_sha256"] != sha256_file(FEATURE_MANIFEST_PATH):
                raise ValueError(f"Feature manifest hash mismatch in {run_dir}")
            if expected_variant == "D":
                scaler_paths = [run_dir / "fold_scalers" / f"fold_{fold}.json" for fold in range(5)]
                if not all(path.is_file() for path in scaler_paths):
                    raise ValueError(f"Descriptor fold scalers are incomplete in {run_dir}")
                for scaler_path in scaler_paths:
                    scaler = json.loads(scaler_path.read_text(encoding="utf-8"))
                    expected_fold = int(scaler_path.stem.removeprefix("fold_"))
                    if scaler.get("fit_row_count") not in (6378, 6379):
                        raise ValueError(f"Descriptor scaler fit row count is invalid: {scaler_path}")
                    if scaler.get("fold") != expected_fold:
                        raise ValueError(f"Descriptor scaler fold metadata mismatch: {scaler_path}")
                    if scaler.get("source_data_sha256") != EXPECTED_TRAIN_SHA256:
                        raise ValueError(f"Descriptor scaler source hash mismatch: {scaler_path}")
                    if scaler.get("zero_variance_handling") != "use scale=1 and keep the column":
                        raise ValueError(f"Descriptor zero-variance policy mismatch: {scaler_path}")
                    if scaler.get("feature_names") != json.loads(
                        FEATURE_MANIFEST_PATH.read_text(encoding="utf-8")
                    )["descriptors"]["names_in_order"]:
                        raise ValueError(f"Descriptor scaler schema mismatch: {scaler_path}")
            found[seed].append(
                {
                    "directory": run_dir,
                    "metadata": metadata,
                    "metrics": metrics,
                    "source_manifest": source,
                    "registry": registry,
                    "parameter_metadata": info,
                    "diagnostics": diagnostics,
                    "effective_config_sha256": sha256_file(run_dir / "config.json"),
                }
            )
    selected: dict[str, dict[str, Any]] = {}
    for seed, attempts in found.items():
        if len(attempts) != 1:
            raise ValueError(
                f"Expected one successful {variant} run for seed {seed}; found {len(attempts)}"
            )
        selected[seed] = attempts[0]
    return selected


def _expected_experiment_id(variant: str, seed: int) -> str:
    config = json.loads(
        (TRACK_ROOT / "models" / "global_information_augmentation" / variant / "config.json")
        .read_text(encoding="utf-8")
    )
    base = str(config["experiment_id"])
    return base if seed == 42 else f"{base}_seed_{seed}"


def _historical_g0() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    summary = json.loads(HISTORICAL_G0_PATH.read_text(encoding="utf-8"))
    if summary.get("train_data_sha256") != EXPECTED_TRAIN_SHA256:
        raise ValueError("Historical G0 train-data hash does not match Stage 4A")
    if summary.get("folds_sha256") != EXPECTED_FOLDS_SHA256:
        raise ValueError("Historical G0 fold hash does not match Stage 4A")
    overall = {
        str(seed): float(value)
        for seed, value in summary["overall_wmae"]["R0"]["by_seed"].items()
    }
    targets = {
        target: {
            str(seed): float(value)
            for seed, value in summary["per_target_mae"][target]["models"]["R0"]["by_seed"].items()
        }
        for target in TARGETS
    }
    if set(overall) != {str(seed) for seed in FORMAL_SEEDS}:
        raise ValueError("Historical G0 does not contain all five registered seeds")
    if overall["42"] != 0.022772872219188344:
        raise ValueError("Historical G0 does not match the frozen Stage 4A baseline")
    return overall, targets


def _build_summary() -> dict[str, Any]:
    descriptor = _successful_runs("descriptor")
    morgan = _successful_runs("morgan")
    g0_overall, g0_targets = _historical_g0()
    overall: dict[str, dict[str, float]] = {"G0": g0_overall}
    target_values: dict[str, dict[str, dict[str, float]]] = {
        target: {"G0": g0_targets[target]} for target in TARGETS
    }
    sources: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    model_runs = {"D": descriptor, "M": morgan}
    for code, runs in model_runs.items():
        overall[code] = {
            seed: float(run["metrics"]["overall_oof_wmae"])
            for seed, run in runs.items()
        }
        for target in TARGETS:
            target_values[target][code] = {
                seed: float(run["metrics"]["target_mae"][target])
                for seed, run in runs.items()
            }
        commits = {run["metadata"]["git_commit"] for run in runs.values()}
        if len(commits) != 1:
            raise ValueError(f"All {code} formal runs must share one source commit")
        sources[code] = {
            "source_commit": next(iter(commits)),
            "effective_config_sha256_by_seed": {
                seed: run["effective_config_sha256"] for seed, run in runs.items()
            },
            "run_directories_by_seed": {
                seed: str(run["directory"].relative_to(TRACK_ROOT)) for seed, run in runs.items()
            },
            "benchmark_hashes": runs["42"]["source_manifest"]["benchmark_files_sha256"],
        }
        diagnostics[code] = {
            "projection_l2_norm_by_seed_by_fold": {
                seed: run["diagnostics"]["projection_l2_norm_by_fold"]
                for seed, run in runs.items()
            },
            "projection_l2_norm_mean_by_seed": {
                seed: run["diagnostics"]["projection_l2_norm_mean"]
                for seed, run in runs.items()
            },
        }
        if code == "D":
            diagnostics[code]["projection_column_l2_norm_by_seed"] = {
                seed: run["diagnostics"]["descriptor_projection_column_l2_norm_fold_4"]
                for seed, run in runs.items()
            }
        else:
            diagnostics[code]["average_bits_on"] = {
                seed: run["diagnostics"]["morgan_average_bits_on"]
                for seed, run in runs.items()
            }
            diagnostics[code]["duplicate_fingerprint_rows"] = {
                seed: run["diagnostics"]["morgan_duplicate_fingerprint_rows"]
                for seed, run in runs.items()
            }
    commits = {value["source_commit"] for value in sources.values()}
    if len(commits) != 1:
        raise ValueError("D and M formal runs do not share the same source commit")

    comparisons = (
        ("D-G0", "D", "G0"),
        ("M-G0", "M", "G0"),
        ("D-M", "D", "M"),
    )
    paired = {
        name: _paired(overall[left], overall[right])
        for name, left, right in comparisons
    }
    per_target = {}
    for target in TARGETS:
        model_stats = {
            code: _stats(target_values[target][code]) for code in ("G0", "D", "M")
        }
        per_target[target] = {
            "models": model_stats,
            "paired_deltas": {
                name: _paired(target_values[target][left], target_values[target][right])
                for name, left, right in comparisons
            },
        }
    overall_stats = {code: _stats(values) for code, values in overall.items()}
    formal_runs = [run for runs in model_runs.values() for run in runs.values()]
    fold_rows = [fold for run in formal_runs for fold in run["metadata"]["fold_metrics"]]
    runtimes = [float(run["metadata"]["duration_seconds"]) for run in formal_runs]
    utilization = [
        float(fold["gpu_utilization_mean_percent"])
        for fold in fold_rows
        if fold.get("gpu_utilization_mean_percent") is not None
    ]
    epoch_times = [
        float(fold["seconds_per_epoch"])
        for fold in fold_rows
        if fold.get("seconds_per_epoch") is not None
    ]
    execution_notes = json.loads(EXECUTION_NOTES_PATH.read_text(encoding="utf-8"))
    parameter_counts = {
        code: next(iter(runs.values()))["parameter_metadata"]["parameter_count"]
        for code, runs in model_runs.items()
    }
    parameter_counts["G0"] = {
        "total_trainable": parameter_counts["D"]["G0_total_trainable"],
        "projection_trainable": 0,
        "new_parameters_vs_G0": 0,
    }
    for code in ("D", "M"):
        if any(run["parameter_metadata"]["parameter_count"] != parameter_counts[code] for run in model_runs[code].values()):
            raise ValueError(f"{code} parameter counts differ across seeds")

    stable = {
        code: bool(paired[f"{code}-G0"]["mean"] < 0 and paired[f"{code}-G0"]["left_lower_seed_count"] >= 4)
        for code in ("D", "M")
    }
    conclusion = {
        "stable_gain_rule": "paired mean delta < 0 and improvement in at least 4 of 5 seeds",
        "D_stable_improvement": stable["D"],
        "M_stable_improvement": stable["M"],
        "D_plus_M_combination_warranted": stable["D"] and stable["M"],
        "next_model_to_carry": (
            "Stage 4B combination confirmation; preserve G0, D, and M results"
            if stable["D"] and stable["M"]
            else "D" if stable["D"] else "M" if stable["M"] else "G0"
        ),
        "note": "No Stage 4B run is started by this aggregator.",
    }
    return {
        "experiment": "Stage 4A global information augmentation",
        "analysis_only": False,
        "seeds": list(FORMAL_SEEDS),
        "train_data_sha256": EXPECTED_TRAIN_SHA256,
        "folds_sha256": EXPECTED_FOLDS_SHA256,
        "formal_source_commit": next(iter(commits)),
        "source_provenance": sources,
        "overall_wmae": overall_stats,
        "paired_overall_delta": paired,
        "per_target_mae": per_target,
        "parameter_counts": parameter_counts,
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "projection_and_feature_diagnostics": diagnostics,
        "formal_execution": {
            "G0_runs_retrained": 0,
            "D_successful_runs": len(descriptor),
            "M_successful_runs": len(morgan),
            "new_formal_runs": len(formal_runs),
            "training_runtime_seconds_from_run_metadata": sum(runtimes),
            "training_runtime_hours_from_run_metadata": sum(runtimes) / 3600.0,
            "formal_wall_seconds": execution_notes["formal_wall_seconds"],
            "device_names": sorted({
                str(run["metadata"]["runtime"].get("device_name")) for run in formal_runs
            }),
            "max_peak_vram_allocated_mb": max(float(fold["peak_vram_allocated_mb"] or 0.0) for fold in fold_rows),
            "max_peak_vram_reserved_mb": max(float(fold["peak_vram_reserved_mb"] or 0.0) for fold in fold_rows),
            "max_nvidia_smi_memory_mb": max(float(fold["nvidia_smi_peak_memory_used_mb"] or 0.0) for fold in fold_rows),
            "mean_gpu_utilization_percent_across_folds": statistics.fmean(utilization) if utilization else None,
            "max_gpu_utilization_percent_across_folds": max(
                (float(fold["gpu_utilization_max_percent"]) for fold in fold_rows if fold.get("gpu_utilization_max_percent") is not None),
                default=None,
            ),
            "mean_seconds_per_epoch_across_folds": statistics.fmean(epoch_times) if epoch_times else None,
            "concurrency": execution_notes["concurrency"],
            "throughput_pilot": execution_notes["throughput_pilot"],
            "cpu_ram_bottleneck": execution_notes["cpu_ram_bottleneck"],
            "engineering_failures": execution_notes["engineering_failures"],
        },
        "conclusion": conclusion,
    }


def _format_stats(stats: Mapping[str, Any]) -> str:
    return (
        f"{stats['mean']:.10f} ± {stats['sample_std']:.10f}; "
        f"{stats['min']:.10f}–{stats['max']:.10f}"
    )


def _write_markdown(summary: Mapping[str, Any]) -> None:
    lines = [
        "# Stage 4A — global information augmentation",
        "",
        f"Formal source commit: `{summary['formal_source_commit']}`.",
        "G0 reuses the five historical Stage 3A.1 / Stage 3C R0 runs; D and M each add five frozen five-fold OOF runs.",
        "",
        "## Overall OOF wMAE",
        "",
        "| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for code in ("G0", "D", "M"):
        stats = summary["overall_wmae"][code]
        row = " | ".join(f"{stats['by_seed'][str(seed)]:.10f}" for seed in FORMAL_SEEDS)
        lines.append(f"| {code} | {row} | {stats['mean']:.10f} ± {stats['sample_std']:.10f} | {stats['min']:.10f}–{stats['max']:.10f} |")
    lines.extend(["", "## Paired overall deltas (left minus right)", "", "| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Left lower |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for name in ("D-G0", "M-G0", "D-M"):
        stats = summary["paired_overall_delta"][name]
        row = " | ".join(f"{stats['by_seed'][str(seed)]:+.10f}" for seed in FORMAL_SEEDS)
        lines.append(f"| {name} | {row} | {stats['mean']:+.10f} ± {stats['sample_std']:.10f} | {stats['left_lower_seed_count']}/5 |")
    lines.extend(["", "## Per-target MAE", "", "| Target | G0 mean ± SD | D mean ± SD | M mean ± SD | D−G0 delta mean ± SD | M−G0 delta mean ± SD |", "|---|---:|---:|---:|---:|---:|"])
    for target in TARGETS:
        item = summary["per_target_mae"][target]
        d = item["paired_deltas"]["D-G0"]
        m = item["paired_deltas"]["M-G0"]
        lines.append(
            f"| {target} | {_format_stats(item['models']['G0'])} | {_format_stats(item['models']['D'])} | {_format_stats(item['models']['M'])} | {d['mean']:+.10g} ± {d['sample_std']:.10g} | {m['mean']:+.10g} ± {m['sample_std']:.10g} |"
        )
    execution = summary["formal_execution"]
    lines.extend([
        "",
        "## Parameters and execution",
        "",
        f"- G0/D/M parameter counts: `{summary['parameter_counts']}`.",
        f"- Morgan adds {summary['parameter_counts']['M']['projection_trainable']:,} trainable parameters; a Morgan score gain cannot be attributed to fingerprint information alone because model capacity also increases.",
        f"- New formal runs: {execution['new_formal_runs']}; G0 retrains: 0.",
        f"- Training time from run metadata: {execution['training_runtime_hours_from_run_metadata']:.2f} h; formal queue wall time: {execution['formal_wall_seconds']:.1f} s.",
        f"- RTX device(s): {', '.join(execution['device_names'])}; concurrency: {execution['concurrency']}.",
        f"- Maximum VRAM: allocated {execution['max_peak_vram_allocated_mb']:.1f} MiB, reserved {execution['max_peak_vram_reserved_mb']:.1f} MiB; nvidia-smi {execution['max_nvidia_smi_memory_mb']:.1f} MiB.",
        f"- Mean GPU utilization across folds: {execution['mean_gpu_utilization_percent_across_folds']}; max sampled utilization: {execution['max_gpu_utilization_percent_across_folds']}; mean epoch time: {execution['mean_seconds_per_epoch_across_folds']} s.",
        f"- Throughput pilot: `{execution['throughput_pilot']}`.",
        f"- CPU/RAM bottleneck: `{execution['cpu_ram_bottleneck']}`; engineering failures: `{execution['engineering_failures']}`.",
        "",
        "## Diagnostics",
        "",
        f"- D projection mean L2 norm by seed: `{summary['projection_and_feature_diagnostics']['D']['projection_l2_norm_mean_by_seed']}`.",
        f"- M projection mean L2 norm by seed: `{summary['projection_and_feature_diagnostics']['M']['projection_l2_norm_mean_by_seed']}`.",
        f"- M average bits on: `{summary['projection_and_feature_diagnostics']['M']['average_bits_on']}`; duplicate fingerprint rows: `{summary['projection_and_feature_diagnostics']['M']['duplicate_fingerprint_rows']}`.",
        "- D training-fold means/stds and the complete descriptor audit are retained in each run's scaler files and feature manifest.",
        "- Diagnostics are descriptive only and were not used for feature selection or tuning.",
        "",
        "## Conclusion",
        "",
        f"- D stable improvement: **{summary['conclusion']['D_stable_improvement']}**.",
        f"- M stable improvement: **{summary['conclusion']['M_stable_improvement']}**.",
        f"- D+M combination warranted: **{summary['conclusion']['D_plus_M_combination_warranted']}**.",
        f"- Next model to carry: **{summary['conclusion']['next_model_to_carry']}**.",
        "- Stage 4B was not run.",
        "",
    ])
    (EXPERIMENT_ROOT / "aggregate_summary.md").write_text("\n".join(lines), encoding="utf-8")


def _update_results_registry(summary: Mapping[str, Any]) -> None:
    registry_path = TRACK_ROOT / "results.csv"
    with registry_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    if not fields or "experiment_id" not in fields:
        raise ValueError("results.csv has no experiment_id column")
    updates = []
    for variant in ("descriptor", "morgan"):
        selected_runs = _successful_runs(variant)
        for seed in FORMAL_SEEDS:
            run_dir = selected_runs[str(seed)]["directory"]
            meta = json.loads((run_dir / "run_metadata.json").read_text(encoding="utf-8"))
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
            config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
            row = {field: "" for field in fields}
            experiment_id = _expected_experiment_id(variant, seed)
            values = {
                "experiment_id": experiment_id,
                "model_name": config["model_name"],
                "category": "own_model",
                "benchmark_version": config["benchmark_version"],
                "git_commit": meta["git_commit"],
                "seed": seed,
                "oof_wmae": metrics["overall_oof_wmae"],
                **{f"{target.lower()}_mae": metrics["target_mae"][target] for target in TARGETS},
                "status": "formal_oof_model",
                "notes": config["registry_notes"],
                "folds_sha256": EXPECTED_FOLDS_SHA256,
                "train_data_sha256": EXPECTED_TRAIN_SHA256,
                "run_metadata": str((run_dir / "run_metadata.json").relative_to(TRACK_ROOT)),
            }
            for key, value in values.items():
                if key in row:
                    row[key] = value
            updates.append(row)
    ids = [row["experiment_id"] for row in updates]
    if len(ids) != len(set(ids)):
        raise ValueError("Stage 4A generated duplicate registry experiment IDs")
    for update in updates:
        matches = [i for i, row in enumerate(rows) if row.get("experiment_id") == update["experiment_id"]]
        if len(matches) > 1:
            raise ValueError(f"Duplicate existing registry ID: {update['experiment_id']}")
        if matches:
            rows[matches[0]] = update
        else:
            rows.append(update)
    temp = registry_path.with_suffix(".csv.stage4a.tmp")
    with temp.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, registry_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    summary = _build_summary()
    write_json(EXPERIMENT_ROOT / "aggregate_summary.json", summary)
    _write_markdown(summary)
    _update_results_registry(summary)
    print(json.dumps(summary["conclusion"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
