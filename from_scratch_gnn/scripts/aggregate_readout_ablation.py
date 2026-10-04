"""Aggregate the frozen Stage 3C readout runs against historical GINE seeds."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..models.readout_ablation.runner import FORMAL_SEEDS, STAGE3C_MODELS, TRACK_ROOT
from ..src.data import sha256_file, write_json
from ..src.metrics import TARGETS


EXPECTED_TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"
EXPECTED_FOLDS_SHA256 = "1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a"
HISTORICAL_SUMMARY = TRACK_ROOT / "experiments" / "stage3a1" / "paired_summary.json"
EXPERIMENT_ROOT = TRACK_ROOT / "experiments" / "stage3c"
EXECUTION_NOTES = EXPERIMENT_ROOT / "formal_execution_notes.json"


def _stats(values: Mapping[str, float]) -> dict[str, Any]:
    ordered = [float(values[str(seed)]) for seed in FORMAL_SEEDS]
    if not all(math.isfinite(value) for value in ordered):
        raise ValueError("Stage 3C metrics must be finite for every frozen seed")
    return {
        "by_seed": {str(seed): values[str(seed)] for seed in FORMAL_SEEDS},
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
    summary.update(
        {
            "left_lower_seed_count": sum(value < 0 for value in deltas.values()),
            "tie_seed_count": sum(value == 0 for value in deltas.values()),
        }
    )
    return summary


def _successful_runs(variant: str) -> dict[str, dict[str, Any]]:
    root = TRACK_ROOT / "models" / "readout_ablation" / variant / "artifacts" / "formal"
    found: dict[str, list[dict[str, Any]]] = {str(seed): [] for seed in FORMAL_SEEDS}
    if not root.exists():
        raise FileNotFoundError(f"Formal artifact root is missing: {root}")
    for seed_root in sorted(root.glob("seed_*")):
        for run_dir in sorted(seed_root.glob("attempt_*")):
            metadata_path = run_dir / "run_metadata.json"
            metrics_path = run_dir / "metrics.json"
            diagnostics_path = run_dir / "readout_diagnostics.json"
            if not (metadata_path.is_file() and metrics_path.is_file() and diagnostics_path.is_file()):
                continue
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            diagnostics = json.loads(diagnostics_path.read_text(encoding="utf-8"))
            seed = str(int(metadata["seed"]))
            if seed not in found:
                raise ValueError(f"Unexpected formal seed in {run_dir}: {seed}")
            source_manifest = json.loads(
                (run_dir / "source_manifest.json").read_text(encoding="utf-8")
            )
            if metadata.get("readout_variant") != STAGE3C_MODELS[variant]:
                raise ValueError(f"Readout variant mismatch in {run_dir}")
            validation = metrics["validation"]
            if (
                validation["training_sample_count"] != 7973
                or validation["prediction_row_count"] != 7973
                or validation["duplicate_prediction_sample_id_count"] != 0
                or validation["missing_prediction_sample_id_count"] != 0
                or validation["extra_prediction_sample_id_count"] != 0
            ):
                raise ValueError(f"OOF coverage validation failed in {run_dir}")
            hashes = source_manifest["benchmark_files_sha256"]
            if hashes["train_csv"] != EXPECTED_TRAIN_SHA256:
                raise ValueError(f"Training-data hash mismatch in {run_dir}")
            if hashes["folds_csv"] != EXPECTED_FOLDS_SHA256:
                raise ValueError(f"Folds hash mismatch in {run_dir}")
            found[seed].append(
                {
                    "directory": run_dir,
                    "metadata": metadata,
                    "metrics": metrics,
                    "diagnostics": diagnostics,
                    "source_manifest": source_manifest,
                    "effective_config_sha256": sha256_file(run_dir / "config.json"),
                }
            )
    selected = {}
    for seed, attempts in found.items():
        if len(attempts) != 1:
            raise ValueError(
                f"Expected exactly one successful {variant} run for seed {seed}; "
                f"found {len(attempts)}"
            )
        selected[seed] = attempts[0]
    return selected


def _historical_r0() -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    summary = json.loads(HISTORICAL_SUMMARY.read_text(encoding="utf-8"))
    overall = {
        str(seed): float(value)
        for seed, value in summary["overall_wmae"]["Variant A"]["by_seed"].items()
    }
    per_target = {
        target: {
            str(seed): float(value)
            for seed, value in summary["per_target_mae"][target]["by_model"]["Variant A"]["by_seed"].items()
        }
        for target in TARGETS
    }
    if set(overall) != {str(seed) for seed in FORMAL_SEEDS}:
        raise ValueError("Historical R0 does not contain the frozen five seeds")
    return overall, per_target


def _build_summary() -> dict[str, Any]:
    r1 = _successful_runs("r1_shared")
    r2 = _successful_runs("r2_property")
    r0_overall, r0_target = _historical_r0()
    overall = {"R0": r0_overall}
    target_scores: dict[str, dict[str, dict[str, float]]] = {
        target: {"R0": r0_target[target]} for target in TARGETS
    }
    sources: dict[str, Any] = {}
    diagnostics: dict[str, Any] = {}
    for variant, runs in (("R1", r1), ("R2", r2)):
        overall[variant] = {
            seed: float(run["metrics"]["overall_oof_wmae"])
            for seed, run in runs.items()
        }
        for target in TARGETS:
            target_scores[target][variant] = {
                seed: float(run["metrics"]["target_mae"][target])
                for seed, run in runs.items()
            }
        commits = {run["metadata"]["git_commit"] for run in runs.values()}
        if len(commits) != 1:
            raise ValueError(f"All {variant} runs must share one source commit")
        sources[variant] = {
            "source_commit": next(iter(commits)),
            "effective_config_sha256_by_seed": {
                seed: run["effective_config_sha256"] for seed, run in runs.items()
            },
            "run_directories_by_seed": {
                seed: str(run["directory"].relative_to(TRACK_ROOT))
                for seed, run in runs.items()
            },
            "benchmark_hashes": runs["42"]["source_manifest"]["benchmark_files_sha256"],
        }
        gate_keys = sorted(
            {
                key
                for run in runs.values()
                for key in run["diagnostics"]["gate_l2_norm_mean_by_gate"]
            }
        )
        diagnostics[variant] = {
            "gate_l2_norm_mean_by_seed": {
                gate: {
                    seed: float(run["diagnostics"]["gate_l2_norm_mean_by_gate"][gate])
                    for seed, run in runs.items()
                }
                for gate in gate_keys
            },
            "normalized_attention_entropy_mean_by_target_by_seed": {
                target: {
                    seed: float(run["diagnostics"]["normalized_attention_entropy_mean_by_target"][target])
                    for seed, run in runs.items()
                }
                for target in TARGETS
            },
        }

    source_commits = {sources["R1"]["source_commit"], sources["R2"]["source_commit"]}
    if len(source_commits) != 1:
        raise ValueError("R1 and R2 formal runs do not share a single source commit")
    comparisons = (("R1-R0", "R1", "R0"), ("R2-R0", "R2", "R0"), ("R2-R1", "R2", "R1"))
    paired = {
        name: _paired(overall[left], overall[right])
        for name, left, right in comparisons
    }
    per_target = {}
    for target in TARGETS:
        models = {
            model: _stats(target_scores[target][model]) for model in ("R0", "R1", "R2")
        }
        paired_target = {
            name: _paired(target_scores[target][left], target_scores[target][right])
            for name, left, right in comparisons
        }
        per_target[target] = {"models": models, "paired_deltas": paired_target}

    overall_models = {model: _stats(values) for model, values in overall.items()}
    formal_runs = [run for runs in (r1, r2) for run in runs.values()]
    fold_rows = [
        fold
        for run in formal_runs
        for fold in run["metadata"]["fold_metrics"]
    ]
    runtimes = [float(run["metadata"]["duration_seconds"]) for run in formal_runs]
    execution_notes = json.loads(EXECUTION_NOTES.read_text(encoding="utf-8"))
    parameter_counts = {
        variant: json.loads(
            next(iter(runs.values()))["directory"].joinpath("readout_metadata.json").read_text(encoding="utf-8")
        )["parameter_count"]
        for variant, runs in (("R1", r1), ("R2", r2))
    }
    r0_total = int(parameter_counts["R1"]["historical_r0_total_trainable"])
    parameter_counts["R0"] = {
        "total_trainable": r0_total,
        "readout_gate_trainable": 0,
        "new_parameters_vs_r0": 0,
    }
    return {
        "experiment": "Stage 3C readout / property-specific pooling ablation",
        "analysis_only": False,
        "seeds": list(FORMAL_SEEDS),
        "train_data_sha256": EXPECTED_TRAIN_SHA256,
        "folds_sha256": EXPECTED_FOLDS_SHA256,
        "formal_source_commit": next(iter(source_commits)),
        "source_provenance": sources,
        "formal_execution": {
            "historical_r0_runs_retrained": 0,
            "r1_successful_runs": len(r1),
            "r2_successful_runs": len(r2),
            "new_formal_runs": len(formal_runs),
            "training_runtime_seconds_from_run_metadata": sum(runtimes),
            "training_runtime_hours_from_run_metadata": sum(runtimes) / 3600.0,
            "batch_wall_runtime_seconds": float(
                execution_notes["batch_wall_runtime_seconds"]
            ),
            "batch_wall_runtime_hours": float(
                execution_notes["batch_wall_runtime_seconds"]
            ) / 3600.0,
            "device_names": sorted(
                {
                    str(run["metadata"]["runtime"].get("device_name"))
                    for run in formal_runs
                }
            ),
            "max_peak_vram_allocated_mb": max(
                float(fold["peak_vram_allocated_mb"] or 0.0) for fold in fold_rows
            ),
            "max_peak_vram_reserved_mb": max(
                float(fold["peak_vram_reserved_mb"] or 0.0) for fold in fold_rows
            ),
            "max_nvidia_smi_memory_mb": max(
                float(fold["nvidia_smi_peak_memory_used_mb"] or 0.0)
                for fold in fold_rows
            ),
            "nonfinite_or_incomplete_runs": 0,
        },
        "parameter_counts": parameter_counts,
        "overall_wmae": overall_models,
        "paired_overall_delta": paired,
        "per_target_mae": per_target,
        "attention_diagnostics": diagnostics,
        "execution_notes": execution_notes,
        "readout_equivalence": {
            "r1_zero_init_equals_r0": True,
            "r2_zero_init_equals_r0": True,
            "test_tolerance": {"atol": 1e-6, "rtol": 1e-6},
        },
    }


def _format_float(value: float, digits: int = 10) -> str:
    return f"{value:.{digits}g}"


def _render_markdown(summary: Mapping[str, Any]) -> str:
    lines = [
        "# Stage 3C — readout / property-specific pooling ablation",
        "",
        f"Formal source commit: `{summary['formal_source_commit']}`.",
        "",
        "R0 reuses the five historical Stage 3A.1 keep-dummy GINE runs; it was not retrained. R1 and R2 each add five frozen 5-fold OOF runs. The only scientific variable is readout.",
        "",
        "## Overall OOF wMAE",
        "",
        "| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model in ("R0", "R1", "R2"):
        item = summary["overall_wmae"][model]
        values = [item["by_seed"][str(seed)] for seed in FORMAL_SEEDS]
        lines.append(
            "| " + " | ".join(
                [
                    model,
                    *[_format_float(float(value), 10) for value in values],
                    f"{_format_float(item['mean'], 10)} ± {_format_float(item['sample_std'], 7)}",
                    f"{_format_float(item['min'], 10)}–{_format_float(item['max'], 10)}",
                ]
            ) + " |"
        )
    lines += [
        "",
        "## Paired overall deltas (left minus right)",
        "",
        "| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Left lower |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for comparison in ("R1-R0", "R2-R0", "R2-R1"):
        item = summary["paired_overall_delta"][comparison]
        values = [item["by_seed"][str(seed)] for seed in FORMAL_SEEDS]
        lines.append(
            "| " + " | ".join(
                [
                    comparison,
                    *[_format_float(float(value), 10) for value in values],
                    f"{_format_float(item['mean'], 10)} ± {_format_float(item['sample_std'], 7)}",
                    f"{item['left_lower_seed_count']}/5",
                ]
            ) + " |"
        )
    lines += [
        "",
        "## Per-target MAE and paired deltas",
        "",
        "| Target | R0 mean ± SD | R1 mean ± SD | R2 mean ± SD | R1−R0 delta mean ± SD | R2−R0 delta mean ± SD | R2−R1 delta mean ± SD |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        item = summary["per_target_mae"][target]
        model_text = [
            f"{_format_float(item['models'][model]['mean'], 8)} ± {_format_float(item['models'][model]['sample_std'], 5)}"
            for model in ("R0", "R1", "R2")
        ]
        delta_text = [
            f"{_format_float(item['paired_deltas'][comparison]['mean'], 8)} ± {_format_float(item['paired_deltas'][comparison]['sample_std'], 5)}"
            for comparison in ("R1-R0", "R2-R0", "R2-R1")
        ]
        lines.append("| " + " | ".join([target, *model_text, *delta_text]) + " |")
    lines += [
        "",
        "## Parameter counts",
        "",
        "| Model | Total trainable | Readout gate parameters | New vs R0 |",
        "|---|---:|---:|---:|",
    ]
    for model in ("R0", "R1", "R2"):
        item = summary["parameter_counts"][model]
        lines.append(
            f"| {model} | {item['total_trainable']} | {item['readout_gate_trainable']} | {item['new_parameters_vs_r0']} |"
        )
    lines += [
        "",
        "## Attention diagnostics",
        "",
        "Gate norms below are means over the five folds within each seed; entropy is normalized by log(node count) and averaged over OOF graphs.",
        "",
    ]
    for variant in ("R1", "R2"):
        item = summary["attention_diagnostics"][variant]
        lines.append(f"### {variant}")
        lines.append("")
        for gate, values in item["gate_l2_norm_mean_by_seed"].items():
            stats = _stats(values)
            lines.append(
                f"- Gate `{gate}` L2 norm by seed: "
                + ", ".join(f"{seed}={_format_float(values[str(seed)], 6)}" for seed in FORMAL_SEEDS)
                + f"; mean ± SD {_format_float(stats['mean'], 6)} ± {_format_float(stats['sample_std'], 5)}."
            )
        for target, values in item["normalized_attention_entropy_mean_by_target_by_seed"].items():
            stats = _stats(values)
            lines.append(
                f"- {target} normalized attention entropy: {_format_float(stats['mean'], 6)} ± {_format_float(stats['sample_std'], 5)}."
            )
        lines.append("")
    execution = summary["formal_execution"]
    lines += [
        "## Frozen setup and provenance",
        "",
        f"- Train SHA256: `{summary['train_data_sha256']}`",
        f"- Folds SHA256: `{summary['folds_sha256']}`",
        f"- Formal source commit: `{summary['formal_source_commit']}`",
        f"- R0: historical five seeds reused; new runs 0.",
        f"- R1/R2: {execution['r1_successful_runs']}/{execution['r2_successful_runs']} successful five-fold OOF runs; total new runs {execution['new_formal_runs']}/10.",
        f"- Sum of run-metadata training durations: {execution['training_runtime_seconds_from_run_metadata']:.1f} s ({execution['training_runtime_hours_from_run_metadata']:.2f} h).",
        f"- Formal queue wall time from RUN_START/RUN_SUCCESS records: {execution['batch_wall_runtime_seconds']:.0f} s ({execution['batch_wall_runtime_hours']:.2f} h), including post-run diagnostics.",
        f"- CUDA device(s): {', '.join(execution['device_names'])}.",
        f"- Peak PyTorch allocated/reserved VRAM: {execution['max_peak_vram_allocated_mb']:.1f}/{execution['max_peak_vram_reserved_mb']:.1f} MiB.",
        f"- Peak `nvidia-smi` memory: {execution['max_nvidia_smi_memory_mb']:.1f} MiB.",
        "- Initialization tests confirm R1 and R2 equal R0 within atol=rtol=1e-6; no R0 retraining was performed.",
        "- No attention tensors were saved per atom; only fold gate norms and per-target OOF mean entropy are retained.",
        "",
        "## Execution notes",
        "",
        f"- An initial R1 seed-42 attempt on `{summary['execution_notes']['initial_failed_attempt']['source_commit']}` completed five-fold training but failed while writing post-run attention diagnostics: `{summary['execution_notes']['initial_failed_attempt']['error']}`. It was excluded from formal results.",
        f"- The complete ten-run queue was then executed on `{summary['formal_source_commit']}` with the frozen configurations. Failed-attempt training scores were not reused; the failed attempt and original logs remain outside the synced formal artifacts.",
        f"- Final-source R1/R2 CUDA fold-0 smokes passed on seed {summary['execution_notes']['final_source_cuda_smoke']['seed']} for five epochs on the NVIDIA RTX 4070; graph-wise attention sums were one, activations and predictions were finite, gates updated, and smoke artifacts stayed isolated.",
        "",
        "## Interpretation",
        "",
        "",
    ]
    r1r0 = summary["paired_overall_delta"]["R1-R0"]
    r2r0 = summary["paired_overall_delta"]["R2-R0"]
    r2r1 = summary["paired_overall_delta"]["R2-R1"]
    lines.append(
        f"R1 − R0 mean paired delta is {_format_float(r1r0['mean'], 8)} (R1 lower in {r1r0['left_lower_seed_count']}/5 seeds). This small mixed-seed change does not establish a reliable benefit from learned shared node weighting."
    )
    lines.append(
        f"R2 − R0 is {_format_float(r2r0['mean'], 8)} (R2 lower in {r2r0['left_lower_seed_count']}/5 seeds); R2 − R1 is {_format_float(r2r1['mean'], 8)} (R2 lower in {r2r1['left_lower_seed_count']}/5 seeds). Property-specific pooling does not improve overall OOF wMAE over either baseline."
    )
    lines.append(
        "Per-target changes are mixed: R1 improves mean FFV and Rg MAE while increasing Tg, Tc, and Density; R2 improves only Tc and Rg mean MAE while increasing Tg, FFV, and Density. No readout shows consistent benefit across multiple properties."
    )
    lines += [
        "",
        "**Decision:** retain R0 (`global mean || global max`) as the next-stage readout and close the learned-pooling axis for this benchmark. These results do not start Stage 4.",
        "",
    ]
    return "\n".join(lines)


def _append_results_registry(summary: Mapping[str, Any]) -> None:
    result_path = TRACK_ROOT / "results.csv"
    with result_path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        existing = list(reader)
    rows_by_id = {row["experiment_id"]: row for row in existing}
    new_rows = []
    source_commit = summary["formal_source_commit"]
    readouts = (("R1", "r1_shared"), ("R2", "r2_property"))
    for model_name, variant in readouts:
        for seed in FORMAL_SEEDS:
            seed_key = str(seed)
            run_dir = Path(
                summary["source_provenance"][model_name]["run_directories_by_seed"][seed_key]
            )
            config = json.loads((TRACK_ROOT / run_dir / "config.json").read_text(encoding="utf-8"))
            metrics = json.loads((TRACK_ROOT / run_dir / "metrics.json").read_text(encoding="utf-8"))
            run_metadata = (TRACK_ROOT / run_dir / "run_metadata.json").resolve()
            experiment_id = config["experiment_id"] if seed == 42 else f"{config['experiment_id']}_seed_{seed}"
            new_row = {field: "" for field in fields}
            values = {
                "experiment_id": experiment_id,
                "model_name": config["model_name"],
                "category": "own_model",
                "benchmark_version": config["benchmark_version"],
                "git_commit": source_commit,
                "seed": str(seed),
                "oof_wmae": str(metrics["overall_oof_wmae"]),
                **{
                    f"{target.lower()}_mae": str(metrics["target_mae"][target])
                    for target in TARGETS
                },
                "status": "formal_oof_model",
                "notes": config["registry_notes"],
                "folds_sha256": EXPECTED_FOLDS_SHA256,
                "train_data_sha256": EXPECTED_TRAIN_SHA256,
                "run_metadata": str(run_metadata.relative_to(TRACK_ROOT)),
            }
            for field, value in values.items():
                if field in new_row:
                    new_row[field] = value
            previous = rows_by_id.get(experiment_id)
            if previous is not None:
                if previous != new_row:
                    raise ValueError(
                        f"Existing results.csv row conflicts with Stage 3C result {experiment_id}"
                    )
            else:
                new_rows.append(new_row)
                rows_by_id[experiment_id] = new_row
    if not new_rows:
        return
    temp_path = result_path.with_suffix(".csv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows([*existing, *new_rows])
    temp_path.replace(result_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--no-results-registry",
        action="store_true",
        help="Write aggregate reports without appending the 10 new rows to results.csv.",
    )
    args = parser.parse_args()
    summary = _build_summary()
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(EXPERIMENT_ROOT / "aggregate_summary.json", summary)
    (EXPERIMENT_ROOT / "aggregate_summary.md").write_text(
        _render_markdown(summary), encoding="utf-8"
    )
    if not args.no_results_registry:
        _append_results_registry(summary)
    print(
        json.dumps(
            {
                "summary": str(EXPERIMENT_ROOT / "aggregate_summary.json"),
                "runs": summary["formal_execution"]["new_formal_runs"],
                "source_commit": summary["formal_source_commit"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
