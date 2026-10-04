"""Validate and aggregate the five Stage 4B E seed runs, then update results.csv."""
from __future__ import annotations

import csv
import copy
import json
import math
import os
import statistics
from pathlib import Path
from typing import Any

from ..models.elemental_physical_priors.features import FEATURE_NAMES
from ..models.elemental_physical_priors.runner import (
    EXPERIMENT_ROOT,
    FEATURE_MANIFEST_PATH,
    FORMAL_SEEDS,
    MODEL_ROOT,
    _validate_config,
)
from ..src.data import sha256_file, write_json
from ..src.metrics import TARGETS

TRACK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRACK_ROOT.parent
RESULTS_PATH = TRACK_ROOT / "results.csv"
STAGE4A_SUMMARY = EXPERIMENT_ROOT.parent / "stage4a" / "aggregate_summary.json"


def _summary(values: dict[str, float]) -> dict[str, Any]:
    ordered = [float(values[str(seed)]) for seed in FORMAL_SEEDS]
    return {
        "by_seed": {str(seed): float(values[str(seed)]) for seed in FORMAL_SEEDS},
        "mean": statistics.fmean(ordered),
        "sample_std": statistics.stdev(ordered),
        "min": min(ordered),
        "max": max(ordered),
    }


def _paired_summary(values: dict[str, float]) -> dict[str, Any]:
    result = _summary(values)
    result["e_lower_seed_count"] = sum(float(values[str(seed)]) < 0 for seed in FORMAL_SEEDS)
    result["tie_seed_count"] = sum(float(values[str(seed)]) == 0 for seed in FORMAL_SEEDS)
    return result


def _formal_attempt(seed: int) -> Path:
    seed_root = MODEL_ROOT / "artifacts" / "formal" / f"seed_{seed}"
    complete = sorted(path for path in seed_root.glob("attempt_*") if (path / "run_metadata.json").is_file())
    if len(complete) != 1:
        raise RuntimeError(f"Expected one completed formal E run for seed {seed}, found {len(complete)}")
    return complete[0]


def _load_runs() -> dict[str, dict[str, Any]]:
    expected_train_sha = json.loads(
        (TRACK_ROOT / "benchmark" / "data_manifest.json").read_text(encoding="utf-8")
    )["source"]["sha256"]
    expected_folds_sha = json.loads(
        (TRACK_ROOT / "benchmark" / "data_manifest.json").read_text(encoding="utf-8")
    )["folds"]["sha256"]
    feature_hash = sha256_file(FEATURE_MANIFEST_PATH)
    runs: dict[str, dict[str, Any]] = {}
    source_commits: set[str] = set()
    for seed in FORMAL_SEEDS:
        path = _formal_attempt(seed)
        metadata = json.loads((path / "run_metadata.json").read_text(encoding="utf-8"))
        registry = json.loads((path / "registry_row.json").read_text(encoding="utf-8"))
        config = json.loads((path / "config.json").read_text(encoding="utf-8"))
        if int(metadata["seed"]) != seed or int(registry["seed"]) != seed:
            raise RuntimeError(f"Formal seed metadata mismatch under {path}")
        if int(config.get("seed", -1)) != seed:
            raise RuntimeError(f"Formal config seed mismatch under {path}")
        full_config = copy.deepcopy(config)
        # The checked-in config's seed is the baseline seed 42; the other
        # four values are the pre-registered per-run override only.
        full_config["seed"] = 42
        _validate_config(full_config)
        if metadata.get("stage") != "4B" or metadata.get("variant") != "E":
            raise RuntimeError(f"Unexpected formal model identity under {path}")
        if metadata["source_train_sha256"] != expected_train_sha:
            raise RuntimeError(f"Training data hash mismatch in seed {seed}")
        if metadata["benchmark_fold_sha256"] != expected_folds_sha:
            raise RuntimeError(f"Frozen folds hash mismatch in seed {seed}")
        if metadata["element_feature_manifest_sha256"] != feature_hash:
            raise RuntimeError(f"Feature manifest hash mismatch in seed {seed}")
        if registry["effective_config_sha256"] != sha256_file(path / "config.json"):
            raise RuntimeError(f"Effective config hash mismatch in seed {seed}")
        if registry["source_commit"] != metadata["git_commit"]:
            raise RuntimeError(f"Registry source commit mismatch in seed {seed}")
        validation = metadata["oof_validation"]
        if (
            int(validation["training_sample_count"]) != 7973
            or int(validation["prediction_row_count"]) != 7973
            or int(validation["duplicate_training_sample_id_count"]) != 0
            or int(validation["duplicate_prediction_sample_id_count"]) != 0
            or int(validation["missing_prediction_sample_id_count"]) != 0
            or int(validation["extra_prediction_sample_id_count"]) != 0
        ):
            raise RuntimeError(f"Stage 0 rejected formal OOF predictions for seed {seed}")
        source_commits.add(str(metadata["git_commit"]))
        runs[str(seed)] = {
            "directory": path,
            "metadata": metadata,
            "registry": registry,
            "config": config,
            "metrics": json.loads((path / "metrics.json").read_text(encoding="utf-8")),
            "fold_metrics": list(csv.DictReader((path / "fold_metrics.csv").open(encoding="utf-8-sig", newline=""))),
            "diagnostics": json.loads((path / "elemental_physical_diagnostics.json").read_text(encoding="utf-8")),
        }
    if len(source_commits) != 1:
        raise RuntimeError(f"Stage 4B seeds do not share one source commit: {sorted(source_commits)}")
    return runs


def _load_execution_notes(expected_commit: str, expected_train_sha: str) -> dict[str, Any]:
    notes_path = EXPERIMENT_ROOT / "formal_execution_notes.json"
    if not notes_path.is_file():
        raise FileNotFoundError(
            f"Stage 4B queue/resource notes are required before aggregation: {notes_path}"
        )
    notes = json.loads(notes_path.read_text(encoding="utf-8"))
    if notes.get("status") != "passed" or notes.get("concurrency") != 2:
        raise RuntimeError("Stage 4B formal execution notes do not confirm a successful two-slot run")
    if sorted(notes.get("completed_seeds", [])) != list(FORMAL_SEEDS):
        raise RuntimeError("Stage 4B execution notes do not cover exactly the five registered seeds")
    if notes.get("source_commit") != expected_commit:
        raise RuntimeError("Stage 4B execution monitor source commit differs from the formal runs")
    if notes.get("train_csv_sha256") != expected_train_sha:
        raise RuntimeError("Stage 4B execution monitor training-data hash differs from the formal runs")
    return notes


def _build_summary(runs: dict[str, dict[str, Any]], notes: dict[str, Any]) -> dict[str, Any]:
    stage4a = json.loads(STAGE4A_SUMMARY.read_text(encoding="utf-8"))
    if stage4a["train_data_sha256"] != runs["42"]["metadata"]["source_train_sha256"]:
        raise RuntimeError("Stage 4A G0 history uses a different frozen training snapshot")
    if stage4a["folds_sha256"] != runs["42"]["metadata"]["benchmark_fold_sha256"]:
        raise RuntimeError("Stage 4A G0 history uses different frozen folds")
    g0_overall = stage4a["overall_wmae"]["G0"]["by_seed"]
    e_overall = {
        seed: float(runs[seed]["metrics"]["overall_oof_wmae"]) for seed in runs
    }
    paired_overall = {seed: e_overall[seed] - float(g0_overall[seed]) for seed in e_overall}
    per_target: dict[str, Any] = {}
    for target in TARGETS:
        g0 = stage4a["per_target_mae"][target]["models"]["G0"]["by_seed"]
        e = {seed: float(runs[seed]["metrics"]["target_mae"][target]) for seed in runs}
        per_target[target] = {
            "G0": _summary(g0),
            "E": _summary(e),
            "E_minus_G0": _paired_summary(
                {seed: e[seed] - float(g0[seed]) for seed in e}
            ),
        }
    mean_delta = statistics.fmean(paired_overall.values())
    lower_count = sum(value < 0 for value in paired_overall.values())
    stable_gain = mean_delta < 0 and lower_count >= 4
    source_commit = next(iter(runs.values()))["metadata"]["git_commit"]
    fold_runtime = [
        fold
        for seed in FORMAL_SEEDS
        for fold in runs[str(seed)]["metadata"]["fold_metrics"]
    ]
    projection_norms = {
        seed: float(runs[seed]["diagnostics"]["projection_l2_norm_mean_over_folds"])
        for seed in runs
    }
    feature_column_norms = {
        name: {
            seed: float(
                runs[seed]["diagnostics"]["projection_column_l2_norm_mean_over_folds"][name]
            )
            for seed in runs
        }
        for name in FEATURE_NAMES
    }
    result = {
        "experiment": "Stage 4B elemental physical priors",
        "analysis_only": False,
        "seeds": list(FORMAL_SEEDS),
        "train_data_sha256": runs["42"]["metadata"]["source_train_sha256"],
        "folds_sha256": runs["42"]["metadata"]["benchmark_fold_sha256"],
        "formal_source_commit": source_commit,
        "G0_history": {
            "reused": True,
            "retrained_runs": 0,
            "source_summary": "Stage 4A aggregate; G0 is the historical keep-dummy Own-GNN Variant A",
            "overall_wmae": stage4a["overall_wmae"]["G0"],
        },
        "E_runs": {
            "new_formal_runs": len(runs),
            "by_seed": {
                seed: {
                    "run_directory": str(runs[seed]["directory"].relative_to(TRACK_ROOT)),
                    "oof_wmae": e_overall[seed],
                    "target_mae": runs[seed]["metrics"]["target_mae"],
                    "source_commit": runs[seed]["metadata"]["git_commit"],
                    "effective_config_sha256": runs[seed]["registry"]["effective_config_sha256"],
                }
                for seed in runs
            },
            "overall_wmae": _summary(e_overall),
        },
        "paired_overall_delta_E_minus_G0": _paired_summary(paired_overall),
        "per_target_mae": per_target,
        "parameter_counts": runs["42"]["metadata"]["parameter_count"],
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "reference_table_sha256": json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))[
            "reference_table_sha256"
        ],
        "observed_element_table_sha256": json.loads(
            FEATURE_MANIFEST_PATH.read_text(encoding="utf-8")
        )["coverage"]["observed_element_table_sha256"],
        "projection_diagnostics": {
            "final_W_phys_frobenius_l2_norm_mean_across_folds_by_seed": projection_norms,
            "feature_column_l2_norm_mean_across_folds_by_feature_and_seed": feature_column_norms,
            "selection_role": "diagnostic only; no feature selection or model selection",
        },
        "formal_execution": {
            "concurrency": notes["concurrency"],
            "formal_wall_seconds": notes["formal_wall_seconds"],
            "summed_fold_training_seconds": float(sum(float(item["duration_seconds"]) for item in fold_runtime)),
            "mean_gpu_utilization_percent": notes["resource_monitor"]["mean_gpu_utilization_percent"],
            "peak_gpu_utilization_percent": notes["resource_monitor"]["peak_gpu_utilization_percent"],
            "nvidia_smi_peak_vram_mb": notes["resource_monitor"]["peak_nvidia_smi_memory_used_mb"],
            "pytorch_peak_allocated_mb": max(float(item["peak_vram_allocated_mb"]) for item in fold_runtime),
            "pytorch_peak_reserved_mb": max(float(item["peak_vram_reserved_mb"]) for item in fold_runtime),
            "host_peak_cpu_percent": notes["resource_monitor"]["peak_host_cpu_percent"],
            "host_peak_ram_used_mb": notes["resource_monitor"]["peak_host_ram_used_mb"],
            "engineering_failures": notes.get("engineering_failures", []),
            "G0_retrained": 0,
            "E_runs": len(runs),
        },
        "stability_rule": "negative paired mean E-G0 and E lower in at least four of five seeds",
        "conclusion": {
            "stable_improvement": stable_gain,
            "E_lower_seed_count": lower_count,
            "paired_mean_delta": mean_delta,
            "carry_forward": "E" if stable_gain else "G0",
            "stop_simple_2d_feature_engineering": not stable_gain,
            "interpretation": (
                "Elemental physical priors show a stable OOF improvement under the preregistered rule."
                if stable_gain
                else "Elemental physical priors show no stable OOF improvement; retain G0 and stop further simple 2D feature engineering for now."
            ),
        },
    }
    return result


def _update_results_registry(runs: dict[str, dict[str, Any]]) -> None:
    with RESULTS_PATH.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fields = list(reader.fieldnames or [])
        rows = list(reader)
    train_hash = runs["42"]["metadata"]["source_train_sha256"]
    folds_hash = runs["42"]["metadata"]["benchmark_fold_sha256"]
    for seed in FORMAL_SEEDS:
        key = str(seed)
        run = runs[key]
        metrics = run["metrics"]
        values = {
            "experiment_id": f"own_gnn_elemental_physical_priors_seed_{seed}",
            "model_name": "Own-GNN E — elemental physical priors",
            "category": "own_model",
            "benchmark_version": "nopp2025_train_v1",
            "git_commit": run["metadata"]["git_commit"],
            "seed": str(seed),
            "oof_wmae": str(metrics["overall_oof_wmae"]),
            "tg_mae": str(metrics["target_mae"]["Tg"]),
            "ffv_mae": str(metrics["target_mae"]["FFV"]),
            "tc_mae": str(metrics["target_mae"]["Tc"]),
            "density_mae": str(metrics["target_mae"]["Density"]),
            "rg_mae": str(metrics["target_mae"]["Rg"]),
            "status": "formal_oof_model",
            "notes": "Stage 4B E; fixed node-level RDKit elemental physical priors; frozen G0 graph and training protocol.",
            "folds_sha256": folds_hash,
            "train_data_sha256": train_hash,
            "run_metadata": str((run["directory"] / "run_metadata.json").relative_to(TRACK_ROOT)),
        }
        row = {field: values.get(field, "") for field in fields}
        matches = [index for index, existing in enumerate(rows) if existing.get("experiment_id") == values["experiment_id"]]
        if len(matches) > 1:
            raise RuntimeError(f"results.csv contains duplicate {values['experiment_id']}")
        if matches:
            rows[matches[0]] = row
        else:
            rows.append(row)
    temp_path = RESULTS_PATH.with_suffix(".csv.tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp_path, RESULTS_PATH)


def _write_markdown(summary: dict[str, Any], path: Path) -> None:
    overall_g0 = summary["G0_history"]["overall_wmae"]["by_seed"]
    overall_e = summary["E_runs"]["overall_wmae"]["by_seed"]
    paired = summary["paired_overall_delta_E_minus_G0"]["by_seed"]
    lines = [
        "# Stage 4B — elemental physical priors",
        "",
        f"Formal source commit: `{summary['formal_source_commit']}`. G0 reuses its frozen five historical seeds; E completed five new five-fold OOF runs.",
        "",
        "## Overall OOF wMAE",
        "",
        "| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for label, stats in (("G0", summary["G0_history"]["overall_wmae"]), ("E", summary["E_runs"]["overall_wmae"])):
        vals = stats["by_seed"]
        lines.append(
            f"| {label} | "
            + " | ".join(f"{float(vals[str(seed)]):.10f}" for seed in FORMAL_SEEDS)
            + f" | {stats['mean']:.10f} ± {stats['sample_std']:.10f} | {stats['min']:.10f}–{stats['max']:.10f} |"
        )
    delta_stats = summary["paired_overall_delta_E_minus_G0"]
    lines += [
        "",
        "## Paired E − G0 wMAE",
        "",
        "| Seed 42 | Seed 43 | Seed 44 | Seed 45 | Seed 46 | Mean ± sample SD | E lower |",
        "|---:|---:|---:|---:|---:|---:|---:|",
        "| " + " | ".join(f"{float(paired[str(seed)]):+.10f}" for seed in FORMAL_SEEDS)
        + f" | {delta_stats['mean']:+.10f} ± {delta_stats['sample_std']:.10f} | {delta_stats['e_lower_seed_count']}/5 |",
        "",
        "## Per-target MAE",
        "",
        "| Target | G0 mean ± SD | E mean ± SD | E − G0 paired mean ± SD |",
        "|---|---:|---:|---:|",
    ]
    for target in TARGETS:
        item = summary["per_target_mae"][target]
        lines.append(
            f"| {target} | {item['G0']['mean']:.10g} ± {item['G0']['sample_std']:.5g} | "
            f"{item['E']['mean']:.10g} ± {item['E']['sample_std']:.5g} | "
            f"{item['E_minus_G0']['mean']:+.10g} ± {item['E_minus_G0']['sample_std']:.5g} |"
        )
    lines += [
        "",
        "## Execution and parameter counts",
        "",
        f"- New formal runs: {summary['formal_execution']['E_runs']}; G0 retrains: 0.",
        f"- Concurrency: {summary['formal_execution']['concurrency']}; formal wall: {summary['formal_execution']['formal_wall_seconds']} s; summed fold training: {summary['formal_execution']['summed_fold_training_seconds']:.1f} s.",
        f"- GPU utilization mean/peak: {summary['formal_execution']['mean_gpu_utilization_percent']}% / {summary['formal_execution']['peak_gpu_utilization_percent']}%; `nvidia-smi` peak VRAM: {summary['formal_execution']['nvidia_smi_peak_vram_mb']} MiB.",
        f"- PyTorch peak allocated/reserved VRAM: {summary['formal_execution']['pytorch_peak_allocated_mb']} / {summary['formal_execution']['pytorch_peak_reserved_mb']} MiB.",
        f"- Host CPU/RAM peak: {summary['formal_execution']['host_peak_cpu_percent']}% / {summary['formal_execution']['host_peak_ram_used_mb']} MiB.",
        f"- Parameters: G0 {summary['parameter_counts']['G0_total_trainable']:,}; E {summary['parameter_counts']['E_total_trainable']:,}; added {summary['parameter_counts']['physical_projection_trainable']:,}.",
        "",
        "## Projection diagnostics",
        "",
        "| Seed | Mean fold W_phys Frobenius norm | " + " | ".join(FEATURE_NAMES) + " |",
        "|---:|---:|" + "---:|" * len(FEATURE_NAMES),
    ]
    norms = summary["projection_diagnostics"]["final_W_phys_frobenius_l2_norm_mean_across_folds_by_seed"]
    columns = summary["projection_diagnostics"]["feature_column_l2_norm_mean_across_folds_by_feature_and_seed"]
    for seed in FORMAL_SEEDS:
        lines.append(
            f"| {seed} | {norms[str(seed)]:.6f} | "
            + " | ".join(f"{columns[name][str(seed)]:.6f}" for name in FEATURE_NAMES)
            + " |"
        )
    lines += [
        "",
        "## Conclusion",
        "",
        summary["conclusion"]["interpretation"],
        "",
        f"Carry-forward candidate: **{summary['conclusion']['carry_forward']}**. Stable improvement under the preregistered rule: **{summary['conclusion']['stable_improvement']}**.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    runs = _load_runs()
    expected_commit = runs["42"]["metadata"]["git_commit"]
    expected_train_sha = runs["42"]["metadata"]["source_train_sha256"]
    notes = _load_execution_notes(expected_commit, expected_train_sha)
    summary = _build_summary(runs, notes)
    _update_results_registry(runs)
    EXPERIMENT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(EXPERIMENT_ROOT / "aggregate_summary.json", summary)
    _write_markdown(summary, EXPERIMENT_ROOT / "aggregate_summary.md")
    print(json.dumps(summary["conclusion"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
