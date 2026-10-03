#!/usr/bin/env python3
"""Aggregate the preregistered Stage 3A.1 paired-seed registry rows.

This is an analysis-only tool. It consumes completed, validated OOF metrics from
results.csv; it never reads predictions or launches training.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import statistics
from typing import Any

SEEDS = (42, 43, 44, 45, 46)
TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"
FOLDS_SHA256 = "1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a"
MODELS = {
    "own_gnn_v0_frozen_oof_v1": "Own-GNN v0",
    "own_gnn_repr_keep_dummy": "Variant A",
    "own_gnn_repr_endpoint_marker": "Variant B",
}
TARGET_COLUMNS = {
    "Tg": "tg_mae",
    "FFV": "ffv_mae",
    "Tc": "tc_mae",
    "Density": "density_mae",
    "Rg": "rg_mae",
}
OVERALL_COLUMN = "oof_wmae"


def _num(value: str | None, *, field: str, experiment_id: str) -> float:
    try:
        result = float(value or "")
    except ValueError as exc:
        raise ValueError(f"{experiment_id}: missing or invalid {field}: {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{experiment_id}: {field} must be finite")
    return result


def _stats(values: list[float]) -> dict[str, float | None]:
    return {
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values) if len(values) > 1 else None,
        "min": min(values),
        "max": max(values),
    }


def _delta_summary(values: dict[int, float]) -> dict[str, Any]:
    deltas = [values[seed] for seed in SEEDS]
    return {
        "by_seed": {str(seed): values[seed] for seed in SEEDS},
        "mean": statistics.mean(deltas),
        "sample_std": statistics.stdev(deltas),
        "positive_count": sum(value > 0 for value in deltas),
        "negative_count": sum(value < 0 for value in deltas),
        "tie_count": sum(value == 0 for value in deltas),
    }


def load_and_validate_registry(path: str | Path) -> dict[str, dict[int, dict[str, Any]]]:
    """Return metrics keyed by model ID and seed, rejecting incomplete/mixed runs."""
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {
            "experiment_id", "seed", "status", "folds_sha256", "train_data_sha256",
            OVERALL_COLUMN, *TARGET_COLUMNS.values(),
        }
        missing = sorted(required.difference(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Registry is missing columns: {missing}")
        rows: dict[str, dict[int, dict[str, Any]]] = {model: {} for model in MODELS}
        for row in reader:
            experiment_id = row.get("experiment_id", "")
            model = next(
                (
                    candidate
                    for candidate in MODELS
                    if experiment_id == candidate
                    or experiment_id.startswith(f"{candidate}_seed_")
                ),
                None,
            )
            if model is None:
                continue
            if row.get("status") != "formal_oof_model":
                raise ValueError(f"{model}: expected status formal_oof_model")
            try:
                seed = int(row.get("seed", ""))
            except ValueError as exc:
                raise ValueError(f"{model}: invalid seed {row.get('seed')!r}") from exc
            if seed not in SEEDS:
                raise ValueError(f"{model}: unexpected seed {seed}")
            expected_id = model if seed == 42 else f"{model}_seed_{seed}"
            if experiment_id != expected_id:
                raise ValueError(
                    f"{model} seed {seed}: expected experiment_id {expected_id!r}, "
                    f"got {experiment_id!r}"
                )
            if seed in rows[model]:
                raise ValueError(f"duplicate registry row for {model}, seed {seed}")
            if row.get("train_data_sha256") != TRAIN_SHA256:
                raise ValueError(f"{model} seed {seed}: train SHA256 mismatch")
            if row.get("folds_sha256") != FOLDS_SHA256:
                raise ValueError(f"{model} seed {seed}: folds SHA256 mismatch")
            metrics = {"oof_wmae": _num(row.get(OVERALL_COLUMN), field=OVERALL_COLUMN, experiment_id=model)}
            metrics.update({
                target: _num(row.get(column), field=column, experiment_id=model)
                for target, column in TARGET_COLUMNS.items()
            })
            rows[model][seed] = metrics
    expected = set(SEEDS)
    problems = [f"{model}: missing seeds {sorted(expected.difference(values))}"
                for model, values in rows.items() if set(values) != expected]
    if problems:
        raise ValueError("Incomplete paired-seed registry:\n- " + "\n- ".join(problems))
    return rows


def aggregate_registry(path: str | Path) -> dict[str, Any]:
    runs = load_and_validate_registry(path)
    overall = {
        MODELS[model]: {
            "by_seed": {str(seed): runs[model][seed]["oof_wmae"] for seed in SEEDS},
            **_stats([runs[model][seed]["oof_wmae"] for seed in SEEDS]),
        }
        for model in MODELS
    }
    v0, a, b = ("own_gnn_v0_frozen_oof_v1", "own_gnn_repr_keep_dummy", "own_gnn_repr_endpoint_marker")
    paired: dict[str, Any] = {}
    for label, left, right in (("A-v0", a, v0), ("B-v0", b, v0), ("A-B", a, b)):
        deltas = {seed: runs[left][seed]["oof_wmae"] - runs[right][seed]["oof_wmae"] for seed in SEEDS}
        paired[label] = _delta_summary(deltas)
    per_target: dict[str, Any] = {}
    for target in TARGET_COLUMNS:
        by_model = {
            MODELS[model]: {
                "by_seed": {str(seed): runs[model][seed][target] for seed in SEEDS},
                **_stats([runs[model][seed][target] for seed in SEEDS]),
            }
            for model in MODELS
        }
        per_target[target] = {
            "by_model": by_model,
            "paired_delta": {
                name: _delta_summary({
                    seed: runs[left][seed][target] - runs[right][seed][target]
                    for seed in SEEDS
                })
                for name, left, right in (("A-v0", a, v0), ("B-v0", b, v0))
            },
        }
    return {
        "experiment": "Stage 3A.1 paired-seed representation confirmation",
        "analysis_only": True,
        "seeds": list(SEEDS),
        "train_data_sha256": TRAIN_SHA256,
        "folds_sha256": FOLDS_SHA256,
        "overall_wmae": overall,
        "paired_overall_delta": paired,
        "per_target_mae": per_target,
    }


def render_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Stage 3A.1 paired-seed summary", "",
        f"Seeds: {', '.join(map(str, result['seeds']))}. Sample standard deviations (ddof=1).",
        "", "## Overall OOF wMAE", "", "| Model | Mean | SD | Min | Max |", "|---|---:|---:|---:|---:|",
    ]
    for model, stats in result["overall_wmae"].items():
        lines.append(f"| {model} | {stats['mean']:.10f} | {stats['sample_std']:.10f} | {stats['min']:.10f} | {stats['max']:.10f} |")
    lines += ["", "## Paired overall deltas (left minus right)", "", "| Comparison | " + " | ".join(map(str, result["seeds"])) + " | Mean | SD | + / − / tie |", "|---|" + "---:|" * (len(result["seeds"]) + 2)]
    for name, stats in result["paired_overall_delta"].items():
        values = " | ".join(f"{stats['by_seed'][str(seed)]:.10f}" for seed in result["seeds"])
        lines.append(f"| {name} | {values} | {stats['mean']:.10f} | {stats['sample_std']:.10f} | {stats['positive_count']} / {stats['negative_count']} / {stats['tie_count']} |")
    lines += ["", "## Per-target MAE", "", "Mean ± sample SD across seeds; paired deltas are left minus v0.", "", "| Target | v0 | A | B | A−v0 mean delta | B−v0 mean delta |", "|---|---:|---:|---:|---:|---:|"]
    for target, item in result["per_target_mae"].items():
        cells = []
        for name in MODELS.values():
            stats = item["by_model"][name]
            cells.append(f"{stats['mean']:.6g} ± {stats['sample_std']:.3g}")
        da = item["paired_delta"]["A-v0"]["mean"]
        db = item["paired_delta"]["B-v0"]["mean"]
        lines.append(f"| {target} | {cells[0]} | {cells[1]} | {cells[2]} | {da:.6g} | {db:.6g} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-csv", type=Path, default=root / "results.csv")
    parser.add_argument("--output-json", type=Path, default=root / "experiments/stage3a1/paired_summary.json")
    parser.add_argument("--output-md", type=Path, default=root / "experiments/stage3a1/paired_summary.md")
    args = parser.parse_args()
    result = aggregate_registry(args.results_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(result), encoding="utf-8")
    print(f"Wrote {args.output_json} and {args.output_md}")


if __name__ == "__main__":
    main()
