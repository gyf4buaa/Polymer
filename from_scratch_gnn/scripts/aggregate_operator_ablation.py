"""Aggregate the pre-registered Stage 3B operator comparison."""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Mapping, Sequence

from from_scratch_gnn.src.data import sha256_file, write_json
from from_scratch_gnn.models.polymer_representation_ablation.model import OwnGNNRepresentation

TRACK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRACK_ROOT.parent
BASELINE_CONFIG = REPOSITORY_ROOT / "from_scratch_gnn" / "models" / "own_gnn_repr_keep_dummy" / "config.json"
BASELINE_SUMMARY = TRACK_ROOT / "experiments" / "stage3a1" / "paired_summary.json"
SEEDS = (42, 43, 44, 45, 46)
TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"
FOLDS_SHA256 = "1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a"
TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")
OPERATORS = ("gatv2", "pna")


def _stats(values: Sequence[float]) -> dict[str, Any]:
    if len(values) != len(SEEDS):
        raise ValueError(f"Expected {len(SEEDS)} paired seed values, got {len(values)}")
    return {
        "by_seed": {str(seed): float(value) for seed, value in zip(SEEDS, values)},
        "mean": statistics.mean(values),
        "sample_std": statistics.stdev(values),
        "min": min(values),
        "max": max(values),
    }


def _paired(left: Sequence[float], right: Sequence[float]) -> dict[str, Any]:
    deltas = [float(a - b) for a, b in zip(left, right)]
    result = _stats(deltas)
    result.update(
        {
            "left_lower_count": sum(value < 0 for value in deltas),
            "right_lower_count": sum(value > 0 for value in deltas),
            "tie_count": sum(value == 0 for value in deltas),
            "sign_convention": "left minus right; negative favors left",
        }
    )
    return result


def _load_complete_run(operator: str, seed: int, *, source_commit: str | None) -> dict[str, Any]:
    root = TRACK_ROOT / "models" / "operator_ablation" / operator / "artifacts" / f"seed_{seed}"
    required = ("metrics.json", "config.json", "source_manifest.json", "run_metadata.json", "fold_metrics.csv", "oof_predictions.csv", "operator_metadata.json", "registry_row.json")
    missing = [name for name in required if not (root / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Incomplete formal {operator} seed {seed}: missing {missing}")
    metrics = json.loads((root / "metrics.json").read_text(encoding="utf-8"))
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    run_metadata = json.loads((root / "run_metadata.json").read_text(encoding="utf-8"))
    operator_metadata = json.loads((root / "operator_metadata.json").read_text(encoding="utf-8"))
    frozen_config = json.loads(BASELINE_CONFIG.read_text(encoding="utf-8"))
    if metrics.get("n_samples") != 7973 or metrics.get("overall_oof_wmae") is None:
        raise ValueError(f"{operator} seed {seed} has an invalid Stage 0 metric artifact")
    if metrics.get("validation", {}).get("prediction_row_count") != 7973:
        raise ValueError(f"{operator} seed {seed} did not pass complete Stage 0 OOF validation")
    if metrics.get("validation", {}).get("truth_source_sha256") != TRAIN_SHA256:
        raise ValueError(f"{operator} seed {seed} OOF validation has the wrong train hash")
    if manifest.get("benchmark_files_sha256", {}).get("train_csv") != TRAIN_SHA256:
        raise ValueError(f"{operator} seed {seed} source manifest has the wrong train hash")
    if manifest.get("benchmark_files_sha256", {}).get("folds_csv") != FOLDS_SHA256:
        raise ValueError(f"{operator} seed {seed} source manifest has the wrong folds hash")
    if int(config.get("seed", -1)) != seed or int(manifest.get("seed", -1)) != seed:
        raise ValueError(f"{operator} seed {seed} artifact identity mismatch")
    if config.get("model", {}).get("operator") != operator:
        raise ValueError(f"{operator} seed {seed} config operator mismatch")
    if config.get("training") != frozen_config["training"] or config.get("graph") != frozen_config["graph"]:
        raise ValueError(f"{operator} seed {seed} changed frozen training or graph settings")
    if any(
        config.get("model", {}).get(key) != frozen_config["model"].get(key)
        for key in ("hidden_dim", "num_layers", "dropout", "residual", "normalization", "pooling", "property_heads")
    ):
        raise ValueError(f"{operator} seed {seed} changed a frozen model setting")
    if int(config["model"]["hidden_dim"]) != 256 or int(config["model"]["num_layers"]) != 4:
        raise ValueError(f"{operator} seed {seed} violated the frozen 256×4 dimensions")
    if manifest.get("effective_config_sha256") != sha256_file(root / "config.json"):
        raise ValueError(f"{operator} seed {seed} effective config hash mismatch")
    if operator_metadata.get("graph_schema") != frozen_config["graph"]["schema"]:
        raise ValueError(f"{operator} seed {seed} graph schema mismatch")
    if len(operator_metadata.get("message_passing_output_widths", [])) != 4 or any(
        width != 256 for width in operator_metadata["message_passing_output_widths"]
    ):
        raise ValueError(f"{operator} seed {seed} changed message-passing output width/depth")
    recorded_commit = manifest.get("git_commit")
    if not recorded_commit or run_metadata.get("git_commit") != recorded_commit:
        raise ValueError(f"{operator} seed {seed} source commit provenance mismatch")
    if source_commit is not None and recorded_commit != source_commit:
        raise ValueError(f"{operator} seed {seed} was trained from a different source commit")
    return {
        "operator": operator,
        "seed": seed,
        "root": root,
        "metrics": metrics,
        "config": config,
        "manifest": manifest,
        "run_metadata": run_metadata,
        "operator_metadata": operator_metadata,
    }


def aggregate_operator_ablation(
    *,
    baseline_path: Path = BASELINE_SUMMARY,
    source_commit: str | None = None,
) -> dict[str, Any]:
    baseline_path = baseline_path.resolve()
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    if baseline.get("analysis_only") is not True:
        raise ValueError("Stage 3A.1 summary must be consumed as historical baseline input")
    if baseline.get("train_data_sha256") != TRAIN_SHA256 or baseline.get("folds_sha256") != FOLDS_SHA256:
        raise ValueError("Stage 3A.1 baseline does not match the frozen data/folds hashes")
    if baseline.get("seeds") != list(SEEDS):
        raise ValueError("Stage 3A.1 baseline does not contain the pre-registered seed set")

    runs = {
        operator: {
            seed: _load_complete_run(operator, seed, source_commit=source_commit)
            for seed in SEEDS
        }
        for operator in OPERATORS
    }
    commits = {
        run["manifest"]["git_commit"]
        for operator_runs in runs.values()
        for run in operator_runs.values()
    }
    if len(commits) != 1:
        raise ValueError(f"Formal runs do not share one source commit: {sorted(commits)}")
    formal_commit = next(iter(commits))
    graph_fingerprints = {
        run["manifest"]["graph_provenance"]["graph_set_fingerprint_sha256"]
        for operator_runs in runs.values()
        for run in operator_runs.values()
    }
    if len(graph_fingerprints) != 1 or None in graph_fingerprints:
        raise ValueError("Formal operators did not consume the identical frozen graph set")
    for operator in OPERATORS:
        operator_configs = {
            json.dumps(runs[operator][seed]["config"]["model"]["operator_config"], sort_keys=True)
            for seed in SEEDS
        }
        if len(operator_configs) != 1:
            raise ValueError(f"{operator} operator config differs across paired seeds")
        parameter_counts = {
            (
                runs[operator][seed]["operator_metadata"].get("trainable_parameter_count"),
                runs[operator][seed]["operator_metadata"].get("message_passing_block_parameter_count"),
            )
            for seed in SEEDS
        }
        if len(parameter_counts) != 1:
            raise ValueError(f"{operator} parameter counts differ across seeds")

    overall: dict[str, dict[str, Any]] = {
        "GINE": baseline["overall_wmae"]["Variant A"]
    }
    for operator, display in (("gatv2", "GATv2"), ("pna", "PNA")):
        values = [float(runs[operator][seed]["metrics"]["overall_oof_wmae"]) for seed in SEEDS]
        overall[display] = _stats(values)
    paired_overall = {
        "GATv2-GINE": _paired(
            [overall["GATv2"]["by_seed"][str(seed)] for seed in SEEDS],
            [overall["GINE"]["by_seed"][str(seed)] for seed in SEEDS],
        ),
        "PNA-GINE": _paired(
            [overall["PNA"]["by_seed"][str(seed)] for seed in SEEDS],
            [overall["GINE"]["by_seed"][str(seed)] for seed in SEEDS],
        ),
        "GATv2-PNA": _paired(
            [overall["GATv2"]["by_seed"][str(seed)] for seed in SEEDS],
            [overall["PNA"]["by_seed"][str(seed)] for seed in SEEDS],
        ),
    }

    per_target: dict[str, Any] = {}
    for target in TARGETS:
        values_by_model: dict[str, list[float]] = {
            "GINE": [
                float(baseline["per_target_mae"][target]["by_model"]["Variant A"]["by_seed"][str(seed)])
                for seed in SEEDS
            ]
        }
        for operator, display in (("gatv2", "GATv2"), ("pna", "PNA")):
            values_by_model[display] = [
                float(runs[operator][seed]["metrics"]["target_mae"][target])
                for seed in SEEDS
            ]
        per_target[target] = {
            "by_model": {name: _stats(values) for name, values in values_by_model.items()},
            "paired_delta": {
                "GATv2-GINE": _paired(values_by_model["GATv2"], values_by_model["GINE"]),
                "PNA-GINE": _paired(values_by_model["PNA"], values_by_model["GINE"]),
            },
        }

    formal_runs = [runs[operator][seed] for operator in OPERATORS for seed in SEEDS]
    runtimes = [float(item["run_metadata"]["duration_seconds"]) for item in formal_runs]
    memory_allocated = [
        float(fold["peak_vram_allocated_mb"])
        for item in formal_runs
        for fold in item["run_metadata"]["fold_metrics"]
        if fold.get("peak_vram_allocated_mb") is not None
    ]
    memory_reserved = [
        float(fold["peak_vram_reserved_mb"])
        for item in formal_runs
        for fold in item["run_metadata"]["fold_metrics"]
        if fold.get("peak_vram_reserved_mb") is not None
    ]
    nvidia_memory = [
        float(fold["nvidia_smi_peak_memory_used_mb"])
        for item in formal_runs
        for fold in item["run_metadata"]["fold_metrics"]
        if fold.get("nvidia_smi_peak_memory_used_mb") is not None
    ]
    gpu_samples = [
        int(item["run_metadata"].get("gpu_sampling", {}).get("gpu_utilization_sample_count", 0))
        for item in formal_runs
    ]
    runtime = {
        "formal_run_count": len(formal_runs),
        "successful_runs": {operator: len(runs[operator]) for operator in OPERATORS},
        "total_seconds": sum(runtimes),
        "mean_run_seconds": statistics.mean(runtimes),
        "maximum_pytorch_peak_allocated_mb": max(memory_allocated, default=None),
        "maximum_pytorch_peak_reserved_mb": max(memory_reserved, default=None),
        "maximum_nvidia_smi_observed_memory_mb": max(nvidia_memory, default=None),
        "gpu_utilization_sample_count_per_run": gpu_samples,
        "gpu_utilization": {
            operator: {
                "by_seed": {
                    str(seed): runs[operator][seed]["run_metadata"].get("gpu_sampling")
                    for seed in SEEDS
                }
            }
            for operator in OPERATORS
        },
    }
    gine = OwnGNNRepresentation()
    parameter_counts: dict[str, Any] = {
        "GINE": {
            "operator": "historical GINEConv keep-dummy Variant A",
            "trainable_parameter_count": sum(
                parameter.numel() for parameter in gine.parameters() if parameter.requires_grad
            ),
            "message_passing_block_parameter_count": sum(
                parameter.numel()
                for block in gine.convs
                for parameter in block.parameters()
                if parameter.requires_grad
            ),
        }
    }
    operator_configs: dict[str, Any] = {}
    effective_config_hashes: dict[str, Any] = {}
    degree_provenance: dict[str, Any] | None = None
    for display, operator in (("GINE", None), ("GATv2", "gatv2"), ("PNA", "pna")):
        if operator is None:
            continue
        first = runs[operator][42]
        parameter_counts[display] = first["operator_metadata"]
        operator_configs[display] = first["config"]["model"]["operator_config"]
        effective_config_hashes[display] = {
            str(seed): sha256_file(runs[operator][seed]["root"] / "config.json")
            for seed in SEEDS
        }
        if operator == "pna":
            degree_path = first["root"] / "degree_histogram.json"
            degree_provenance = json.loads(degree_path.read_text(encoding="utf-8"))
            if (
                degree_provenance.get("graph_count") != 7973
                or degree_provenance.get("source_train_sha256") != TRAIN_SHA256
                or degree_provenance.get("uses_labels_or_targets") is not False
                or degree_provenance.get("graph_set_fingerprint_sha256") not in graph_fingerprints
            ):
                raise ValueError("PNA degree statistics are not from the full frozen graph set")
            degree_hashes = {
                sha256_file(runs[operator][seed]["root"] / "degree_histogram.json")
                for seed in SEEDS
            }
            if len(degree_hashes) != 1:
                raise ValueError("PNA degree histogram provenance differs across seeds")

    return {
        "experiment": "Stage 3B message-passing operator ablation",
        "status": "complete",
        "seeds": list(SEEDS),
        "formal_source_commit": formal_commit,
        "train_sha256": TRAIN_SHA256,
        "folds_sha256": FOLDS_SHA256,
        "baseline_source": {
            "stage3a1_paired_summary": str(baseline_path),
            "stage3a1_paired_summary_sha256": sha256_file(baseline_path),
            "gine_retrained": False,
        },
        "overall_wmae": overall,
        "paired_overall_delta": paired_overall,
        "per_target_mae": per_target,
        "runtime": runtime,
        "parameter_count": parameter_counts,
        "operator_config": operator_configs,
        "effective_config_sha256": effective_config_hashes,
        "pna_degree_statistics": degree_provenance,
        "artifact_roots": {
            operator: {
                str(seed): str(runs[operator][seed]["root"].relative_to(TRACK_ROOT))
                for seed in SEEDS
            }
            for operator in OPERATORS
        },
        "selection_policy": "paired seed results on the frozen benchmark; no tuning or post-result reruns",
    }


def write_markdown(summary: Mapping[str, Any], path: Path) -> None:
    lines = [
        "# Stage 3B — Message-passing operator ablation",
        "",
        f"Formal source commit: `{summary['formal_source_commit']}`.",
        "",
        "GINE is the historical keep-dummy Variant A baseline from Stage 3A/3A.1; it was not retrained. Negative paired deltas favor the left-hand operator.",
        "",
        "## Overall OOF wMAE",
        "",
        "| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for model, values in summary["overall_wmae"].items():
        per_seed = " | ".join(f"{values['by_seed'][str(seed)]:.10f}" for seed in SEEDS)
        lines.append(
            f"| {model} | {per_seed} | {values['mean']:.10f} ± {values['sample_std']:.10f} | {values['min']:.10f}–{values['max']:.10f} |"
        )
    lines.extend(
        [
            "",
            "## Paired overall deltas",
            "",
            "| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Left lower |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for comparison, values in summary["paired_overall_delta"].items():
        per_seed = " | ".join(f"{values['by_seed'][str(seed)]:+.10f}" for seed in SEEDS)
        lines.append(
            f"| {comparison} | {per_seed} | {values['mean']:+.10f} ± {values['sample_std']:.10f} | {values['left_lower_count']}/5 |"
        )
    lines.extend(
        [
            "",
            "## Per-target MAE",
            "",
            "| Target | GINE mean ± SD | GATv2 mean ± SD | PNA mean ± SD | GATv2 − GINE | PNA − GINE |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for target, values in summary["per_target_mae"].items():
        gine = values["by_model"]["GINE"]
        gat = values["by_model"]["GATv2"]
        pna = values["by_model"]["PNA"]
        gd = values["paired_delta"]["GATv2-GINE"]
        pd = values["paired_delta"]["PNA-GINE"]
        lines.append(
            f"| {target} | {gine['mean']:.8g} ± {gine['sample_std']:.4g} | {gat['mean']:.8g} ± {gat['sample_std']:.4g} | {pna['mean']:.8g} ± {pna['sample_std']:.4g} | {gd['mean']:+.8g} ± {gd['sample_std']:.4g} | {pd['mean']:+.8g} ± {pd['sample_std']:.4g} |"
        )
    lines.extend(
        [
            "",
            "## Frozen setup and provenance",
            "",
            f"- Train SHA256: `{summary['train_sha256']}`",
            f"- Folds SHA256: `{summary['folds_sha256']}`",
            f"- Formal OOF runs: GATv2 {summary['runtime']['successful_runs']['gatv2']}/5; PNA {summary['runtime']['successful_runs']['pna']}/5.",
            f"- Total formal runtime: {summary['runtime']['total_seconds']:.1f} seconds.",
            f"- Maximum PyTorch allocated VRAM: {summary['runtime']['maximum_pytorch_peak_allocated_mb']:.1f} MiB; maximum `nvidia-smi` observed memory: {summary['runtime']['maximum_nvidia_smi_observed_memory_mb']:.0f} MiB.",
            f"- GATv2: {summary['operator_config']['GATv2']}.",
            f"- PNA: {summary['operator_config']['PNA']}.",
            f"- PNA degree histogram (entire frozen graph set): `{summary['pna_degree_statistics']['histogram_by_degree']}`; graph fingerprint `{summary['pna_degree_statistics']['graph_set_fingerprint_sha256']}`.",
            "- All 10 formal runs use the same source commit; GINE's existing five seeds were reused.",
            "- No readout, global feature, descriptor, ensemble, or later-stage experiments were run.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-summary", type=Path, default=BASELINE_SUMMARY)
    parser.add_argument("--source-commit")
    parser.add_argument(
        "--output-json", type=Path, default=TRACK_ROOT / "experiments" / "stage3b" / "aggregate_summary.json"
    )
    parser.add_argument(
        "--output-markdown", type=Path, default=TRACK_ROOT / "experiments" / "stage3b" / "aggregate_summary.md"
    )
    args = parser.parse_args()
    result = aggregate_operator_ablation(
        baseline_path=args.baseline_summary, source_commit=args.source_commit
    )
    write_json(args.output_json, result)
    write_markdown(result, args.output_markdown)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_markdown}")


if __name__ == "__main__":
    main()
