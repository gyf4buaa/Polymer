"""Run one frozen Stage 5A width/seed through the shared Own-GNN engine."""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch

from ...models.own_gnn_repr_keep_dummy import graph as g0_graph
from ...models.own_gnn_v0 import train_oof as frozen_engine
from ...models.polymer_representation_ablation.model import OwnGNNRepresentation
from ...models.polymer_representation_ablation import runner as representation_runner
from ...src.data import sha256_file, write_json
from .protocol import (
    MODEL_ROOT,
    REPOSITORY_ROOT,
    SEEDS,
    TRACK_ROOT,
    WIDTHS,
    artifact_dir,
    load_base_config,
    run_config,
)


def _config_for(width: int, seed: int | None = None) -> dict[str, Any]:
    config = run_config(width, 42 if seed is None else int(seed))
    return config


def _stage_source_manifest(
    commit: str,
    train_csv: Path,
    folds_csv: Path,
    *,
    config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    effective_config = dict(config if config is not None else load_base_config())
    prefixes = (
        "from_scratch_gnn/models/capacity_scaling/",
        "from_scratch_gnn/models/own_gnn_repr_keep_dummy/",
        "from_scratch_gnn/models/own_gnn_v0/",
        "from_scratch_gnn/models/polymer_representation_ablation/",
        "from_scratch_gnn/scripts/run_stage5a_queue.py",
        "from_scratch_gnn/scripts/aggregate_stage5a.py",
        "from_scratch_gnn/experiments/stage5a/parameter_audit.json",
        "from_scratch_gnn/experiments/stage5a/tiny_overfit_notes.json",
        "from_scratch_gnn/src/data.py",
        "from_scratch_gnn/src/metrics.py",
        "from_scratch_gnn/scripts/evaluate_oof.py",
        "from_scratch_gnn/benchmark/data_manifest.json",
        "from_scratch_gnn/benchmark/folds.csv",
    )
    tracked = frozen_engine._git_output("ls-files", *prefixes).splitlines()
    source_files = {
        relative: sha256_file(REPOSITORY_ROOT / relative)
        for relative in tracked
        if (REPOSITORY_ROOT / relative).is_file()
    }
    return {
        "stage": "5A",
        "experiment_id": effective_config["experiment_id"],
        "hidden_dim": int(effective_config["model"]["hidden_dim"]),
        "seed": int(effective_config["seed"]),
        "git_commit": commit,
        "branch": frozen_engine._git_output("branch", "--show-current"),
        "source_files_sha256": source_files,
        "source_config_sha256": sha256_file(MODEL_ROOT / "config.json"),
        "effective_config_sha256": frozen_engine._config_sha256(effective_config),
        "benchmark_files_sha256": {
            "train_csv": sha256_file(train_csv),
            "folds_csv": sha256_file(folds_csv),
            "manifest_json": sha256_file(TRACK_ROOT / "benchmark" / "data_manifest.json"),
        },
        "historical_c256_reused": True,
    }


def _write_registry_row(
    _result_path: Path,
    *,
    config: Mapping[str, Any],
    commit: str,
    metrics: Mapping[str, Any],
    folds_sha256: str,
    train_sha256: str,
    run_metadata_path: Path,
) -> None:
    """Write a run-local row; the queue never concurrently appends results.csv."""
    width = int(config["model"]["hidden_dim"])
    seed = int(config["seed"])
    write_json(
        run_metadata_path.parent / "registry_row.json",
        {
            "experiment_id": f"stage5a_capacity_c{width}_seed_{seed}",
            "model_name": f"Stage 5A G0 C{width}",
            "category": "own_model",
            "stage": "5A",
            "hidden_dim": width,
            "seed": seed,
            "status": "formal_oof_model",
            "source_commit": commit,
            "train_sha256": train_sha256,
            "folds_sha256": folds_sha256,
            "effective_config_sha256": sha256_file(run_metadata_path.parent / "config.json"),
            "overall_oof_wmae": metrics["overall_oof_wmae"],
            "target_mae": metrics["target_mae"],
            "run_metadata": (
                str(run_metadata_path.relative_to(TRACK_ROOT))
                if run_metadata_path.is_relative_to(TRACK_ROOT)
                else str(run_metadata_path)
            ),
        },
    )


def _verify_artifact_path(args: argparse.Namespace, width: int, phase: str) -> None:
    expected_parent = (MODEL_ROOT / "artifacts" / phase / f"C{width}").resolve()
    actual = args.output_dir.resolve()
    if expected_parent not in actual.parents:
        raise ValueError(
            f"{phase} artifacts must be isolated beneath {expected_parent}, got {actual}"
        )
    if phase == "formal":
        expected_seed_parent = expected_parent / f"seed_{int(args.seed if args.seed is not None else 42)}"
        if expected_seed_parent not in actual.parents:
            raise ValueError("Formal output path must also be isolated by seed")
        if not actual.name.startswith("attempt_"):
            raise ValueError("Formal output path must use an attempt_NN directory")


def _verify_smoke(width: int, output_dir: Path) -> dict[str, Any]:
    metadata_path = output_dir / "gpu_smoke_fold_0" / "smoke_fold_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    summary = metadata["fold_summary"]
    if metadata.get("status") != "passed" or metadata.get("device") != "cuda":
        raise RuntimeError("Stage 5A smoke did not complete on CUDA")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA became unavailable while verifying the smoke")
    device_name = torch.cuda.get_device_name(torch.device("cuda"))
    if "RTX 4070" not in device_name:
        raise RuntimeError(f"Stage 5A smoke used unexpected CUDA device: {device_name}")
    if int(metadata["fold_summary"]["seed"]) not in SEEDS:
        raise RuntimeError("Stage 5A smoke metadata has an unexpected seed")
    if not torch.isfinite(torch.tensor(float(summary["best_validation_wmae"]))):
        raise RuntimeError("Stage 5A smoke produced a non-finite validation metric")
    checkpoint_path = Path(summary["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model = OwnGNNRepresentation(hidden_dim=width, num_layers=4, dropout=0.1)
    model.load_state_dict(checkpoint["state_dict"])
    result = {
        "status": "passed",
        "stage": "5A",
        "hidden_dim": width,
        "device": device_name,
        "cuda_active": True,
        "fold": 0,
        "seed": int(metadata["seed"]),
        "requested_epochs": int(metadata["requested_epochs"]),
        "validation_wmae": float(summary["best_validation_wmae"]),
        "checkpoint_round_trip": True,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "peak_vram_allocated_mb": summary["peak_vram_allocated_mb"],
        "peak_vram_reserved_mb": summary["peak_vram_reserved_mb"],
        "nvidia_smi_peak_memory_used_mb": summary["nvidia_smi_peak_memory_used_mb"],
        "gpu_utilization_mean_percent": summary["gpu_utilization_mean_percent"],
        "artifact_phase": "smoke; isolated from formal artifacts",
    }
    write_json(output_dir / "stage5a_cuda_smoke_verification.json", result)
    return result


def run_one(argv: list[str] | None = None) -> dict[str, Any] | None:
    width_parser = argparse.ArgumentParser(add_help=False)
    width_parser.add_argument("--hidden-dim", type=int, required=True, choices=WIDTHS)
    width_args, engine_argv = width_parser.parse_known_args(argv)
    width = int(width_args.hidden_dim)
    phase = (
        "tiny" if "--tiny-overfit" in engine_argv
        else "smoke" if "--smoke-fold" in engine_argv
        else "formal"
    )

    previous_argv = sys.argv
    saved_names = (
        "MODEL_ROOT",
        "GRAPH_SCHEMA",
        "build_polymer_graph",
        "_load_config",
        "_build_graphs",
        "_source_manifest",
        "_append_result",
        "_default_output_dir",
        "OwnGNNv0",
    )
    previous = {name: getattr(frozen_engine, name) for name in saved_names}
    bound_builder = representation_runner._graph_builder_for(
        g0_graph.GRAPH_SCHEMA, g0_graph.REPRESENTATION, g0_graph.build_polymer_graph
    )
    try:
        sys.argv = [previous_argv[0], *engine_argv]
        frozen_engine.MODEL_ROOT = MODEL_ROOT.resolve()
        frozen_engine.GRAPH_SCHEMA = g0_graph.GRAPH_SCHEMA
        frozen_engine.build_polymer_graph = bound_builder
        frozen_engine._load_config = partial(_config_for, width)
        frozen_engine._build_graphs = representation_runner._build_graphs_for_variant
        frozen_engine._source_manifest = _stage_source_manifest
        frozen_engine._append_result = _write_registry_row
        frozen_engine.OwnGNNv0 = OwnGNNRepresentation
        frozen_engine._default_output_dir = lambda _root, seed, _base_seed=42: artifact_dir(
            phase, width, int(seed)
        )
        args = frozen_engine.parse_args()
        config = _config_for(width, args.seed)
        if phase == "formal":
            if width == 256:
                raise ValueError("C256 is the historical G0 reference and must not be retrained")
            if args.seed is None or int(args.seed) not in SEEDS:
                raise ValueError("Formal Stage 5A jobs require a preregistered seed 42–46")
            if args.device != "cuda":
                raise ValueError("Formal Stage 5A runs must explicitly request --device cuda")
            if any(value is not None for value in (args.epochs, args.patience, args.batch_size)):
                raise ValueError("Formal Stage 5A training settings cannot be overridden")
            _verify_artifact_path(args, width, phase)
            frozen_engine._require_clean_source()
            metadata = frozen_engine._run_formal(args)
            metadata_path = args.output_dir / "run_metadata.json"
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored.update(
                {
                    "stage": "5A",
                    "hidden_dim": width,
                    "historical_c256_reused": True,
                    "formal_matrix_job": f"C{width}-seed{int(config['seed'])}",
                }
            )
            write_json(metadata_path, stored)
            return stored

        _verify_artifact_path(args, width, phase)
        if phase == "tiny":
            if args.device != "cpu":
                raise ValueError("Stage 5A CPU tiny-overfit must explicitly request --device cpu")
            if width == 256:
                raise ValueError("Tiny-overfit is preregistered only for new widths")
            return frozen_engine._run_tiny_overfit(args)

        if args.device != "cuda":
            raise ValueError("Stage 5A CUDA smokes must explicitly request --device cuda")
        if width == 256:
            raise ValueError("CUDA smokes are preregistered only for new widths")
        frozen_engine._require_clean_source()
        frozen_engine._run_smoke_fold(args)
        return _verify_smoke(width, args.output_dir.resolve())
    finally:
        sys.argv = previous_argv
        for name, value in previous.items():
            setattr(frozen_engine, name, value)


def main() -> None:
    result = run_one()
    if result is not None:
        print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
