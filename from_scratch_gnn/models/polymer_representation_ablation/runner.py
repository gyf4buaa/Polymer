"""Bind representation-specific inputs to the frozen Own-GNN v0 run engine.

The v0 implementation remains unchanged. Its training, fold validation,
target normalization, loss, metric, checkpoints, and OOF validator are reused
as-is; this adapter supplies variant graph/model/config/provenance hooks in a
single-process context and restores every hook when the run ends.
"""
from __future__ import annotations

import csv
import json
import os
from collections import Counter
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import torch
from torch_geometric.data import Data

from ...src.data import sha256_file, write_json
from ..own_gnn_v0 import train_oof as frozen_engine
from .model import OwnGNNRepresentation

V0_MODEL_ROOT = frozen_engine.MODEL_ROOT


def _graph_builder_for(schema: str, representation: str, builder: Callable[..., Any]):
    def build(smiles: str, *, sample_id: str | None = None):
        return builder(smiles, sample_id=sample_id)

    build.graph_schema = schema  # type: ignore[attr-defined]
    build.representation = representation  # type: ignore[attr-defined]
    return build


def _build_graphs_for_variant(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_path: Path,
    source_sha256: str,
    diagnostics_path: Path,
) -> tuple[list[Data], dict[str, Any]]:
    import rdkit

    graph_schema = frozen_engine.GRAPH_SCHEMA
    builder = frozen_engine.build_polymer_graph
    cache_metadata = {
        "cache_schema": graph_schema,
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
            diagnostics = saved["diagnostics"]
            write_json(diagnostics_path, diagnostics)
            print(f"Reusing graph cache: {cache_path}", flush=True)
            return saved["graphs"], diagnostics

    graphs: list[Data] = []
    sample_diagnostics: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    dummy_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    fallback_reasons: Counter[str] = Counter()
    for index, row in enumerate(rows):
        if index and index % 1000 == 0:
            print(f"building polymer graphs {index}/{len(rows)}", flush=True)
        sample_id = str(row["sample_id"])
        try:
            graph, info = builder(str(row["SMILES"]), sample_id=sample_id)
        except frozen_engine.GraphBuildError as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            continue
        graphs.append(graph)
        detail = info.as_dict()
        sample_diagnostics.append(detail)
        dummy_counts[str(info.dummy_atom_count)] += 1
        topology_counts[str(info.topology)] += 1
        if info.fallback_reason is not None:
            fallback_reasons[str(info.fallback_reason)] += 1

    summary = {
        "graph_schema": graph_schema,
        "representation": builder.representation,  # type: ignore[attr-defined]
        "source_sha256": source_sha256,
        "source_sample_count": len(rows),
        "graph_count": len(graphs),
        "source_rows_dropped": 0,
        "rdkit_parse_failures": failures,
        "dummy_atom_count_distribution": dict(
            sorted(dummy_counts.items(), key=lambda item: int(item[0]))
        ),
        "topology_distribution": dict(sorted(topology_counts.items())),
        "polymer_endpoint_graph_count": sum(
            int(item["endpoint_marker_applied"]) for item in sample_diagnostics
        ),
        "polymer_endpoint_node_count": sum(
            int(item["polymer_endpoint_count"]) for item in sample_diagnostics
        ),
        "endpoint_neighbors_already_bonded_count": sum(
            int(item["endpoint_neighbors_already_bonded"])
            for item in sample_diagnostics
        ),
        "endpoint_neighbors_already_bonded_marker_count": sum(
            int(
                item["endpoint_neighbors_already_bonded"]
                and item["endpoint_marker_applied"]
            )
            for item in sample_diagnostics
        ),
        "endpoint_closure_count": 0,
        "polymerization_edge_count": sum(
            int(item["polymerization_edge_count"]) for item in sample_diagnostics
        ),
        "atomic_number_zero_node_count": sum(
            int(item["atomic_number_zero_node_count"])
            for item in sample_diagnostics
        ),
        "retained_dummy_node_count": sum(
            int(item["retained_dummy_node_count"]) for item in sample_diagnostics
        ),
        "fallback_count": sum(fallback_reasons.values()),
        "fallback_reasons": dict(sorted(fallback_reasons.items())),
        "fallback_samples": [
            {
                "sample_id": item["sample_id"],
                "dummy_atom_count": item["dummy_atom_count"],
                "reason": item["fallback_reason"],
            }
            for item in sample_diagnostics
            if item["fallback_reason"] is not None
        ],
        "samples": sample_diagnostics,
    }
    write_json(diagnostics_path, summary)
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
            "graph_diagnostics": sample_diagnostics,
            "diagnostics": summary,
        },
        cache_path,
    )
    return graphs, summary


def _source_manifest_for_variant(
    commit: str,
    train_csv: Path,
    folds_csv: Path,
    *,
    variant_root: Path,
    graph_schema: str,
    representation: str,
) -> dict[str, Any]:
    variant_path = variant_root.resolve().relative_to(
        frozen_engine.REPOSITORY_ROOT
    ).as_posix()
    prefixes = (
        "from_scratch_gnn/models/polymer_representation_ablation/",
        f"{variant_path}/",
    )
    tracked = frozen_engine._git_output(
        "ls-files",
        *prefixes,
        "from_scratch_gnn/models/own_gnn_v0/graph.py",
        "from_scratch_gnn/models/own_gnn_v0/model.py",
        "from_scratch_gnn/models/own_gnn_v0/train_oof.py",
    ).splitlines()
    files = {
        relative: sha256_file(frozen_engine.REPOSITORY_ROOT / relative)
        for relative in tracked
        if (frozen_engine.REPOSITORY_ROOT / relative).is_file()
    }
    return {
        "experiment_id": json.loads(
            (variant_root / "config.json").read_text(encoding="utf-8")
        )["experiment_id"],
        "representation": representation,
        "graph_schema": graph_schema,
        "git_commit": commit,
        "branch": frozen_engine._git_output("branch", "--show-current"),
        "source_files_sha256": files,
        "variant_config_sha256": sha256_file(variant_root / "config.json"),
        "benchmark_files_sha256": {
            "train_csv": sha256_file(train_csv),
            "folds_csv": sha256_file(folds_csv),
            "manifest_json": sha256_file(
                frozen_engine.TRACK_ROOT / "benchmark" / "data_manifest.json"
            ),
        },
    }


def _load_variant_config(
    *, variant_root: Path, graph_schema: str, representation: str
) -> dict[str, Any]:
    config = json.loads((variant_root / "config.json").read_text(encoding="utf-8"))
    baseline = json.loads(
        (V0_MODEL_ROOT / "config.json").read_text(encoding="utf-8")
    )
    if config.get("benchmark_version") != baseline["benchmark_version"]:
        raise ValueError("Representation variant changed the frozen benchmark version")
    if config.get("seed") != baseline["seed"]:
        raise ValueError("Representation variant changed the frozen seed")
    if config.get("training") != baseline["training"]:
        raise ValueError("Representation variant changed frozen training settings")
    fixed_model_keys = (
        "operator",
        "hidden_dim",
        "num_layers",
        "dropout",
        "residual",
        "normalization",
        "pooling",
        "property_heads",
    )
    if any(
        config["model"].get(key) != baseline["model"].get(key)
        for key in fixed_model_keys
    ):
        raise ValueError("Representation variant changed the frozen v0 model settings")
    graph_config = config.get("graph", {})
    if graph_config.get("schema") != graph_schema:
        raise ValueError("Variant graph schema differs from its independently versioned config")
    if graph_config.get("representation") != representation:
        raise ValueError("Variant representation differs from its graph config")
    if config.get("category") != "own_model":
        raise ValueError("Representation variants must be registered as owned models")
    return config


def _append_variant_result(
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
    values = {
        "experiment_id": config["experiment_id"],
        "model_name": config["model_name"],
        "category": "own_model",
        "benchmark_version": config["benchmark_version"],
        "git_commit": commit,
        "seed": config["seed"],
        "oof_wmae": metrics["overall_oof_wmae"],
        **{
            f"{target.lower()}_mae": metrics["target_mae"][target]
            for target in frozen_engine.TARGETS
        },
        "status": "formal_oof_model",
        "notes": config["registry_notes"],
        "folds_sha256": folds_sha256,
        "train_data_sha256": train_sha256,
        "run_metadata": str(run_metadata_path.relative_to(frozen_engine.TRACK_ROOT)),
    }
    for key, value in values.items():
        if key in new_row:
            new_row[key] = value
    existing.append(new_row)
    temp_path = result_path.with_suffix(result_path.suffix + ".tmp")
    with temp_path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(existing)
    os.replace(temp_path, result_path)


def main_for_variant(
    *,
    variant_root: Path,
    graph_schema: str,
    representation: str,
    graph_builder: Callable[..., Any],
) -> None:
    """Run one variant through the unmodified, frozen v0 CLI and train engine."""
    bound_builder = _graph_builder_for(graph_schema, representation, graph_builder)
    replacements = {
        "MODEL_ROOT": variant_root.resolve(),
        "GRAPH_SCHEMA": graph_schema,
        "build_polymer_graph": bound_builder,
        "_load_config": lambda: _load_variant_config(
            variant_root=variant_root,
            graph_schema=graph_schema,
            representation=representation,
        ),
        "OwnGNNv0": OwnGNNRepresentation,
        "_build_graphs": _build_graphs_for_variant,
        "_source_manifest": partial(
            _source_manifest_for_variant,
            variant_root=variant_root.resolve(),
            graph_schema=graph_schema,
            representation=representation,
        ),
        "_append_result": _append_variant_result,
    }
    previous = {name: getattr(frozen_engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(frozen_engine, name, value)
        args = frozen_engine.parse_args()
        if args.tiny_overfit:
            frozen_engine._run_tiny_overfit(args)
        elif args.smoke_fold is not None:
            frozen_engine._run_smoke_fold(args)
        else:
            frozen_engine._run_formal(args)
    finally:
        for name, value in previous.items():
            setattr(frozen_engine, name, value)
