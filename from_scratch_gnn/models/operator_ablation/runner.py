"""Run one Stage 3B operator through the frozen Stage 0 training engine."""
from __future__ import annotations

import gzip
import json
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

from torch_geometric.data import Data

from ...src.data import sha256_file, write_json
from ..own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA as KEEP_DUMMY_SCHEMA,
    REPRESENTATION as KEEP_DUMMY_REPRESENTATION,
    build_polymer_graph as build_keep_dummy_graph,
)
from ..polymer_representation_ablation import runner as representation_runner
from .degree_stats import graph_set_fingerprint, write_degree_provenance
from .model import build_operator_model

frozen_engine = representation_runner.frozen_engine
TRACK_ROOT = frozen_engine.TRACK_ROOT
BASELINE_ROOT = TRACK_ROOT / "models" / "own_gnn_repr_keep_dummy"


def _load_operator_config(
    *, operator_root: Path, operator: str, seed_override: int | None = None
) -> dict[str, Any]:
    config = json.loads((operator_root / "config.json").read_text(encoding="utf-8"))
    baseline = json.loads((BASELINE_ROOT / "config.json").read_text(encoding="utf-8"))
    if operator not in {"gatv2", "pna"}:
        raise ValueError(f"Unsupported Stage 3B operator: {operator}")
    if config.get("benchmark_version") != baseline["benchmark_version"]:
        raise ValueError("Operator variant changed the frozen benchmark version")
    if config.get("seed") != baseline["seed"]:
        raise ValueError("Operator variant changed the checked-in seed")
    if config.get("training") != baseline["training"]:
        raise ValueError("Operator variant changed frozen training settings")
    if config.get("graph") != baseline["graph"]:
        raise ValueError("Operator variant changed the frozen keep-dummy graph")
    if config.get("category") != "own_model":
        raise ValueError("Operator variants must be registered as owned models")
    model = config.get("model", {})
    baseline_model = baseline["model"]
    fixed_keys = (
        "hidden_dim",
        "num_layers",
        "dropout",
        "residual",
        "normalization",
        "pooling",
        "property_heads",
    )
    if any(model.get(key) != baseline_model.get(key) for key in fixed_keys):
        raise ValueError("Operator variant changed a frozen architecture setting")
    if model.get("operator") != operator:
        raise ValueError("Operator name differs from its isolated config directory")
    if model.get("hidden_dim") != 256 or model.get("num_layers") != 4:
        raise ValueError("Stage 3B requires hidden_dim=256 and num_layers=4")
    if model.get("dropout") != 0.1:
        raise ValueError("Stage 3B requires dropout=0.1")
    if operator == "gatv2":
        settings = model.get("operator_config", {})
        if settings != {
            "heads": 4,
            "channels_per_head": 64,
            "concat": True,
            "edge_aware": True,
            "attention_dropout": 0.0,
            "add_self_loops": False,
        }:
            raise ValueError("GATv2 operator config differs from the frozen 4×64 design")
    else:
        settings = model.get("operator_config", {})
        if settings != {
            "aggregators": ["mean", "min", "max", "std"],
            "scalers": ["identity", "amplification", "attenuation"],
            "towers": 1,
            "edge_aware": True,
        }:
            raise ValueError("PNA operator config differs from the fixed Stage 3B design")
    if seed_override is not None:
        if int(seed_override) < 0:
            raise ValueError("Seed override must be non-negative")
        config["seed"] = int(seed_override)
    return config


def _output_dir(operator_root: Path, seed: int) -> Path:
    return operator_root / "artifacts" / f"seed_{seed}"


def _tracked_sources(operator_root: Path) -> list[str]:
    operator_rel = operator_root.resolve().relative_to(frozen_engine.REPOSITORY_ROOT).as_posix()
    files = frozen_engine._git_output(
        "ls-files",
        "from_scratch_gnn/models/operator_ablation/",
        f"{operator_rel}/",
        "from_scratch_gnn/models/own_gnn_repr_keep_dummy/graph.py",
        "from_scratch_gnn/models/polymer_representation_ablation/graph_builder.py",
        "from_scratch_gnn/models/polymer_representation_ablation/runner.py",
        "from_scratch_gnn/models/own_gnn_v0/graph.py",
        "from_scratch_gnn/models/own_gnn_v0/model.py",
        "from_scratch_gnn/models/own_gnn_v0/train_oof.py",
    ).splitlines()
    return sorted(set(files))


def _source_manifest(
    commit: str,
    train_csv: Path,
    folds_csv: Path,
    *,
    operator_root: Path,
    operator: str,
    config: Mapping[str, Any],
    context: Mapping[str, Any],
) -> dict[str, Any]:
    relative_files = _tracked_sources(operator_root)
    source_files = {
        relative: sha256_file(frozen_engine.REPOSITORY_ROOT / relative)
        for relative in relative_files
        if (frozen_engine.REPOSITORY_ROOT / relative).is_file()
    }
    degree_path = context.get("degree_statistics_path")
    diagnostics_path = context.get("graph_diagnostics_path")
    return {
        "experiment_id": config["experiment_id"],
        "operator": operator,
        "representation": KEEP_DUMMY_REPRESENTATION,
        "graph_schema": KEEP_DUMMY_SCHEMA,
        "seed": int(config["seed"]),
        "git_commit": commit,
        "branch": frozen_engine._git_output("branch", "--show-current"),
        "source_files_sha256": source_files,
        "source_config_sha256": sha256_file(operator_root / "config.json"),
        "effective_config_sha256": frozen_engine._config_sha256(config),
        "benchmark_files_sha256": {
            "train_csv": sha256_file(train_csv),
            "folds_csv": sha256_file(folds_csv),
            "manifest_json": sha256_file(TRACK_ROOT / "benchmark" / "data_manifest.json"),
        },
        "graph_provenance": {
            "graph_set_fingerprint_sha256": context.get("graph_set_fingerprint_sha256"),
            "degree_statistics_path": (
                str(Path(degree_path).relative_to(TRACK_ROOT)) if degree_path else None
            ),
            "degree_statistics_sha256": sha256_file(Path(degree_path)) if degree_path else None,
            "graph_diagnostics_path": (
                str(Path(diagnostics_path).relative_to(TRACK_ROOT))
                if diagnostics_path
                else None
            ),
            "graph_diagnostics_sha256": (
                sha256_file(Path(diagnostics_path)) if diagnostics_path else None
            ),
        },
    }


def _append_operator_result(
    result_path: Path,
    *,
    config: Mapping[str, Any],
    commit: str,
    metrics: Mapping[str, Any],
    folds_sha256: str,
    train_sha256: str,
    run_metadata_path: Path,
    operator: str,
    context: Mapping[str, Any],
) -> None:
    metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    metadata["operator"] = operator
    metadata["operator_config"] = config["model"]["operator_config"]
    metadata["graph_set_fingerprint_sha256"] = context["graph_set_fingerprint_sha256"]
    metadata["parameter_count"] = context["parameter_count"]
    degree_path = context.get("degree_statistics_path")
    if degree_path:
        metadata["degree_statistics_path"] = str(Path(degree_path).relative_to(TRACK_ROOT))
        metadata["degree_statistics_sha256"] = sha256_file(Path(degree_path))
    write_json(run_metadata_path, metadata)
    row = {
        "experiment_id": config["experiment_id"],
        "operator": operator,
        "seed": int(config["seed"]),
        "status": "formal_oof_model",
        "source_commit": commit,
        "train_sha256": train_sha256,
        "folds_sha256": folds_sha256,
        "effective_config_sha256": sha256_file(run_metadata_path.parent / "config.json"),
        "overall_oof_wmae": metrics["overall_oof_wmae"],
        "target_mae": metrics["target_mae"],
        "run_metadata": str(run_metadata_path.relative_to(TRACK_ROOT)),
        "parameter_count": context["parameter_count"],
    }
    write_json(run_metadata_path.parent / "registry_row.json", row)


def _write_operator_metadata(
    directory: Path,
    *,
    operator: str,
    operator_config: Mapping[str, Any],
    context: dict[str, Any],
) -> None:
    config = frozen_engine._load_config()
    constructor = context["model_constructor"]
    model = constructor(
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    block_parameters = sum(
        parameter.numel()
        for block in model.convs
        for parameter in block.parameters()
        if parameter.requires_grad
    )
    metadata = {
        "operator": operator,
        "operator_config": dict(operator_config),
        "graph_schema": KEEP_DUMMY_SCHEMA,
        "representation": KEEP_DUMMY_REPRESENTATION,
        "hidden_dim": int(config["model"]["hidden_dim"]),
        "num_layers": int(config["model"]["num_layers"]),
        "message_passing_output_widths": [model.hidden_dim] * model.num_layers,
        "trainable_parameter_count": parameter_count,
        "message_passing_block_parameter_count": block_parameters,
        "graph_set_fingerprint_sha256": context["graph_set_fingerprint_sha256"],
        "selection_role": "formal configuration; smoke metrics are non-selection",
    }
    if context.get("degree_statistics_path"):
        metadata["degree_statistics_path"] = str(
            Path(context["degree_statistics_path"]).relative_to(TRACK_ROOT)
        )
        metadata["degree_statistics_sha256"] = sha256_file(
            Path(context["degree_statistics_path"])
        )
    context["parameter_count"] = {
        "trainable": parameter_count,
        "message_passing_blocks": block_parameters,
    }
    write_json(directory / "operator_metadata.json", metadata)


def _graph_builder_for_operator(context: dict[str, Any], operator: str):
    def build_graphs(
        rows: Sequence[Mapping[str, Any]],
        *,
        cache_path: Path,
        source_sha256: str,
        diagnostics_path: Path,
    ) -> tuple[list[Data], dict[str, Any]]:
        graphs, diagnostics = representation_runner._build_graphs_for_variant(
            rows,
            cache_path=cache_path,
            source_sha256=source_sha256,
            diagnostics_path=diagnostics_path,
        )
        fingerprint = graph_set_fingerprint(graphs, KEEP_DUMMY_SCHEMA)
        context["graphs"] = graphs
        context["graph_diagnostics"] = diagnostics
        context["graph_set_fingerprint_sha256"] = fingerprint
        directory = diagnostics_path.parent
        # The detailed diagnostic is retained compactly; the shared engine
        # does not read it again after graph preparation.
        if diagnostics_path.exists():
            compressed_path = diagnostics_path.with_suffix(".json.gz")
            with diagnostics_path.open("rb") as source, gzip.open(compressed_path, "wb") as target:
                target.write(source.read())
            diagnostics_path.unlink()
            context["graph_diagnostics_path"] = compressed_path
        if operator == "pna":
            degree_path = directory / "degree_histogram.json"
            provenance = write_degree_provenance(
                degree_path,
                graphs=graphs,
                graph_schema=KEEP_DUMMY_SCHEMA,
                source_train_sha256=source_sha256,
                source_commit=frozen_engine._git_output("rev-parse", "HEAD"),
                generator_path=Path(__file__).with_name("degree_stats.py"),
            )
            context["degree_histogram"] = provenance["histogram_tensor"]
            context["degree_statistics_path"] = degree_path
        config = frozen_engine._load_config()
        _write_operator_metadata(
            directory,
            operator=operator,
            operator_config=config["model"]["operator_config"],
            context=context,
        )
        return graphs, diagnostics

    build_graphs.graph_schema = KEEP_DUMMY_SCHEMA  # type: ignore[attr-defined]
    build_graphs.representation = KEEP_DUMMY_REPRESENTATION  # type: ignore[attr-defined]
    return build_graphs


def main_for_operator(operator_root: Path, operator: str) -> None:
    operator_root = operator_root.resolve()
    context: dict[str, Any] = {"operator": operator}
    bound_builder = representation_runner._graph_builder_for(
        KEEP_DUMMY_SCHEMA, KEEP_DUMMY_REPRESENTATION, build_keep_dummy_graph
    )

    def load_config() -> dict[str, Any]:
        return _load_operator_config(operator_root=operator_root, operator=operator)

    def load_effective_config(seed_override: int | None = None) -> dict[str, Any]:
        return _load_operator_config(
            operator_root=operator_root, operator=operator, seed_override=seed_override
        )

    def model_constructor(**kwargs: Any):
        return build_operator_model(
            operator=operator,
            operator_config=load_config()["model"]["operator_config"],
            degree_histogram=context.get("degree_histogram"),
            **kwargs,
        )

    context["model_constructor"] = model_constructor
    replacements: dict[str, Any] = {
        "MODEL_ROOT": operator_root,
        "GRAPH_SCHEMA": KEEP_DUMMY_SCHEMA,
        "build_polymer_graph": bound_builder,
        "_load_config": load_config,
        "OwnGNNv0": model_constructor,
        "_model_config": lambda config: {
            key: config["model"][key] for key in ("hidden_dim", "num_layers", "dropout")
        },
        "_build_graphs": _graph_builder_for_operator(context, operator),
        "_source_manifest": partial(
            _source_manifest,
            operator_root=operator_root,
            operator=operator,
            context=context,
        ),
        "_append_result": partial(
            _append_operator_result,
            operator=operator,
            context=context,
        ),
        "_default_output_dir": lambda model_root, seed, configured_seed=42: _output_dir(
            operator_root, seed
        ),
    }
    previous = {name: getattr(frozen_engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(frozen_engine, name, value)
        args = frozen_engine.parse_args()
        config = load_effective_config(args.seed)
        if not args.tiny_overfit and args.smoke_fold is None:
            if int(config["seed"]) not in {42, 43, 44, 45, 46}:
                raise ValueError("Formal Stage 3B seeds are frozen to 42–46")
            if any(value is not None for value in (args.epochs, args.patience, args.batch_size)):
                raise ValueError("Formal Stage 3B training hyperparameters cannot be overridden")
            expected_output = _output_dir(operator_root, int(config["seed"])).resolve()
            if "--output-dir" in sys.argv and args.output_dir.resolve() != expected_output:
                raise ValueError("Formal artifacts must use the operator/seed-specific directory")
            args.output_dir = expected_output
            if any(
                (expected_output / name).exists()
                for name in ("metrics.json", "oof_predictions.csv", "run_metadata.json")
            ):
                raise FileExistsError(
                    f"Formal {operator} seed {config['seed']} already has completed artifacts"
                )
            frozen_engine._require_clean_source()
        elif args.smoke_fold is not None:
            frozen_engine._require_clean_source()
            if args.device != "cuda":
                raise ValueError("Stage 3B CUDA smoke must explicitly request --device cuda")
        if args.smoke_fold is not None and "--output-dir" not in sys.argv:
            args.output_dir = operator_root / "artifacts" / "smoke" / f"seed_{config['seed']}"
        elif args.tiny_overfit and "--output-dir" not in sys.argv:
            args.output_dir = operator_root / "artifacts" / "tiny" / f"seed_{config['seed']}"
        if args.smoke_fold is not None and "smoke" not in args.output_dir.parts:
            raise ValueError("CUDA smoke artifacts must be written under a smoke directory")
        if args.tiny_overfit and "tiny" not in args.output_dir.parts:
            raise ValueError("Tiny-overfit artifacts must be written under a tiny directory")
        if args.smoke_fold is not None and (
            args.output_dir / f"gpu_smoke_fold_{args.smoke_fold}" / "smoke_fold_metadata.json"
        ).exists():
            raise FileExistsError("This CUDA smoke artifact path already exists")

        # The same label-free all-benchmark graph topology supplies PNA's
        # degree statistics before any train/validation fold is selected.
        if args.tiny_overfit:
            rows, _, _ = frozen_engine._load_frozen_inputs(args.train_csv, args.folds_csv)
            cache_path = args.output_dir.parent / "cache" / "polymer_graphs_v1.pt"
            diagnostics_path = args.output_dir / "graph_diagnostics.json"
            frozen_engine._build_graphs(
                rows,
                cache_path=cache_path,
                source_sha256=sha256_file(args.train_csv),
                diagnostics_path=diagnostics_path,
            )
            tiny_root = args.output_dir / "tiny_overfit"
            _write_operator_metadata(
                tiny_root,
                operator=operator,
                operator_config=config["model"]["operator_config"],
                context=context,
            )
            result = frozen_engine._run_tiny_overfit(args)
            tiny_result_path = args.output_dir / "tiny_overfit" / "tiny_overfit.json"
            tiny_result = json.loads(tiny_result_path.read_text(encoding="utf-8"))
            tiny_result.update(
                {"operator": operator, "selection_role": "non-selection CPU smoke"}
            )
            write_json(tiny_result_path, tiny_result)
            del result
        elif args.smoke_fold is not None:
            metadata = frozen_engine._run_smoke_fold(args)
            smoke_root = args.output_dir / f"gpu_smoke_fold_{args.smoke_fold}"
            smoke_path = smoke_root / "smoke_fold_metadata.json"
            smoke_data = json.loads(smoke_path.read_text(encoding="utf-8"))
            smoke_config = json.loads((smoke_root / "config.json").read_text(encoding="utf-8"))
            checkpoint_path = Path(smoke_data["fold_summary"]["checkpoint_path"])
            smoke_data.update(
                {
                    "operator": operator,
                    "operator_config": config["model"]["operator_config"],
                    "graph_schema": KEEP_DUMMY_SCHEMA,
                    "representation": KEEP_DUMMY_REPRESENTATION,
                    "train_sha256": sha256_file(args.train_csv),
                    "folds_sha256": sha256_file(args.folds_csv),
                    "graph_set_fingerprint_sha256": context["graph_set_fingerprint_sha256"],
                    "parameter_count": context["parameter_count"],
                    "effective_config_sha256": sha256_file(smoke_root / "config.json"),
                    "cuda_available": bool(frozen_engine.torch.cuda.is_available()),
                    "device_name": smoke_config["runtime"]["device_name"],
                    "model_and_batch_device": smoke_data["device"],
                    "edge_attr_passed_to_operator": True,
                    "checkpoint_present": checkpoint_path.is_file(),
                    "selection_role": "non-selection CUDA smoke",
                }
            )
            if (
                not smoke_data["cuda_available"]
                or smoke_data["device"] != "cuda"
                or "RTX 4070" not in smoke_data["device_name"]
                or not smoke_data["checkpoint_present"]
            ):
                raise RuntimeError("CUDA smoke did not meet the RTX 4070/device/checkpoint requirements")
            degree_path = context.get("degree_statistics_path")
            if degree_path:
                smoke_data["degree_statistics_path"] = str(
                    Path(degree_path).relative_to(TRACK_ROOT)
                )
                smoke_data["degree_statistics_sha256"] = sha256_file(Path(degree_path))
            write_json(
                smoke_root / "source_manifest.json",
                _source_manifest(
                    frozen_engine._git_output("rev-parse", "HEAD"),
                    args.train_csv,
                    args.folds_csv,
                    operator_root=operator_root,
                    operator=operator,
                    config=smoke_config,
                    context=context,
                ),
            )
            write_json(smoke_path, smoke_data)
            del metadata
        else:
            frozen_engine._run_formal(args)
    finally:
        for name, value in previous.items():
            setattr(frozen_engine, name, value)
