"""Run Stage 3C readouts through the frozen Stage 0 OOF engine."""
from __future__ import annotations

import gzip
import json
import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping

import torch
from torch_geometric.data import Batch
from torch_geometric.loader import DataLoader
from torch_geometric.nn import global_add_pool

from ...src.data import sha256_file, write_json
from ...src.metrics import TARGETS
from ..own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA,
    REPRESENTATION,
    build_polymer_graph,
)
from ..polymer_representation_ablation import runner as representation_runner
from .model import ReadoutGNN, normalized_attention_entropy


frozen_engine = representation_runner.frozen_engine
TRACK_ROOT = frozen_engine.TRACK_ROOT
MODEL_ROOT = TRACK_ROOT / "models" / "readout_ablation"
BASELINE_ROOT = TRACK_ROOT / "models" / "own_gnn_repr_keep_dummy"
FORMAL_SEEDS = (42, 43, 44, 45, 46)
STAGE3C_MODELS = {
    "r1_shared": "R1",
    "r2_property": "R2",
}


def _load_readout_config(
    *, variant_root: Path, variant: str, seed_override: int | None = None
) -> dict[str, Any]:
    if variant not in STAGE3C_MODELS:
        raise ValueError(f"Unsupported Stage 3C variant: {variant}")
    config = json.loads((variant_root / "config.json").read_text(encoding="utf-8"))
    baseline = json.loads((BASELINE_ROOT / "config.json").read_text(encoding="utf-8"))
    if config.get("benchmark_version") != baseline["benchmark_version"]:
        raise ValueError("Stage 3C changed the frozen benchmark version")
    if config.get("seed") != baseline["seed"]:
        raise ValueError("Stage 3C changed the checked-in base seed")
    if config.get("training") != baseline["training"]:
        raise ValueError("Stage 3C changed the frozen training protocol")
    if config.get("graph") != baseline["graph"]:
        raise ValueError("Stage 3C changed the frozen keep-dummy graph")
    if config.get("category") != "own_model":
        raise ValueError("Stage 3C models must be registered as owned models")

    model = config.get("model", {})
    if not isinstance(model, dict):
        raise ValueError("Stage 3C model config must be an object")
    baseline_model = baseline["model"]
    if set(model) != set(baseline_model) | {"readout"}:
        raise ValueError("Stage 3C may add only its explicit readout configuration")
    for key, value in baseline_model.items():
        if key == "pooling":
            continue
        if model.get(key) != value:
            raise ValueError(f"Stage 3C changed frozen model setting: {key}")
    expected_readout = "R1" if variant == "r1_shared" else "R2"
    expected_pooling = (
        ["shared_attentive_mean", "global_max"]
        if expected_readout == "R1"
        else ["property_specific_attentive_mean", "global_max"]
    )
    expected_readout_config = (
        {
            "variant": "R1",
            "mean_pooling": "shared_attentive_mean",
            "max_pooling": "global_max",
            "gate": "Linear(256, 1, bias=False)",
            "gate_initialization": "all_zero",
        }
        if expected_readout == "R1"
        else {
            "variant": "R2",
            "mean_pooling": "property_specific_attentive_mean",
            "max_pooling": "global_max",
            "gate_per_target": "Linear(256, 1, bias=False)",
            "gate_initialization": "all_zero",
            "targets": list(TARGETS),
        }
    )
    readout_config = model.get("readout")
    if not isinstance(readout_config, dict) or readout_config != expected_readout_config:
        raise ValueError("Stage 3C readout details differ from the frozen design")
    if model.get("pooling") != expected_pooling:
        raise ValueError("Stage 3C pooling differs from the selected readout variant")
    if seed_override is not None:
        if int(seed_override) not in FORMAL_SEEDS:
            raise ValueError(f"Formal Stage 3C seeds are frozen to {FORMAL_SEEDS}")
        config["seed"] = int(seed_override)
    return config


def _output_dir(variant_root: Path, seed: int) -> Path:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Formal Stage 3C seeds are frozen to {FORMAL_SEEDS}")
    seed_root = variant_root / "artifacts" / "formal" / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    attempts = sorted(seed_root.glob("attempt_*"))
    for attempt in attempts:
        if (attempt / "run_metadata.json").is_file():
            raise FileExistsError(f"A completed formal run already exists: {attempt}")
    next_index = 1 + max(
        (int(path.name.split("_")[-1]) for path in attempts if path.name.split("_")[-1].isdigit()),
        default=0,
    )
    return seed_root / f"attempt_{next_index:02d}"


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    model = config["model"]
    return {
        key: model[key] for key in ("hidden_dim", "num_layers", "dropout")
    }


def _model_constructor(variant: str):
    readout_variant = STAGE3C_MODELS[variant]

    def construct(**kwargs: Any) -> ReadoutGNN:
        return ReadoutGNN(readout_variant=readout_variant, **kwargs)

    return construct


def _source_files(variant_root: Path) -> dict[str, str]:
    tracked = frozen_engine._git_output(
        "ls-files",
        "from_scratch_gnn/models/readout_ablation/",
        "from_scratch_gnn/models/own_gnn_v0/",
        "from_scratch_gnn/models/own_gnn_repr_keep_dummy/",
        "from_scratch_gnn/models/polymer_representation_ablation/model.py",
        "from_scratch_gnn/models/polymer_representation_ablation/graph_builder.py",
        "from_scratch_gnn/src/data.py",
        "from_scratch_gnn/src/metrics.py",
        "from_scratch_gnn/scripts/evaluate_oof.py",
        "from_scratch_gnn/benchmark/data_manifest.json",
    ).splitlines()
    files = {}
    for relative in tracked:
        path = frozen_engine.REPOSITORY_ROOT / relative
        if path.is_file():
            files[relative] = sha256_file(path)
    config_path = variant_root / "config.json"
    if config_path.is_file():
        relative = config_path.resolve().relative_to(
            frozen_engine.REPOSITORY_ROOT
        ).as_posix()
        files[relative] = sha256_file(config_path)
    return files


def _source_manifest(
    commit: str,
    train_csv: Path,
    folds_csv: Path,
    *,
    variant_root: Path,
    variant: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "experiment_id": config["experiment_id"],
        "stage": "3C",
        "readout_variant": STAGE3C_MODELS[variant],
        "representation": REPRESENTATION,
        "graph_schema": GRAPH_SCHEMA,
        "seed": int(config["seed"]),
        "git_commit": commit,
        "branch": frozen_engine._git_output("branch", "--show-current"),
        "source_files_sha256": _source_files(variant_root),
        "source_config_sha256": sha256_file(variant_root / "config.json"),
        "effective_config_sha256": frozen_engine._config_sha256(config),
        "benchmark_files_sha256": {
            "train_csv": sha256_file(train_csv),
            "folds_csv": sha256_file(folds_csv),
            "manifest_json": sha256_file(
                frozen_engine.TRACK_ROOT / "benchmark" / "data_manifest.json"
            ),
        },
    }


def _append_registry_row(
    result_path: Path,
    *,
    config: Mapping[str, Any],
    commit: str,
    metrics: Mapping[str, Any],
    folds_sha256: str,
    train_sha256: str,
    run_metadata_path: Path,
) -> None:
    """Keep formal run registration inside artifacts until every run finishes."""
    write_json(
        run_metadata_path.parent / "registry_row.json",
        {
            "experiment_id": config["experiment_id"],
            "readout_variant": config["model"]["readout"]["variant"],
            "seed": int(config["seed"]),
            "status": "formal_oof_model",
            "source_commit": commit,
            "train_sha256": train_sha256,
            "folds_sha256": folds_sha256,
            "effective_config_sha256": sha256_file(
                run_metadata_path.parent / "config.json"
            ),
            "overall_oof_wmae": metrics["overall_oof_wmae"],
            "target_mae": metrics["target_mae"],
            "run_metadata": str(
                run_metadata_path.relative_to(frozen_engine.TRACK_ROOT)
            ),
        },
    )


def _parameter_counts(variant: str, config: Mapping[str, Any]) -> dict[str, int]:
    model = _model_constructor(variant)(**_model_kwargs(config))
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    gates = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and (name.startswith("shared_gate.") or name.startswith("property_gates."))
    )
    base_model = ReadoutGNN(readout_variant="R0", **_model_kwargs(config))
    base_total = sum(
        parameter.numel()
        for parameter in base_model.parameters()
        if parameter.requires_grad
    )
    return {
        "total_trainable": total,
        "historical_r0_total_trainable": base_total,
        "readout_gate_trainable": gates,
        "new_parameters_vs_r0": total - base_total,
    }


def _write_model_metadata(
    output_dir: Path, *, variant: str, config: Mapping[str, Any]
) -> dict[str, Any]:
    counts = _parameter_counts(variant, config)
    expected_gate_count = 256 if variant == "r1_shared" else 5 * 256
    if counts["new_parameters_vs_r0"] != expected_gate_count:
        raise RuntimeError(
            "Stage 3C added an unexpected number of parameters: "
            f"expected {expected_gate_count}, got {counts['new_parameters_vs_r0']}"
        )
    metadata = {
        "stage": "3C",
        "readout_variant": STAGE3C_MODELS[variant],
        "readout": config["model"]["readout"],
        "parameter_count": counts,
        "readout_dimension": 2 * int(config["model"]["hidden_dim"]),
        "selection_role": "formal readout ablation; smoke outputs are non-selection",
    }
    write_json(output_dir / "readout_metadata.json", metadata)
    return metadata


def _load_graph_cache(cache_path: Path):
    saved = torch.load(cache_path, map_location="cpu", weights_only=False)
    return saved["graphs"]


def _graph_builder_for_readout(variant_root: Path):
    shared_cache = variant_root / "artifacts" / "cache" / "polymer_graphs_v1.pt"

    def build_graphs(
        rows,
        *,
        cache_path: Path,
        source_sha256: str,
        diagnostics_path: Path,
    ):
        graphs, diagnostics = representation_runner._build_graphs_for_variant(
            rows,
            cache_path=shared_cache,
            source_sha256=source_sha256,
            diagnostics_path=diagnostics_path,
        )
        if diagnostics_path.exists():
            compressed_path = diagnostics_path.with_suffix(".json.gz")
            with diagnostics_path.open("rb") as source, gzip.open(
                compressed_path, "wb"
            ) as target:
                target.write(source.read())
            diagnostics_path.unlink()
        return graphs, diagnostics

    build_graphs.graph_schema = GRAPH_SCHEMA  # type: ignore[attr-defined]
    build_graphs.representation = REPRESENTATION  # type: ignore[attr-defined]
    return build_graphs


def _attention_entropy_for_run(
    *,
    variant: str,
    config: Mapping[str, Any],
    output_dir: Path,
    train_csv: Path,
    folds_csv: Path,
    device: torch.device,
) -> dict[str, Any]:
    rows, fold_ids, _ = frozen_engine._load_frozen_inputs(train_csv, folds_csv)
    graphs = _load_graph_cache(
        _safe_variant_root(variant) / "artifacts" / "cache" / "polymer_graphs_v1.pt"
    )
    if len(graphs) != len(rows) or len(fold_ids) != len(rows):
        raise RuntimeError("Frozen graph cache does not align with the benchmark rows")

    per_fold: dict[str, Any] = {}
    gate_norms_by_fold: dict[str, dict[str, float]] = {}
    entropy_values = {target: [] for target in TARGETS}
    for fold in range(5):
        checkpoint_path = output_dir / "checkpoints" / f"fold_{fold}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        model = _model_constructor(variant)(**_model_kwargs(config)).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        model.eval()
        gate_norms = model.gate_l2_norms()
        gate_norms_by_fold[str(fold)] = gate_norms
        validation_indices = [index for index, assigned in enumerate(fold_ids) if assigned == fold]
        validation_graphs = [graphs[index] for index in validation_indices]
        fold_entropy: dict[str, list[float]] = {target: [] for target in TARGETS}
        loader = DataLoader(validation_graphs, batch_size=64, shuffle=False)
        with torch.no_grad():
            for batch in loader:
                batch = batch.to(device)
                hidden = model.encode_nodes(batch)
                for target in TARGETS:
                    weights = model.attention_weights(
                        hidden,
                        batch.batch,
                        target=target if variant == "r2_property" else None,
                    )
                    entropy = normalized_attention_entropy(weights, batch.batch)
                    values = entropy.reshape(-1).detach().cpu().tolist()
                    fold_entropy[target].extend(float(value) for value in values)
                    entropy_values[target].extend(float(value) for value in values)
        per_fold[str(fold)] = {
            "gate_l2_norms": gate_norms,
            "normalized_attention_entropy_mean_by_target": {
                target: statistics.fmean(values) if values else None
                for target, values in fold_entropy.items()
            },
            "oof_graph_count": len(validation_indices),
        }
        del model, checkpoint

    return {
        "readout_variant": STAGE3C_MODELS[variant],
        "gate_l2_norms_by_fold": gate_norms_by_fold,
        "gate_l2_norm_mean_by_gate": {
            key: statistics.fmean(values)
            for key, values in _transpose_gate_norms(gate_norms_by_fold).items()
        },
        "normalized_attention_entropy_mean_by_target": {
            target: statistics.fmean(values) if values else None
            for target, values in entropy_values.items()
        },
        "folds": per_fold,
        "oof_graph_count": sum(
            sum(assigned == fold for assigned in fold_ids) for fold in range(5)
        ),
        "stored_atom_attention_tensors": False,
    }


def _transpose_gate_norms(
    values: Mapping[str, Mapping[str, float]]
) -> dict[str, list[float]]:
    keys = sorted({key for fold in values.values() for key in fold})
    return {key: [float(fold[key]) for fold in values.values()] for key in keys}


def _augment_formal_run(
    *, variant: str, config: Mapping[str, Any], output_dir: Path,
    train_csv: Path, folds_csv: Path, device: torch.device,
) -> None:
    model_metadata = _write_model_metadata(output_dir, variant=variant, config=config)
    diagnostics = _attention_entropy_for_run(
        variant=variant,
        config=config,
        output_dir=output_dir,
        train_csv=train_csv,
        folds_csv=folds_csv,
        device=device,
    )
    write_json(output_dir / "readout_diagnostics.json", diagnostics)
    run_metadata_path = output_dir / "run_metadata.json"
    run_metadata = json.loads(run_metadata_path.read_text(encoding="utf-8"))
    run_metadata.update(
        {
            "stage": "3C",
            "readout_variant": STAGE3C_MODELS[variant],
            "parameter_count": model_metadata["parameter_count"],
            "readout_diagnostics_path": "readout_diagnostics.json",
        }
    )
    write_json(run_metadata_path, run_metadata)


def _smoke_attention_check(
    *, variant: str, config: Mapping[str, Any], output_dir: Path,
    device: torch.device, fold: int, train_csv: Path, folds_csv: Path,
) -> dict[str, Any]:
    fold_metadata = json.loads(
        (output_dir / "smoke_fold_metadata.json").read_text(encoding="utf-8")
    )
    checkpoint_path = Path(fold_metadata["fold_summary"]["checkpoint_path"])
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = _model_constructor(variant)(**_model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    graphs = _load_graph_cache(
        _safe_variant_root(variant) / "artifacts" / "cache" / "polymer_graphs_v1.pt"
    )
    rows, fold_ids, _ = frozen_engine._load_frozen_inputs(train_csv, folds_csv)
    selected = [index for index, assigned in enumerate(fold_ids) if assigned == fold][:16]
    batch = Batch.from_data_list([graphs[index] for index in selected]).to(device)
    with torch.no_grad():
        hidden = model.encode_nodes(batch)
        prediction = model(batch)
        sums_by_target = {}
        for target in TARGETS:
            weights = model.attention_weights(
                hidden,
                batch.batch,
                target=target if variant == "r2_property" else None,
            )
            sums = global_add_pool(weights, batch.batch)
            sums_by_target[target] = sums.detach().cpu().flatten().tolist()
            if not torch.isfinite(weights).all() or not torch.allclose(
                sums, torch.ones_like(sums), atol=1e-6, rtol=1e-6
            ):
                raise RuntimeError("CUDA attention weights failed graph-wise softmax checks")
        if not torch.isfinite(hidden).all() or not torch.isfinite(prediction).all():
            raise RuntimeError("CUDA smoke produced non-finite activations or predictions")
    model_metadata = _write_model_metadata(output_dir, variant=variant, config=config)
    gate_norms = model.gate_l2_norms()
    if not gate_norms or any(value <= 0.0 for value in gate_norms.values()):
        raise RuntimeError("CUDA smoke gate weights did not update from zero initialization")
    runtime = json.loads((output_dir / "config.json").read_text(encoding="utf-8"))["runtime"]
    device_name = runtime.get("device_name")
    if not torch.cuda.is_available() or device.type != "cuda" or "RTX 4070" not in str(device_name):
        raise RuntimeError("Stage 3C CUDA smoke did not use an NVIDIA RTX 4070")
    result = {
        "status": "passed",
        "stage": "3C",
        "readout_variant": STAGE3C_MODELS[variant],
        "seed": int(config["seed"]),
        "source_commit": frozen_engine._git_output("rev-parse", "HEAD"),
        "effective_config_sha256": sha256_file(output_dir / "config.json"),
        "device": str(device),
        "device_name": device_name,
        "fold": fold,
        "parameter_count": model_metadata["parameter_count"],
        "gate_l2_norms_after_smoke": gate_norms,
        "attention_sum_per_graph_by_target": sums_by_target,
        "finite_activations_and_predictions": True,
        "artifacts_under_smoke_path": "smoke" in output_dir.parts,
        "selection_role": "non-selection CUDA execution check",
    }
    write_json(output_dir / "smoke_verification.json", result)
    fold_metadata.update(result)
    write_json(output_dir / "smoke_fold_metadata.json", fold_metadata)
    write_json(
        output_dir / "source_manifest.json",
        _source_manifest(
            result["source_commit"],
            train_csv,
            folds_csv,
            variant_root=_safe_variant_root(variant),
            variant=variant,
            config=json.loads((output_dir / "config.json").read_text(encoding="utf-8")),
        ),
    )


def _safe_variant_root(variant: str) -> Path:
    if variant not in STAGE3C_MODELS:
        raise ValueError(f"Unknown Stage 3C variant: {variant}")
    return MODEL_ROOT / variant


def _default_output_dir(variant_root: Path, seed: int) -> Path:
    if "--tiny-overfit" in sys.argv:
        return variant_root / "artifacts" / "tiny" / f"seed_{seed}"
    if "--smoke-fold" in sys.argv:
        return variant_root / "artifacts" / "smoke" / f"seed_{seed}"
    return _output_dir(variant_root, seed)


def main_for_readout(variant: str) -> None:
    variant_root = _safe_variant_root(variant).resolve()
    readout_variant = STAGE3C_MODELS[variant]

    def load_config(seed_override: int | None = None) -> dict[str, Any]:
        return _load_readout_config(
            variant_root=variant_root, variant=variant, seed_override=seed_override
        )

    def construct(**kwargs: Any) -> ReadoutGNN:
        return ReadoutGNN(readout_variant=readout_variant, **kwargs)

    bound_graph_builder = representation_runner._graph_builder_for(
        GRAPH_SCHEMA, REPRESENTATION, build_polymer_graph
    )
    replacements = {
        "MODEL_ROOT": variant_root,
        "GRAPH_SCHEMA": GRAPH_SCHEMA,
        "build_polymer_graph": bound_graph_builder,
        "_load_config": load_config,
        "_model_config": _model_kwargs,
        "OwnGNNv0": construct,
        "_build_graphs": _graph_builder_for_readout(variant_root),
        "_source_manifest": partial(
            _source_manifest, variant_root=variant_root, variant=variant
        ),
        "_append_result": _append_registry_row,
        "_default_output_dir": lambda model_root, seed, configured_seed=42: _default_output_dir(
            variant_root, seed
        ),
    }
    previous = {name: getattr(frozen_engine, name) for name in replacements}
    try:
        for name, value in replacements.items():
            setattr(frozen_engine, name, value)
        args = frozen_engine.parse_args()
        config = load_config(args.seed)
        args.output_dir = args.output_dir.resolve()

        if args.tiny_overfit:
            if args.device != "cpu":
                raise ValueError("Stage 3C tiny-overfit must explicitly run on CPU")
            if "tiny" not in args.output_dir.parts:
                raise ValueError("Tiny-overfit artifacts must be isolated under a tiny path")
            if (args.output_dir / "tiny_overfit" / "tiny_overfit.json").exists():
                raise FileExistsError("This Stage 3C CPU tiny-overfit path is already complete")
            frozen_engine._run_tiny_overfit(args)
            tiny_result_path = args.output_dir / "tiny_overfit" / "tiny_overfit.json"
            tiny_result = json.loads(tiny_result_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(
                tiny_result["checkpoint_path"], map_location="cpu", weights_only=True
            )
            model = construct(**_model_kwargs(config))
            model.load_state_dict(checkpoint["state_dict"])
            gate_norms = model.gate_l2_norms()
            if not gate_norms or any(value <= 0 for value in gate_norms.values()):
                raise RuntimeError("CPU tiny-overfit did not move every attention gate")
            tiny_result.update(
                {
                    "readout_variant": readout_variant,
                    "gate_l2_norms_after_tiny_overfit": gate_norms,
                    "selection_role": "non-selection CPU execution check",
                }
            )
            write_json(tiny_result_path, tiny_result)
        elif args.smoke_fold is not None:
            if args.device != "cuda":
                raise ValueError("Stage 3C RTX smoke must explicitly request --device cuda")
            if int(config["seed"]) != 43:
                raise ValueError("Stage 3C CUDA smoke uses the designated seed 43")
            if "smoke" not in args.output_dir.parts:
                raise ValueError("CUDA smoke artifacts must be isolated under a smoke path")
            if (args.output_dir / f"gpu_smoke_fold_{args.smoke_fold}" / "smoke_fold_metadata.json").exists():
                raise FileExistsError("This Stage 3C CUDA smoke path is already complete")
            frozen_engine._require_clean_source()
            frozen_engine._run_smoke_fold(args)
            smoke_path = args.output_dir / f"gpu_smoke_fold_{args.smoke_fold}"
            _smoke_attention_check(
                variant=variant,
                config=json.loads((smoke_path / "config.json").read_text(encoding="utf-8")),
                output_dir=smoke_path,
                device=torch.device("cuda"),
                fold=int(args.smoke_fold),
                train_csv=args.train_csv,
                folds_csv=args.folds_csv,
            )
        else:
            if int(config["seed"]) not in FORMAL_SEEDS:
                raise ValueError(f"Formal Stage 3C seeds are frozen to {FORMAL_SEEDS}")
            if args.device != "cuda":
                raise ValueError("Formal Stage 3C runs must explicitly request --device cuda")
            if any(value is not None for value in (args.epochs, args.patience, args.batch_size)):
                raise ValueError("Formal Stage 3C training settings cannot be overridden")
            expected_seed_root = (variant_root / "artifacts" / "formal" / f"seed_{config['seed']}").resolve()
            if expected_seed_root not in args.output_dir.parents:
                raise ValueError("Formal artifacts must use the variant- and seed-specific path")
            frozen_engine._require_clean_source()
            frozen_engine._run_formal(args)
            effective_config = json.loads((args.output_dir / "config.json").read_text(encoding="utf-8"))
            _augment_formal_run(
                variant=variant,
                config=effective_config,
                output_dir=args.output_dir,
                train_csv=args.train_csv,
                folds_csv=args.folds_csv,
                device=torch.device("cuda"),
            )
    finally:
        for name, value in previous.items():
            setattr(frozen_engine, name, value)
