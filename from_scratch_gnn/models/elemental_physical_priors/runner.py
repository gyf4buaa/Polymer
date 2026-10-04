"""Stage 4B adapter around the frozen Stage 0 five-fold training engine."""
from __future__ import annotations

import csv
import json
import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from ...scripts.audit_stage4b_elements import load_smiles_only, sha256_file
from ...src.data import write_json
from ...src.metrics import TARGETS
from ..global_information_augmentation.model import GlobalInformationGNN
from ..own_gnn_repr_keep_dummy.graph import GRAPH_SCHEMA, REPRESENTATION, build_polymer_graph
from ..polymer_representation_ablation import runner as representation_runner
from ..own_gnn_v0 import train_oof as frozen_engine
from .features import (
    FEATURE_NAMES,
    PHYSICAL_FEATURE_DIM,
    build_element_feature_manifest,
    manifest_json_bytes,
    normalize_atomic_numbers,
)
from .model import ElementalPhysicalGNN

TRACK_ROOT = frozen_engine.TRACK_ROOT
MODEL_ROOT = TRACK_ROOT / "models" / "elemental_physical_priors"
BASELINE_ROOT = TRACK_ROOT / "models" / "own_gnn_repr_keep_dummy"
EXPERIMENT_ROOT = TRACK_ROOT / "experiments" / "stage4b"
FEATURE_MANIFEST_PATH = EXPERIMENT_ROOT / "element_feature_manifest.json"
FORMAL_SEEDS = (42, 43, 44, 45, 46)


def _expected_physical_config() -> dict[str, Any]:
    return {
        "feature_names_in_order": list(FEATURE_NAMES),
        "source": "fixed RDKit PeriodicTable APIs for atomic numbers 1..118",
        "input_dim": PHYSICAL_FEATURE_DIM,
        "normalization": "benchmark-independent fixed periodic-table reference Z=1..118; population mean/std",
        "fusion": "h_cat + Linear(5, 256, bias=False)(x_physical) before GINE; node-wise additive residual",
        "projection_initialization": "all_zero; created after all G0 common parameters",
        "dummy_atomic_number_zero": "physical vector fixed to five zeros",
    }


def _validate_config(
    config: Mapping[str, Any], *, baseline: Mapping[str, Any] | None = None
) -> None:
    if baseline is None:
        baseline = json.loads((BASELINE_ROOT / "config.json").read_text(encoding="utf-8"))
    for key in ("benchmark_version", "seed", "training", "graph", "category"):
        expected = "own_model" if key == "category" else baseline[key]
        if config.get(key) != expected:
            raise ValueError(f"Stage 4B changed the frozen {key} setting")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Stage 4B model config must be an object")
    baseline_model = baseline["model"]
    if set(model) != set(baseline_model) | {"elemental_physical"}:
        raise ValueError("Stage 4B may add only the declared elemental_physical config")
    if any(model.get(key) != value for key, value in baseline_model.items()):
        raise ValueError("Stage 4B changed a frozen G0 architecture setting")
    if model.get("elemental_physical") != _expected_physical_config():
        raise ValueError("Stage 4B physical feature configuration differs from registration")


def _load_config(seed_override: int | None = None) -> dict[str, Any]:
    config = json.loads((MODEL_ROOT / "config.json").read_text(encoding="utf-8"))
    _validate_config(config)
    if seed_override is not None:
        if int(seed_override) not in FORMAL_SEEDS:
            raise ValueError(f"Stage 4B seeds are frozen to {FORMAL_SEEDS}")
        config["seed"] = int(seed_override)
    return config


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config["model"][key] for key in ("hidden_dim", "num_layers", "dropout")}


def _parameter_counts(config: Mapping[str, Any]) -> dict[str, int]:
    kwargs = _model_kwargs(config)
    e_model = ElementalPhysicalGNN(**kwargs)
    g0_model = GlobalInformationGNN(variant="G0", **kwargs)
    e_total = sum(parameter.numel() for parameter in e_model.parameters() if parameter.requires_grad)
    g0_total = sum(parameter.numel() for parameter in g0_model.parameters() if parameter.requires_grad)
    projection_count = sum(
        parameter.numel()
        for parameter in e_model.physical_projection.parameters()
        if parameter.requires_grad
    )
    expected = PHYSICAL_FEATURE_DIM * int(config["model"]["hidden_dim"])
    if projection_count != expected or e_total - g0_total != expected:
        raise RuntimeError(
            f"Unexpected Stage 4B parameter difference: expected {expected}, "
            f"got total delta={e_total - g0_total}, projection={projection_count}"
        )
    return {
        "G0_total_trainable": g0_total,
        "E_total_trainable": e_total,
        "physical_projection_trainable": projection_count,
        "new_parameters_vs_G0": e_total - g0_total,
    }


def _output_dir(seed: int, phase: str) -> Path:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Stage 4B seeds are frozen to {FORMAL_SEEDS}")
    phase_dir = {"formal": "formal", "smoke": "smoke", "tiny": "tiny"}.get(phase)
    if phase_dir is None:
        raise ValueError(f"Unknown Stage 4B artifact phase: {phase}")
    seed_root = MODEL_ROOT / "artifacts" / phase_dir / f"seed_{seed}"
    seed_root.mkdir(parents=True, exist_ok=True)
    if phase != "formal":
        return seed_root
    attempts = sorted(seed_root.glob("attempt_*"))
    for attempt in attempts:
        if (attempt / "run_metadata.json").is_file():
            raise FileExistsError(f"A completed formal seed {seed} already exists: {attempt}")
    next_index = 1 + max(
        (
            int(path.name.split("_")[-1])
            for path in attempts
            if path.name.split("_")[-1].isdigit()
        ),
        default=0,
    )
    return seed_root / f"attempt_{next_index:02d}"


def _graph_cache_path(output_dir: Path) -> Path:
    """Keep cache writes isolated to one seed/phase/attempt directory."""
    return output_dir.resolve() / "cache" / "polymer_graphs_v1.pt"


def _default_output_dir(model_root: Path, seed: int, configured_seed: int = 42) -> Path:
    phase = "tiny" if "--tiny-overfit" in sys.argv else "smoke" if "--smoke-fold" in sys.argv else "formal"
    return _output_dir(int(seed), phase)


def _read_and_validate_manifest(train_csv: Path) -> dict[str, Any]:
    smiles = load_smiles_only(train_csv)
    current = build_element_feature_manifest(
        smiles,
        source_train_sha256=sha256_file(train_csv),
    )
    if not FEATURE_MANIFEST_PATH.is_file():
        raise FileNotFoundError(
            f"Run the Stage 4B element audit and commit its manifest first: {FEATURE_MANIFEST_PATH}"
        )
    saved = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
    if saved != current:
        raise ValueError("Runtime Stage 4B audit differs from the committed feature manifest")
    if current["source"]["sample_count"] != 7973:
        raise RuntimeError("Stage 4B audit does not cover all 7,973 frozen source SMILES")
    coverage = current["coverage"]
    if not coverage["all_observed_real_elements_finite_all_five"]:
        raise RuntimeError("An observed real element lacks a finite five-feature vector")
    if len(coverage["observed_element_table"]) == 0:
        raise RuntimeError("Stage 4B observed-element audit is empty")
    return current


def _feature_aware_dataset_factory(original_factory, *, manifest: Mapping[str, Any]):
    lookup = normalize_atomic_numbers(np.arange(119, dtype=np.int64), _reference_from_manifest(manifest))

    def make_dataset(
        graphs: Sequence[Data],
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        stats: Mapping[str, Any],
    ):
        dataset = original_factory(graphs, rows, indices, stats)
        if len(dataset) != len(indices):
            raise RuntimeError("Stage 4B dataset size differs from the requested source indices")
        for item, source_index in zip(dataset, indices):
            # The feature vector is derived from the graph's preserved Z field.
            atomic_numbers = item.x[:, 0].detach().cpu().numpy().astype(np.int64, copy=False)
            if np.any(atomic_numbers >= len(lookup)):
                raise RuntimeError("Graph contains an atomic number outside the audited lookup")
            item.physical_features = torch.as_tensor(
                lookup[atomic_numbers], dtype=torch.float32
            )
            if item.physical_features.shape != (item.x.size(0), PHYSICAL_FEATURE_DIM):
                raise RuntimeError(f"Physical feature alignment failed at source row {source_index}")
            dummy_mask = item.x[:, 0] == 0
            if dummy_mask.any() and not torch.equal(
                item.physical_features[dummy_mask],
                torch.zeros((int(dummy_mask.sum()), PHYSICAL_FEATURE_DIM), dtype=torch.float32),
            ):
                raise RuntimeError("Dummy nodes must receive an exact all-zero physical vector")
            if not torch.isfinite(item.physical_features).all():
                raise RuntimeError("Dataset contains non-finite elemental physical features")
        return dataset

    return make_dataset


def _reference_from_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    ref = manifest["features"]["reference"]
    rows = ref["table_rows"]
    raw = np.asarray(
        [
            [row["atomic_number"], *[row["values"][name] for name in FEATURE_NAMES]]
            for row in rows
        ],
        dtype=np.float64,
    )
    means = np.asarray([ref["mean_ref"][name] for name in FEATURE_NAMES], dtype=np.float64)
    stds = np.asarray([ref["std_ref"][name] for name in FEATURE_NAMES], dtype=np.float64)
    return {"_numeric_table": raw, "_means": means, "_stds": stds}


def _source_manifest(
    commit: str,
    train_csv: Path,
    folds_csv: Path,
    *,
    config: Mapping[str, Any],
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    prefixes = (
        "from_scratch_gnn/models/elemental_physical_priors/",
        "from_scratch_gnn/scripts/audit_stage4b_elements.py",
        "from_scratch_gnn/scripts/aggregate_stage4b.py",
        "from_scratch_gnn/scripts/run_stage4b_formal_queue.py",
        "from_scratch_gnn/models/polymer_representation_ablation/",
        "from_scratch_gnn/models/own_gnn_repr_keep_dummy/",
        "from_scratch_gnn/models/own_gnn_v0/",
        "from_scratch_gnn/models/global_information_augmentation/model.py",
        "from_scratch_gnn/src/data.py",
        "from_scratch_gnn/src/metrics.py",
        "from_scratch_gnn/scripts/evaluate_oof.py",
        "from_scratch_gnn/benchmark/data_manifest.json",
        "from_scratch_gnn/benchmark/folds.csv",
        "from_scratch_gnn/experiments/stage4b/element_feature_manifest.json",
    )
    tracked = frozen_engine._git_output("ls-files", *prefixes).splitlines()
    source_files = {
        relative: sha256_file(frozen_engine.REPOSITORY_ROOT / relative)
        for relative in tracked
        if (frozen_engine.REPOSITORY_ROOT / relative).is_file()
    }
    return {
        "stage": "4B",
        "experiment_id": config["experiment_id"],
        "variant": "E",
        "representation": REPRESENTATION,
        "graph_schema": GRAPH_SCHEMA,
        "seed": int(config["seed"]),
        "git_commit": commit,
        "branch": frozen_engine._git_output("branch", "--show-current"),
        "source_files_sha256": source_files,
        "source_config_sha256": sha256_file(MODEL_ROOT / "config.json"),
        "effective_config_sha256": frozen_engine._config_sha256(config),
        "element_feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "element_feature_manifest": str(FEATURE_MANIFEST_PATH.relative_to(TRACK_ROOT)),
        "reference_table_sha256": feature_manifest["reference_table_sha256"],
        "observed_element_table_sha256": feature_manifest["coverage"][
            "observed_element_table_sha256"
        ],
        "benchmark_files_sha256": {
            "train_csv": sha256_file(train_csv),
            "folds_csv": sha256_file(folds_csv),
            "manifest_json": sha256_file(TRACK_ROOT / "benchmark" / "data_manifest.json"),
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
    seed = int(config["seed"])
    write_json(
        run_metadata_path.parent / "registry_row.json",
        {
            "experiment_id": f"{config['experiment_id']}_seed_{seed}",
            "variant": "E",
            "seed": seed,
            "status": "formal_oof_model",
            "source_commit": commit,
            "train_sha256": train_sha256,
            "folds_sha256": folds_sha256,
            "effective_config_sha256": sha256_file(run_metadata_path.parent / "config.json"),
            "element_feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
            "overall_oof_wmae": metrics["overall_oof_wmae"],
            "target_mae": metrics["target_mae"],
            "run_metadata": str(run_metadata_path.relative_to(TRACK_ROOT)),
        },
    )


def _projection_diagnostics(output_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    norms: dict[str, float] = {}
    columns_by_fold: dict[str, dict[str, float]] = {}
    kwargs = _model_kwargs(config)
    for fold in range(5):
        checkpoint_path = output_dir / "checkpoints" / f"fold_{fold}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = ElementalPhysicalGNN(**kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        norms[str(fold)] = model.projection_l2_norm()
        columns_by_fold[str(fold)] = {
            name: float(value)
            for name, value in zip(FEATURE_NAMES, model.projection_column_l2_norms())
        }
        del model, checkpoint
    feature_means = {
        name: statistics.fmean(columns_by_fold[str(fold)][name] for fold in range(5))
        for name in FEATURE_NAMES
    }
    return {
        "stage": "4B",
        "variant": "E",
        "projection_l2_norm_by_fold": norms,
        "projection_l2_norm_mean_over_folds": statistics.fmean(norms.values()),
        "projection_column_l2_norm_by_fold": columns_by_fold,
        "projection_column_l2_norm_mean_over_folds": feature_means,
        "selection_role": "diagnostic only; no feature selection or model selection",
    }


def _initial_projection_gradient_norm(
    *,
    args,
    config: Mapping[str, Any],
    original_dataset_factory,
    manifest: Mapping[str, Any],
) -> float:
    rows = frozen_engine.load_training_data(args.train_csv.resolve())
    indices = frozen_engine._tiny_sample_indices(rows, limit=32)
    selected_rows = [rows[index] for index in indices]
    stats = frozen_engine._target_normalizer(rows, indices)
    graphs = [
        build_polymer_graph(str(row["SMILES"]), sample_id=str(row["sample_id"]))[0]
        for row in selected_rows
    ]
    dataset_factory = _feature_aware_dataset_factory(
        original_dataset_factory, manifest=manifest
    )
    dataset = dataset_factory(graphs, selected_rows, list(range(len(selected_rows))), stats)
    from torch_geometric.loader import DataLoader

    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    frozen_engine._set_seed(int(config["seed"]))
    model = ElementalPhysicalGNN(**_model_kwargs(config))
    loss = frozen_engine.masked_huber_loss(model(batch), batch.y)
    loss.backward()
    gradient = model.physical_projection.weight.grad
    if gradient is None:
        return 0.0
    return float(gradient.detach().norm().cpu())


def _run() -> None:
    variant_root = MODEL_ROOT.resolve()
    bound_builder = representation_runner._graph_builder_for(
        GRAPH_SCHEMA, REPRESENTATION, build_polymer_graph
    )
    graph_cache_path: Path | None = None
    patch_names = (
        "MODEL_ROOT",
        "GRAPH_SCHEMA",
        "build_polymer_graph",
        "_load_config",
        "_model_config",
        "OwnGNNv0",
        "_build_graphs",
        "_source_manifest",
        "_append_result",
        "_default_output_dir",
        "_make_dataset",
    )
    previous = {name: getattr(frozen_engine, name) for name in patch_names}
    try:
        frozen_engine.MODEL_ROOT = variant_root
        frozen_engine.GRAPH_SCHEMA = GRAPH_SCHEMA
        frozen_engine.build_polymer_graph = bound_builder
        frozen_engine._load_config = lambda: _load_config()
        frozen_engine._model_config = _model_kwargs
        frozen_engine.OwnGNNv0 = ElementalPhysicalGNN

        def build_graphs(rows, *, cache_path: Path, source_sha256: str, diagnostics_path: Path):
            if graph_cache_path is None:
                raise RuntimeError("Stage 4B graph cache path was not isolated for this run")
            return representation_runner._build_graphs_for_variant(
                rows,
                cache_path=graph_cache_path,
                source_sha256=source_sha256,
                diagnostics_path=diagnostics_path,
            )

        frozen_engine._build_graphs = build_graphs
        frozen_engine._append_result = _append_registry_row
        frozen_engine._default_output_dir = lambda model_root, seed, configured_seed=42: _output_dir(
            int(seed),
            "tiny" if "--tiny-overfit" in sys.argv
            else "smoke" if "--smoke-fold" in sys.argv
            else "formal",
        )

        args = frozen_engine.parse_args()
        graph_cache_path = _graph_cache_path(args.output_dir)
        config = _load_config(seed_override=args.seed)
        if args.seed is not None and int(args.seed) not in FORMAL_SEEDS:
            raise ValueError(f"Stage 4B seeds are frozen to {FORMAL_SEEDS}")
        args.output_dir = args.output_dir.resolve()
        if args.device == "auto":
            raise ValueError("Stage 4B requires an explicit --device cpu or --device cuda")
        feature_manifest = _read_and_validate_manifest(args.train_csv.resolve())
        manifest_copy = args.output_dir / "element_feature_manifest.json"
        manifest_copy.parent.mkdir(parents=True, exist_ok=True)
        manifest_copy.write_bytes(manifest_json_bytes(feature_manifest))

        original_dataset_factory = previous["_make_dataset"]
        frozen_engine._make_dataset = _feature_aware_dataset_factory(
            original_dataset_factory, manifest=feature_manifest
        )
        frozen_engine._source_manifest = partial(
            _source_manifest,
            feature_manifest=feature_manifest,
        )

        if args.tiny_overfit:
            if args.device != "cpu":
                raise ValueError("Stage 4B tiny-overfit must explicitly request --device cpu")
            if "tiny" not in args.output_dir.parts:
                raise ValueError("Tiny-overfit artifacts must be isolated under a tiny path")
            gradient_norm = _initial_projection_gradient_norm(
                args=args,
                config=config,
                original_dataset_factory=original_dataset_factory,
                manifest=feature_manifest,
            )
            if not np.isfinite(gradient_norm) or gradient_norm <= 0:
                raise RuntimeError("Physical projection did not receive a finite nonzero initial gradient")
            frozen_engine._run_tiny_overfit(args)
            tiny_path = args.output_dir / "tiny_overfit" / "tiny_overfit.json"
            result = json.loads(tiny_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(result["checkpoint_path"], map_location="cpu", weights_only=True)
            model = ElementalPhysicalGNN(**_model_kwargs(config))
            model.load_state_dict(checkpoint["state_dict"])
            projection_norm = model.projection_l2_norm()
            if projection_norm <= 0:
                raise RuntimeError("Tiny-overfit physical projection did not move from zero")
            result.update(
                {
                    "stage": "4B",
                    "variant": "E",
                    "initial_projection_gradient_l2_norm": gradient_norm,
                    "projection_l2_norm_after_tiny_overfit": projection_norm,
                    "projection_nonzero_gradient": True,
                    "projection_moved_from_zero": True,
                    "selection_role": "non-selection CPU execution check",
                    "element_feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
                }
            )
            write_json(tiny_path, result)
        elif args.smoke_fold is not None:
            if args.device != "cuda":
                raise ValueError("Stage 4B CUDA smoke must explicitly request --device cuda")
            if int(config["seed"]) != 43:
                raise ValueError("Stage 4B CUDA smoke is preregistered to seed 43")
            if "smoke" not in args.output_dir.parts:
                raise ValueError("CUDA smoke artifacts must be isolated under a smoke path")
            frozen_engine._require_clean_source()
            frozen_engine._run_smoke_fold(args)
            smoke_path = args.output_dir / f"gpu_smoke_fold_{int(args.smoke_fold)}"
            _verify_smoke(smoke_path, config, feature_manifest)
        else:
            if int(config["seed"]) not in FORMAL_SEEDS:
                raise ValueError(f"Stage 4B formal seeds are frozen to {FORMAL_SEEDS}")
            if args.device != "cuda":
                raise ValueError("Stage 4B formal runs must explicitly request --device cuda")
            if any(value is not None for value in (args.epochs, args.patience, args.batch_size)):
                raise ValueError("Stage 4B formal training settings cannot be overridden")
            expected_root = (
                variant_root / "artifacts" / "formal" / f"seed_{config['seed']}"
            ).resolve()
            if expected_root not in args.output_dir.parents:
                raise ValueError("Formal artifacts must use the E/seed-specific path")
            counts = _parameter_counts(config)
            frozen_engine._require_clean_source()
            metadata = frozen_engine._run_formal(args)
            output_dir = args.output_dir
            projection = _projection_diagnostics(output_dir, config)
            write_json(output_dir / "elemental_physical_diagnostics.json", projection)
            metadata_path = output_dir / "run_metadata.json"
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored.update(
                {
                    "stage": "4B",
                    "variant": "E",
                    "parameter_count": counts,
                    "element_feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
                    "reference_table_sha256": feature_manifest["reference_table_sha256"],
                    "observed_element_table_sha256": feature_manifest["coverage"][
                        "observed_element_table_sha256"
                    ],
                    "element_feature_manifest_path": str(
                        FEATURE_MANIFEST_PATH.relative_to(TRACK_ROOT)
                    ),
                    "projection_diagnostics_path": "elemental_physical_diagnostics.json",
                }
            )
            write_json(metadata_path, stored)
            write_json(
                output_dir / "source_manifest.json",
                _source_manifest(
                    frozen_engine._git_output("rev-parse", "HEAD"),
                    args.train_csv.resolve(),
                    args.folds_csv.resolve(),
                    config=json.loads((output_dir / "config.json").read_text(encoding="utf-8")),
                    feature_manifest=feature_manifest,
                ),
            )
            _append_registry_row(
                TRACK_ROOT / "results.csv",
                config=json.loads((output_dir / "config.json").read_text(encoding="utf-8")),
                commit=frozen_engine._git_output("rev-parse", "HEAD"),
                metrics=metadata,
                folds_sha256=sha256_file(args.folds_csv.resolve()),
                train_sha256=sha256_file(args.train_csv.resolve()),
                run_metadata_path=metadata_path,
            )
    finally:
        for name, value in previous.items():
            setattr(frozen_engine, name, value)


def _verify_smoke(
    output_dir: Path, config: Mapping[str, Any], feature_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    metadata = json.loads((output_dir / "smoke_fold_metadata.json").read_text(encoding="utf-8"))
    fold_summary = metadata["fold_summary"]
    device = torch.device("cuda")
    if not torch.cuda.is_available() or metadata.get("device") != "cuda":
        raise RuntimeError("Stage 4B smoke did not use CUDA")
    gpu_name = torch.cuda.get_device_name(device)
    if "RTX 4070" not in gpu_name:
        raise RuntimeError(f"Stage 4B smoke used unexpected CUDA device: {gpu_name}")
    if not np.isfinite(float(fold_summary["best_validation_wmae"])):
        raise RuntimeError("Stage 4B smoke produced a non-finite validation score")
    checkpoint = torch.load(fold_summary["checkpoint_path"], map_location=device, weights_only=True)
    model = ElementalPhysicalGNN(**_model_kwargs(config)).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    projection_norm = model.projection_l2_norm()
    if projection_norm <= 0:
        raise RuntimeError("Stage 4B smoke projection did not update from zero")
    result = {
        "status": "passed",
        "stage": "4B",
        "variant": "E",
        "source_commit": frozen_engine._git_output("rev-parse", "HEAD"),
        "device": gpu_name,
        "seed": int(metadata["seed"]),
        "fold": int(metadata["fold"]),
        "requested_epochs": int(metadata["requested_epochs"]),
        "validation_wmae": float(fold_summary["best_validation_wmae"]),
        "projection_l2_norm": projection_norm,
        "physical_input_device_verified": True,
        "finite_forward_and_validation": True,
        "cuda_oom": False,
        "element_feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "reference_table_sha256": feature_manifest["reference_table_sha256"],
        "artifact_phase": "smoke; isolated from formal artifacts",
    }
    write_json(output_dir / "stage4b_cuda_smoke_verification.json", result)
    return result


def main() -> None:
    _validate_config(_load_config())
    _run()
