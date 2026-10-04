"""Stage 4A adapter around the frozen Stage 0 training engine."""
from __future__ import annotations

import json
import statistics
import sys
from functools import partial
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
from torch_geometric.data import Data

from ...src.data import sha256_file, write_json
from ...src.metrics import TARGETS
from ..own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA,
    REPRESENTATION,
    build_polymer_graph,
)
from ..polymer_representation_ablation import runner as representation_runner
from ..own_gnn_v0 import train_oof as frozen_engine
from .features import (
    DESCRIPTOR_NAMES,
    fit_descriptor_scaler,
    generate_feature_matrices,
    load_feature_cache,
    save_feature_cache,
    transform_descriptors,
)
from .model import GLOBAL_INPUT_DIMS, GlobalInformationGNN

TRACK_ROOT = frozen_engine.TRACK_ROOT
MODEL_ROOT = TRACK_ROOT / "models" / "global_information_augmentation"
BASELINE_ROOT = TRACK_ROOT / "models" / "own_gnn_repr_keep_dummy"
FEATURE_MANIFEST_PATH = TRACK_ROOT / "experiments" / "stage4a" / "feature_manifest.json"
FORMAL_SEEDS = (42, 43, 44, 45, 46)
VARIANTS = {"descriptor": "D", "morgan": "M"}


def _variant_root(variant: str) -> Path:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported Stage 4A variant: {variant}")
    return MODEL_ROOT / variant


def _expected_global_config(variant: str) -> dict[str, Any]:
    if variant == "descriptor":
        return {
            "variant": "D",
            "source": "fixed RDKit global scalar descriptors",
            "feature_names": list(DESCRIPTOR_NAMES),
            "input_dim": 20,
            "standardization": "fold_train_mean_std",
            "fusion": "z_graph + Linear(20, 512, bias=False)(x_D_std)",
            "projection_initialization": "all_zero",
        }
    return {
        "variant": "M",
        "source": "binary Morgan fingerprint from raw benchmark SMILES",
        "radius": 2,
        "bit_count": 2048,
        "chirality": True,
        "input_dim": 2048,
        "standardization": "none",
        "target_based_selection": False,
        "fusion": "z_graph + Linear(2048, 512, bias=False)(x_M)",
        "projection_initialization": "all_zero",
    }


def _validate_variant_config(
    config: Mapping[str, Any], *, variant: str, baseline: Mapping[str, Any] | None = None
) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported Stage 4A variant: {variant}")
    if baseline is None:
        baseline = json.loads((BASELINE_ROOT / "config.json").read_text(encoding="utf-8"))
    if config.get("benchmark_version") != baseline["benchmark_version"]:
        raise ValueError("Stage 4A changed the frozen benchmark version")
    if config.get("seed") != baseline["seed"]:
        raise ValueError("Stage 4A changed the checked-in base seed")
    if config.get("training") != baseline["training"]:
        raise ValueError("Stage 4A changed the frozen training protocol")
    if config.get("graph") != baseline["graph"]:
        raise ValueError("Stage 4A changed the frozen keep-dummy graph")
    if config.get("category") != "own_model":
        raise ValueError("Stage 4A variants must be registered as owned models")
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("Stage 4A model config must be an object")
    baseline_model = baseline["model"]
    if set(model) != set(baseline_model) | {"global_information"}:
        raise ValueError("Stage 4A may add only the declared global_information config")
    if any(model.get(key) != value for key, value in baseline_model.items()):
        raise ValueError("Stage 4A changed a frozen graph architecture setting")
    if model.get("global_information") != _expected_global_config(variant):
        raise ValueError("Stage 4A global information configuration differs from registration")


def _load_config(
    *, variant_root: Path, variant: str, seed_override: int | None = None
) -> dict[str, Any]:
    config = json.loads((variant_root / "config.json").read_text(encoding="utf-8"))
    _validate_variant_config(config, variant=variant)
    if seed_override is not None:
        if int(seed_override) not in FORMAL_SEEDS:
            raise ValueError(f"Stage 4A seeds are frozen to {FORMAL_SEEDS}")
        config["seed"] = int(seed_override)
    return config


def _model_kwargs(config: Mapping[str, Any]) -> dict[str, Any]:
    return {key: config["model"][key] for key in ("hidden_dim", "num_layers", "dropout")}


def _model_constructor(variant: str):
    code = VARIANTS[variant]

    def construct(**kwargs: Any) -> GlobalInformationGNN:
        return GlobalInformationGNN(variant=code, **kwargs)

    return construct


def _parameter_counts(variant: str, config: Mapping[str, Any]) -> dict[str, int]:
    kwargs = _model_kwargs(config)
    model = _model_constructor(variant)(**kwargs)
    total = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    baseline = GlobalInformationGNN(variant="G0", **kwargs)
    baseline_total = sum(
        parameter.numel() for parameter in baseline.parameters() if parameter.requires_grad
    )
    projection = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("global_projection.") and parameter.requires_grad
    )
    expected = GLOBAL_INPUT_DIMS[VARIANTS[variant]] * (2 * int(config["model"]["hidden_dim"]))
    if total - baseline_total != expected or projection != expected:
        raise RuntimeError(
            f"Unexpected Stage 4A parameter difference: expected {expected}, "
            f"got total delta={total - baseline_total}, projection={projection}"
        )
    return {
        "G0_total_trainable": baseline_total,
        f"{VARIANTS[variant]}_total_trainable": total,
        "projection_trainable": projection,
        "new_parameters_vs_G0": total - baseline_total,
    }


def _feature_cache_path(variant_root: Path) -> Path:
    return variant_root / "artifacts" / "cache" / "global_feature_matrices.npz"


def _load_or_build_feature_data(
    *, variant_root: Path, train_csv: Path, rows: Sequence[Mapping[str, Any]]
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    source_hash = sha256_file(train_csv)
    cache_path = _feature_cache_path(variant_root)
    if cache_path.exists():
        descriptor_matrix, morgan_matrix = load_feature_cache(cache_path)
        manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
        if manifest["source"]["data_sha256"] != source_hash:
            raise ValueError("Feature manifest belongs to a different frozen training CSV")
        if descriptor_matrix.shape != tuple(manifest["descriptors"]["shape"]):
            raise ValueError("Cached descriptor matrix shape differs from the manifest")
        if morgan_matrix.shape != tuple(manifest["morgan"]["shape"]):
            raise ValueError("Cached Morgan matrix shape differs from the manifest")
        from .features import _canonical_matrix_sha256

        if _canonical_matrix_sha256(descriptor_matrix, "<f8") != manifest["descriptors"]["raw_matrix_sha256"]:
            raise ValueError("Cached descriptors differ from the committed matrix hash")
        if _canonical_matrix_sha256(morgan_matrix, "|u1") != manifest["morgan"]["raw_matrix_sha256"]:
            raise ValueError("Cached Morgan fingerprints differ from the committed matrix hash")
    else:
        generated = generate_feature_matrices([str(row["SMILES"]) for row in rows])
        generated.manifest["source"]["data_sha256"] = source_hash
        generated.manifest["descriptors"]["source_data_sha256"] = source_hash
        generated.manifest["morgan"]["source_data_sha256"] = source_hash
        if FEATURE_MANIFEST_PATH.is_file():
            manifest = json.loads(FEATURE_MANIFEST_PATH.read_text(encoding="utf-8"))
            for key in ("rdkit_version", "descriptors", "morgan", "source"):
                if generated.manifest.get(key) != manifest.get(key):
                    raise ValueError(f"Runtime feature audit differs from source manifest: {key}")
        else:
            raise FileNotFoundError(
                f"Committed Stage 4A feature audit is missing: {FEATURE_MANIFEST_PATH}"
            )
        descriptor_matrix, morgan_matrix, manifest = (
            generated.descriptors,
            generated.morgan,
            generated.manifest,
        )
        save_feature_cache(cache_path, generated)
    if manifest["descriptors"]["nonfinite_count"] != 0:
        raise RuntimeError("Descriptor audit contains non-finite values; formal runs are blocked")
    if len(rows) != 7973 or descriptor_matrix.shape != (7973, 20) or morgan_matrix.shape != (7973, 2048):
        raise RuntimeError("Stage 4A features do not cover exactly 7,973 frozen rows")
    return descriptor_matrix, morgan_matrix, manifest


def _output_dir(variant_root: Path, seed: int, phase: str) -> Path:
    if seed not in FORMAL_SEEDS:
        raise ValueError(f"Stage 4A seeds are frozen to {FORMAL_SEEDS}")
    if phase == "formal":
        seed_root = variant_root / "artifacts" / "formal" / f"seed_{seed}"
    elif phase == "smoke":
        seed_root = variant_root / "artifacts" / "smoke" / f"seed_{seed}"
    elif phase == "tiny":
        seed_root = variant_root / "artifacts" / "tiny" / f"seed_{seed}"
    else:
        raise ValueError(f"Unknown Stage 4A artifact phase: {phase}")
    seed_root.mkdir(parents=True, exist_ok=True)
    if phase != "formal":
        return seed_root
    attempts = sorted(seed_root.glob("attempt_*"))
    for attempt in attempts:
        if (attempt / "run_metadata.json").is_file():
            raise FileExistsError(f"A completed formal run already exists: {attempt}")
    next_index = 1 + max(
        (int(path.name.split("_")[-1]) for path in attempts if path.name.split("_")[-1].isdigit()),
        default=0,
    )
    return seed_root / f"attempt_{next_index:02d}"


def _default_output_dir(variant_root: Path, seed: int, configured_seed: int = 42) -> Path:
    if "--tiny-overfit" in sys.argv:
        return _output_dir(variant_root, seed, "tiny")
    if "--smoke-fold" in sys.argv:
        return _output_dir(variant_root, seed, "smoke")
    return _output_dir(variant_root, seed, "formal")


def _source_manifest(
    commit: str,
    train_csv: Path,
    folds_csv: Path,
    *,
    variant_root: Path,
    variant: str,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    prefixes = (
        "from_scratch_gnn/models/global_information_augmentation/",
        "from_scratch_gnn/scripts/audit_stage4a_features.py",
        "from_scratch_gnn/scripts/aggregate_stage4a.py",
        "from_scratch_gnn/models/polymer_representation_ablation/",
        "from_scratch_gnn/models/own_gnn_repr_keep_dummy/",
        "from_scratch_gnn/models/own_gnn_v0/",
        "from_scratch_gnn/src/data.py",
        "from_scratch_gnn/src/metrics.py",
        "from_scratch_gnn/scripts/evaluate_oof.py",
        "from_scratch_gnn/benchmark/data_manifest.json",
        "from_scratch_gnn/benchmark/folds.csv",
    )
    tracked = frozen_engine._git_output("ls-files", *prefixes).splitlines()
    source_files = {
        relative: sha256_file(frozen_engine.REPOSITORY_ROOT / relative)
        for relative in tracked
        if (frozen_engine.REPOSITORY_ROOT / relative).is_file()
    }
    return {
        "stage": "4A",
        "experiment_id": config["experiment_id"],
        "variant": VARIANTS[variant],
        "representation": REPRESENTATION,
        "graph_schema": GRAPH_SCHEMA,
        "seed": int(config["seed"]),
        "git_commit": commit,
        "branch": frozen_engine._git_output("branch", "--show-current"),
        "source_files_sha256": source_files,
        "source_config_sha256": sha256_file(variant_root / "config.json"),
        "effective_config_sha256": frozen_engine._config_sha256(config),
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "feature_manifest": str(FEATURE_MANIFEST_PATH.relative_to(TRACK_ROOT)),
        "benchmark_files_sha256": {
            "train_csv": sha256_file(train_csv),
            "folds_csv": sha256_file(folds_csv),
            "manifest_json": sha256_file(TRACK_ROOT / "benchmark" / "data_manifest.json"),
        },
    }


def _write_parameter_metadata(
    output_dir: Path, *, variant: str, config: Mapping[str, Any], feature_manifest: Mapping[str, Any]
) -> dict[str, Any]:
    counts = _parameter_counts(variant, config)
    metadata = {
        "stage": "4A",
        "variant": VARIANTS[variant],
        "parameter_count": counts,
        "graph_readout_dim": 512,
        "fusion": config["model"]["global_information"]["fusion"],
        "projection_initialization": "all_zero",
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "descriptor_nonfinite_count": feature_manifest["descriptors"]["nonfinite_count"],
        "morgan_average_bits_on": feature_manifest["morgan"]["average_bits_on"],
        "selection_role": "formal global-information augmentation; smoke outputs are non-selection",
    }
    write_json(output_dir / "global_information_metadata.json", metadata)
    return metadata


def _registry_experiment_id(config: Mapping[str, Any]) -> str:
    seed = int(config["seed"])
    return (
        str(config["experiment_id"])
        if seed == 42
        else f"{config['experiment_id']}_seed_{seed}"
    )


def _write_projection_diagnostics(
    *, variant: str, config: Mapping[str, Any], output_dir: Path,
    feature_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    model_constructor = _model_constructor(variant)
    kwargs = _model_kwargs(config)
    norms: dict[str, float] = {}
    descriptor_column_norms: dict[str, float] | None = None
    for fold in range(5):
        checkpoint_path = output_dir / "checkpoints" / f"fold_{fold}_best.pt"
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model = model_constructor(**kwargs)
        model.load_state_dict(checkpoint["state_dict"])
        norms[str(fold)] = model.projection_l2_norm()
        if variant == "descriptor":
            assert model.global_projection is not None
            values = model.global_projection.weight.detach().norm(dim=0).cpu().tolist()
            descriptor_column_norms = {
                name: float(value) for name, value in zip(DESCRIPTOR_NAMES, values)
            }
        del model, checkpoint
    result = {
        "stage": "4A",
        "variant": VARIANTS[variant],
        "projection_l2_norm_by_fold": norms,
        "projection_l2_norm_mean": statistics.fmean(norms.values()),
        "descriptor_projection_column_l2_norm_fold_4": descriptor_column_norms,
        "descriptor_train_fold_scaler_files": (
            [f"fold_scalers/fold_{fold}.json" for fold in range(5)]
            if variant == "descriptor"
            else []
        ),
        "morgan_average_bits_on": feature_manifest["morgan"]["average_bits_on"],
        "morgan_duplicate_fingerprint_rows": feature_manifest["morgan"]["duplicate_fingerprint_rows"],
        "selection_role": "diagnostic only; no feature selection or model selection",
    }
    write_json(output_dir / "global_information_diagnostics.json", result)
    return result


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
    # Never append concurrently to results.csv. The single-process aggregator
    # consumes one registry_row.json from each independent attempt directory.
    write_json(
        run_metadata_path.parent / "registry_row.json",
        {
            "experiment_id": _registry_experiment_id(config),
            "variant": config["model"]["global_information"]["variant"],
            "seed": int(config["seed"]),
            "status": "formal_oof_model",
            "source_commit": commit,
            "train_sha256": train_sha256,
            "folds_sha256": folds_sha256,
            "effective_config_sha256": sha256_file(run_metadata_path.parent / "config.json"),
            "overall_oof_wmae": metrics["overall_oof_wmae"],
            "target_mae": metrics["target_mae"],
            "run_metadata": str(run_metadata_path.relative_to(TRACK_ROOT)),
        },
    )


def _write_fold_scaler(
    *,
    output_dir: Path,
    fold: int | None,
    scaler: Mapping[str, Any],
    variant: str,
) -> None:
    if fold is None:
        path = output_dir / "tiny_descriptor_scaler.json"
    else:
        path = output_dir / "fold_scalers" / f"fold_{fold}.json"
    value = dict(scaler)
    value["variant"] = VARIANTS[variant]
    value["fold"] = fold
    write_json(path, value)


def _transformed_features(
    *, variant: str, raw_descriptors: np.ndarray, morgan: np.ndarray,
    scaler: Mapping[str, Any] | None,
) -> np.ndarray:
    if VARIANTS[variant] == "D":
        if scaler is None:
            raise RuntimeError("Descriptor variant must have an active fold-train scaler")
        return transform_descriptors(raw_descriptors, scaler)
    if VARIANTS[variant] == "M":
        return np.asarray(morgan, dtype=np.float32)
    raise ValueError(f"Unsupported Stage 4A variant {variant}")


def _feature_aware_train_fold(
    *,
    original_train_fold,
    variant: str,
    rows: Sequence[Mapping[str, Any]],
    fold_ids: Sequence[int],
    descriptors: np.ndarray,
    morgan: np.ndarray,
    source_data_sha256: str,
    output_dir: Path,
    **kwargs: Any,
):
    fold = int(kwargs["fold"])
    training_indices = [index for index, assigned in enumerate(fold_ids) if assigned != fold]
    scaler = (
        fit_descriptor_scaler(
            descriptors, training_indices, source_data_sha256=source_data_sha256
        )
        if VARIANTS[variant] == "D"
        else None
    )
    if scaler is not None:
        _write_fold_scaler(output_dir=output_dir, fold=fold, scaler=scaler, variant=variant)
    feature_matrix = _transformed_features(
        variant=variant, raw_descriptors=descriptors, morgan=morgan, scaler=scaler
    )
    context = {
        "sample_ids": [str(row["sample_id"]) for row in rows],
        "feature_matrix": feature_matrix,
    }
    previous = _FEATURE_CONTEXT.get("value")
    _FEATURE_CONTEXT["value"] = context
    try:
        return original_train_fold(**kwargs)
    finally:
        if previous is None:
            _FEATURE_CONTEXT.pop("value", None)
        else:
            _FEATURE_CONTEXT["value"] = previous


_FEATURE_CONTEXT: dict[str, Any] = {}


def _feature_aware_dataset_factory(
    original_factory,
    *,
    variant: str,
    all_sample_ids: Sequence[str],
    descriptors: np.ndarray,
    morgan: np.ndarray,
    source_data_sha256: str,
    tiny_output_dir: Path,
    ):
    id_to_full_index = {sample_id: index for index, sample_id in enumerate(all_sample_ids)}

    def make_dataset(
        graphs: Sequence[Data],
        rows: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        stats: Mapping[str, Any],
    ):
        dataset = original_factory(graphs, rows, indices, stats)
        context = _FEATURE_CONTEXT.get("value")
        if context is not None:
            values = context["feature_matrix"]
            sample_ids = context["sample_ids"]
            vectors = [values[id_to_full_index[str(rows[index]["sample_id"])]] for index in indices]
        else:
            sample_ids = [str(rows[index]["sample_id"]) for index in indices]
            full_indices = [id_to_full_index[sample_id] for sample_id in sample_ids]
            scaler = (
                fit_descriptor_scaler(
                    descriptors, full_indices, source_data_sha256=source_data_sha256
                )
                if VARIANTS[variant] == "D"
                else None
            )
            values = _transformed_features(
                variant=variant, raw_descriptors=descriptors, morgan=morgan, scaler=scaler
            )
            vectors = [values[full_index] for full_index in full_indices]
            if scaler is not None:
                _write_fold_scaler(
                    output_dir=tiny_output_dir,
                    fold=None,
                    scaler=scaler,
                    variant=variant,
                )
        if len(dataset) != len(vectors):
            raise RuntimeError("Global feature vectors do not align with dataset samples")
        for item, vector in zip(dataset, vectors):
            item.global_features = torch.as_tensor(vector, dtype=torch.float32).reshape(1, -1)
        return dataset

    return make_dataset


def _run_for_variant(variant: str) -> None:
    variant_root = _variant_root(variant).resolve()
    code = VARIANTS[variant]
    bound_builder = representation_runner._graph_builder_for(
        GRAPH_SCHEMA, REPRESENTATION, build_polymer_graph
    )
    shared_graph_cache = variant_root / "artifacts" / "cache" / "polymer_graphs_v1.pt"
    previous = {
        name: getattr(frozen_engine, name)
        for name in (
            "MODEL_ROOT", "GRAPH_SCHEMA", "build_polymer_graph", "_load_config",
            "_model_config", "OwnGNNv0", "_build_graphs", "_source_manifest",
            "_append_result", "_default_output_dir", "_train_fold", "_make_dataset",
        )
    }
    try:
        frozen_engine.MODEL_ROOT = variant_root
        frozen_engine.GRAPH_SCHEMA = GRAPH_SCHEMA
        frozen_engine.build_polymer_graph = bound_builder
        frozen_engine._load_config = lambda: _load_config(
            variant_root=variant_root, variant=variant
        )
        frozen_engine._model_config = _model_kwargs
        frozen_engine.OwnGNNv0 = _model_constructor(variant)
        def build_graphs(
            rows, *, cache_path: Path, source_sha256: str, diagnostics_path: Path
        ):
            return representation_runner._build_graphs_for_variant(
                rows,
                cache_path=shared_graph_cache,
                source_sha256=source_sha256,
                diagnostics_path=diagnostics_path,
            )

        frozen_engine._build_graphs = build_graphs
        frozen_engine._source_manifest = partial(
            _source_manifest,
            variant_root=variant_root,
            variant=variant,
        )
        frozen_engine._append_result = _append_registry_row
        frozen_engine._default_output_dir = lambda model_root, seed, configured_seed=42: _default_output_dir(
            variant_root, seed, "tiny" if "--tiny-overfit" in sys.argv
            else "smoke" if "--smoke-fold" in sys.argv else "formal"
        )

        args = frozen_engine.parse_args()
        config = _load_config(
            variant_root=variant_root, variant=variant, seed_override=args.seed
        )
        if args.seed is not None and int(args.seed) not in FORMAL_SEEDS:
            raise ValueError(f"Stage 4A seed is frozen to {FORMAL_SEEDS}")
        args.output_dir = args.output_dir.resolve()
        rows = frozen_engine.load_training_data(args.train_csv.resolve())
        source_data_sha256 = sha256_file(args.train_csv.resolve())
        descriptors, morgan, feature_manifest = _load_or_build_feature_data(
            variant_root=variant_root,
            train_csv=args.train_csv.resolve(),
            rows=rows,
        )
        if feature_manifest["source"]["data_sha256"] != source_data_sha256:
            raise ValueError("Stage 4A feature source hash does not match the frozen train CSV")
        if feature_manifest["descriptors"]["nonfinite_count"] != 0:
            raise RuntimeError("Non-finite descriptor audit blocks all Stage 4A training")
        feature_manifest_copy = args.output_dir / "feature_manifest.json"
        feature_manifest_copy.parent.mkdir(parents=True, exist_ok=True)
        write_json(feature_manifest_copy, feature_manifest)

        original_train_fold = previous["_train_fold"]
        original_dataset_factory = previous["_make_dataset"]
        frozen_engine._make_dataset = _feature_aware_dataset_factory(
            original_dataset_factory,
            variant=variant,
            all_sample_ids=[str(row["sample_id"]) for row in rows],
            descriptors=descriptors,
            morgan=morgan,
            source_data_sha256=source_data_sha256,
            tiny_output_dir=args.output_dir,
        )

        def wrapped_train_fold(**kwargs: Any):
            return _feature_aware_train_fold(
                original_train_fold=original_train_fold,
                variant=variant,
                descriptors=descriptors,
                morgan=morgan,
                source_data_sha256=source_data_sha256,
                **kwargs,
            )

        frozen_engine._train_fold = wrapped_train_fold

        if args.tiny_overfit:
            if args.device != "cpu":
                raise ValueError("Stage 4A CPU tiny-overfit must explicitly request --device cpu")
            if "tiny" not in args.output_dir.parts:
                raise ValueError("Tiny-overfit artifacts must be isolated under a tiny path")
            initial_gradient_norm = _initial_projection_gradient_norm(
                variant=variant,
                config=config,
                train_csv=args.train_csv.resolve(),
                descriptors=descriptors,
                morgan=morgan,
                source_data_sha256=source_data_sha256,
            )
            frozen_engine._run_tiny_overfit(args)
            tiny_json_path = args.output_dir / "tiny_overfit" / "tiny_overfit.json"
            tiny_result = json.loads(tiny_json_path.read_text(encoding="utf-8"))
            checkpoint = torch.load(
                tiny_result["checkpoint_path"], map_location="cpu", weights_only=True
            )
            restored = _model_constructor(variant)(**_model_kwargs(config))
            restored.load_state_dict(checkpoint["state_dict"])
            projection_norm = restored.projection_l2_norm()
            if initial_gradient_norm <= 0 or projection_norm <= 0:
                raise RuntimeError("CPU tiny-overfit did not produce/update a global projection")
            tiny_result.update(
                {
                    "variant": code,
                    "initial_projection_gradient_l2_norm": initial_gradient_norm,
                    "projection_l2_norm_after_tiny_overfit": projection_norm,
                    "projection_nonzero_gradient": True,
                    "projection_moved_from_zero": True,
                    "selection_role": "non-selection CPU execution check",
                }
            )
            write_json(tiny_json_path, tiny_result)
        elif args.smoke_fold is not None:
            if args.device != "cuda":
                raise ValueError("Stage 4A CUDA smoke must explicitly request --device cuda")
            if int(config["seed"]) != 43:
                raise ValueError("Stage 4A CUDA smoke uses the designated seed 43")
            if "smoke" not in args.output_dir.parts:
                raise ValueError("CUDA smoke artifacts must be isolated under a smoke path")
            frozen_engine._require_clean_source()
            frozen_engine._run_smoke_fold(args)
            smoke_path = args.output_dir / f"gpu_smoke_fold_{int(args.smoke_fold)}"
            _verify_smoke(
                variant=variant,
                config=json.loads((smoke_path / "config.json").read_text(encoding="utf-8")),
                output_dir=smoke_path,
                feature_manifest=feature_manifest,
                train_csv=args.train_csv.resolve(),
            )
        else:
            if int(config["seed"]) not in FORMAL_SEEDS:
                raise ValueError(f"Stage 4A formal seed is frozen to {FORMAL_SEEDS}")
            if args.device != "cuda":
                raise ValueError("Stage 4A formal runs must explicitly request --device cuda")
            if any(value is not None for value in (args.epochs, args.patience, args.batch_size)):
                raise ValueError("Stage 4A formal training settings cannot be overridden")
            expected_root = (variant_root / "artifacts" / "formal" / f"seed_{config['seed']}").resolve()
            if expected_root not in args.output_dir.parents:
                raise ValueError("Formal artifacts must use the variant/seed-specific path")
            frozen_engine._require_clean_source()
            metadata = frozen_engine._run_formal(args)
            _write_parameter_metadata(
                args.output_dir,
                variant=variant,
                config=json.loads((args.output_dir / "config.json").read_text(encoding="utf-8")),
                feature_manifest=feature_manifest,
            )
            _write_projection_diagnostics(
                variant=variant,
                config=json.loads((args.output_dir / "config.json").read_text(encoding="utf-8")),
                output_dir=args.output_dir,
                feature_manifest=feature_manifest,
            )
            metadata_path = args.output_dir / "run_metadata.json"
            stored = json.loads(metadata_path.read_text(encoding="utf-8"))
            stored.update(
                {
                    "stage": "4A",
                    "variant": code,
                    "parameter_count": _parameter_counts(variant, config),
                    "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
                    "fold_scalers_path": "fold_scalers/" if code == "D" else None,
                    "feature_cache_path": str(_feature_cache_path(variant_root).relative_to(TRACK_ROOT)),
                }
            )
            write_json(metadata_path, stored)
            # Re-write source provenance after the formal metadata enrichments.
            write_json(
                args.output_dir / "source_manifest.json",
                _source_manifest(
                    frozen_engine._git_output("rev-parse", "HEAD"),
                    args.train_csv.resolve(),
                    args.folds_csv.resolve(),
                    variant_root=variant_root,
                    variant=variant,
                    config=json.loads((args.output_dir / "config.json").read_text(encoding="utf-8")),
                ),
            )
            _append_registry_row(
                TRACK_ROOT / "results.csv",
                config=json.loads((args.output_dir / "config.json").read_text(encoding="utf-8")),
                commit=frozen_engine._git_output("rev-parse", "HEAD"),
                metrics=metadata,
                folds_sha256=sha256_file(args.folds_csv.resolve()),
                train_sha256=source_data_sha256,
                run_metadata_path=metadata_path,
            )
    finally:
        for name, value in previous.items():
            setattr(frozen_engine, name, value)


def _verify_smoke(
    *, variant: str, config: Mapping[str, Any], output_dir: Path,
    feature_manifest: Mapping[str, Any], train_csv: Path,
) -> dict[str, Any]:
    metadata = json.loads((output_dir / "smoke_fold_metadata.json").read_text(encoding="utf-8"))
    fold_summary = metadata["fold_summary"]
    checkpoint = torch.load(
        fold_summary["checkpoint_path"], map_location="cuda", weights_only=True
    )
    model = _model_constructor(variant)(**_model_kwargs(config)).to("cuda")
    model.load_state_dict(checkpoint["state_dict"])
    projection_norm = model.projection_l2_norm()
    if not torch.cuda.is_available() or metadata.get("device") != "cuda":
        raise RuntimeError("Stage 4A smoke did not use CUDA")
    if not np.isfinite(float(fold_summary["best_validation_wmae"])):
        raise RuntimeError("Stage 4A CUDA smoke produced a non-finite validation score")
    if projection_norm <= 0:
        raise RuntimeError("Stage 4A smoke projection did not update from zero initialization")
    gpu_name = torch.cuda.get_device_name(torch.device("cuda"))
    if "RTX 4070" not in gpu_name:
        raise RuntimeError(f"Stage 4A smoke used unexpected CUDA device: {gpu_name}")
    result = {
        "status": "passed",
        "stage": "4A",
        "variant": VARIANTS[variant],
        "source_commit": frozen_engine._git_output("rev-parse", "HEAD"),
        "device": gpu_name,
        "feature_source_sha256": sha256_file(train_csv),
        "feature_manifest_sha256": sha256_file(FEATURE_MANIFEST_PATH),
        "feature_manifest_rdkit_version": feature_manifest["rdkit_version"],
        "projection_l2_norm": projection_norm,
        "validation_wmae": float(fold_summary["best_validation_wmae"]),
        "cuda_used": True,
        "finite_validation_metric": True,
        "smoke_artifacts_isolated": "smoke" in output_dir.parts,
        "selection_role": "non-selection CUDA execution check",
    }
    write_json(output_dir / "smoke_verification.json", result)
    metadata.update(result)
    write_json(output_dir / "smoke_fold_metadata.json", metadata)
    return result


def _initial_projection_gradient_norm(
    *,
    variant: str,
    config: Mapping[str, Any],
    train_csv: Path,
    descriptors: np.ndarray,
    morgan: np.ndarray,
    source_data_sha256: str,
) -> float:
    """Check at zero initialization that a tiny real batch reaches the projection."""
    from torch_geometric.loader import DataLoader

    rows = frozen_engine.load_training_data(train_csv)
    indices = frozen_engine._tiny_sample_indices(rows, limit=32)
    selected_rows = [rows[index] for index in indices]
    scaler = (
        fit_descriptor_scaler(descriptors, indices, source_data_sha256=source_data_sha256)
        if VARIANTS[variant] == "D"
        else None
    )
    transformed = _transformed_features(
        variant=variant, raw_descriptors=descriptors, morgan=morgan, scaler=scaler
    )
    stats = frozen_engine._target_normalizer(rows, indices)
    graphs = [
        build_polymer_graph(str(row["SMILES"]), sample_id=str(row["sample_id"]))[0]
        for row in selected_rows
    ]
    dataset = frozen_engine._make_dataset(
        graphs, selected_rows, list(range(len(selected_rows))), stats
    )
    for item, index in zip(dataset, indices):
        item.global_features = torch.as_tensor(transformed[index], dtype=torch.float32).reshape(1, -1)
    batch = next(iter(DataLoader(dataset, batch_size=len(dataset), shuffle=False)))
    frozen_engine._set_seed(int(config["seed"]))
    model = _model_constructor(variant)(**_model_kwargs(config))
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    optimizer.zero_grad(set_to_none=True)
    loss = frozen_engine.masked_huber_loss(model(batch), batch.y)
    loss.backward()
    assert model.global_projection is not None
    gradient = model.global_projection.weight.grad
    if gradient is None:
        return 0.0
    return float(gradient.detach().norm().cpu())


def main_for_variant(variant: str) -> None:
    if variant not in VARIANTS:
        raise ValueError(f"Unsupported Stage 4A variant: {variant}")
    _run_for_variant(variant)
