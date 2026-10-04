"""Frozen Stage 5A config, job matrix, and artifact-path rules."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Mapping

TRACK_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = TRACK_ROOT.parent
MODEL_ROOT = Path(__file__).resolve().parent
EXPERIMENT_ROOT = TRACK_ROOT / "experiments" / "stage5a"
CONFIG_PATH = MODEL_ROOT / "config.json"
G0_CONFIG_PATH = TRACK_ROOT / "models" / "own_gnn_repr_keep_dummy" / "config.json"
WIDTHS = (128, 256, 384, 512)
NEW_WIDTHS = (128, 384, 512)
SEEDS = (42, 43, 44, 45, 46)
TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"
FOLDS_SHA256 = "1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a"
BENCHMARK_VERSION = "nopp2025_train_v1"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _canonical_model(model: Mapping[str, Any]) -> dict[str, Any]:
    normalized = copy.deepcopy(dict(model))
    normalized["hidden_dim"] = 256
    return normalized


def load_base_config() -> dict[str, Any]:
    config = _read_json(CONFIG_PATH)
    validate_frozen_config(config)
    return config


def validate_frozen_config(config: Mapping[str, Any]) -> None:
    """Reject any scientific drift from accepted keep-dummy G0 C256."""
    baseline = _read_json(G0_CONFIG_PATH)
    if config.get("benchmark_version") != BENCHMARK_VERSION:
        raise ValueError("Stage 5A benchmark version is frozen")
    if config.get("benchmark_version") != baseline.get("benchmark_version"):
        raise ValueError("Stage 5A benchmark differs from the G0 benchmark")
    if config.get("graph") != baseline.get("graph"):
        raise ValueError("Stage 5A graph representation or features drifted from G0")
    if config.get("training") != baseline.get("training"):
        raise ValueError("Stage 5A training protocol drifted from G0")
    if _canonical_model(config.get("model", {})) != _canonical_model(
        baseline.get("model", {})
    ):
        raise ValueError("Stage 5A model architecture drifted beyond hidden_dim")
    width = int(config.get("model", {}).get("hidden_dim", 0))
    if width not in WIDTHS:
        raise ValueError(f"Stage 5A hidden_dim must be one of {WIDTHS}, got {width}")
    if int(config.get("seed", -1)) not in SEEDS:
        raise ValueError(f"Stage 5A seed must be one of {SEEDS}")


def run_config(width: int, seed: int) -> dict[str, Any]:
    if width not in WIDTHS:
        raise ValueError(f"Unsupported Stage 5A width: {width}")
    if seed not in SEEDS:
        raise ValueError(f"Unsupported Stage 5A seed: {seed}")
    config = copy.deepcopy(load_base_config())
    config["experiment_id"] = f"stage5a_capacity_c{width}"
    config["model"]["hidden_dim"] = width
    config["seed"] = seed
    validate_frozen_config(config)
    return config


def _walk_differences(left: Any, right: Any, prefix: str = "") -> list[str]:
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        paths: list[str] = []
        for key in sorted(set(left) | set(right)):
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left or key not in right:
                paths.append(path)
            else:
                paths.extend(_walk_differences(left[key], right[key], path))
        return paths
    if isinstance(left, list) and isinstance(right, list):
        if len(left) != len(right):
            return [prefix]
        paths = []
        for index, (left_value, right_value) in enumerate(zip(left, right)):
            paths.extend(_walk_differences(left_value, right_value, f"{prefix}[{index}]"))
        return paths
    return [] if left == right else [prefix]


def config_diff_paths(left: Mapping[str, Any], right: Mapping[str, Any]) -> list[str]:
    return _walk_differences(left, right)


def allowed_config_diff(path: str) -> bool:
    return path in {"experiment_id", "seed", "model.hidden_dim"}


def formal_job_matrix() -> list[dict[str, int]]:
    return [
        {"hidden_dim": width, "seed": seed}
        for seed in SEEDS
        for width in NEW_WIDTHS
    ]


def artifact_dir(phase: str, width: int, seed: int, *, attempt: int = 1) -> Path:
    if phase not in {"formal", "smoke", "tiny"}:
        raise ValueError(f"Unknown Stage 5A artifact phase: {phase}")
    if width not in WIDTHS or seed not in SEEDS:
        raise ValueError("Artifact path width/seed is outside the frozen matrix")
    path = MODEL_ROOT / "artifacts" / phase / f"C{width}" / f"seed_{seed}"
    return path / f"attempt_{attempt:02d}" if phase == "formal" else path


def historical_c256_artifact_dir(seed: int) -> Path:
    if seed not in SEEDS:
        raise ValueError(f"Unexpected historical C256 seed: {seed}")
    root = TRACK_ROOT / "models" / "own_gnn_repr_keep_dummy" / "artifacts"
    if seed == 42:
        return root / "production_gpu_20261003"
    return root / f"paired_seed_{seed}"
