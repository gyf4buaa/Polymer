"""Parameter-count and config-diff audit for the four frozen widths."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ...models.polymer_representation_ablation.model import OwnGNNRepresentation
from ...src.data import write_json
from .protocol import EXPERIMENT_ROOT, MODEL_ROOT, WIDTHS, config_diff_paths, run_config

HISTORICAL_C256_PARAMETER_COUNT = 1_243_657
ALLOWED_DIFFS = {"experiment_id", "seed", "model.hidden_dim"}


def parameter_breakdown(width: int) -> dict[str, int]:
    config = run_config(width, 42)
    model_config = config["model"]
    model = OwnGNNRepresentation(
        hidden_dim=int(model_config["hidden_dim"]),
        num_layers=int(model_config["num_layers"]),
        dropout=float(model_config["dropout"]),
    )
    counts = {
        "embedding_parameters": sum(
            parameter.numel()
            for parameter in model.atom_embeddings.parameters()
            if parameter.requires_grad
        ) + model.polymer_endpoint_embedding.numel(),
        "gine_trunk_parameters": sum(
            parameter.numel()
            for module in (model.convs, model.norms)
            for parameter in module.parameters()
            if parameter.requires_grad
        ),
        "readout_head_parameters": sum(
            parameter.numel()
            for parameter in model.heads.parameters()
            if parameter.requires_grad
        ),
    }
    counts["total_trainable_parameters"] = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    if sum(counts[key] for key in counts if key != "total_trainable_parameters") != counts[
        "total_trainable_parameters"
    ]:
        raise AssertionError("Parameter groups do not partition every trainable parameter")
    return counts


def config_diff_audit() -> dict[str, Any]:
    configs = {str(width): run_config(width, 42) for width in WIDTHS}
    reference = configs["256"]
    comparisons: dict[str, Any] = {}
    for width in WIDTHS:
        changed = config_diff_paths(reference, configs[str(width)])
        unexpected = sorted(path for path in changed if path not in ALLOWED_DIFFS)
        if unexpected:
            raise AssertionError(
                f"C{width} differs from C256 outside the frozen allow-list: {unexpected}"
            )
        comparisons[str(width)] = {
            "changed_paths_vs_c256": changed,
            "unexpected_paths": unexpected,
            "passed": not unexpected,
        }
    seed_configs = [run_config(128, seed) for seed in (42, 43)]
    seed_diff = config_diff_paths(seed_configs[0], seed_configs[1])
    unexpected_seed = sorted(path for path in seed_diff if path not in {"experiment_id", "seed"})
    if unexpected_seed:
        raise AssertionError(f"Seed changed frozen config fields: {unexpected_seed}")
    return {
        "reference": "C256 effective Stage 5A config; architecture/training matches historical G0",
        "allowed_diff_paths": sorted(ALLOWED_DIFFS),
        "width_comparisons": comparisons,
        "seed_comparison": {
            "changed_paths": seed_diff,
            "unexpected_paths": unexpected_seed,
            "passed": not unexpected_seed,
        },
    }


def build_audit() -> dict[str, Any]:
    parameters = {str(width): parameter_breakdown(width) for width in WIDTHS}
    totals = [parameters[str(width)]["total_trainable_parameters"] for width in WIDTHS]
    if not all(left < right for left, right in zip(totals, totals[1:])):
        raise AssertionError(f"Parameter counts are not strictly increasing: {totals}")
    if parameters["256"]["total_trainable_parameters"] != HISTORICAL_C256_PARAMETER_COUNT:
        raise AssertionError(
            "C256 parameter count differs from historical G0: "
            f"{parameters['256']['total_trainable_parameters']} != "
            f"{HISTORICAL_C256_PARAMETER_COUNT}"
        )
    return {
        "stage": "5A",
        "architecture": {
            "operator": "GINEConv",
            "num_layers": 4,
            "readout": ["global_mean", "global_max"],
            "heads": ["Tg", "FFV", "Tc", "Density", "Rg"],
        },
        "parameter_counts": parameters,
        "strictly_increasing": True,
        "historical_c256_count_verified": True,
        "config_diff": config_diff_audit(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=EXPERIMENT_ROOT / "parameter_audit.json",
    )
    args = parser.parse_args()
    payload = build_audit()
    write_json(args.output.resolve(), payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
