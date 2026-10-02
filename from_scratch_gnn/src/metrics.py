"""The single authoritative competition-style OOF metric implementation."""
from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

TARGETS = ("Tg", "FFV", "Tc", "Density", "Rg")


def _number(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, str) and value.strip().lower() in {
        "", "na", "n/a", "nan", "null", "none"
    }:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Expected a numeric value for {context}; got {value!r}.") from exc
    if math.isnan(result):
        return None
    if not math.isfinite(result):
        raise ValueError(f"Expected a finite value for {context}; got {value!r}.")
    return result


def competition_weights(
    y_true: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, float | int]]:
    """Compute Kaggle-style weights from this benchmark's observed labels.

    Local benchmark counts/ranges come from the official training CSV because
    hidden test labels are unavailable and are never read.
    """
    if not y_true:
        raise ValueError("Cannot compute metric weights for an empty dataset.")

    count_by_target: dict[str, int] = {}
    range_by_target: dict[str, float] = {}
    for target in TARGETS:
        values = [
            value
            for row_index, row in enumerate(y_true)
            if (value := _number(row.get(target), context=f"truth[{row_index}].{target}")
            ) is not None
        ]
        if not values:
            raise ValueError(f"Target {target} has no observed labels.")
        count_by_target[target] = len(values)
        value_range = max(values) - min(values)
        if value_range <= 0:
            raise ValueError(f"Target {target} has a zero or negative range.")
        range_by_target[target] = value_range

    normalizer = sum(math.sqrt(1.0 / count_by_target[target]) for target in TARGETS)
    weights: dict[str, dict[str, float | int]] = {}
    for target in TARGETS:
        count = count_by_target[target]
        balance = len(TARGETS) * math.sqrt(1.0 / count) / normalizer
        weight = balance / range_by_target[target]
        weights[target] = {
            "valid_count": count,
            "value_range": range_by_target[target],
            "task_balance_factor": balance,
            "weight": weight,
        }
    return weights


def evaluate_oof(
    y_true: Sequence[Mapping[str, Any]],
    y_pred: Sequence[Mapping[str, Any]],
    *,
    target_weights: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Score aligned OOF rows, masking missing truth labels.

    Kaggle-style rule:
      wMAE = (1/N) * sum_rows sum_observed_targets(weight[target] * abs(error))
    Each reported target MAE uses only observed labels for that target. Pass
    target_weights from the frozen manifest when scoring one held-out fold; the
    default derives weights from the supplied full truth set.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Truth/prediction row counts differ: {len(y_true)} != {len(y_pred)}."
        )
    if not y_true:
        raise ValueError("Cannot score an empty OOF dataset.")

    if target_weights is None:
        weights = competition_weights(y_true)
    else:
        if set(target_weights) != set(TARGETS):
            raise ValueError(f"target_weights must contain exactly {TARGETS}.")
        weights = {}
        for target in TARGETS:
            supplied = target_weights[target]
            supplied_weight = supplied.get("weight") if isinstance(supplied, Mapping) else supplied
            numeric_weight = _number(supplied_weight, context=f"target_weights.{target}")
            if numeric_weight is None or numeric_weight <= 0:
                raise ValueError(f"target_weights.{target} must be finite and positive.")
            weights[target] = (
                dict(supplied) if isinstance(supplied, Mapping) else {"weight": numeric_weight}
            )
            weights[target]["weight"] = numeric_weight
    absolute_error_sum = {target: 0.0 for target in TARGETS}
    valid_count = {target: 0 for target in TARGETS}

    for row_index, (truth_row, prediction_row) in enumerate(zip(y_true, y_pred)):
        for target in TARGETS:
            truth = _number(
                truth_row.get(target), context=f"truth[{row_index}].{target}"
            )
            if truth is None:
                continue
            prediction = _number(
                prediction_row.get(target), context=f"prediction[{row_index}].{target}"
            )
            if prediction is None:
                raise ValueError(
                    f"Missing prediction for observed {target} label at row {row_index}."
                )
            absolute_error_sum[target] += abs(prediction - truth)
            valid_count[target] += 1

    row_count = len(y_true)
    target_mae = {
        target: (
            absolute_error_sum[target] / valid_count[target]
            if valid_count[target]
            else None
        )
        for target in TARGETS
    }
    target_contribution = {
        target: (
            float(weights[target]["weight"]) * absolute_error_sum[target] / row_count
            if valid_count[target]
            else 0.0
        )
        for target in TARGETS
    }
    return {
        "metric": "competition_style_weighted_mae",
        "n_samples": row_count,
        "overall_oof_wmae": sum(target_contribution.values()),
        "target_mae": target_mae,
        "target_contribution": target_contribution,
        "target_counts": valid_count,
        "target_weights": weights,
    }
