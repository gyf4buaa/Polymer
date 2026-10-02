#!/usr/bin/env python3
"""Validate a keyed OOF CSV against the frozen training snapshot and score it."""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

TRACK_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = TRACK_ROOT.parent
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from from_scratch_gnn.src.data import (  # noqa: E402
    TARGETS, load_training_data, sha256_file, write_json
)
from from_scratch_gnn.src.metrics import evaluate_oof  # noqa: E402


def _parse_prediction(value: str | None, *, row_number: int, target: str) -> float | None:
    text = "" if value is None else value.strip()
    if text.lower() in {"", "na", "n/a", "nan", "null", "none"}:
        return None
    try:
        number = float(text)
    except ValueError as exc:
        raise ValueError(
            f"Non-numeric prediction in CSV row {row_number}, target {target}: {value!r}"
        ) from exc
    if math.isnan(number):
        return None
    return number


def load_oof_predictions(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        fieldnames = reader.fieldnames or []
        required = {"sample_id", *TARGETS}
        missing = sorted(required.difference(fieldnames))
        if missing:
            raise ValueError(f"OOF CSV is missing required columns: {missing}")
        if len(fieldnames) != len(set(fieldnames)):
            raise ValueError("OOF CSV contains duplicate column names.")

        result: list[dict[str, Any]] = []
        for row_number, raw in enumerate(reader, start=2):
            row: dict[str, Any] = {
                "sample_id": "" if raw.get("sample_id") is None else raw["sample_id"]
            }
            for target in TARGETS:
                row[target] = _parse_prediction(
                    raw.get(target), row_number=row_number, target=target
                )
            result.append(row)
    return result


def validate_and_score(
    train_csv: Path, oof_csv: Path, output_json: Path
) -> dict[str, Any]:
    truth_rows = load_training_data(train_csv)
    prediction_rows = load_oof_predictions(oof_csv)

    truth_ids = [str(row["sample_id"]) for row in truth_rows]
    prediction_ids = [str(row["sample_id"]) for row in prediction_rows]
    duplicate_truth = sorted(
        sample_id for sample_id, count in Counter(truth_ids).items() if count > 1
    )
    duplicate_predictions = sorted(
        sample_id for sample_id, count in Counter(prediction_ids).items() if count > 1
    )
    truth_id_set = set(truth_ids)
    prediction_id_set = set(prediction_ids)
    blank_truth_ids = [index for index, sample_id in enumerate(truth_ids) if not sample_id.strip()]
    missing_prediction_id = sorted(truth_id_set.difference(prediction_id_set))
    extra_prediction_id = sorted(prediction_id_set.difference(truth_id_set))
    blank_prediction_ids = [
        index for index, sample_id in enumerate(prediction_ids) if not sample_id.strip()
    ]

    problems: list[str] = []
    if blank_truth_ids:
        problems.append(f"training truth has blank sample_id on zero-based rows: {blank_truth_ids[:10]}")
    if duplicate_truth:
        problems.append(f"training truth has duplicate sample_id values: {duplicate_truth[:10]}")
    if duplicate_predictions:
        problems.append(
            f"OOF predictions have duplicate sample_id values: {duplicate_predictions[:10]}"
        )
    if blank_prediction_ids:
        problems.append(
            f"OOF predictions have blank sample_id on zero-based rows: {blank_prediction_ids[:10]}"
        )
    if missing_prediction_id:
        problems.append(
            f"OOF predictions are missing {len(missing_prediction_id)} training sample_id values"
        )
    if extra_prediction_id:
        problems.append(
            f"OOF predictions contain {len(extra_prediction_id)} unknown sample_id values"
        )
    if problems:
        raise ValueError("Invalid OOF coverage:\n- " + "\n- ".join(problems))

    prediction_by_id = {str(row["sample_id"]): row for row in prediction_rows}
    aligned_predictions = [prediction_by_id[sample_id] for sample_id in truth_ids]
    metrics = evaluate_oof(truth_rows, aligned_predictions)
    result = {
        **metrics,
        "validation": {
            "training_sample_count": len(truth_rows),
            "prediction_row_count": len(prediction_rows),
            "duplicate_training_sample_id_count": 0,
            "duplicate_prediction_sample_id_count": 0,
            "missing_prediction_sample_id_count": 0,
            "extra_prediction_sample_id_count": 0,
            "aligned_by": "sample_id",
            "truth_source_sha256": sha256_file(train_csv),
        },
        "input_oof_csv": str(oof_csv),
    }
    write_json(output_json, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--oof-csv", type=Path, required=True)
    parser.add_argument(
        "--output-json", type=Path,
        help="Defaults to metrics.json next to the OOF CSV.",
    )
    args = parser.parse_args()
    output_path = args.output_json or args.oof_csv.with_name("metrics.json")
    result = validate_and_score(args.train_csv, args.oof_csv, output_path)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
