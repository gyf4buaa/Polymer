from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import pytest

from from_scratch_gnn.models.own_gnn_v0 import train_oof


def test_seed_override_changes_only_the_effective_seed():
    source = train_oof._load_config()
    default = train_oof._effective_config(argparse.Namespace(seed=None))
    seeded = train_oof._effective_config(argparse.Namespace(seed=43))

    assert default == source
    assert seeded["seed"] == 43
    seeded["seed"] = source["seed"]
    assert seeded == source
    assert train_oof._load_config() == source


def test_seed_parser_preserves_seed_42_default_and_isolates_added_seeds(
    monkeypatch: pytest.MonkeyPatch,
):
    model_root = train_oof.MODEL_ROOT
    monkeypatch.setattr(sys, "argv", ["train_oof"])
    default_args = train_oof.parse_args()
    assert default_args.seed is None
    assert default_args.output_dir == model_root / "artifacts" / "production"

    monkeypatch.setattr(sys, "argv", ["train_oof", "--seed", "43"])
    seeded_args = train_oof.parse_args()
    assert seeded_args.seed == 43
    assert seeded_args.output_dir == model_root / "artifacts" / "paired_seed_43"
    assert train_oof._default_output_dir(model_root, 44) != seeded_args.output_dir


def test_cross_seed_artifact_overwrite_is_rejected(tmp_path: Path):
    output_dir = tmp_path / "seed_43"
    output_dir.mkdir()
    (output_dir / "config.json").write_text(json.dumps({"seed": 43}))

    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        train_oof._refuse_cross_seed_artifact_reuse(output_dir, seed=44)
    train_oof._refuse_cross_seed_artifact_reuse(output_dir, seed=43)


def test_v0_source_manifest_records_effective_seed_schema_and_hashes(tmp_path: Path):
    config = train_oof._load_config()
    config["seed"] = 45
    train_csv = tmp_path / "train.csv"
    folds_csv = tmp_path / "folds.csv"
    train_csv.write_text("train fixture\n")
    folds_csv.write_text("fold fixture\n")

    manifest = train_oof._source_manifest(
        "stage3a1-test-commit", train_csv, folds_csv, config=config
    )

    assert manifest["git_commit"] == "stage3a1-test-commit"
    assert manifest["seed"] == 45
    assert manifest["graph_schema"] == train_oof.GRAPH_SCHEMA
    assert manifest["representation"] == "endpoint_closure"
    assert manifest["effective_config_sha256"] == train_oof._config_sha256(config)
    assert manifest["benchmark_files_sha256"]["train_csv"] == train_oof.sha256_file(train_csv)
    assert manifest["benchmark_files_sha256"]["folds_csv"] == train_oof.sha256_file(folds_csv)


def test_v0_registry_seed_result_is_unique_and_idempotent(tmp_path: Path):
    track_root = train_oof.TRACK_ROOT
    source = track_root / "results.csv"
    destination = tmp_path / "results.csv"
    with source.open("r", encoding="utf-8-sig", newline="") as original:
        original_rows = list(csv.DictReader(original))
    target_id = "own_gnn_v0_frozen_oof_v1_seed_43"
    original_rows = [row for row in original_rows if row["experiment_id"] != target_id]
    with destination.open("w", encoding="utf-8", newline="") as fixture:
        writer = csv.DictWriter(
            fixture,
            fieldnames=list(original_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(original_rows)

    config = train_oof._load_config()
    config["seed"] = 43
    metrics = {
        "overall_oof_wmae": 0.1,
        "target_mae": {name: 1.0 for name in train_oof.TARGETS},
    }
    metadata_path = track_root / "models/own_gnn_v0/artifacts/seed43/run_metadata.json"
    for score in (0.1, 0.2):
        metrics["overall_oof_wmae"] = score
        train_oof._append_result(
            destination,
            config=config,
            commit="stage3a1-test-commit",
            metrics=metrics,
            folds_sha256="frozen-folds",
            train_sha256="frozen-train",
            run_metadata_path=metadata_path,
        )

    with destination.open("r", encoding="utf-8-sig", newline="") as result:
        rows = list(csv.DictReader(result))
    matches = [
        row for row in rows
        if row["experiment_id"] == target_id
    ]
    assert rows[: len(original_rows)] == original_rows
    assert len(matches) == 1
    assert matches[0]["seed"] == "43"
    assert matches[0]["category"] == "internal_baseline"
    assert matches[0]["oof_wmae"] == "0.2"
