from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import pytest

from from_scratch_gnn.models.own_gnn_repr_endpoint_marker.graph import (
    GRAPH_SCHEMA as ENDPOINT_SCHEMA,
    REPRESENTATION as ENDPOINT_REPRESENTATION,
    build_polymer_graph as build_endpoint_graph,
)
from from_scratch_gnn.models.own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA as DUMMY_SCHEMA,
    REPRESENTATION as DUMMY_REPRESENTATION,
    build_polymer_graph as build_dummy_graph,
)
from from_scratch_gnn.models.polymer_representation_ablation import runner


@pytest.mark.parametrize(
    ("variant_name", "schema", "representation", "builder", "expected"),
    [
        (
            "own_gnn_repr_keep_dummy",
            DUMMY_SCHEMA,
            DUMMY_REPRESENTATION,
            build_dummy_graph,
            {"fallback_count": 0, "endpoint_graphs": 0, "endpoint_nodes": 0},
        ),
        (
            "own_gnn_repr_endpoint_marker",
            ENDPOINT_SCHEMA,
            ENDPOINT_REPRESENTATION,
            build_endpoint_graph,
            {"fallback_count": 1, "endpoint_graphs": 1, "endpoint_nodes": 2},
        ),
    ],
)
def test_graph_diagnostic_cache_is_representation_specific_and_lossless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    variant_name: str,
    schema: str,
    representation: str,
    builder,
    expected: dict[str, int],
):
    engine = runner.frozen_engine
    monkeypatch.setattr(engine, "GRAPH_SCHEMA", schema)
    monkeypatch.setattr(
        engine,
        "build_polymer_graph",
        runner._graph_builder_for(schema, representation, builder),
    )
    rows = [
        {"sample_id": "valid", "SMILES": "[*]CCO[*]"},
        {"sample_id": "shared", "SMILES": "[*]C(*)"},
        {"sample_id": "plain", "SMILES": "CCO"},
    ]
    cache = tmp_path / variant_name / "graphs.pt"
    diagnostics_path = tmp_path / variant_name / "graph_diagnostics.json"

    graphs, diagnostics = runner._build_graphs_for_variant(
        rows,
        cache_path=cache,
        source_sha256="frozen-source",
        diagnostics_path=diagnostics_path,
    )
    cached_graphs, cached_diagnostics = runner._build_graphs_for_variant(
        rows,
        cache_path=cache,
        source_sha256="frozen-source",
        diagnostics_path=diagnostics_path,
    )

    assert len(graphs) == len(cached_graphs) == 3
    assert diagnostics == cached_diagnostics
    assert diagnostics["graph_schema"] == schema
    assert diagnostics["source_rows_dropped"] == 0
    assert diagnostics["endpoint_closure_count"] == 0
    assert diagnostics["polymerization_edge_count"] == 0
    assert diagnostics["fallback_count"] == expected["fallback_count"]
    assert diagnostics["polymer_endpoint_graph_count"] == expected["endpoint_graphs"]
    assert diagnostics["polymer_endpoint_node_count"] == expected["endpoint_nodes"]
    assert len(diagnostics["samples"]) == 3
    assert json.loads(diagnostics_path.read_text())["graph_schema"] == schema


@pytest.mark.parametrize(
    ("variant_name", "schema", "representation"),
    [
        ("own_gnn_repr_keep_dummy", DUMMY_SCHEMA, DUMMY_REPRESENTATION),
        (
            "own_gnn_repr_endpoint_marker",
            ENDPOINT_SCHEMA,
            ENDPOINT_REPRESENTATION,
        ),
    ],
)
def test_variant_config_rejects_training_setting_drift(
    tmp_path: Path, variant_name: str, schema: str, representation: str
):
    source = (
        runner.frozen_engine.TRACK_ROOT
        / "models"
        / variant_name
        / "config.json"
    )
    config = json.loads(source.read_text(encoding="utf-8"))
    config["training"]["learning_rate"] *= 2
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="frozen training settings"):
        runner._load_variant_config(
            variant_root=tmp_path,
            graph_schema=schema,
            representation=representation,
        )


@pytest.mark.parametrize(
    ("variant_name", "schema", "representation"),
    [
        ("own_gnn_repr_keep_dummy", DUMMY_SCHEMA, DUMMY_REPRESENTATION),
        (
            "own_gnn_repr_endpoint_marker",
            ENDPOINT_SCHEMA,
            ENDPOINT_REPRESENTATION,
        ),
    ],
)
def test_seed_override_changes_only_effective_seed_and_keeps_source_config(
    variant_name: str, schema: str, representation: str
):
    root = runner.frozen_engine.TRACK_ROOT / "models" / variant_name
    source_path = root / "config.json"
    source_bytes = source_path.read_bytes()
    source_config = json.loads(source_bytes)
    default_config = runner._load_variant_config(
        variant_root=root,
        graph_schema=schema,
        representation=representation,
    )
    seeded_config = runner._load_variant_config(
        variant_root=root,
        graph_schema=schema,
        representation=representation,
        seed_override=43,
    )

    assert default_config == source_config
    assert seeded_config["seed"] == 43
    seeded_config["seed"] = source_config["seed"]
    assert seeded_config == source_config
    assert source_path.read_bytes() == source_bytes


def test_variant_source_manifest_records_seed_schema_representation_and_hashes(
    tmp_path: Path,
):
    variant_root = runner.frozen_engine.TRACK_ROOT / "models/own_gnn_repr_keep_dummy"
    config = runner._load_variant_config(
        variant_root=variant_root,
        graph_schema=DUMMY_SCHEMA,
        representation=DUMMY_REPRESENTATION,
        seed_override=44,
    )
    train_csv = tmp_path / "train.csv"
    folds_csv = tmp_path / "folds.csv"
    train_csv.write_text("train fixture\n")
    folds_csv.write_text("fold fixture\n")

    manifest = runner._source_manifest_for_variant(
        "stage3a1-test-commit",
        train_csv,
        folds_csv,
        variant_root=variant_root,
        graph_schema=DUMMY_SCHEMA,
        representation=DUMMY_REPRESENTATION,
        config=config,
    )

    assert manifest["git_commit"] == "stage3a1-test-commit"
    assert manifest["seed"] == 44
    assert manifest["graph_schema"] == DUMMY_SCHEMA
    assert manifest["representation"] == DUMMY_REPRESENTATION
    assert manifest["benchmark_files_sha256"]["train_csv"] == runner.sha256_file(train_csv)
    assert manifest["benchmark_files_sha256"]["folds_csv"] == runner.sha256_file(folds_csv)
    assert manifest["variant_config_sha256"] == runner.sha256_file(variant_root / "config.json")
    assert manifest["effective_config_sha256"] == runner.frozen_engine._config_sha256(config)


def test_variant_registry_append_preserves_v0_row_and_uses_owned_category(
    tmp_path: Path,
):
    track_root = runner.frozen_engine.TRACK_ROOT
    source = track_root / "results.csv"
    destination = tmp_path / "results.csv"
    shutil.copyfile(source, destination)
    with source.open("r", encoding="utf-8-sig", newline="") as original:
        original_rows = list(csv.DictReader(original))
    before_bytes = source.read_bytes()

    config = json.loads(
        (track_root / "models/own_gnn_repr_keep_dummy/config.json").read_text()
    )
    metrics = {
        "overall_oof_wmae": 0.1,
        "target_mae": {name: 1.0 for name in runner.frozen_engine.TARGETS},
    }
    metadata_path = track_root / "models/own_gnn_repr_keep_dummy/artifacts/run_metadata.json"
    runner._append_variant_result(
        destination,
        config=config,
        commit="stage3a-test-commit",
        metrics=metrics,
        folds_sha256="frozen-folds",
        train_sha256="frozen-train",
        run_metadata_path=metadata_path,
    )

    with destination.open("r", encoding="utf-8-sig", newline="") as updated:
        rows = list(csv.DictReader(updated))
    assert rows[:-1] == original_rows
    assert rows[-1]["experiment_id"] == "own_gnn_repr_keep_dummy"
    assert rows[-1]["category"] == "own_model"
    assert rows[-1]["git_commit"] == "stage3a-test-commit"
    assert source.read_bytes() == before_bytes


def test_variant_seed_result_is_unique_and_idempotent(tmp_path: Path):
    track_root = runner.frozen_engine.TRACK_ROOT
    destination = tmp_path / "results.csv"
    shutil.copyfile(track_root / "results.csv", destination)
    config = json.loads(
        (track_root / "models/own_gnn_repr_keep_dummy/config.json").read_text()
    )
    config["seed"] = 43
    metrics = {
        "overall_oof_wmae": 0.1,
        "target_mae": {name: 1.0 for name in runner.frozen_engine.TARGETS},
    }
    metadata_path = track_root / "models/own_gnn_repr_keep_dummy/artifacts/seed43/run_metadata.json"

    for score in (0.1, 0.2):
        metrics["overall_oof_wmae"] = score
        runner._append_variant_result(
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
    matches = [row for row in rows if row["experiment_id"] == "own_gnn_repr_keep_dummy_seed_43"]
    assert len(matches) == 1
    assert matches[0]["seed"] == "43"
    assert matches[0]["oof_wmae"] == "0.2"
    assert matches[0]["category"] == "own_model"


def test_cli_adapter_restores_frozen_engine_hooks_after_argparse_exit(
    monkeypatch: pytest.MonkeyPatch,
):
    engine = runner.frozen_engine
    original = {
        key: getattr(engine, key)
        for key in (
            "MODEL_ROOT",
            "GRAPH_SCHEMA",
            "build_polymer_graph",
            "_load_config",
            "OwnGNNv0",
            "_build_graphs",
            "_source_manifest",
            "_append_result",
        )
    }
    variant_root = (
        engine.TRACK_ROOT / "models/own_gnn_repr_keep_dummy"
    )
    monkeypatch.setattr(sys, "argv", ["train_oof", "--help"])

    with pytest.raises(SystemExit) as exc:
        runner.main_for_variant(
            variant_root=variant_root,
            graph_schema=DUMMY_SCHEMA,
            representation=DUMMY_REPRESENTATION,
            graph_builder=build_dummy_graph,
        )

    assert exc.value.code == 0
    for key, value in original.items():
        assert getattr(engine, key) is value
