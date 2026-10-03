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
