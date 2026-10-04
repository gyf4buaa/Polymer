from __future__ import annotations

import csv
import json
import threading
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch

from from_scratch_gnn.models.capacity_scaling import runner
from from_scratch_gnn.models.capacity_scaling.parameter_audit import (
    HISTORICAL_C256_PARAMETER_COUNT,
    build_audit,
    parameter_breakdown,
)
from from_scratch_gnn.models.capacity_scaling.protocol import (
    FOLDS_SHA256,
    G0_CONFIG_PATH,
    MODEL_ROOT,
    NEW_WIDTHS,
    SEEDS,
    TRACK_ROOT,
    TRAIN_SHA256,
    WIDTHS,
    artifact_dir,
    config_diff_paths,
    formal_job_matrix,
    historical_c256_artifact_dir,
    load_base_config,
    run_config,
    validate_frozen_config,
)
from from_scratch_gnn.models.own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA,
    build_polymer_graph,
)
from from_scratch_gnn.models.polymer_representation_ablation.graph_builder import (
    build_graph,
)
from from_scratch_gnn.models.own_gnn_v0.model import masked_huber_loss
from from_scratch_gnn.models.polymer_representation_ablation.model import OwnGNNRepresentation
from from_scratch_gnn.models.capacity_scaling import protocol
from from_scratch_gnn.scripts import aggregate_stage5a
from from_scratch_gnn.scripts.aggregate_stage5a import (
    _update_results_registry,
    load_historical_c256,
)
from from_scratch_gnn.scripts.run_stage5a_queue import _job_matrix
from from_scratch_gnn.src.data import sha256_file


def _batch():
    first, _ = build_polymer_graph("[*]CCO[*]", sample_id="first")
    second, _ = build_polymer_graph("c1ccccc1", sample_id="second")
    return Batch.from_data_list([first, second])


def test_g0_graph_construction_is_unchanged_for_representative_smiles():
    assert GRAPH_SCHEMA == "own_gnn_repr_keep_dummy_raw_graph_node7_edge16_endpoint1_v1"
    for smiles in ("[*]CCO[*]", "[*]CC[*]", "CCO", "[*]CC(*)C[*]"):
        actual, actual_info = build_polymer_graph(smiles, sample_id="sample")
        expected, expected_info = build_graph(
            smiles, representation="keep_dummy", sample_id="sample"
        )
        assert torch.equal(actual.x, expected.x)
        assert torch.equal(actual.edge_index, expected.edge_index)
        assert torch.equal(actual.edge_attr, expected.edge_attr)
        assert actual_info.as_dict() == expected_info.as_dict()


def test_c256_architecture_and_training_config_match_historical_g0():
    current = run_config(256, 42)
    historical = json.loads(G0_CONFIG_PATH.read_text(encoding="utf-8"))
    assert current["benchmark_version"] == historical["benchmark_version"]
    assert current["graph"] == historical["graph"]
    assert current["model"] == historical["model"]
    assert current["training"] == historical["training"]
    assert current["model"]["hidden_dim"] == 256
    assert current["model"]["num_layers"] == 4


def test_width_configs_only_change_hidden_dim_and_experiment_id():
    reference = run_config(256, 42)
    allowed = {"model.hidden_dim", "experiment_id"}
    for width in WIDTHS:
        paths = config_diff_paths(reference, run_config(width, 42))
        assert set(paths) <= allowed
        if width == 256:
            assert paths == []
        else:
            assert set(paths) == allowed
    seed_paths = config_diff_paths(run_config(128, 42), run_config(128, 43))
    assert seed_paths == ["seed"]


def test_frozen_config_guard_rejects_scientific_drift():
    base = load_base_config()
    drifted_training = json.loads(json.dumps(base))
    drifted_training["training"]["batch_size"] = 32
    with pytest.raises(ValueError, match="training protocol"):
        validate_frozen_config(drifted_training)
    drifted_operator = json.loads(json.dumps(base))
    drifted_operator["model"]["operator"] = "GATv2"
    with pytest.raises(ValueError, match="architecture drifted"):
        validate_frozen_config(drifted_operator)
    drifted_graph = json.loads(json.dumps(base))
    drifted_graph["graph"]["representation"] = "endpoint_marker"
    with pytest.raises(ValueError, match="graph representation"):
        validate_frozen_config(drifted_graph)


@pytest.mark.parametrize("width", WIDTHS)
def test_forward_shape_all_five_heads_and_finite(width: int):
    config = run_config(width, 42)
    model = OwnGNNRepresentation(**{
        "hidden_dim": width,
        "num_layers": config["model"]["num_layers"],
        "dropout": config["model"]["dropout"],
    })
    output = model(_batch())
    assert output.shape == (2, 5)
    assert tuple(model.heads.keys()) == ("Tg", "FFV", "Tc", "Density", "Rg")
    assert torch.isfinite(output).all()


def test_masked_loss_remains_finite_with_missing_targets():
    prediction = torch.tensor([[0.0, 2.0, 3.0, 1.0, 4.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan"), 3.0, float("nan"), 5.0]])
    loss = masked_huber_loss(prediction, target, delta=1.0)
    assert torch.isfinite(loss)
    loss.backward()
    assert torch.isfinite(prediction.grad).all()
    assert prediction.grad[0, 1].item() == 0.0
    assert prediction.grad[0, 3].item() == 0.0


@pytest.mark.parametrize("width", NEW_WIDTHS)
def test_checkpoint_round_trip_each_new_width(tmp_path: Path, width: int):
    model = OwnGNNRepresentation(hidden_dim=width, num_layers=4, dropout=0.1).eval()
    batch = _batch()
    with torch.no_grad():
        expected = model(batch)
    path = tmp_path / f"C{width}.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    restored = OwnGNNRepresentation(hidden_dim=width, num_layers=4, dropout=0.1).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    with torch.no_grad():
        actual = restored(batch)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("width", NEW_WIDTHS)
def test_same_seed_initialization_is_deterministic_within_width(width: int):
    torch.manual_seed(314159)
    left = OwnGNNRepresentation(hidden_dim=width, num_layers=4, dropout=0.1)
    torch.manual_seed(314159)
    right = OwnGNNRepresentation(hidden_dim=width, num_layers=4, dropout=0.1)
    assert left.state_dict().keys() == right.state_dict().keys()
    assert all(torch.equal(left.state_dict()[key], right.state_dict()[key]) for key in left.state_dict())


def test_parameter_audit_ordering_and_historical_c256_count():
    audit = build_audit()
    totals = [audit["parameter_counts"][str(width)]["total_trainable_parameters"] for width in WIDTHS]
    assert totals == sorted(totals)
    assert len(set(totals)) == 4
    assert totals[1] == HISTORICAL_C256_PARAMETER_COUNT
    for width in WIDTHS:
        breakdown = parameter_breakdown(width)
        assert breakdown["embedding_parameters"] > 0
        assert breakdown["gine_trunk_parameters"] > 0
        assert breakdown["readout_head_parameters"] > 0
        assert breakdown["total_trainable_parameters"] == sum(
            breakdown[key]
            for key in (
                "embedding_parameters",
                "gine_trunk_parameters",
                "readout_head_parameters",
            )
        )


def test_benchmark_hashes_and_frozen_sample_count():
    manifest = json.loads((TRACK_ROOT / "benchmark" / "data_manifest.json").read_text(encoding="utf-8"))
    assert manifest["source"]["sha256"] == TRAIN_SHA256
    folds_path = TRACK_ROOT / "benchmark" / "folds.csv"
    assert sha256_file(folds_path) == FOLDS_SHA256
    assert manifest["sample_count"] == 7973


def test_formal_matrix_excludes_historical_c256_and_has_15_jobs():
    jobs = formal_job_matrix()
    assert len(jobs) == 15
    assert {job["hidden_dim"] for job in jobs} == set(NEW_WIDTHS)
    assert {job["seed"] for job in jobs} == set(SEEDS)
    assert all(job["hidden_dim"] != 256 for job in jobs)
    assert len(_job_matrix(pilot=False)) == 15


def test_artifact_paths_are_isolated_per_width_and_seed():
    paths = [artifact_dir("formal", width, seed) for width in NEW_WIDTHS for seed in SEEDS]
    assert len(paths) == len(set(paths)) == 15
    assert all(path.name == "attempt_01" for path in paths)
    assert artifact_dir("smoke", 512, 42) != artifact_dir("tiny", 512, 42)


def test_aggregator_reuses_all_historical_c256_artifacts_without_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    for seed in SEEDS:
        directory = tmp_path / f"seed_{seed}"
        directory.mkdir()
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "n_samples": 7973,
                    "overall_oof_wmae": 0.0227,
                    "target_mae": {name: 0.1 for name in ("Tg", "FFV", "Tc", "Density", "Rg")},
                    "validation": {
                        "truth_source_sha256": TRAIN_SHA256,
                        "prediction_row_count": 7973,
                    },
                }
            ),
            encoding="utf-8",
        )
        (directory / "run_metadata.json").write_text(
            json.dumps(
                {
                    "benchmark_fold_sha256": FOLDS_SHA256,
                    "source_train_sha256": TRAIN_SHA256,
                }
            ),
            encoding="utf-8",
        )
        (directory / "config.json").write_text(
            json.dumps(run_config(256, seed)), encoding="utf-8"
        )
    monkeypatch.setattr(
        aggregate_stage5a,
        "historical_c256_artifact_dir",
        lambda seed: tmp_path / f"seed_{seed}",
    )
    historical = load_historical_c256()
    assert set(historical) == set(SEEDS)
    assert all(run["historical_reuse"] for run in historical.values())
    assert all(run["config"]["model"]["hidden_dim"] == 256 for run in historical.values())
    assert all(run["metrics"]["validation"]["prediction_row_count"] == 7973 for run in historical.values())
    assert protocol.historical_c256_artifact_dir(42).name == "production_gpu_20261003"
    assert not any(job["hidden_dim"] == 256 for job in formal_job_matrix())


def test_registry_rows_write_per_run_without_mutating_shared_csv(tmp_path: Path):
    from from_scratch_gnn.models.capacity_scaling.runner import _write_registry_row

    registry = tmp_path / "results.csv"
    registry.write_text("experiment_id,seed\nold,42\n", encoding="utf-8")
    original = registry.read_bytes()
    rows = []
    lock = threading.Lock()

    def write(width: int, seed: int):
        output = tmp_path / f"C{width}" / f"seed_{seed}" / "attempt_01"
        output.mkdir(parents=True)
        config = run_config(width, seed)
        (output / "config.json").write_text(json.dumps(config), encoding="utf-8")
        metadata = output / "run_metadata.json"
        metadata.write_text("{}", encoding="utf-8")
        _write_registry_row(
            registry,
            config=config,
            commit="source-sha",
            metrics={"overall_oof_wmae": 0.02, "target_mae": {name: 0.1 for name in ("Tg", "FFV", "Tc", "Density", "Rg")}},
            folds_sha256=FOLDS_SHA256,
            train_sha256=TRAIN_SHA256,
            run_metadata_path=metadata,
        )
        row = json.loads((output / "registry_row.json").read_text(encoding="utf-8"))
        with lock:
            rows.append(row)

    threads = [threading.Thread(target=write, args=(width, seed)) for width, seed in ((128, 42), (384, 43))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(rows) == 2
    assert registry.read_bytes() == original


def test_registry_aggregation_is_atomic_and_preserves_historical_rows(tmp_path: Path):
    registry = tmp_path / "results.csv"
    with registry.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("experiment_id", "seed", "oof_wmae"))
        writer.writeheader()
        writer.writerow({"experiment_id": "historical", "seed": 42, "oof_wmae": 0.03})
    _update_results_registry(
        registry,
        [
            {
                "experiment_id": "stage5a_capacity_c128_seed_42",
                "model_name": "Stage 5A G0 C128",
                "category": "own_model",
                "seed": 42,
                "overall_oof_wmae": 0.02,
                "target_mae": {name: 0.1 for name in ("Tg", "FFV", "Tc", "Density", "Rg")},
                "status": "formal_oof_model",
                "source_commit": "source-sha",
                "train_sha256": TRAIN_SHA256,
                "folds_sha256": FOLDS_SHA256,
                "run_metadata": "models/capacity_scaling/artifacts/formal/C128/seed_42/attempt_01/run_metadata.json",
            }
        ],
    )
    with registry.open("r", encoding="utf-8", newline="") as source:
        rows = list(csv.DictReader(source))
    assert [row["experiment_id"] for row in rows] == [
        "historical",
        "stage5a_capacity_c128_seed_42",
    ]
    assert not registry.with_suffix(".csv.tmp").exists()
