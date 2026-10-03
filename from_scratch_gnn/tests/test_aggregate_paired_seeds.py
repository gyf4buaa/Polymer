import csv

import pytest

from from_scratch_gnn.scripts.aggregate_paired_seeds import (
    FOLDS_SHA256,
    MODELS,
    SEEDS,
    TARGET_COLUMNS,
    TRAIN_SHA256,
    aggregate_registry,
)


def _write_registry(path, *, duplicate=False, bad_hash=False):
    fields = [
        "experiment_id", "seed", "status", "train_data_sha256", "folds_sha256",
        "oof_wmae", *TARGET_COLUMNS.values(),
    ]
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        rows = []
        for model_index, model in enumerate(MODELS):
            for seed_index, seed in enumerate(SEEDS):
                # Variant A ties v0 at seed 42, improves at 43/44, and regresses at 45/46.
                offsets = {
                    "own_gnn_v0_frozen_oof_v1": 0.0,
                    "own_gnn_repr_keep_dummy": (0.0, -0.2, -0.1, 0.1, 0.2)[seed_index],
                    "own_gnn_repr_endpoint_marker": 0.3,
                }
                row = {
                    "experiment_id": model if seed == 42 else f"{model}_seed_{seed}",
                    "seed": seed,
                    "status": "formal_oof_model",
                    "train_data_sha256": "wrong" if bad_hash else TRAIN_SHA256,
                    "folds_sha256": FOLDS_SHA256,
                    "oof_wmae": 1.0 + seed_index * 0.1 + offsets[model],
                }
                for target_index, column in enumerate(TARGET_COLUMNS.values()):
                    row[column] = model_index + seed_index + target_index / 10
                rows.append(row)
        writer.writerows(rows)
        if duplicate:
            writer.writerow(rows[0])


def test_aggregate_registry_calculates_overall_paired_and_target_stats(tmp_path):
    path = tmp_path / "results.csv"
    _write_registry(path)

    result = aggregate_registry(path)

    assert result["seeds"] == list(SEEDS)
    assert result["overall_wmae"]["Own-GNN v0"]["mean"] == pytest.approx(1.2)
    assert result["overall_wmae"]["Own-GNN v0"]["sample_std"] == pytest.approx(0.158113883)
    a_delta = result["paired_overall_delta"]["A-v0"]
    assert [a_delta["by_seed"][str(seed)] for seed in SEEDS] == pytest.approx([0, -0.2, -0.1, 0.1, 0.2])
    assert (a_delta["positive_count"], a_delta["negative_count"], a_delta["tie_count"]) == (2, 2, 1)
    target = result["per_target_mae"]["Tg"]
    assert target["by_model"]["Variant A"]["mean"] == pytest.approx(3.0)
    assert target["paired_delta"]["A-v0"]["mean"] == pytest.approx(1.0)


@pytest.mark.parametrize("kwargs, message", [
    ({"duplicate": True}, "duplicate registry row"),
    ({"bad_hash": True}, "train SHA256 mismatch"),
])
def test_aggregate_registry_rejects_duplicate_or_unfrozen_rows(tmp_path, kwargs, message):
    path = tmp_path / "results.csv"
    _write_registry(path, **kwargs)
    with pytest.raises(ValueError, match=message):
        aggregate_registry(path)


def test_aggregate_registry_rejects_missing_seed(tmp_path):
    path = tmp_path / "results.csv"
    _write_registry(path)
    contents = path.read_text()
    lines = contents.splitlines()
    # Drop the last (seed 46) model row, leaving an incomplete paired design.
    path.write_text("\n".join(lines[:-1]) + "\n")
    with pytest.raises(ValueError, match="missing seeds \\[46\\]"):
        aggregate_registry(path)


def test_aggregate_registry_rejects_seed_id_mismatch(tmp_path):
    path = tmp_path / "results.csv"
    _write_registry(path)
    contents = path.read_text().replace(
        "own_gnn_repr_keep_dummy_seed_43,43,",
        "own_gnn_repr_keep_dummy_seed_44,43,",
    )
    path.write_text(contents)
    with pytest.raises(ValueError, match="expected experiment_id"):
        aggregate_registry(path)
