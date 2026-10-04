from __future__ import annotations

import json

import pytest

from from_scratch_gnn.scripts import aggregate_operator_ablation as aggregate


def _write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def _write_fixture_run(root, operator, seed, score, source_commit="clean-source"):
    run = root / "models" / "operator_ablation" / operator / "artifacts" / f"seed_{seed}"
    frozen_config = json.loads(aggregate.BASELINE_CONFIG.read_text(encoding="utf-8"))
    config = {
        **frozen_config,
        "seed": seed,
        "model": {
            **frozen_config["model"],
            "operator": operator,
            "operator_config": {"edge_aware": True},
        },
    }
    _write_json(
        run / "metrics.json",
        {
            "n_samples": 7973,
            "overall_oof_wmae": score,
            "target_mae": {target: score + index for index, target in enumerate(aggregate.TARGETS)},
            "validation": {
                "prediction_row_count": 7973,
                "truth_source_sha256": aggregate.TRAIN_SHA256,
            },
        },
    )
    _write_json(run / "config.json", config)
    _write_json(
        run / "source_manifest.json",
        {
            "seed": seed,
            "git_commit": source_commit,
            "benchmark_files_sha256": {
                "train_csv": aggregate.TRAIN_SHA256,
                "folds_csv": aggregate.FOLDS_SHA256,
            },
            "effective_config_sha256": aggregate.sha256_file(run / "config.json"),
            "graph_provenance": {
                "graph_set_fingerprint_sha256": "same-graph-set"
            },
        },
    )
    _write_json(
        run / "run_metadata.json",
        {
            "git_commit": source_commit,
            "duration_seconds": 10.0,
            "fold_metrics": [
                {
                    "peak_vram_allocated_mb": 100.0,
                    "peak_vram_reserved_mb": 120.0,
                    "nvidia_smi_peak_memory_used_mb": 1500.0,
                }
            ],
            "gpu_sampling": {"gpu_utilization_sample_count": 1},
        },
    )
    _write_json(
        run / "operator_metadata.json",
        {
            "graph_schema": frozen_config["graph"]["schema"],
            "message_passing_output_widths": [256] * 4,
            "trainable_parameter_count": 1000,
            "message_passing_block_parameter_count": 500,
        },
    )
    _write_json(run / "registry_row.json", {"seed": seed})
    (run / "fold_metrics.csv").write_text("fold\n0\n", encoding="utf-8")
    (run / "oof_predictions.csv").write_text("sample_id\n1\n", encoding="utf-8")
    if operator == "pna":
        _write_json(
            run / "degree_histogram.json",
            {
                "graph_schema": "frozen",
                "histogram_by_degree": {"0": 1, "1": 2},
                "graph_count": 7973,
                "source_train_sha256": aggregate.TRAIN_SHA256,
                "uses_labels_or_targets": False,
                "graph_set_fingerprint_sha256": "same-graph-set",
            },
        )


def _baseline():
    return {
        "analysis_only": True,
        "seeds": list(aggregate.SEEDS),
        "train_data_sha256": aggregate.TRAIN_SHA256,
        "folds_sha256": aggregate.FOLDS_SHA256,
        "overall_wmae": {
            "Variant A": {
                "by_seed": {str(seed): 1.0 + index for index, seed in enumerate(aggregate.SEEDS)},
                "mean": 3.0,
                "sample_std": 1.5811388300841898,
                "min": 1.0,
                "max": 5.0,
            }
        },
        "per_target_mae": {
            target: {
                "by_model": {
                    "Variant A": {
                        "by_seed": {str(seed): float(index) for index, seed in enumerate(aggregate.SEEDS)}
                    }
                }
            }
            for target in aggregate.TARGETS
        },
    }


def test_aggregate_operator_ablation_calculates_paired_results_and_checks_provenance(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(aggregate, "TRACK_ROOT", tmp_path)
    baseline_path = tmp_path / "baseline.json"
    _write_json(baseline_path, _baseline())
    for operator, offset in (("gatv2", -0.1), ("pna", 0.2)):
        for index, seed in enumerate(aggregate.SEEDS):
            _write_fixture_run(
                tmp_path,
                operator,
                seed,
                1.0 + index + offset,
            )

    result = aggregate.aggregate_operator_ablation(baseline_path=baseline_path)
    assert result["overall_wmae"]["GINE"]["mean"] == pytest.approx(3.0)
    assert result["paired_overall_delta"]["GATv2-GINE"]["mean"] == pytest.approx(-0.1)
    assert result["paired_overall_delta"]["GATv2-GINE"]["left_lower_count"] == 5
    assert result["paired_overall_delta"]["PNA-GINE"]["mean"] == pytest.approx(0.2)
    assert result["runtime"]["formal_run_count"] == 10
    assert result["pna_degree_statistics"]["graph_set_fingerprint_sha256"] == "same-graph-set"


def test_aggregate_operator_ablation_rejects_mixed_source_commits(tmp_path, monkeypatch):
    monkeypatch.setattr(aggregate, "TRACK_ROOT", tmp_path)
    baseline_path = tmp_path / "baseline.json"
    _write_json(baseline_path, _baseline())
    for operator in aggregate.OPERATORS:
        for seed in aggregate.SEEDS:
            commit = "different" if operator == "pna" and seed == 46 else "clean-source"
            _write_fixture_run(tmp_path, operator, seed, 1.0, source_commit=commit)
    with pytest.raises(ValueError, match="one source commit"):
        aggregate.aggregate_operator_ablation(baseline_path=baseline_path)
