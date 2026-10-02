from __future__ import annotations

import csv
import tempfile
import unittest
from collections import Counter
from pathlib import Path

from from_scratch_gnn.scripts.evaluate_oof import validate_and_score
from from_scratch_gnn.src.data import (
    TARGETS,
    build_fold_assignments,
    diagnose_training_data,
    fold_diagnostics,
    load_training_data,
)
from from_scratch_gnn.src.metrics import competition_weights, evaluate_oof


def example_rows():
    return [
        {"sample_id": "a", "SMILES": "C", "Tg": 0.0, "FFV": 0.1, "Tc": 1.0, "Density": 1.0, "Rg": 10.0},
        {"sample_id": "b", "SMILES": "CC", "Tg": 2.0, "FFV": None, "Tc": 2.0, "Density": None, "Rg": 12.0},
        {"sample_id": "c", "SMILES": "CO", "Tg": None, "FFV": 0.3, "Tc": 3.0, "Density": 2.0, "Rg": None},
        {"sample_id": "d", "SMILES": "CN", "Tg": 6.0, "FFV": 0.5, "Tc": 4.0, "Density": 3.0, "Rg": 16.0},
    ]


class DataTests(unittest.TestCase):
    def test_loader_canonical_schema_and_diagnostics_keep_all_rows(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "train.csv"
            path.write_text(
                "id,SMILES,Tg,FFV,Tc,Density,Rg\n"
                "dup,CCO,,,,,\n"
                "dup,OCC,1,0.2,0.1,1.1,5\n",
                encoding="utf-8",
            )
            rows = load_training_data(path)
            self.assertEqual(len(rows), 2)
            self.assertEqual(tuple(rows[0]), ("sample_id", "SMILES", *TARGETS))
            report = diagnose_training_data(rows)
            self.assertEqual(report["duplicate_sample_id_group_count"], 1)
            self.assertEqual(report["duplicate_smiles_group_count"], 0)
            if report["canonical_smiles_audit"]["available"]:
                self.assertEqual(
                    report["canonical_smiles_audit"]["duplicate_group_count"], 1
                )
            self.assertEqual(report["all_targets_missing_count"], 1)

    def test_folds_are_reproducible_balanced_and_order_independent(self):
        ids = [f"id-{index}" for index in range(13)]
        first = build_fold_assignments(ids, n_splits=5, seed=42)
        repeated = build_fold_assignments(ids, n_splits=5, seed=42)
        reversed_input = build_fold_assignments(list(reversed(ids)), n_splits=5, seed=42)
        other_seed = build_fold_assignments(ids, n_splits=5, seed=43)
        self.assertEqual(first, repeated)
        self.assertEqual(first, reversed_input)
        self.assertNotEqual(first, other_seed)
        self.assertEqual(sorted(Counter(first.values()).values()), [2, 2, 3, 3, 3])

    def test_fold_diagnostics_conserve_sample_and_label_counts(self):
        rows = example_rows()
        assignments = build_fold_assignments(
            (row["sample_id"] for row in rows), n_splits=2, seed=42
        )
        diagnostics = fold_diagnostics(rows, assignments, n_splits=2)
        self.assertEqual(sum(item["sample_count"] for item in diagnostics.values()), 4)
        for target in TARGETS:
            self.assertEqual(
                sum(item["valid_target_counts"][target] for item in diagnostics.values()),
                sum(row[target] is not None for row in rows),
            )

    def test_folds_reject_non_unique_ids(self):
        with self.assertRaisesRegex(ValueError, "repeat"):
            build_fold_assignments(["x", "x", "y"], n_splits=2)


class MetricTests(unittest.TestCase):
    def test_perfect_prediction_is_zero_and_missing_labels_are_masked(self):
        truth = example_rows()
        predictions = [
            {target: row[target] for target in TARGETS}
            for row in truth
        ]
        score = evaluate_oof(truth, predictions)
        self.assertAlmostEqual(score["overall_oof_wmae"], 0.0, places=15)
        self.assertEqual(score["target_counts"], {
            "Tg": 3, "FFV": 3, "Tc": 4, "Density": 3, "Rg": 3
        })
        self.assertTrue(all(value == 0.0 for value in score["target_mae"].values()))

    def test_constant_mean_prediction_has_finite_positive_score(self):
        truth = example_rows()
        means = {
            target: sum(float(row[target]) for row in truth if row[target] is not None)
            / sum(row[target] is not None for row in truth)
            for target in TARGETS
        }
        predictions = [
            {
                target: (None if row[target] is None else means[target])
                for target in TARGETS
            }
            for row in truth
        ]
        score = evaluate_oof(truth, predictions)
        self.assertGreater(score["overall_oof_wmae"], 0.0)
        self.assertTrue(all(value >= 0 for value in score["target_mae"].values()))
        self.assertAlmostEqual(
            score["overall_oof_wmae"], sum(score["target_contribution"].values())
        )

    def test_fixed_weights_allow_a_fold_without_one_targets_labels(self):
        truth = example_rows()
        weights = competition_weights(truth)
        fold_truth = [truth[2]]
        fold_pred = [{target: truth[2][target] for target in TARGETS}]
        result = evaluate_oof(fold_truth, fold_pred, target_weights=weights)
        self.assertIsNone(result["target_mae"]["Tg"])
        self.assertIsNone(result["target_mae"]["Rg"])
        self.assertEqual(result["target_contribution"]["Tg"], 0.0)

    def test_observed_label_requires_finite_prediction(self):
        truth = example_rows()
        predictions = [
            {target: row[target] for target in TARGETS}
            for row in truth
        ]
        predictions[0]["Tg"] = None
        with self.assertRaisesRegex(ValueError, "Missing prediction"):
            evaluate_oof(truth, predictions)


class ValidatorTests(unittest.TestCase):
    def _write_csv(self, path, columns, rows):
        with path.open("w", encoding="utf-8", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)

    def test_validator_aligns_by_id_and_writes_metrics_json(self):
        truth = example_rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_path = root / "train.csv"
            oof_path = root / "oof.csv"
            metrics_path = root / "metrics.json"
            train_data = []
            for row in truth:
                train_data.append({
                    "id": row["sample_id"],
                    "SMILES": row["SMILES"],
                    **{target: "" if row[target] is None else row[target] for target in TARGETS},
                })
            self._write_csv(train_path, ["id", "SMILES", *TARGETS], train_data)
            perfect = [
                {"sample_id": row["sample_id"], **{target: row[target] for target in TARGETS}}
                for row in reversed(truth)
            ]
            self._write_csv(oof_path, ["sample_id", *TARGETS], perfect)
            result = validate_and_score(train_path, oof_path, metrics_path)
            self.assertAlmostEqual(result["overall_oof_wmae"], 0.0, places=15)
            self.assertEqual(result["validation"]["prediction_row_count"], len(truth))
            self.assertTrue(metrics_path.exists())

    def test_validator_rejects_duplicate_and_missing_ids(self):
        truth = example_rows()
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            train_path = root / "train.csv"
            oof_path = root / "oof.csv"
            self._write_csv(
                train_path, ["id", "SMILES", *TARGETS],
                [
                    {"id": row["sample_id"], "SMILES": row["SMILES"],
                     **{target: "" if row[target] is None else row[target] for target in TARGETS}}
                    for row in truth
                ],
            )
            bad = [
                {"sample_id": "a", **{target: truth[0][target] for target in TARGETS}},
                {"sample_id": "a", **{target: truth[0][target] for target in TARGETS}},
            ]
            self._write_csv(oof_path, ["sample_id", *TARGETS], bad)
            with self.assertRaisesRegex(ValueError, "Invalid OOF coverage"):
                validate_and_score(train_path, oof_path, root / "metrics.json")


if __name__ == "__main__":
    unittest.main()
