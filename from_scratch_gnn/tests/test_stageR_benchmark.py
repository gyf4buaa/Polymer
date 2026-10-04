from __future__ import annotations

import unittest

import numpy as np

from from_scratch_gnn.scripts import stageR_benchmark as stage_r


class StageRBenchmarkTests(unittest.TestCase):
    def test_median_predictions_use_only_fold_training_labels(self):
        y = np.repeat(np.asarray([[100.0], [1.0], [3.0], [5.0], [7.0]]), 5, axis=1)
        observed = np.ones_like(y, dtype=bool)
        folds = np.arange(5, dtype=np.int8)
        before = stage_r.median_oof(y, observed, folds)
        y[0, 0] = -10000.0
        after = stage_r.median_oof(y, observed, folds)
        self.assertEqual(before[0, 0], 4.0)
        self.assertEqual(after[0, 0], before[0, 0])
        self.assertEqual(after[1, 0], np.median([-10000.0, 3.0, 5.0, 7.0]))

    def test_knn_returns_fold_training_similarity_and_predictions(self):
        _, chem, _, _, generator_factory = stage_r._rdkit()
        generator = generator_factory.GetMorganGenerator(radius=2, fpSize=2048)
        smiles = ["CCO", "CCN", "CCCC", "CCCl", "CCBr"]
        fps = [generator.GetFingerprint(chem.MolFromSmiles(value)) for value in smiles]
        y = np.repeat(np.asarray([[0.0], [1.0], [2.0], [3.0], [4.0]]), 5, axis=1)
        observed = np.ones_like(y, dtype=bool)
        folds = np.arange(5, dtype=np.int8)
        prediction, max_similarity = stage_r.knn_oof(y, observed, folds, fps, k=2)
        self.assertTrue(np.isfinite(prediction).all())
        self.assertTrue(np.isfinite(max_similarity).all())
        self.assertTrue(((max_similarity >= 0) & (max_similarity <= 1)).all())

    def test_similarity_bins_assign_boundaries_once(self):
        similarity = np.asarray([0.699, 0.7, 0.9, 0.901])
        observed = np.ones((4, 5), dtype=bool)
        y = np.repeat(np.arange(4, dtype=float).reshape(-1, 1), 5, axis=1)
        mae = np.ones_like(y)
        report, _, indices = stage_r._similarity_diagnostics(similarity, observed, y, mae)
        self.assertEqual(indices.tolist(), [0, 1, 1, 2])
        self.assertEqual(report["bin_sample_counts"], {"<0.7": 1, "0.7–0.9": 2, ">0.9": 1})

    def test_paired_sample_contributions_reconcile_to_mean_delta(self):
        y = np.repeat(np.asarray([[1.0], [2.0], [4.0]]), 5, axis=1)
        observed = np.ones_like(y, dtype=bool)
        prediction = np.repeat(np.asarray([[2.0], [1.0], [2.0]]), 5, axis=1)
        c128_abs = np.full_like(y, 0.5)
        weights = {target: {"weight": 2.0} for target in stage_r.TARGETS}
        contribution = stage_r._paired_row_contributions(y, observed, prediction, c128_abs, weights)
        baseline_mae = np.abs(prediction - y).mean()
        c128_mae = c128_abs.mean()
        self.assertAlmostEqual(float(contribution.mean()), 2.0 * (baseline_mae - c128_mae) * 5)

    def test_fingerprint_audit_does_not_conflate_graphs_and_fingerprints(self):
        bits = np.asarray([[1, 0], [1, 0]], dtype=np.uint8)
        report, groups = stage_r._fingerprint_audit(
            ["a", "b"],
            ["C[*]", "N[*]"],
            bits,
            ["C[*]", "N[*]"],
            ["C[*]", "N[*]"],
        )
        self.assertEqual(report["duplicate_fingerprint_group_count"], 1)
        self.assertEqual(report["groups_different_achiral_canonical_graphs"], 1)
        self.assertEqual(groups[0]["polymer_equivalence"], "UNKNOWN")


if __name__ == "__main__":
    unittest.main()
