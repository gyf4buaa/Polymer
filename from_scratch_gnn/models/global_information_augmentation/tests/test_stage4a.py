from __future__ import annotations

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from torch_geometric.data import Batch

from ....models.own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA as BASELINE_GRAPH_SCHEMA,
    build_polymer_graph as build_baseline_graph,
)
from ....models.polymer_representation_ablation.model import OwnGNNRepresentation
from ....models.global_information_augmentation.features import (
    DESCRIPTOR_NAMES,
    fit_descriptor_scaler,
    generate_feature_matrices,
    transform_descriptors,
)
from ....models.global_information_augmentation.model import GlobalInformationGNN
from ....models.global_information_augmentation import runner
from ....src.data import load_training_data, sha256_file

TRACK_ROOT = Path(__file__).resolve().parents[3]
REPOSITORY_ROOT = TRACK_ROOT.parent
BASELINE_CONFIG = json.loads(
    (TRACK_ROOT / "models" / "own_gnn_repr_keep_dummy" / "config.json").read_text()
)
EXPECTED_DESCRIPTOR_NAMES = (
    "MolWt", "MolLogP", "MolMR", "TPSA", "LabuteASA", "HeavyAtomCount",
    "NumHeteroatoms", "NumHDonors", "NumHAcceptors", "NumRotatableBonds",
    "RingCount", "NumAromaticRings", "NumAliphaticRings", "NumSaturatedRings",
    "NumAromaticHeterocycles", "NumAliphaticHeterocycles",
    "NumSaturatedHeterocycles", "NumAromaticCarbocycles",
    "NumAliphaticCarbocycles", "FractionCSP3",
)


def _train_csv() -> Path:
    candidates = []
    configured = os.environ.get("POLYMER_TRAIN_CSV")
    if configured:
        candidates.append(Path(configured))
    candidates.extend(
        [
            REPOSITORY_ROOT / "data" / "competition_raw" / "train.csv",
            Path("/Users/gyf/聚合物/data/competition_raw/train.csv"),
            Path("/home/gyf/work/polymer/data/competition_raw/train.csv"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    raise FileNotFoundError("Frozen 7,973-row Stage 0 train.csv was not found")


class Stage4AFeatureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.train_csv = _train_csv()
        cls.rows = load_training_data(cls.train_csv)
        cls.features = generate_feature_matrices([row["SMILES"] for row in cls.rows])

    def test_historical_g0_forward_path_is_unchanged(self) -> None:
        graph, _ = build_baseline_graph("*CCO*", sample_id="g0-test")
        batch = Batch.from_data_list([graph, graph.clone()])
        torch.manual_seed(407)
        historical = OwnGNNRepresentation().eval()
        torch.manual_seed(407)
        stage4_g0 = GlobalInformationGNN(variant="G0").eval()
        for name, value in historical.state_dict().items():
            self.assertTrue(torch.equal(value, stage4_g0.state_dict()[name]), name)
        with torch.no_grad():
            self.assertTrue(torch.equal(historical(batch), stage4_g0(batch)))

    def test_keep_dummy_graph_is_identical_to_stage3_baseline(self) -> None:
        from ....models.global_information_augmentation.runner import build_polymer_graph

        smiles = "*CC(O)C*"
        old_graph, old_info = build_baseline_graph(smiles, sample_id="same")
        new_graph, new_info = build_polymer_graph(smiles, sample_id="same")
        self.assertEqual(BASELINE_GRAPH_SCHEMA, "own_gnn_repr_keep_dummy_raw_graph_node7_edge16_endpoint1_v1")
        self.assertEqual(old_info.as_dict(), new_info.as_dict())
        self.assertGreater(int((old_graph.x[:, 0] == 0).sum()), 0)
        for field in ("x", "edge_index", "edge_attr", "polymer_endpoint"):
            self.assertTrue(torch.equal(getattr(old_graph, field), getattr(new_graph, field)))

    def test_descriptor_schema_has_the_preregistered_20_names(self) -> None:
        self.assertEqual(DESCRIPTOR_NAMES, EXPECTED_DESCRIPTOR_NAMES)
        self.assertEqual(len(DESCRIPTOR_NAMES), 20)

    def test_descriptor_order_is_deterministic(self) -> None:
        first = generate_feature_matrices(["*CCO*", "*CCN*"])
        second = generate_feature_matrices(["*CCO*", "*CCN*"])
        self.assertEqual(first.manifest["descriptors"]["names_in_order"], list(DESCRIPTOR_NAMES))
        self.assertEqual(
            first.manifest["descriptors"]["raw_matrix_sha256"],
            second.manifest["descriptors"]["raw_matrix_sha256"],
        )

    def test_descriptor_and_morgan_coverage_is_7973_of_7973(self) -> None:
        self.assertEqual(len(self.rows), 7973)
        self.assertEqual(self.features.descriptors.shape, (7973, 20))
        self.assertEqual(self.features.morgan.shape, (7973, 2048))
        self.assertEqual(self.features.manifest["source"]["sample_count"], 7973)
        self.assertEqual(self.features.manifest["descriptors"]["nonfinite_count"], 0)
        self.assertTrue(np.isfinite(self.features.descriptors).all())

    def test_descriptor_audit_reports_stats_and_constant_columns(self) -> None:
        audit = self.features.manifest["descriptors"]
        self.assertEqual(set(audit["statistics"]), set(DESCRIPTOR_NAMES))
        for name in DESCRIPTOR_NAMES:
            self.assertEqual(
                set(audit["statistics"][name]),
                {"min", "max", "mean", "std_population", "constant"},
            )
        self.assertIsInstance(audit["constant_columns"], list)

    def test_morgan_configuration_and_hash_are_deterministic(self) -> None:
        first = self.features.manifest["morgan"]
        second = generate_feature_matrices([row["SMILES"] for row in self.rows]).manifest["morgan"]
        self.assertEqual(first["radius"], 2)
        self.assertEqual(first["bit_count"], 2048)
        self.assertIs(first["chirality"], True)
        self.assertEqual(first["representation"], "binary bit fingerprint")
        self.assertEqual(first["raw_matrix_sha256"], second["raw_matrix_sha256"])

    def test_feature_matrix_repeated_generation_has_identical_hashes(self) -> None:
        smiles = [row["SMILES"] for row in self.rows[:32]]
        first = generate_feature_matrices(smiles)
        second = generate_feature_matrices(smiles)
        self.assertEqual(
            first.manifest["descriptors"]["raw_matrix_sha256"],
            second.manifest["descriptors"]["raw_matrix_sha256"],
        )
        self.assertEqual(
            first.manifest["morgan"]["raw_matrix_sha256"],
            second.manifest["morgan"]["raw_matrix_sha256"],
        )

    def test_descriptor_scaler_uses_only_fold_training_rows(self) -> None:
        matrix = np.arange(80, dtype=np.float64).reshape(4, 20)
        scaler = fit_descriptor_scaler(matrix, [0, 1, 2], source_data_sha256="train-hash")
        self.assertEqual(scaler["fit_row_count"], 3)
        self.assertEqual(scaler["train_mean"][0], 20.0)
        self.assertEqual(scaler["source_data_sha256"], "train-hash")

    def test_held_out_row_does_not_change_fold_mean_or_std(self) -> None:
        matrix = np.arange(80, dtype=np.float64).reshape(4, 20)
        changed = matrix.copy()
        changed[3] = 1.0e12
        first = fit_descriptor_scaler(matrix, [0, 1, 2], source_data_sha256="x")
        second = fit_descriptor_scaler(changed, [0, 1, 2], source_data_sha256="x")
        self.assertEqual(first["train_mean"], second["train_mean"])
        self.assertEqual(first["train_std"], second["train_std"])
        self.assertEqual(first["scale_used"], second["scale_used"])

    def test_zero_variance_descriptor_is_kept_with_scale_one(self) -> None:
        matrix = np.ones((3, 20), dtype=np.float64)
        scaler = fit_descriptor_scaler(matrix, [0, 1], source_data_sha256="x")
        self.assertEqual(scaler["zero_variance_columns"], list(DESCRIPTOR_NAMES))
        self.assertEqual(scaler["scale_used"], [1.0] * 20)
        self.assertEqual(transform_descriptors(matrix, scaler).shape, (3, 20))

    def test_morgan_has_no_target_selection_api_or_config(self) -> None:
        import inspect

        self.assertEqual(tuple(inspect.signature(generate_feature_matrices).parameters), ("smiles_values",))
        morgan_config = runner._expected_global_config("morgan")
        self.assertIs(morgan_config["target_based_selection"], False)

    def test_feature_aware_fold_adapter_forwards_frozen_fold_inputs(self) -> None:
        rows = [{"sample_id": f"sample-{index}"} for index in range(3)]
        fold_ids = [0, 1, 2]
        output_dir = Path("/tmp/stage4a-adapter-test")
        observed = {}

        def original_train_fold(*, rows, fold_ids, output_dir, **kwargs):
            observed["rows"] = rows
            observed["fold_ids"] = fold_ids
            observed["output_dir"] = output_dir
            observed["fold"] = kwargs["fold"]
            observed["feature_shape"] = runner._FEATURE_CONTEXT["value"]["feature_matrix"].shape
            return "fold-result"

        result = runner._feature_aware_train_fold(
            original_train_fold=original_train_fold,
            variant="morgan",
            rows=rows,
            fold_ids=fold_ids,
            descriptors=np.zeros((3, 20), dtype=np.float64),
            morgan=np.ones((3, 2048), dtype=np.uint8),
            source_data_sha256="frozen-train-hash",
            output_dir=output_dir,
            fold=1,
        )

        self.assertEqual(result, "fold-result")
        self.assertEqual(observed["rows"], rows)
        self.assertEqual(observed["fold_ids"], fold_ids)
        self.assertEqual(observed["output_dir"], output_dir)
        self.assertEqual(observed["fold"], 1)
        self.assertEqual(observed["feature_shape"], (3, 2048))
        self.assertNotIn("value", runner._FEATURE_CONTEXT)

    def test_descriptor_projection_is_zero_initialized(self) -> None:
        model = GlobalInformationGNN(variant="D")
        self.assertEqual(tuple(model.global_projection.weight.shape), (512, 20))
        self.assertTrue(torch.count_nonzero(model.global_projection.weight).item() == 0)

    def test_morgan_projection_is_zero_initialized(self) -> None:
        model = GlobalInformationGNN(variant="M")
        self.assertEqual(tuple(model.global_projection.weight.shape), (512, 2048))
        self.assertTrue(torch.count_nonzero(model.global_projection.weight).item() == 0)

    def test_same_seed_common_parameters_match_g0(self) -> None:
        for variant in ("D", "M"):
            torch.manual_seed(915)
            baseline = GlobalInformationGNN(variant="G0")
            torch.manual_seed(915)
            augmented = GlobalInformationGNN(variant=variant)
            baseline_parameters = dict(baseline.named_parameters())
            augmented_parameters = dict(augmented.named_parameters())
            self.assertEqual(set(baseline_parameters), set(augmented_parameters) - {"global_projection.weight"})
            for name, value in baseline_parameters.items():
                self.assertTrue(torch.equal(value, augmented_parameters[name]), f"{variant}:{name}")

    def test_zero_init_forward_matches_g0_for_both_variants(self) -> None:
        graph, _ = build_baseline_graph("*CCO*", sample_id="zero-init")
        batch = Batch.from_data_list([graph, graph.clone()])
        for variant, n_features in (("D", 20), ("M", 2048)):
            batch.global_features = torch.arange(2 * n_features, dtype=torch.float32).reshape(2, n_features)
            torch.manual_seed(245)
            baseline = GlobalInformationGNN(variant="G0").eval()
            torch.manual_seed(245)
            augmented = GlobalInformationGNN(variant=variant).eval()
            with torch.no_grad():
                base_prediction = baseline(batch)
                variant_prediction = augmented(batch)
            torch.testing.assert_close(base_prediction, variant_prediction, atol=1e-6, rtol=1e-6)

    def test_checkpoint_round_trip_preserves_predictions(self) -> None:
        graph, _ = build_baseline_graph("*CCO*", sample_id="checkpoint")
        batch = Batch.from_data_list([graph])
        for variant, width in (("D", 20), ("M", 2048)):
            batch.global_features = torch.ones((1, width), dtype=torch.float32)
            torch.manual_seed(88)
            model = GlobalInformationGNN(variant=variant).eval()
            with torch.no_grad():
                expected = model(batch)
            with tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "checkpoint.pt"
                torch.save({"state_dict": model.state_dict()}, path)
                restored = GlobalInformationGNN(variant=variant).eval()
                restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
                with torch.no_grad():
                    actual = restored(batch)
            torch.testing.assert_close(expected, actual, atol=0, rtol=0)

    def test_d_and_m_artifacts_are_isolated_and_registry_is_per_run(self) -> None:
        original_track_root = runner.TRACK_ROOT
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            runner.TRACK_ROOT = root
            try:
                descriptor_run = runner._output_dir(root / "D", 42, "formal")
                morgan_run = runner._output_dir(root / "M", 42, "formal")
                self.assertNotEqual(descriptor_run, morgan_run)
                self.assertIn("D", descriptor_run.parts)
                self.assertIn("M", morgan_run.parts)
                for variant, code, run_dir in (
                    ("descriptor", "D", descriptor_run),
                    ("morgan", "M", morgan_run),
                ):
                    run_dir.mkdir(parents=True, exist_ok=True)
                    config = {
                        "experiment_id": f"own_gnn_stage4a_{variant}",
                        "seed": 42,
                        "model": {"global_information": {"variant": code}},
                    }
                    (run_dir / "config.json").write_text(json.dumps(config))
                    metadata_path = run_dir / "run_metadata.json"
                    metadata_path.write_text("{}")
                    runner._append_registry_row(
                        root / "results.csv",
                        config=config,
                        commit="source-commit",
                        metrics={"overall_oof_wmae": 0.1, "target_mae": {target: 0.1 for target in ("Tg", "FFV", "Tc", "Density", "Rg")}},
                        folds_sha256="fold-hash",
                        train_sha256="train-hash",
                        run_metadata_path=metadata_path,
                    )
                self.assertTrue((descriptor_run / "registry_row.json").is_file())
                self.assertTrue((morgan_run / "registry_row.json").is_file())
                self.assertFalse((root / "results.csv").exists())
            finally:
                runner.TRACK_ROOT = original_track_root

    def test_registry_experiment_ids_are_unique_by_variant_and_seed(self) -> None:
        identifiers = []
        for variant in ("descriptor", "morgan"):
            base = {
                "experiment_id": f"own_gnn_stage4a_{variant}",
                "seed": 42,
            }
            for seed in (42, 43, 44, 45, 46):
                identifiers.append(
                    runner._registry_experiment_id({**base, "seed": seed})
                )
        self.assertEqual(len(identifiers), 10)
        self.assertEqual(len(set(identifiers)), 10)

    def test_frozen_config_invariant_rejects_architecture_and_training_changes(self) -> None:
        config = copy.deepcopy(BASELINE_CONFIG)
        config["experiment_id"] = "test"
        config["model"]["global_information"] = runner._expected_global_config("descriptor")
        runner._validate_variant_config(config, variant="descriptor", baseline=BASELINE_CONFIG)
        changed = copy.deepcopy(config)
        changed["model"]["num_layers"] = 5
        with self.assertRaises(ValueError):
            runner._validate_variant_config(changed, variant="descriptor", baseline=BASELINE_CONFIG)
        changed = copy.deepcopy(config)
        changed["model"]["operator"] = "PNA"
        with self.assertRaises(ValueError):
            runner._validate_variant_config(changed, variant="descriptor", baseline=BASELINE_CONFIG)
        changed = copy.deepcopy(config)
        changed["training"]["learning_rate"] *= 2
        with self.assertRaises(ValueError):
            runner._validate_variant_config(changed, variant="descriptor", baseline=BASELINE_CONFIG)


if __name__ == "__main__":
    unittest.main()
