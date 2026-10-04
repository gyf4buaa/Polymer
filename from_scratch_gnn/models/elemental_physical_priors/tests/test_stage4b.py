from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import torch
from torch_geometric.data import Batch

from ...global_information_augmentation.model import GlobalInformationGNN
from ...own_gnn_repr_keep_dummy.graph import GRAPH_SCHEMA, build_polymer_graph
from .. import runner
from ..features import (
    FEATURE_NAMES,
    build_element_feature_manifest,
    build_periodic_table_reference,
    normalize_atomic_numbers,
)
from ..model import ElementalPhysicalGNN


BASELINE_CONFIG = json.loads(
    (Path(__file__).resolve().parents[2] / "own_gnn_repr_keep_dummy" / "config.json").read_text(
        encoding="utf-8"
    )
)
ELEMENTAL_CONFIG = json.loads(
    (Path(__file__).resolve().parents[1] / "config.json").read_text(encoding="utf-8")
)


class Stage4BFeatureTests(unittest.TestCase):
    def test_five_elemental_features_have_fixed_order(self) -> None:
        self.assertEqual(
            FEATURE_NAMES,
            ("atomic_weight", "covalent_radius", "vdw_radius", "outer_electrons", "period"),
        )
        self.assertEqual(ELEMENTAL_CONFIG["model"]["elemental_physical"]["feature_names_in_order"], list(FEATURE_NAMES))

    def test_rdkit_feature_lookup_is_deterministic_and_matches_api(self) -> None:
        first = build_periodic_table_reference()
        second = build_periodic_table_reference()
        self.assertEqual(first["table_sha256"], second["table_sha256"])
        self.assertEqual(first["table_rows"], second["table_rows"])
        carbon = first["table_rows"][5]["values"]
        self.assertEqual(
            [carbon[name] for name in FEATURE_NAMES],
            [12.011, 0.76, 1.7, 4.0, 2.0],
        )

    def test_periodic_reference_scaler_is_data_independent_and_deterministic(self) -> None:
        one = build_periodic_table_reference()
        two = build_periodic_table_reference()
        self.assertEqual(one["mean_ref"], two["mean_ref"])
        self.assertEqual(one["std_ref"], two["std_ref"])
        self.assertEqual(one["table_sha256"], two["table_sha256"])
        # The reference builder accepts no benchmark, label, or fold inputs.
        self.assertEqual(set(one["mean_ref"]), set(FEATURE_NAMES))

    def test_dummy_atomic_number_has_an_exact_zero_vector(self) -> None:
        values = normalize_atomic_numbers([0, 6, 0])
        np.testing.assert_array_equal(values[0], np.zeros(5, dtype=np.float32))
        np.testing.assert_array_equal(values[2], np.zeros(5, dtype=np.float32))
        self.assertTrue(np.isfinite(values).all())

    def test_observed_element_audit_covers_finite_raw_and_normalized_values(self) -> None:
        manifest = build_element_feature_manifest(
            ["[*]CCO[*]", "c1ccccc1"], source_train_sha256="fixed-test-hash"
        )
        coverage = manifest["coverage"]
        self.assertEqual(coverage["observed_atomic_numbers"], [0, 6, 8])
        self.assertEqual(coverage["dummy_atom_count"], 2)
        self.assertEqual(coverage["real_atom_count"], 9)
        self.assertEqual(coverage["observed_real_element_count"], 2)
        self.assertTrue(coverage["all_observed_real_elements_finite_all_five"])
        for row in coverage["observed_element_table"]:
            self.assertEqual(len(row["raw_values"]), 5)
            self.assertEqual(len(row["normalized_values"]), 5)
            self.assertTrue(row["finite_all_five"])
        self.assertEqual(
            coverage["observed_element_table"][0]["normalized_values"],
            {name: 0.0 for name in FEATURE_NAMES},
        )

    def test_graph_schema_and_keep_dummy_topology_are_unchanged(self) -> None:
        first, info = build_polymer_graph("[*]CCO[*]", sample_id="sample")
        second, _ = build_polymer_graph("[*]CCO[*]", sample_id="sample")
        self.assertEqual(GRAPH_SCHEMA, "own_gnn_repr_keep_dummy_raw_graph_node7_edge16_endpoint1_v1")
        self.assertEqual(info.dummy_atom_count, 2)
        self.assertEqual(int((first.x[:, 0] == 0).sum()), 2)
        self.assertEqual(first.x.shape[0], 5)
        self.assertEqual(first.edge_index.shape[1], 8)
        self.assertTrue(torch.equal(first.x, second.x))
        self.assertTrue(torch.equal(first.edge_index, second.edge_index))
        self.assertTrue(torch.equal(first.edge_attr, second.edge_attr))
        self.assertTrue(torch.equal(first.polymer_endpoint, torch.zeros(5, dtype=torch.float32)))

    def test_normalization_manifest_does_not_read_labels_or_folds(self) -> None:
        manifest = build_element_feature_manifest(
            ["CCO"], source_train_sha256="same-source-id"
        )
        ref = manifest["features"]["reference"]
        self.assertEqual(ref["atomic_number_range_inclusive"], [1, 118])
        self.assertEqual(ref["ddof"], 0)
        self.assertFalse(manifest["source"]["labels_read_by_feature_extractor"])
        self.assertFalse(manifest["source"]["folds_read_by_feature_extractor"])
        self.assertEqual(manifest["reference_table_sha256"], ref["table_sha256"])


class Stage4BModelTests(unittest.TestCase):
    def test_projection_shape_bias_and_zero_initialization(self) -> None:
        model = ElementalPhysicalGNN()
        self.assertEqual(tuple(model.physical_projection.weight.shape), (256, 5))
        self.assertIsNone(model.physical_projection.bias)
        self.assertTrue(torch.equal(model.physical_projection.weight, torch.zeros_like(model.physical_projection.weight)))
        self.assertEqual(sum(p.numel() for p in model.physical_projection.parameters()), 5 * 256)

    def test_same_seed_common_initialization_matches_historical_g0(self) -> None:
        torch.manual_seed(731)
        baseline = GlobalInformationGNN(variant="G0")
        torch.manual_seed(731)
        elemental = ElementalPhysicalGNN()
        baseline_state = baseline.state_dict()
        elemental_state = elemental.state_dict()
        for name, value in baseline_state.items():
            self.assertTrue(torch.equal(value, elemental_state[name]), name)

    def test_zero_init_forward_matches_historical_g0(self) -> None:
        graph, _ = build_polymer_graph("[*]CCO[*]", sample_id="sample")
        graph.physical_features = torch.as_tensor(normalize_atomic_numbers(graph.x[:, 0].numpy()))
        batch = Batch.from_data_list([graph])
        torch.manual_seed(912)
        baseline = GlobalInformationGNN(variant="G0").eval()
        torch.manual_seed(912)
        elemental = ElementalPhysicalGNN().eval()
        with torch.no_grad():
            baseline_output = baseline(batch)
            elemental_output = elemental(batch)
        torch.testing.assert_close(elemental_output, baseline_output, atol=1e-6, rtol=1e-6)

    def test_physical_input_shape_device_and_finiteness_are_enforced(self) -> None:
        graph, _ = build_polymer_graph("CCO")
        model = ElementalPhysicalGNN().eval()
        with self.assertRaisesRegex(ValueError, "physical_features"):
            model(graph)
        graph.physical_features = torch.zeros((graph.x.size(0), 4), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "shape"):
            model(graph)

    def test_checkpoint_round_trip_preserves_prediction(self) -> None:
        graph, _ = build_polymer_graph("[*]CCO[*]")
        graph.physical_features = torch.as_tensor(normalize_atomic_numbers(graph.x[:, 0].numpy()))
        batch = Batch.from_data_list([graph])
        model = ElementalPhysicalGNN().eval()
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "model.pt"
            torch.save({"state_dict": model.state_dict()}, path)
            restored = ElementalPhysicalGNN().eval()
            restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
        with torch.no_grad():
            torch.testing.assert_close(model(batch), restored(batch), atol=0, rtol=0)


class Stage4BRunnerTests(unittest.TestCase):
    def test_frozen_config_invariant_rejects_model_graph_and_training_drift(self) -> None:
        config = copy.deepcopy(ELEMENTAL_CONFIG)
        runner._validate_config(config, baseline=BASELINE_CONFIG)
        for section, key, value in (
            ("model", "num_layers", 5),
            ("model", "operator", "PNA"),
            ("model", "pooling", ["global_mean"]),
            ("training", "learning_rate", 0.002),
            ("graph", "dummy_policy", "remove_dummy"),
        ):
            changed = copy.deepcopy(config)
            changed[section][key] = value
            with self.assertRaises(ValueError, msg=f"{section}.{key}"):
                runner._validate_config(changed, baseline=BASELINE_CONFIG)

    def test_parameter_count_matches_1280_new_weights(self) -> None:
        counts = runner._parameter_counts(ELEMENTAL_CONFIG)
        self.assertEqual(counts["G0_total_trainable"], 1_243_657)
        self.assertEqual(counts["physical_projection_trainable"], 1_280)
        self.assertEqual(counts["E_total_trainable"], 1_244_937)

    def test_seed_and_phase_artifacts_are_isolated(self) -> None:
        with tempfile.TemporaryDirectory() as temporary, mock.patch.object(
            runner, "MODEL_ROOT", Path(temporary)
        ):
            formal42 = runner._output_dir(42, "formal")
            formal42.mkdir(parents=True)
            formal42_second = runner._output_dir(42, "formal")
            formal43 = runner._output_dir(43, "formal")
            smoke43 = runner._output_dir(43, "smoke")
            tiny43 = runner._output_dir(43, "tiny")
            self.assertNotEqual(formal42, formal42_second)
            self.assertNotEqual(formal42, formal43)
            self.assertNotEqual(formal43, smoke43)
            self.assertNotEqual(smoke43, tiny43)
            cache_paths = {
                runner._graph_cache_path(path)
                for path in (formal42, formal43, smoke43, tiny43)
            }
            self.assertEqual(len(cache_paths), 4)
            self.assertTrue(all(path.parent.name == "cache" for path in cache_paths))
            self.assertIn("seed_42", str(formal42))
            self.assertIn("seed_43", str(formal43))

    def test_only_registered_seeds_are_accepted(self) -> None:
        with self.assertRaises(ValueError):
            runner._load_config(seed_override=47)


if __name__ == "__main__":
    unittest.main()
