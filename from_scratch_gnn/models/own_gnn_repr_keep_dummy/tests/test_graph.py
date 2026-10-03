from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from from_scratch_gnn.models.own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA,
    build_polymer_graph,
)
from from_scratch_gnn.models.own_gnn_v0.graph import (
    NODE_CARDINALITIES,
    POLYMERIZATION_EDGE_COLUMN,
    GraphBuildError,
)
from from_scratch_gnn.models.own_gnn_v0.model import OwnGNNv0
from from_scratch_gnn.models.polymer_representation_ablation.model import (
    OwnGNNRepresentation,
)


def test_raw_graph_schema_keeps_dummy_atoms_and_has_a_valid_zero_atomic_number_bucket():
    graph, info = build_polymer_graph("[*]CCO[*]", sample_id="normal")

    assert GRAPH_SCHEMA.endswith("_v1")
    assert info.graph_node_count == info.original_atom_count == 5
    assert info.retained_dummy_node_count == info.atomic_number_zero_node_count == 2
    assert info.polymer_endpoint_count == 0
    assert info.endpoint_closure_applied is False
    assert info.polymerization_edge_count == 0
    assert graph.x.shape == (5, len(NODE_CARDINALITIES))
    assert graph.x[:, 0].tolist().count(0) == 2
    assert graph.edge_index.size(1) == 8
    assert not graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].any()


def test_adjacent_endpoints_keep_original_bond_without_closure_or_endpoint_marker():
    graph, info = build_polymer_graph("[*]CC[*]")

    assert info.endpoint_neighbors_already_bonded
    assert graph.x.size(0) == 4
    assert graph.edge_index.size(1) == 6
    assert info.retained_dummy_node_count == 2
    assert not graph.polymer_endpoint.any()
    assert not graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].any()


@pytest.mark.parametrize(
    ("smiles", "expected_nodes", "expected_edges", "expected_dummies"),
    [
        ("CCO", 3, 4, 0),
        ("[*]CC", 3, 4, 1),
        ("[*]CC(*)C[*]", 6, 10, 3),
        ("[*]CC(*)C(*)C[*]", 8, 14, 4),
        ("[*]C(*)", 3, 4, 2),
        ("*[*]", 2, 2, 2),
    ],
)
def test_nonstandard_topologies_are_retained_without_mutation(
    smiles: str, expected_nodes: int, expected_edges: int, expected_dummies: int
):
    graph, info = build_polymer_graph(smiles)

    assert info.graph_node_count == info.original_atom_count == expected_nodes
    assert info.retained_dummy_node_count == expected_dummies
    assert info.directed_edge_count == expected_edges
    assert info.fallback_applied is False
    assert info.polymer_endpoint_count == 0
    assert not graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].any()


def test_invalid_smiles_is_reported_with_sample_id():
    with pytest.raises(GraphBuildError, match="sample_id=bad-row"):
        build_polymer_graph("[not-valid", sample_id="bad-row")


def test_shared_representation_model_accepts_raw_dummy_graphs_and_batches():
    first, _ = build_polymer_graph("[*]CCO[*]")
    second, _ = build_polymer_graph("CCO")
    model = OwnGNNRepresentation(hidden_dim=32, num_layers=2, dropout=0.0)

    output = model(Batch.from_data_list([first, second]))

    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()


def test_zero_endpoint_feature_keeps_the_frozen_v0_forward_identical():
    graph, _ = build_polymer_graph("[*]CCO[*]")
    batch = Batch.from_data_list([graph])

    torch.manual_seed(314)
    baseline = OwnGNNv0(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    torch.manual_seed(314)
    representation = OwnGNNRepresentation(
        hidden_dim=32, num_layers=2, dropout=0.0
    ).eval()
    with torch.no_grad():
        expected = baseline(batch)
        actual = representation(batch)

    assert torch.equal(expected, actual)
