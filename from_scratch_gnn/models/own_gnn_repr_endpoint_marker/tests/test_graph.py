from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Batch

from from_scratch_gnn.models.own_gnn_repr_endpoint_marker.graph import (
    GRAPH_SCHEMA,
    build_polymer_graph,
)
from from_scratch_gnn.models.own_gnn_v0.graph import (
    POLYMERIZATION_EDGE_COLUMN,
    GraphBuildError,
)
from from_scratch_gnn.models.polymer_representation_ablation.model import (
    OwnGNNRepresentation,
)


def test_normal_pair_is_removed_and_both_real_endpoints_are_marked_without_closure():
    graph, info = build_polymer_graph("[*]CCO[*]", sample_id="normal")

    assert GRAPH_SCHEMA.endswith("_v1")
    assert info.original_atom_count == 5
    assert info.graph_node_count == 3
    assert info.retained_dummy_node_count == 0
    assert info.polymer_endpoint_count == 2
    assert info.endpoint_marker_applied
    assert not info.endpoint_closure_applied
    assert info.polymerization_edge_count == 0
    assert graph.polymer_endpoint.tolist() == [1.0, 0.0, 1.0]
    assert graph.edge_index.size(1) == 4
    assert not graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].any()


def test_preexisting_endpoint_bond_is_kept_once_and_no_parallel_closure_is_added():
    graph, info = build_polymer_graph("[*]CC[*]")

    assert info.graph_node_count == 2
    assert info.endpoint_neighbors_already_bonded
    assert info.polymer_endpoint_count == 2
    assert graph.edge_index.size(1) == 2
    assert int(graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].sum().item()) == 0


def test_no_dummy_graph_is_unchanged_and_has_no_fallback():
    graph, info = build_polymer_graph("CCO")

    assert info.topology == "no_dummy_atoms"
    assert not info.fallback_applied
    assert info.fallback_reason is None
    assert info.graph_node_count == info.original_atom_count == 3
    assert info.polymer_endpoint_count == 0
    assert graph.edge_index.size(1) == 4
    assert not graph.polymer_endpoint.any()


@pytest.mark.parametrize(
    ("smiles", "expected_reason", "expected_dummy_count"),
    [
        ("[*]CC", "dummy_atom_count_1", 1),
        ("[*]C(*)", "shared_endpoint", 2),
        ("[*]CC(*)C[*]", "dummy_atom_count_3", 3),
        ("[*]CC(*)C(*)C[*]", "dummy_atom_count_4", 4),
        ("[*].CC[*]", "endpoint_degree_not_one", 2),
        ("*[*]", "endpoint_neighbor_is_dummy", 2),
    ],
)
def test_ambiguous_dummy_topology_uses_lossless_fallback(
    smiles: str, expected_reason: str, expected_dummy_count: int
):
    graph, info = build_polymer_graph(smiles, sample_id="fallback")

    assert info.fallback_applied
    assert info.fallback_reason == expected_reason
    assert info.dummy_atom_count == expected_dummy_count
    assert info.graph_node_count == info.original_atom_count
    assert info.retained_dummy_node_count == expected_dummy_count
    assert info.polymer_endpoint_count == 0
    assert graph.polymer_endpoint.numel() == info.original_atom_count
    assert not graph.polymer_endpoint.any()
    assert not graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].any()


def test_invalid_smiles_is_reported_with_sample_id():
    with pytest.raises(GraphBuildError, match="sample_id=bad-row"):
        build_polymer_graph("[not-valid", sample_id="bad-row")


def test_shared_representation_model_accepts_marked_graphs_and_batches():
    first, _ = build_polymer_graph("[*]CCO[*]")
    second, _ = build_polymer_graph("CCO")
    model = OwnGNNRepresentation(hidden_dim=32, num_layers=2, dropout=0.0)

    output = model(Batch.from_data_list([first, second]))

    assert output.shape == (2, 5)
    assert torch.isfinite(output).all()
