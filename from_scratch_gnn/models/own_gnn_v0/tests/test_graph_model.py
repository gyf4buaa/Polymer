from __future__ import annotations

import torch
import pytest
from torch_geometric.data import Batch

from from_scratch_gnn.models.own_gnn_v0.graph import (
    EDGE_FEATURE_DIM,
    NODE_CARDINALITIES,
    POLYMERIZATION_EDGE_COLUMN,
    GraphBuildError,
    build_polymer_graph,
)
from from_scratch_gnn.models.own_gnn_v0.model import OwnGNNv0, masked_huber_loss


def test_endpoint_closure_removes_dummies_and_adds_one_marked_undirected_edge():
    graph, info = build_polymer_graph("[*]CCO[*]", sample_id="endpoint-example")

    assert info.sample_id == "endpoint-example"
    assert info.endpoint_closure_applied
    assert info.fallback_reason is None
    assert info.dummy_atom_count == 2
    assert info.original_atom_count == 5
    assert info.graph_node_count == 3
    assert graph.x.shape == (3, len(NODE_CARDINALITIES))
    assert graph.edge_index.shape == (2, 6)
    assert graph.edge_attr.shape == (6, EDGE_FEATURE_DIM)
    assert int(graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].sum().item()) == 2


def test_existing_endpoint_bond_is_preserved_alongside_marked_polymer_edge():
    graph, info = build_polymer_graph("[*]CC[*]")

    assert info.endpoint_closure_applied
    assert info.endpoint_neighbors_already_bonded
    # One ordinary C-C bond and one special polymerization edge, both directed.
    assert graph.x.size(0) == 2
    assert graph.edge_index.size(1) == 4
    assert int(graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].sum().item()) == 2


@pytest.mark.parametrize(
    ("smiles", "expected_reason"),
    [
        ("CCO", "dummy_atom_count_0"),
        ("[*]CC", "dummy_atom_count_1"),
        ("*[*]", "endpoint_neighbor_is_dummy"),
        ("[*]C(*)", "shared_endpoint"),
        ("[*]C(*)C(*)", "dummy_atom_count_3"),
    ],
)
def test_ambiguous_or_nonpolymer_inputs_use_lossless_deterministic_fallback(
    smiles: str, expected_reason: str
):
    graph, info = build_polymer_graph(smiles, sample_id="fallback-example")

    assert not info.endpoint_closure_applied
    assert info.fallback_reason == expected_reason
    assert info.graph_node_count == info.original_atom_count
    assert info.sample_id == "fallback-example"
    assert graph.x.size(0) == info.original_atom_count
    assert graph.edge_attr.shape == (info.directed_edge_count, EDGE_FEATURE_DIM)
    assert not graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].any()


def test_invalid_smiles_fails_explicitly_instead_of_dropping_the_sample():
    with pytest.raises(GraphBuildError, match="sample_id=broken-row"):
        build_polymer_graph("[not-valid", sample_id="broken-row")


def test_masked_huber_loss_masks_missing_labels_and_weights_observed_tasks_equally():
    prediction = torch.tensor([[0.0, 5.0], [100.0, 2.0]], requires_grad=True)
    target = torch.tensor([[1.0, float("nan")], [float("nan"), 4.0]])

    loss = masked_huber_loss(prediction, target, delta=1.0)
    assert torch.allclose(loss, torch.tensor(1.0))
    loss.backward()
    assert prediction.grad is not None
    assert prediction.grad[1, 0].item() == 0.0
    assert prediction.grad[0, 1].item() == 0.0


def _sample_batch():
    graph_a, _ = build_polymer_graph("[*]CCO[*]")
    graph_b, _ = build_polymer_graph("c1ccccc1")
    return Batch.from_data_list([graph_a, graph_b])


def test_forward_shape_and_five_independent_property_heads():
    model = OwnGNNv0(hidden_dim=32, num_layers=2, dropout=0.1)
    output = model(_sample_batch())

    assert output.shape == (2, 5)
    assert tuple(model.heads.keys()) == ("Tg", "FFV", "Tc", "Density", "Rg")
    assert len({id(head) for head in model.heads.values()}) == 5


def test_checkpoint_save_load_round_trip(tmp_path):
    model = OwnGNNv0(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    batch = _sample_batch()
    with torch.no_grad():
        expected = model(batch)

    checkpoint = tmp_path / "own_gnn.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored = OwnGNNv0(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    with torch.no_grad():
        actual = restored(batch)
    assert torch.equal(expected, actual)
