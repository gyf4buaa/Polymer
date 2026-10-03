from __future__ import annotations

import copy
import json
import pytest
import torch
from torch_geometric.data import Batch

from from_scratch_gnn.models.own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA,
    build_polymer_graph,
)
from from_scratch_gnn.models.operator_ablation.degree_stats import (
    degree_histogram,
    graph_set_fingerprint,
)
from from_scratch_gnn.models.operator_ablation.model import build_operator_model
from from_scratch_gnn.models.operator_ablation.runner import (
    BASELINE_ROOT,
    _load_operator_config,
    _output_dir,
)
from from_scratch_gnn.models.own_gnn_v0.model import OwnGNNv0
from from_scratch_gnn.models.own_gnn_v0.graph import EDGE_FEATURE_DIM
from from_scratch_gnn.models.polymer_representation_ablation.model import OwnGNNRepresentation
from from_scratch_gnn.src.metrics import TARGETS


GAT_CONFIG = {
    "heads": 4,
    "channels_per_head": 64,
    "concat": True,
    "edge_aware": True,
    "attention_dropout": 0.0,
    "add_self_loops": False,
}
PNA_CONFIG = {
    "aggregators": ["mean", "min", "max", "std"],
    "scalers": ["identity", "amplification", "attenuation"],
    "towers": 1,
    "edge_aware": True,
}


def _batch():
    graphs = [
        build_polymer_graph("[*]CCO[*]", sample_id="dummy")[0],
        build_polymer_graph("CCO", sample_id="plain")[0],
    ]
    return Batch.from_data_list(graphs), graphs


def _model(operator: str, degree=None, *, hidden_dim=256, num_layers=4):
    operator_config = dict(GAT_CONFIG if operator == "gatv2" else PNA_CONFIG)
    if operator == "gatv2":
        operator_config["channels_per_head"] = hidden_dim // int(operator_config["heads"])
    return build_operator_model(
        operator=operator,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=0.1,
        operator_config=operator_config,
        degree_histogram=degree,
    )


def test_historical_keep_dummy_gine_path_preserves_v0_forward_and_parameterization():
    batch, _ = _batch()
    torch.manual_seed(99)
    baseline = OwnGNNv0(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    torch.manual_seed(99)
    historical = OwnGNNRepresentation(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    with torch.no_grad():
        expected = baseline(batch)
        actual = historical(batch)
    assert torch.equal(expected, actual)
    assert all("GINEConv" in type(layer).__name__ for layer in historical.convs)


@pytest.mark.parametrize("operator", ["gatv2", "pna"])
def test_edge_aware_operators_share_graph_schema_output_width_and_checkpoint(operator, tmp_path):
    batch, graphs = _batch()
    degree = degree_histogram(graphs)["histogram_tensor"]
    model = _model(operator, degree).eval()
    assert model.hidden_dim == 256 and model.num_layers == 4
    assert all(
        (block.out_channels * block.heads if operator == "gatv2" else block.out_channels)
        == 256
        for block in model.convs
    )
    assert all(getattr(block, "edge_dim", None) == EDGE_FEATURE_DIM for block in model.convs)
    assert graph_set_fingerprint(graphs, GRAPH_SCHEMA) == graph_set_fingerprint(
        [build_polymer_graph("[*]CCO[*]")[0], build_polymer_graph("CCO")[0]], GRAPH_SCHEMA
    )
    with torch.no_grad():
        hidden = model.encode_nodes(batch)
        predictions = model(batch)
    assert hidden.shape == (batch.x.size(0), 256)
    assert predictions.shape == (2, len(TARGETS))
    assert torch.isfinite(predictions).all()

    checkpoint = tmp_path / f"{operator}.pt"
    torch.save(model.state_dict(), checkpoint)
    restored = _model(operator, degree).eval()
    restored.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    with torch.no_grad():
        restored_predictions = restored(batch)
    assert torch.equal(predictions, restored_predictions)


@pytest.mark.parametrize("operator", ["gatv2", "pna"])
def test_operator_forward_backward_tiny_overfit_and_edge_attr_consumption(operator):
    batch, graphs = _batch()
    degree = degree_histogram(graphs)["histogram_tensor"]
    torch.manual_seed(123)
    model = _model(operator, degree, hidden_dim=32, num_layers=2)
    batch.y = torch.tensor([[0.5, -0.2, 0.1, 0.7, -0.4], [0.2, 0.3, -0.6, 0.1, 0.8]])
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    model.eval()
    with torch.no_grad():
        initial = torch.nn.functional.mse_loss(model(batch), batch.y).item()
    model.train()
    for _ in range(60):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.mse_loss(model(batch), batch.y)
        loss.backward()
        optimizer.step()
    model.eval()
    with torch.no_grad():
        final = torch.nn.functional.mse_loss(model(batch), batch.y).item()
    assert final < initial

    changed_edges = batch.clone()
    changed_edges.edge_attr = torch.zeros_like(batch.edge_attr)
    with torch.no_grad():
        altered = model(changed_edges)
        original = model(batch)
    assert not torch.equal(original, altered), "operator did not consume encoded edge_attr"


def test_degree_histogram_is_deterministic_and_label_free():
    _, graphs = _batch()
    first = degree_histogram(graphs)
    second = degree_histogram(graphs)
    assert first == second
    assert first["graph_count"] == 2
    assert first["node_count"] == sum(int(graph.x.size(0)) for graph in graphs)
    assert first["uses_labels_or_targets"] is False
    assert sum(first["histogram_tensor"]) == first["node_count"]


@pytest.mark.parametrize("operator", ["gatv2", "pna"])
@pytest.mark.parametrize("mutation", ["hidden_dim", "num_layers", "pooling", "training"])
def test_operator_config_rejects_frozen_invariant_drift(tmp_path, operator, mutation):
    config_path = BASELINE_ROOT.parents[0] / "operator_ablation" / operator / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    if mutation == "training":
        config["training"]["learning_rate"] *= 2
    elif mutation == "pooling":
        config["model"]["pooling"] = ["global_sum"]
    else:
        config["model"][mutation] += 1
    tmp_path.joinpath("config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_operator_config(operator_root=tmp_path, operator=operator)


@pytest.mark.parametrize("operator", ["gatv2", "pna"])
def test_seed_override_and_artifact_paths_are_isolated(operator, tmp_path):
    operator_root = tmp_path / operator
    config_path = operator_root / "config.json"
    source_config_path = (
        BASELINE_ROOT.parents[0] / "operator_ablation" / operator / "config.json"
    )
    config_path.parent.mkdir(parents=True)
    config_path.write_bytes(source_config_path.read_bytes())
    source_bytes = config_path.read_bytes()
    config = _load_operator_config(
        operator_root=operator_root, operator=operator, seed_override=43
    )
    assert config["seed"] == 43
    assert config_path.read_bytes() == source_bytes
    assert _output_dir(operator_root, 42) != _output_dir(operator_root, 43)
    assert _output_dir(operator_root, 42) != _output_dir(
        BASELINE_ROOT, 42
    )
