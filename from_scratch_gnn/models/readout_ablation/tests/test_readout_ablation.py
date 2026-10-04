from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest
import torch
from torch_geometric.data import Batch
from torch_geometric.nn import global_add_pool

from from_scratch_gnn.models.own_gnn_repr_keep_dummy.graph import (
    GRAPH_SCHEMA,
    build_polymer_graph,
)
from from_scratch_gnn.models.polymer_representation_ablation.model import OwnGNNRepresentation
from from_scratch_gnn.models.own_gnn_v0.model import OwnGNNv0, masked_huber_loss
from from_scratch_gnn.models.readout_ablation.model import (
    ReadoutGNN,
    normalized_attention_entropy,
)
from from_scratch_gnn.models.readout_ablation.runner import (
    _load_readout_config,
    _output_dir,
    _parameter_counts,
)
from from_scratch_gnn.scripts.aggregate_readout_ablation import (
    _historical_r0,
    _paired,
    _stats,
)
from from_scratch_gnn.src.metrics import TARGETS


VARIANTS = ("R0", "R1", "R2")


def _graphs():
    return [
        build_polymer_graph("[*]CCO[*]", sample_id="dummy")[0],
        build_polymer_graph("c1ccccc1", sample_id="plain")[0],
    ]


def _batch():
    return Batch.from_data_list(_graphs())


def _gate_parameters(model: ReadoutGNN):
    return [
        parameter
        for name, parameter in model.named_parameters()
        if name.startswith("shared_gate.") or name.startswith("property_gates.")
    ]


def test_keep_dummy_graph_and_historical_gine_path_are_unchanged():
    batch = _batch()
    assert GRAPH_SCHEMA == "own_gnn_repr_keep_dummy_raw_graph_node7_edge16_endpoint1_v1"
    torch.manual_seed(99)
    historical = OwnGNNRepresentation(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    torch.manual_seed(99)
    baseline = OwnGNNv0(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    torch.manual_seed(99)
    r0 = ReadoutGNN(
        readout_variant="R0", hidden_dim=32, num_layers=2, dropout=0.0
    ).eval()
    with torch.no_grad():
        expected = historical(batch)
        assert torch.equal(expected, baseline(batch))
        assert torch.equal(expected, r0(batch))
        assert torch.equal(historical.encode_nodes(batch), r0.encode_nodes(batch))
    assert all("GINEConv" in type(layer).__name__ for layer in r0.convs)
    assert r0.polymer_endpoint_embedding.shape == (32,)
    graphs_again = _graphs()
    assert torch.equal(graphs_again[0].x, batch.to_data_list()[0].x)
    assert torch.equal(graphs_again[0].edge_index, batch.to_data_list()[0].edge_index)
    assert torch.equal(graphs_again[0].edge_attr, batch.to_data_list()[0].edge_attr)
    assert torch.equal(
        graphs_again[0].polymer_endpoint,
        batch.to_data_list()[0].polymer_endpoint,
    )


@pytest.mark.parametrize("variant", VARIANTS)
def test_output_shape_readout_width_and_checkpoint_round_trip(variant, tmp_path):
    batch = _batch()
    model = ReadoutGNN(readout_variant=variant, hidden_dim=32, num_layers=2, dropout=0.0).eval()
    with torch.no_grad():
        expected = model(batch)
    assert expected.shape == (2, len(TARGETS))
    assert all(head[0].in_features == 64 for head in model.heads.values())
    assert model.graph_embedding(
        model.encode_nodes(batch), batch.batch, target="Tg" if variant == "R2" else None
    ).shape == (2, 64)

    checkpoint = tmp_path / f"{variant}.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored = ReadoutGNN(
        readout_variant=variant, hidden_dim=32, num_layers=2, dropout=0.0
    ).eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    with torch.no_grad():
        actual = restored(batch)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("variant", ["R1", "R2"])
def test_zero_initialized_gates_match_historical_forward_and_preserve_rng(variant):
    batch = _batch()
    torch.manual_seed(713)
    historical = OwnGNNRepresentation(hidden_dim=32, num_layers=2, dropout=0.0).eval()
    torch.manual_seed(713)
    attentive = ReadoutGNN(
        readout_variant=variant, hidden_dim=32, num_layers=2, dropout=0.0
    ).eval()
    assert set(historical.state_dict()).issubset(attentive.state_dict())
    for name, value in historical.state_dict().items():
        assert torch.equal(value, attentive.state_dict()[name]), name
    gate_params = _gate_parameters(attentive)
    assert gate_params
    assert all(parameter.numel() in (32,) for parameter in gate_params)
    assert all(torch.count_nonzero(parameter).item() == 0 for parameter in gate_params)
    if variant == "R1":
        assert attentive.shared_gate is not None
        assert attentive.shared_gate.bias is None
    else:
        assert tuple(attentive.property_gates.keys()) == TARGETS
        assert all(gate.bias is None for gate in attentive.property_gates.values())
    with torch.no_grad():
        historical_output = historical(batch)
        attentive_output = attentive(batch)
    assert torch.allclose(historical_output, attentive_output, atol=1e-6, rtol=1e-6)


def test_all_three_readouts_are_equal_at_initialization_with_common_parameters():
    batch = _batch()
    models = {}
    outputs = {}
    for variant in VARIANTS:
        torch.manual_seed(909)
        model = ReadoutGNN(
            readout_variant=variant, hidden_dim=32, num_layers=2, dropout=0.0
        ).eval()
        models[variant] = model
        with torch.no_grad():
            outputs[variant] = model(batch)
    shared_names = set(models["R0"].state_dict())
    for variant in ("R1", "R2"):
        for name in shared_names:
            assert torch.equal(
                models["R0"].state_dict()[name], models[variant].state_dict()[name]
            ), name
        assert torch.allclose(outputs["R0"], outputs[variant], atol=1e-6, rtol=1e-6)
    assert torch.allclose(outputs["R1"], outputs["R2"], atol=1e-6, rtol=1e-6)


@pytest.mark.parametrize("variant", ["R1", "R2"])
def test_attention_softmax_is_graph_wise_and_entropy_is_normalized(variant):
    batch = _batch()
    model = ReadoutGNN(readout_variant=variant, hidden_dim=32, num_layers=2, dropout=0.0).eval()
    if variant == "R1":
        model.shared_gate.weight.data.copy_(torch.linspace(-0.2, 0.2, 32).view(1, 32))
    else:
        for index, target in enumerate(TARGETS):
            model.property_gates[target].weight.data.fill_((index + 1) * 0.015)
    with torch.no_grad():
        hidden = model.encode_nodes(batch)
        graph_a = Batch.from_data_list([_graphs()[0]])
        hidden_a = model.encode_nodes(graph_a)
        for target in TARGETS:
            selected_target = target if variant == "R2" else None
            weights = model.attention_weights(hidden, batch.batch, selected_target)
            sums = global_add_pool(weights, batch.batch)
            assert torch.allclose(sums, torch.ones_like(sums), atol=1e-6)
            node_count_a = int((batch.batch == 0).sum().item())
            weights_a_alone = model.attention_weights(
                hidden_a, graph_a.batch, selected_target
            )
            assert torch.allclose(weights[:node_count_a], weights_a_alone, atol=1e-6)
            entropy = normalized_attention_entropy(weights, batch.batch)
            assert entropy.shape == (2,)
            assert torch.all((entropy >= 0) & (entropy <= 1.0 + 1e-6))


@pytest.mark.parametrize("variant", ["R1", "R2"])
def test_cpu_tiny_overfit_moves_gates_and_round_trips_checkpoint(variant, tmp_path):
    torch.manual_seed(151)
    batch = _batch()
    batch.y = torch.tensor(
        [[0.5, -0.2, 0.1, 0.7, -0.4], [0.2, 0.3, -0.6, 0.1, 0.8]],
        dtype=torch.float32,
    )
    model = ReadoutGNN(readout_variant=variant, hidden_dim=32, num_layers=2, dropout=0.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003, weight_decay=0.0)
    model.eval()
    with torch.no_grad():
        initial = float(masked_huber_loss(model(batch), batch.y))
    model.train()
    for step in range(80):
        optimizer.zero_grad(set_to_none=True)
        loss = masked_huber_loss(model(batch), batch.y)
        loss.backward()
        if step == 0:
            assert all(parameter.grad is not None for parameter in _gate_parameters(model))
            assert all(parameter.grad.norm().item() > 0 for parameter in _gate_parameters(model))
        optimizer.step()
    model.eval()
    with torch.no_grad():
        expected = model(batch)
        final = float(masked_huber_loss(expected, batch.y))
    assert final < initial
    assert all(value > 0 for value in model.gate_l2_norms().values())

    path = tmp_path / "tiny.pt"
    torch.save({"state_dict": model.state_dict()}, path)
    restored = ReadoutGNN(readout_variant=variant, hidden_dim=32, num_layers=2, dropout=0.0).eval()
    restored.load_state_dict(torch.load(path, map_location="cpu", weights_only=True)["state_dict"])
    with torch.no_grad():
        actual = restored(batch)
    assert torch.equal(expected, actual)


@pytest.mark.parametrize("variant,expected_gate_count", [("r1_shared", 256), ("r2_property", 1280)])
def test_parameter_counts_only_add_expected_gate_weights(variant, expected_gate_count):
    config_path = Path(__file__).resolve().parents[1] / variant / "config.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    counts = _parameter_counts(variant, config)
    assert counts["historical_r0_total_trainable"] == 1_243_657
    assert counts["readout_gate_trainable"] == expected_gate_count
    assert counts["new_parameters_vs_r0"] == expected_gate_count
    assert counts["total_trainable"] == 1_243_657 + expected_gate_count


@pytest.mark.parametrize(
    "mutation",
    [
        "operator",
        "hidden_dim",
        "num_layers",
        "dropout",
        "training",
        "pooling",
        "readout_gate",
    ],
)
def test_frozen_configuration_rejects_architecture_or_training_drift(tmp_path, mutation):
    variant = "r1_shared"
    source = Path(__file__).resolve().parents[1] / variant / "config.json"
    variant_root = tmp_path / variant
    variant_root.mkdir()
    config = json.loads(source.read_text(encoding="utf-8"))
    config = copy.deepcopy(config)
    if mutation == "training":
        config["training"]["learning_rate"] *= 2
    elif mutation == "pooling":
        config["model"]["pooling"] = ["global_sum"]
    elif mutation == "readout_gate":
        config["model"]["readout"]["gate"] = "hidden_mlp"
    else:
        if mutation == "operator":
            config["model"][mutation] = "GATv2"
        elif mutation == "dropout":
            config["model"][mutation] = 0.2
        else:
            config["model"][mutation] += 1
    (variant_root / "config.json").write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError):
        _load_readout_config(variant_root=variant_root, variant=variant)


def test_seed_and_variant_artifact_paths_are_isolated(tmp_path):
    r1_root = tmp_path / "r1_shared"
    r2_root = tmp_path / "r2_property"
    r1_42 = _output_dir(r1_root, 42)
    r1_43 = _output_dir(r1_root, 43)
    r2_42 = _output_dir(r2_root, 42)
    assert r1_42 != r1_43 and r1_42 != r2_42
    (r1_42 / "run_metadata.json").parent.mkdir(parents=True, exist_ok=True)
    (r1_42 / "run_metadata.json").write_text("{}", encoding="utf-8")
    with pytest.raises(FileExistsError):
        _output_dir(r1_root, 42)


def test_historical_r0_seed_scores_and_paired_summary_are_reused():
    overall, per_target = _historical_r0()
    assert overall == {
        "42": 0.022772872219188344,
        "43": 0.022648625887830755,
        "44": 0.02288966669365641,
        "45": 0.02273300693214453,
        "46": 0.022882862181573347,
    }
    assert set(per_target) == set(TARGETS)
    example = _paired(
        {"42": 0.9, "43": 1.1, "44": 0.8, "45": 1.0, "46": 0.9},
        {"42": 1.0, "43": 1.0, "44": 1.0, "45": 1.0, "46": 1.0},
    )
    assert example["left_lower_seed_count"] == 3
    assert example["by_seed"]["43"] == pytest.approx(0.1)
    assert _stats(overall)["mean"] == pytest.approx(0.022785406782878676)
