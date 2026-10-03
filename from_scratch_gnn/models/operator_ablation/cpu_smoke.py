"""Run a tiny non-selection CPU fit using real keep-dummy graphs."""
from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import torch
from torch_geometric.data import Batch

from ...src.data import sha256_file, write_json
from ..own_gnn_repr_keep_dummy.graph import GRAPH_SCHEMA
from ..own_gnn_v0.model import masked_huber_loss
from .degree_stats import degree_histogram, graph_set_fingerprint, write_degree_provenance
from .model import build_operator_model
from .runner import BASELINE_ROOT

TRACK_ROOT = Path(__file__).resolve().parents[2]
MODEL_ROOT = Path(__file__).resolve().parent
EXPECTED_TRAIN_SHA256 = "1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1"


def run_cpu_smoke(operator: str, *, graph_cache: Path, steps: int = 120) -> dict[str, object]:
    cache = torch.load(graph_cache, map_location="cpu", weights_only=False)
    metadata = cache.get("metadata", {})
    if metadata.get("cache_schema") != GRAPH_SCHEMA:
        raise ValueError("CPU smoke graph cache does not use the frozen keep-dummy schema")
    if metadata.get("source_train_sha256") != EXPECTED_TRAIN_SHA256:
        raise ValueError("CPU smoke graph cache does not match the frozen training data")
    graphs = cache.get("graphs", [])
    if len(graphs) != 7973:
        raise ValueError(f"Expected all 7,973 frozen graphs, got {len(graphs)}")
    selected = sorted(graphs, key=lambda graph: int(graph.x.size(0)))[:3]
    batch = Batch.from_data_list(selected)
    batch.y = torch.tensor(
        [[0.25, -0.4, 0.1, 0.7, -0.6], [-0.2, 0.35, -0.5, 0.15, 0.8], [0.5, 0.1, 0.2, -0.3, 0.4]],
        dtype=torch.float32,
    )
    config = json.loads((MODEL_ROOT / operator / "config.json").read_text(encoding="utf-8"))
    degree_stats = degree_histogram(graphs)
    seed = 4200 + (0 if operator == "gatv2" else 1)
    torch.manual_seed(seed)
    model = build_operator_model(
        operator=operator,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        operator_config=config["model"]["operator_config"],
        degree_histogram=degree_stats["histogram_tensor"],
    ).cpu()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    model.eval()
    with torch.no_grad():
        initial = float(masked_huber_loss(model(batch), batch.y).item())
    started = time.monotonic()
    model.train()
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(batch)
        loss = masked_huber_loss(prediction, batch.y)
        if not torch.isfinite(loss):
            raise RuntimeError(f"{operator} CPU smoke produced a non-finite loss")
        loss.backward()
        optimizer.step()
    duration = time.monotonic() - started
    model.eval()
    with torch.no_grad():
        predictions = model(batch)
        final = float(masked_huber_loss(predictions, batch.y).item())

    output = MODEL_ROOT / operator / "artifacts" / "cpu_smoke" / "seed_42"
    output.mkdir(parents=True, exist_ok=True)
    if operator == "pna":
        degree_path = output / "degree_histogram.json"
        degree_provenance = write_degree_provenance(
            degree_path,
            graphs=graphs,
            graph_schema=GRAPH_SCHEMA,
            source_train_sha256=EXPECTED_TRAIN_SHA256,
            source_commit=subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=TRACK_ROOT.parent, text=True
            ).strip(),
            generator_path=Path(__file__).with_name("degree_stats.py"),
        )
    else:
        degree_provenance = None

    checkpoint = output / "cpu_smoke.pt"
    torch.save({"state_dict": model.state_dict()}, checkpoint)
    restored = build_operator_model(
        operator=operator,
        hidden_dim=int(config["model"]["hidden_dim"]),
        num_layers=int(config["model"]["num_layers"]),
        dropout=float(config["model"]["dropout"]),
        operator_config=config["model"]["operator_config"],
        degree_histogram=degree_stats["histogram_tensor"],
    ).cpu().eval()
    restored.load_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=True)["state_dict"]
    )
    with torch.no_grad():
        restored_predictions = restored(batch)
    checkpoint_delta = float(torch.max(torch.abs(predictions - restored_predictions)).item())
    checkpoint.unlink()
    if not final < initial or checkpoint_delta > 1e-6:
        raise AssertionError(
            f"{operator} CPU smoke failed: {initial:.6f} -> {final:.6f}, "
            f"checkpoint delta {checkpoint_delta:.3g}"
        )
    result = {
        "status": "passed",
        "operator": operator,
        "selection_role": "non-selection CPU tiny-overfit smoke",
        "device": "cpu",
        "steps": steps,
        "sample_count": len(selected),
        "hidden_dim": model.hidden_dim,
        "num_layers": model.num_layers,
        "graph_schema": GRAPH_SCHEMA,
        "graph_set_fingerprint_sha256": graph_set_fingerprint(graphs, GRAPH_SCHEMA),
        "source_train_sha256": EXPECTED_TRAIN_SHA256,
        "effective_config_sha256": sha256_file(MODEL_ROOT / operator / "config.json"),
        "initial_masked_huber_loss": initial,
        "final_masked_huber_loss": final,
        "relative_loss_reduction": 1.0 - final / initial if initial else None,
        "duration_seconds": duration,
        "checkpoint_round_trip": checkpoint_delta <= 1e-6,
        "checkpoint_max_prediction_delta": checkpoint_delta,
        "parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "message_passing_block_parameter_count": sum(
            parameter.numel()
            for block in model.convs
            for parameter in block.parameters()
            if parameter.requires_grad
        ),
        "pna_degree_statistics": degree_provenance,
        "degree_histogram_sha256": sha256_file(output / "degree_histogram.json")
        if operator == "pna"
        else None,
    }
    write_json(output / "cpu_smoke.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--operator", choices=("gatv2", "pna"), required=True)
    parser.add_argument(
        "--graph-cache",
        type=Path,
        default=BASELINE_ROOT / "artifacts" / "cache" / "polymer_graphs_v1.pt",
    )
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    print(json.dumps(run_cpu_smoke(args.operator, graph_cache=args.graph_cache, steps=args.steps), indent=2))


if __name__ == "__main__":
    main()
