"""Deterministic, label-free degree statistics for PNA provenance."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Sequence

import torch
from torch_geometric.data import Data

from ...src.data import sha256_file


def graph_set_fingerprint(graphs: Sequence[Data], graph_schema: str) -> str:
    digest = hashlib.sha256()
    digest.update(graph_schema.encode("utf-8"))
    digest.update(b"\0")
    for graph in graphs:
        for name in ("x", "edge_index", "edge_attr"):
            tensor = getattr(graph, name).detach().cpu().contiguous()
            digest.update(name.encode("ascii"))
            digest.update(str(tuple(tensor.shape)).encode("ascii"))
            digest.update(str(tensor.dtype).encode("ascii"))
            digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def degree_histogram(graphs: Sequence[Data]) -> dict[str, Any]:
    counts: list[int] = []
    node_count = 0
    directed_edge_count = 0
    for graph in graphs:
        number_of_nodes = int(graph.x.size(0))
        degree = torch.bincount(graph.edge_index[1].cpu(), minlength=number_of_nodes)
        if degree.numel() != number_of_nodes:
            raise ValueError("Graph edge_index references a node outside x")
        for value in degree.tolist():
            index = int(value)
            if len(counts) <= index:
                counts.extend([0] * (index + 1 - len(counts)))
            counts[index] += 1
        node_count += number_of_nodes
        directed_edge_count += int(graph.edge_index.size(1))
    if not counts:
        raise ValueError("Cannot compute PNA degree statistics from an empty graph set")
    return {
        "degree_definition": "directed incoming degree; bincount(edge_index[1]) per graph",
        "histogram_by_degree": {str(index): int(count) for index, count in enumerate(counts)},
        "histogram_tensor": counts,
        "node_count": node_count,
        "directed_edge_count": directed_edge_count,
        "graph_count": len(graphs),
        "uses_labels_or_targets": False,
    }


def write_degree_provenance(
    path: Path,
    *,
    graphs: Sequence[Data],
    graph_schema: str,
    source_train_sha256: str,
    source_commit: str,
    generator_path: Path,
) -> dict[str, Any]:
    result = {
        "schema_version": 1,
        "graph_schema": graph_schema,
        "source_train_sha256": source_train_sha256,
        "source_commit": source_commit,
        "generator": str(generator_path),
        "generator_sha256": sha256_file(generator_path),
        "graph_set_fingerprint_sha256": graph_set_fingerprint(graphs, graph_schema),
        **degree_histogram(graphs),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return result
