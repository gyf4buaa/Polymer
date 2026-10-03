"""Variant B graph schema: remove valid dummies and mark their real endpoints."""
from __future__ import annotations

from typing import Any

from ..polymer_representation_ablation.graph_builder import build_graph

GRAPH_SCHEMA = "own_gnn_repr_endpoint_marker_no_closure_node7_edge16_endpoint1_v1"
REPRESENTATION = "endpoint_marker"


def build_polymer_graph(
    smiles: str, *, sample_id: str | None = None
) -> tuple[Any, Any]:
    return build_graph(smiles, representation=REPRESENTATION, sample_id=sample_id)
