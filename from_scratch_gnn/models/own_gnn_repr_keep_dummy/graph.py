"""Variant A graph schema: retain the original RDKit molecular graph."""
from __future__ import annotations

from typing import Any

from ..polymer_representation_ablation.graph_builder import build_graph

GRAPH_SCHEMA = "own_gnn_repr_keep_dummy_raw_graph_node7_edge16_endpoint1_v1"
REPRESENTATION = "keep_dummy"


def build_polymer_graph(
    smiles: str, *, sample_id: str | None = None
) -> tuple[Any, Any]:
    return build_graph(smiles, representation=REPRESENTATION, sample_id=sample_id)
