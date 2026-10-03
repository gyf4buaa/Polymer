"""Graph construction shared by the Stage 3A representation variants.

Chemical node and bond features are delegated to the frozen Own-GNN v0
feature functions. This module changes only how repeat-unit boundaries are
represented and always leaves the polymerization-edge channel at zero.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch
from rdkit import Chem
from torch_geometric.data import Data

from ..own_gnn_v0.graph import (
    EDGE_FEATURE_DIM,
    NODE_CARDINALITIES,
    POLYMERIZATION_EDGE_COLUMN,
    GraphBuildError,
    _atom_features,
    _bond_type_name,
    _edge_features,
    _stereo_name,
)

Representation = Literal["keep_dummy", "endpoint_marker"]


@dataclass(frozen=True)
class GraphBuildInfo:
    sample_id: str | None
    representation: str
    topology: str
    dummy_atom_count: int
    original_atom_count: int
    graph_node_count: int
    directed_edge_count: int
    atomic_number_zero_node_count: int
    retained_dummy_node_count: int
    polymer_endpoint_count: int
    endpoint_marker_applied: bool
    endpoint_closure_applied: bool
    polymerization_edge_count: int
    endpoint_neighbors_already_bonded: bool
    fallback_applied: bool
    fallback_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _endpoint_topology(
    molecule: Chem.Mol, dummy_atoms: list[Chem.Atom]
) -> tuple[str, tuple[int, int] | None, bool, str | None]:
    count = len(dummy_atoms)
    if count == 0:
        return "no_dummy_atoms", None, False, None
    if count != 2:
        return f"dummy_atom_count_{count}", None, False, f"dummy_atom_count_{count}"

    if any(int(atom.GetDegree()) != 1 for atom in dummy_atoms):
        return "two_dummy_endpoint_degree_not_one", None, False, "endpoint_degree_not_one"

    dummy_indices = {int(atom.GetIdx()) for atom in dummy_atoms}
    neighbors = tuple(int(atom.GetNeighbors()[0].GetIdx()) for atom in dummy_atoms)
    if any(index in dummy_indices for index in neighbors):
        return "dummy_atoms_adjacent", None, False, "endpoint_neighbor_is_dummy"
    if neighbors[0] == neighbors[1]:
        return "shared_endpoint", None, False, "shared_endpoint"
    bonded = molecule.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is not None
    return "two_distinct_degree_one_endpoints", neighbors, bonded, None


def build_graph(
    smiles: str,
    *,
    representation: Representation,
    sample_id: str | None = None,
) -> tuple[Data, GraphBuildInfo]:
    """Build one graph using either raw dummy atoms or endpoint identity bits.

    ``endpoint_marker`` removes dummy atoms only for an unambiguous pair of
    degree-one dummies with distinct real neighbors. Those two surviving
    nodes receive ``polymer_endpoint=1``. Every other case retains all source
    atoms and bonds and records a deterministic fallback reason. Neither mode
    adds an endpoint-closure edge.
    """
    if representation not in ("keep_dummy", "endpoint_marker"):
        raise ValueError(f"Unknown graph representation: {representation!r}")

    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        identifier = f" for sample_id={sample_id}" if sample_id is not None else ""
        raise GraphBuildError(
            f"RDKit could not build a non-empty graph{identifier}: {smiles!r}"
        )

    dummy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    topology, endpoints, endpoint_bonded, topology_reason = _endpoint_topology(
        molecule, dummy_atoms
    )
    fallback_applied = representation == "endpoint_marker" and topology_reason is not None

    if representation == "endpoint_marker" and endpoints is not None:
        dummy_indices = {int(atom.GetIdx()) for atom in dummy_atoms}
        kept_atom_indices = [
            index for index in range(molecule.GetNumAtoms()) if index not in dummy_indices
        ]
        marker_indices = set(endpoints)
    else:
        kept_atom_indices = list(range(molecule.GetNumAtoms()))
        marker_indices = set()

    old_to_new = {old: new for new, old in enumerate(kept_atom_indices)}
    node_features = [
        _atom_features(molecule.GetAtomWithIdx(index)) for index in kept_atom_indices
    ]
    polymer_endpoint = [
        float(old_index in marker_indices) for old_index in kept_atom_indices
    ]

    directed_edges: list[tuple[int, int]] = []
    edge_features: list[list[float]] = []
    for bond in molecule.GetBonds():
        left_old = int(bond.GetBeginAtomIdx())
        right_old = int(bond.GetEndAtomIdx())
        if left_old not in old_to_new or right_old not in old_to_new:
            continue
        left, right = old_to_new[left_old], old_to_new[right_old]
        feature = _edge_features(
            _bond_type_name(bond),
            conjugated=bool(bond.GetIsConjugated()),
            in_ring=bool(bond.IsInRing()),
            stereo=_stereo_name(bond),
            polymerization=False,
        )
        directed_edges.extend(((left, right), (right, left)))
        edge_features.extend((feature, feature))

    if directed_edges:
        edge_index = torch.tensor(directed_edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURE_DIM), dtype=torch.float32)

    graph = Data(
        x=torch.tensor(node_features, dtype=torch.long).reshape(-1, len(NODE_CARDINALITIES)),
        edge_index=edge_index,
        edge_attr=edge_attr,
        polymer_endpoint=torch.tensor(polymer_endpoint, dtype=torch.float32),
    )
    zero_nodes = sum(
        molecule.GetAtomWithIdx(index).GetAtomicNum() == 0 for index in kept_atom_indices
    )
    info = GraphBuildInfo(
        sample_id=None if sample_id is None else str(sample_id),
        representation=representation,
        topology=topology,
        dummy_atom_count=len(dummy_atoms),
        original_atom_count=int(molecule.GetNumAtoms()),
        graph_node_count=int(graph.x.size(0)),
        directed_edge_count=int(graph.edge_index.size(1)),
        atomic_number_zero_node_count=int(zero_nodes),
        retained_dummy_node_count=int(zero_nodes),
        polymer_endpoint_count=int(sum(polymer_endpoint)),
        endpoint_marker_applied=bool(marker_indices),
        endpoint_closure_applied=False,
        polymerization_edge_count=int(
            graph.edge_attr[:, POLYMERIZATION_EDGE_COLUMN].sum().item()
        ),
        endpoint_neighbors_already_bonded=bool(endpoint_bonded),
        fallback_applied=bool(fallback_applied),
        fallback_reason=topology_reason if fallback_applied else None,
    )
    return graph, info
