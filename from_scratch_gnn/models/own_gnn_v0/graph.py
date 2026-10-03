"""Polymer repeat-unit graph construction for Own-GNN v0."""
from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
from rdkit import Chem
from torch_geometric.data import Data

GRAPH_SCHEMA = "own_gnn_v0_endpoint_closure_edge16_node7_v1"

# Each atom field is embedded independently and the embeddings are summed.
NODE_FEATURE_NAMES = (
    "atomic_number",
    "degree",
    "formal_charge",
    "hybridization",
    "aromatic",
    "total_hydrogens",
    "chirality",
)
NODE_CARDINALITIES = (120, 7, 8, 8, 2, 6, 4)

BOND_TYPES = ("SINGLE", "DOUBLE", "TRIPLE", "AROMATIC", "DATIVE", "OTHER")
BOND_STEREO = ("STEREONONE", "STEREOANY", "STEREOZ", "STEREOE", "STEREOCIS", "STEREOTRANS", "OTHER")
EDGE_FEATURE_DIM = len(BOND_TYPES) + 1 + 1 + len(BOND_STEREO) + 1
POLYMERIZATION_EDGE_COLUMN = EDGE_FEATURE_DIM - 1

_HYBRIDIZATIONS = {
    "UNSPECIFIED": 0,
    "S": 1,
    "SP": 2,
    "SP2": 3,
    "SP3": 4,
    "SP3D": 5,
    "SP3D2": 6,
}


class GraphBuildError(ValueError):
    """A sample could not be represented as a graph without dropping it."""


@dataclass(frozen=True)
class GraphBuildInfo:
    sample_id: str | None
    dummy_atom_count: int
    original_atom_count: int
    graph_node_count: int
    directed_edge_count: int
    endpoint_closure_applied: bool
    endpoint_neighbors_already_bonded: bool
    fallback_reason: str | None

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


def _atom_features(atom: Chem.Atom) -> list[int]:
    atomic_number = int(atom.GetAtomicNum())
    atomic_number = atomic_number if 0 <= atomic_number <= 118 else 119

    degree = int(atom.GetDegree())
    degree = degree if 0 <= degree <= 5 else 6

    charge = int(atom.GetFormalCharge())
    charge = charge + 3 if -3 <= charge <= 3 else 7

    hybridization_name = str(atom.GetHybridization()).split(".")[-1].upper()
    hybridization = _HYBRIDIZATIONS.get(hybridization_name, 7)

    total_hydrogens = int(atom.GetTotalNumHs(includeNeighbors=True))
    total_hydrogens = total_hydrogens if 0 <= total_hydrogens <= 4 else 5

    chirality = int(atom.GetChiralTag())
    chirality = chirality if 0 <= chirality <= 3 else 3

    return [
        atomic_number,
        degree,
        charge,
        hybridization,
        int(atom.GetIsAromatic()),
        total_hydrogens,
        chirality,
    ]


def _bond_type_name(bond: Chem.Bond) -> str:
    name = str(bond.GetBondType()).split(".")[-1].upper()
    if name == "DATIVE" or name.startswith("DATIVE"):
        return "DATIVE"
    return name if name in BOND_TYPES else "OTHER"


def _stereo_name(bond: Chem.Bond | None) -> str:
    if bond is None:
        return "STEREONONE"
    name = str(bond.GetStereo()).split(".")[-1].upper()
    return name if name in BOND_STEREO else "OTHER"


def _edge_features(
    bond_type: str,
    *,
    conjugated: bool,
    in_ring: bool,
    stereo: str,
    polymerization: bool,
) -> list[float]:
    values = [float(bond_type == item) for item in BOND_TYPES]
    values.extend((float(conjugated), float(in_ring)))
    values.extend(float(stereo == item) for item in BOND_STEREO)
    values.append(float(polymerization))
    return values


def build_polymer_graph(
    smiles: str, *, sample_id: str | None = None
) -> tuple[Data, GraphBuildInfo]:
    """Parse one repeat unit and close valid endpoint pairs with a marked edge.

    When exactly two degree-one dummy atoms have distinct real neighbors, the
    dummy atoms are removed and their neighbors are joined by a special single
    edge. An already-existing neighbor bond is retained as a parallel ordinary
    edge; the additional marked edge expresses the periodic connection.

    Inputs with an ambiguous dummy topology use a deterministic fallback: keep
    every atom and every parsed chemical bond, including the dummy atoms, and
    add no polymerization edge. No valid source sample is silently discarded.
    """
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None or molecule.GetNumAtoms() == 0:
        identifier = f" for sample_id={sample_id}" if sample_id is not None else ""
        raise GraphBuildError(f"RDKit could not build a non-empty graph{identifier}: {smiles!r}")

    dummy_atoms = [atom for atom in molecule.GetAtoms() if atom.GetAtomicNum() == 0]
    dummy_count = len(dummy_atoms)
    fallback_reason: str | None = None
    endpoint_indices: tuple[int, int] | None = None
    endpoint_neighbors_already_bonded = False

    if dummy_count != 2:
        fallback_reason = f"dummy_atom_count_{dummy_count}"
    else:
        degrees = [int(atom.GetDegree()) for atom in dummy_atoms]
        if degrees != [1, 1]:
            fallback_reason = "endpoint_degree_not_one"
        else:
            neighbors = tuple(int(atom.GetNeighbors()[0].GetIdx()) for atom in dummy_atoms)
            dummy_indices = {int(atom.GetIdx()) for atom in dummy_atoms}
            if any(index in dummy_indices for index in neighbors):
                fallback_reason = "endpoint_neighbor_is_dummy"
            elif neighbors[0] == neighbors[1]:
                fallback_reason = "shared_endpoint"
            else:
                endpoint_indices = neighbors
                endpoint_neighbors_already_bonded = (
                    molecule.GetBondBetweenAtoms(neighbors[0], neighbors[1]) is not None
                )

    if endpoint_indices is None:
        kept_atom_indices = list(range(molecule.GetNumAtoms()))
    else:
        dummy_indices = {int(atom.GetIdx()) for atom in dummy_atoms}
        kept_atom_indices = [
            atom_index
            for atom_index in range(molecule.GetNumAtoms())
            if atom_index not in dummy_indices
        ]
    old_to_new = {old: new for new, old in enumerate(kept_atom_indices)}

    node_features = [_atom_features(molecule.GetAtomWithIdx(index)) for index in kept_atom_indices]
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

    if endpoint_indices is not None:
        left, right = (old_to_new[index] for index in endpoint_indices)
        feature = _edge_features(
            "SINGLE",
            conjugated=False,
            in_ring=False,
            stereo="STEREONONE",
            polymerization=True,
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
        x=torch.tensor(node_features, dtype=torch.long),
        edge_index=edge_index,
        edge_attr=edge_attr,
    )
    info = GraphBuildInfo(
        sample_id=None if sample_id is None else str(sample_id),
        dummy_atom_count=dummy_count,
        original_atom_count=int(molecule.GetNumAtoms()),
        graph_node_count=int(graph.x.size(0)),
        directed_edge_count=int(graph.edge_index.size(1)),
        endpoint_closure_applied=endpoint_indices is not None,
        endpoint_neighbors_already_bonded=endpoint_neighbors_already_bonded,
        fallback_reason=fallback_reason,
    )
    return graph, info
