"""Public third-party GATv2 architecture and fold-local molecular preprocessing."""
from __future__ import annotations

import warnings
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from sklearn.feature_selection import SelectKBest, f_regression
from torch_geometric.data import Data
from torch_geometric.nn import BatchNorm, GATv2Conv, global_mean_pool

from ...src.metrics import TARGETS

RDLogger.DisableLog("rdApp.*")

NODE_FEATURES = 7
EDGE_FEATURES = 2
MORGAN_BITS = 1024
MORGAN_RADIUS = 2
SELECTED_MORGAN_BITS = 50


def _atom_features(atom: Chem.Atom) -> list[float]:
    """Seven scalar atom features, matching the public solution."""
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        float(int(atom.GetHybridization())),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ]


def smiles_to_periodic_graph(smiles: str) -> Data:
    """Build the public solution's atom/bond graph with a periodic star edge."""
    molecule = Chem.MolFromSmiles(smiles, sanitize=False)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES: {smiles!r}")
    Chem.SanitizeMol(molecule)

    x = torch.tensor([_atom_features(atom) for atom in molecule.GetAtoms()], dtype=torch.float32)
    edges: list[list[int]] = []
    edge_features: list[list[float]] = []
    star_indices: list[int] = []
    for bond in molecule.GetBonds():
        left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feature = [float(bond.GetBondTypeAsDouble()), float(bond.GetIsConjugated())]
        edges.extend(([left, right], [right, left]))
        edge_features.extend((feature, feature))
    star_indices = [atom.GetIdx() for atom in molecule.GetAtoms() if atom.GetSymbol() == "*"]
    if len(star_indices) == 2:
        left, right = star_indices
        edges.extend(([left, right], [right, left]))
        edge_features.extend(([-1.0, 0.0], [-1.0, 0.0]))

    if edges:
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, EDGE_FEATURES), dtype=torch.float32)
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def morgan_fingerprint(smiles: str) -> torch.Tensor:
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError(f"RDKit could not parse SMILES for Morgan fingerprint: {smiles!r}")
    fingerprint = AllChem.GetMorganFingerprintAsBitVect(
        molecule, MORGAN_RADIUS, nBits=MORGAN_BITS
    )
    values = np.zeros((MORGAN_BITS,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fingerprint, values)
    return torch.from_numpy(values)


def process_smiles(smiles: str) -> Data:
    graph = smiles_to_periodic_graph(smiles)
    graph.morgan_fp = morgan_fingerprint(smiles).unsqueeze(0)
    return graph


def augment_repeat_units(smiles: str, repeats: int = 3) -> str:
    """Reproduce the public three-unit repeat augmentation deterministically."""
    if "*" not in smiles or repeats <= 1:
        return smiles
    try:
        monomer = Chem.MolFromSmiles(smiles)
        if monomer is None:
            return smiles
        star_count = sum(atom.GetSymbol() == "*" for atom in monomer.GetAtoms())
        if star_count != 2:
            core = smiles.replace("[*]", "")
            return f"[*]{core * repeats}[*]"
        reaction = AllChem.ReactionFromSmarts("[*:1].[*:2]>>[*:1]-[*:2]")
        chain = monomer
        for _ in range(repeats - 1):
            products = reaction.RunReactants((chain, monomer))
            if not products:
                return smiles
            chain = products[0][0]
            Chem.SanitizeMol(chain)
        return Chem.MolToSmiles(chain)
    except Exception:
        return smiles


def fit_fold_fp_indices(
    fingerprints: np.ndarray,
    labels: Sequence[Mapping[str, object]],
    train_indices: Sequence[int],
) -> dict[str, list[int]]:
    """Select Morgan bits using only observed labels in the current train fold."""
    fp = np.asarray(fingerprints, dtype=np.float32)
    selected: dict[str, list[int]] = {}
    for target in TARGETS:
        valid_indices = [
            int(index)
            for index in train_indices
            if labels[int(index)].get(target) is not None
        ]
        if len(valid_indices) < 2:
            raise ValueError(f"Fold train split has fewer than two {target} labels.")
        y = np.asarray([float(labels[index][target]) for index in valid_indices])
        selector = SelectKBest(f_regression, k=min(SELECTED_MORGAN_BITS, fp.shape[1]))
        # f_regression emits RuntimeWarnings for constant bits / exact fits;
        # SelectKBest handles their non-finite scores as unselectable ties.
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
            selector.fit(fp[valid_indices], y)
        indices = selector.get_support(indices=True).astype(int).tolist()
        if not indices:
            raise ValueError(f"No Morgan features selected for {target}.")
        selected[target] = indices
    return selected


class PolymerGNNV12Res(nn.Module):
    """Architecture matching PolymerGNN_v12_Res in the cited public source."""

    def __init__(
        self,
        tasks_fp_indices: Mapping[str, Sequence[int]],
        *,
        hidden_dim: int = 384,
        num_layers: int = 6,
        heads: int = 8,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if hidden_dim % heads:
            raise ValueError("hidden_dim must be divisible by the number of attention heads")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        head_dim = hidden_dim // heads
        self.targets = tuple(TARGETS)
        self.tasks_fp_indices = {
            task: tuple(int(i) for i in tasks_fp_indices[task]) for task in TARGETS
        }
        self.dropout = float(dropout)

        self.input_conv = GATv2Conv(
            NODE_FEATURES,
            head_dim,
            heads=heads,
            dropout=dropout,
            edge_dim=EDGE_FEATURES,
        )
        self.input_bn = BatchNorm(hidden_dim)
        self.hidden_convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    dropout=dropout,
                    edge_dim=EDGE_FEATURES,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.hidden_bns = nn.ModuleList(
            [BatchNorm(hidden_dim) for _ in range(num_layers - 1)]
        )
        self.task_predictors = nn.ModuleDict()
        for target in TARGETS:
            fp_count = len(self.tasks_fp_indices[target])
            self.task_predictors[target] = nn.Sequential(
                nn.Linear(hidden_dim + fp_count, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(hidden_dim, 1),
            )

    def forward(self, data: Data) -> torch.Tensor:
        x = F.elu(
            self.input_bn(
                self.input_conv(data.x, data.edge_index, edge_attr=data.edge_attr)
            )
        )
        for conv, batch_norm in zip(self.hidden_convs, self.hidden_bns):
            x = F.elu(batch_norm(conv(x, data.edge_index, edge_attr=data.edge_attr))) + x
        graph_embedding = global_mean_pool(x, data.batch)
        outputs = []
        for target in TARGETS:
            indices = torch.as_tensor(
                self.tasks_fp_indices[target], dtype=torch.long, device=data.morgan_fp.device
            )
            selected_fp = data.morgan_fp.index_select(1, indices)
            outputs.append(
                self.task_predictors[target](torch.cat((graph_embedding, selected_fp), dim=1))
            )
        return torch.cat(outputs, dim=1)
