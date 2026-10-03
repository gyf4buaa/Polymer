"""Reconstructed GATv2 data preparation.

Adapted from the MIT-licensed public third-place solution:
fresnellll/kaggle-NeurIPS-polymer-prediction-solution
Reference commit: f385d220d348283792c9f3dc8ed4ab0619e6f7c4

This is a compatibility reconstruction, not the lost historical local file.
"""

from __future__ import annotations

import argparse
from functools import reduce
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from sklearn.feature_selection import SelectKBest, f_regression
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
SMILES_COL = "SMILES"
MORGAN_FP_DIM = 1024
MORGAN_FP_RADIUS = 2
NUM_BEST_FEATURES = 50


def atom_features(atom: Chem.Atom) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        float(int(atom.GetHybridization())),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ]


def bond_features(bond: Chem.Bond) -> list[float]:
    return [float(bond.GetBondTypeAsDouble()), float(bond.GetIsConjugated())]


def smiles_to_periodic_graph(smiles: str) -> Data | None:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    x = torch.tensor([atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices: list[list[int]] = []
    edge_attrs: list[list[float]] = []
    stars: list[int] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feat = bond_features(bond)
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([feat, feat])

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            stars.append(atom.GetIdx())

    if len(stars) == 2:
        i, j = stars
        edge_indices.extend([[i, j], [j, i]])
        edge_attrs.extend([[-1.0, 0.0], [-1.0, 0.0]])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, 2), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def augment_repeat_units(smiles: str, repeats: int = 3) -> str:
    if "*" not in smiles or repeats <= 1:
        return smiles
    try:
        monomer = Chem.MolFromSmiles(smiles)
        if monomer is None:
            return smiles
        stars = [a.GetIdx() for a in monomer.GetAtoms() if a.GetSymbol() == "*"]
        if len(stars) != 2:
            core = smiles.replace("[*]", "")
            return f"[*]{core * repeats}[*]"

        rxn = AllChem.ReactionFromSmarts("[*:1].[*:2]>>[*:1]-[*:2]")
        chain = monomer
        for _ in range(repeats - 1):
            products = rxn.RunReactants((chain, monomer))
            if not products:
                return smiles
            chain = products[0][0]
            Chem.SanitizeMol(chain)
        return Chem.MolToSmiles(chain)
    except Exception:
        return smiles


def load_targets(raw_dir: Path) -> pd.DataFrame:
    frames = []
    for target in TARGETS:
        path = raw_dir / f"{target}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path)
        if df.shape[1] < 2:
            raise ValueError(f"{path} must contain at least two columns")
        df = df.iloc[:, :2].copy()
        df.columns = [SMILES_COL, target]
        frames.append(df)

    if not frames:
        raise FileNotFoundError(
            f"No target CSVs found in {raw_dir}. Expected files named "
            + ", ".join(f"{t}.csv" for t in TARGETS)
        )

    merged = reduce(lambda left, right: pd.merge(left, right, on=SMILES_COL, how="outer"), frames)
    merged = merged.dropna(subset=[SMILES_COL]).groupby(SMILES_COL, as_index=False).first()
    return merged


def select_fp_features(df: pd.DataFrame) -> dict[str, list[int]]:
    fps = []
    for smi in tqdm(df[SMILES_COL], desc="Morgan fingerprints"):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            fps.append(np.zeros(MORGAN_FP_DIM, dtype=np.float32))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(
            mol, MORGAN_FP_RADIUS, nBits=MORGAN_FP_DIM
        )
        arr = np.zeros(MORGAN_FP_DIM, dtype=np.float32)
        from rdkit import DataStructs

        DataStructs.ConvertToNumpyArray(fp, arr)
        fps.append(arr)

    fp_matrix = np.asarray(fps, dtype=np.float32)
    selected: dict[str, list[int]] = {}
    for target in TARGETS:
        mask = df[target].notna().to_numpy()
        if mask.sum() < 2:
            continue
        selector = SelectKBest(
            f_regression, k=min(NUM_BEST_FEATURES, MORGAN_FP_DIM)
        )
        selector.fit(fp_matrix[mask], df.loc[mask, target].to_numpy())
        selected[target] = selector.get_support(indices=True).tolist()
    return selected


def build_graph(smiles: str, y: torch.Tensor) -> Data | None:
    graph = smiles_to_periodic_graph(smiles)
    if graph is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    from rdkit import DataStructs

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol, MORGAN_FP_RADIUS, nBits=MORGAN_FP_DIM
    )
    arr = np.zeros(MORGAN_FP_DIM, dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    graph.morgan_fp = torch.tensor(arr, dtype=torch.float).unsqueeze(0)
    graph.y = y
    return graph


class PackedDataset(InMemoryDataset):
    def __init__(self, data_list: list[Data]):
        super().__init__()
        self.data, self.slices = self.collate(data_list)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, default=Path("training_artifacts"))
    parser.add_argument("--repeat-units", type=int, default=3)
    args = parser.parse_args()

    assets_dir = args.artifact_root / "assets"
    processed_dir = args.artifact_root / "processed"
    assets_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    df = load_targets(args.raw_dir)
    print(f"Loaded {len(df)} unique SMILES")

    selected = select_fp_features(df)
    joblib.dump(selected, assets_dir / "best_fp_indices.pkl")
    print("Saved task-specific fingerprint indices")

    data_list: list[Data] = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Build PyG graphs"):
        y = torch.tensor(
            [row.get(target, np.nan) for target in TARGETS], dtype=torch.float
        ).unsqueeze(0)
        original = str(row[SMILES_COL])

        graph = build_graph(original, y)
        if graph is not None:
            data_list.append(graph)

        augmented = augment_repeat_units(original, repeats=args.repeat_units)
        if augmented != original:
            graph = build_graph(augmented, y)
            if graph is not None:
                data_list.append(graph)

    if not data_list:
        raise RuntimeError("No valid graph samples were generated")

    dataset = PackedDataset(data_list)
    out = processed_dir / "all_augmented_master.pt"
    torch.save((dataset.data, dataset.slices), out)
    print(f"Saved {len(data_list)} graphs to {out}")


if __name__ == "__main__":
    main()
