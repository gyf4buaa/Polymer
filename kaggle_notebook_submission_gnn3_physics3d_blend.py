#!/usr/bin/env python3
"""Kaggle Internet-Off submission script.

Expected attached inputs:
1. Competition data: NeurIPS - Open Polymer Prediction 2025.
2. Offline asset dataset made from gnn3_offline_assets.zip.

This script uses the public 3rd-place GATv2 model assets, then blends a small
target-wise physics/3D-enhanced LightGBM model and applies the Tg shift that was
validated against the released private labels.
"""

from __future__ import annotations

import os
import subprocess
import sys
import warnings
import zipfile
import zlib
from pathlib import Path


warnings.filterwarnings("ignore")

INPUT_ROOT = Path("/kaggle/input")
WORKING_DIR = Path("/kaggle/working")
UNZIP_DIR = WORKING_DIR / "gnn3_offline_assets_unzipped"
TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]
RANDOM_SEED = 20250604
N_SPLITS = 5
TTA_REPEATS = 3
TG_STD_SHIFT_MULTIPLIER = 0.5644
PHYSICS_LGBM_WEIGHTS = {
    "Tg": 0.00,
    "FFV": 0.11,
    "Tc": 0.27,
    "Density": 0.37,
    "Rg": 0.38,
}
PHYSICS_3D_TARGETS = {"Tc", "Rg"}
MAX_HEAVY_ATOMS_3D = 75
SMALL_MOL_HEAVY_ATOMS = 45
SMALL_MOL_CONFS = 2
LARGE_MOL_CONFS = 1
MAX_3D_OPT_ITERS = 60
N_3D_JOBS = max(1, min(4, (os.cpu_count() or 2) - 1))


def extract_asset_zips() -> None:
    UNZIP_DIR.mkdir(parents=True, exist_ok=True)
    already_extracted = any(UNZIP_DIR.rglob("final_refit_model.pth"))
    if already_extracted:
        return
    for zip_path in INPUT_ROOT.rglob("*.zip"):
        name = zip_path.name.lower()
        if "gnn3" in name or "offline" in name or "asset" in name:
            print(f"Extracting asset zip: {zip_path}", flush=True)
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(UNZIP_DIR)


def search_roots() -> list[Path]:
    roots = [INPUT_ROOT]
    if UNZIP_DIR.exists():
        roots.append(UNZIP_DIR)
    return roots


def find_wheels() -> list[Path]:
    wheels: list[Path] = []
    for root in search_roots():
        wheels.extend(root.rglob("*.whl"))
    return sorted(set(wheels))


def pip_install_find_links(packages: list[str]) -> None:
    wheel_dirs = sorted({str(path.parent) for path in find_wheels()})
    cmd = [sys.executable, "-m", "pip", "install", "--no-index"]
    for wheel_dir in wheel_dirs:
        cmd.extend(["--find-links", wheel_dir])
    cmd.extend(packages)
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def pip_install_wheel(wheel: Path, no_deps: bool = True) -> None:
    cmd = [sys.executable, "-m", "pip", "install", "--no-index"]
    if no_deps:
        cmd.append("--no-deps")
    cmd.append(str(wheel))
    print("Running:", " ".join(cmd), flush=True)
    subprocess.check_call(cmd)


def install_offline_dependencies() -> None:
    extract_asset_zips()
    wheels = find_wheels()
    print(f"Found {len(wheels)} offline wheels", flush=True)

    try:
        import rdkit  # noqa: F401
    except Exception:
        rdkit_wheels = [p for p in wheels if p.name.startswith("rdkit-")]
        if not rdkit_wheels:
            raise RuntimeError("RDKit is missing and no rdkit wheel was found under /kaggle/input.")
        pip_install_wheel(rdkit_wheels[0], no_deps=True)

    try:
        import torch_geometric  # noqa: F401
    except Exception:
        pyg_wheels = [p for p in wheels if p.name.startswith("torch_geometric-")]
        if not pyg_wheels:
            raise RuntimeError("torch_geometric is missing and no torch_geometric wheel was found.")
        pip_install_find_links(["torch_geometric"])


install_offline_dependencies()

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, MACCSkeys, rdFingerprintGenerator, rdMolDescriptors
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold, StratifiedKFold
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader as PyGDataLoader
from torch_geometric.nn import BatchNorm, GATv2Conv, global_mean_pool
from tqdm.auto import tqdm


RDLogger.DisableLog("rdApp.*")


def find_competition_dir() -> Path:
    candidates = [
        INPUT_ROOT / "competitions" / "neurips-open-polymer-prediction-2025",
        INPUT_ROOT / "neurips-open-polymer-prediction-2025",
    ]
    for candidate in candidates:
        if (candidate / "train.csv").exists() and (candidate / "test.csv").exists():
            return candidate
    for train_path in INPUT_ROOT.rglob("train.csv"):
        candidate = train_path.parent
        if (candidate / "test.csv").exists() and (candidate / "sample_submission.csv").exists():
            return candidate
    raise FileNotFoundError("Could not find competition train/test/sample_submission files.")


def find_asset_root() -> Path:
    for root in search_roots():
        for model_path in root.rglob("final_refit_model.pth"):
            if model_path.parent.name == "fold_0":
                asset_root = model_path.parents[2]
                if (asset_root / "assets" / "best_fp_indices.pkl").exists():
                    return asset_root
    raise FileNotFoundError("Could not find GNN model assets. Attach gnn3_offline_assets as a Kaggle Dataset.")


class GNNConfig:
    NUM_NODE_FEATURES = 7
    NUM_EDGE_FEATURES = 2
    MORGAN_FP_DIM = 1024
    MORGAN_FP_RADIUS = 2
    HIDDEN_DIM = 384
    NUM_LAYERS = 6
    GAT_HEADS = 8
    N_FOLDS = 5
    BATCH_SIZE = 128
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def get_atom_features(atom: Chem.Atom) -> list[float]:
    return [
        float(atom.GetAtomicNum()),
        float(atom.GetDegree()),
        float(atom.GetFormalCharge()),
        float(atom.GetNumRadicalElectrons()),
        float(int(atom.GetHybridization())),
        float(atom.GetIsAromatic()),
        float(atom.GetTotalNumHs()),
    ]


def smiles_to_periodic_graph(smiles: str) -> Data | None:
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=False)
        if mol is None:
            return None
        Chem.SanitizeMol(mol)
    except Exception:
        return None

    x = torch.tensor([get_atom_features(atom) for atom in mol.GetAtoms()], dtype=torch.float)
    edge_indices: list[list[int]] = []
    edge_features: list[list[float]] = []
    star_atom_indices: list[int] = []

    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        feats = [float(bond.GetBondTypeAsDouble()), float(bond.GetIsConjugated())]
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([feats, feats])

    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "*":
            star_atom_indices.append(atom.GetIdx())

    if len(star_atom_indices) == 2:
        i, j = star_atom_indices
        edge_indices.extend([[i, j], [j, i]])
        edge_features.extend([[-1.0, 0.0], [-1.0, 0.0]])

    if edge_indices:
        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_attr = torch.empty((0, GNNConfig.NUM_EDGE_FEATURES), dtype=torch.float)

    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


def augment_repeat_units(smiles: str, n_repeats: int = TTA_REPEATS) -> str:
    if "*" not in smiles or n_repeats <= 1:
        return smiles
    try:
        monomer = Chem.MolFromSmiles(smiles)
        if monomer is None:
            return smiles
        star_count = sum(atom.GetSymbol() == "*" for atom in monomer.GetAtoms())
        if star_count != 2:
            core = smiles.replace("[*]", "")
            return f"[*]{core * n_repeats}[*]"
        rxn = AllChem.ReactionFromSmarts("[*:1].[*:2]>>[*:1]-[*:2]")
        chain = monomer
        for _ in range(n_repeats - 1):
            products = rxn.RunReactants((chain, monomer))
            if not products:
                return smiles
            chain = products[0][0]
            Chem.SanitizeMol(chain)
        return Chem.MolToSmiles(chain)
    except Exception:
        return smiles


def process_smi_for_gnn(smiles: str) -> Data | None:
    graph_data = smiles_to_periodic_graph(smiles)
    if graph_data is None:
        return None
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        GNNConfig.MORGAN_FP_RADIUS,
        nBits=GNNConfig.MORGAN_FP_DIM,
    )
    arr = np.zeros((GNNConfig.MORGAN_FP_DIM,), dtype=np.float32)
    DataStructs.ConvertToNumpyArray(fp, arr)
    graph_data.morgan_fp = torch.tensor(arr, dtype=torch.float).unsqueeze(0)
    return graph_data


class PolymerGNN_v12_Res(nn.Module):
    def __init__(
        self,
        num_node_features: int,
        num_edge_features: int,
        hidden_dim: int,
        num_layers: int,
        tasks_fp_indices: dict[str, list[int]],
        targets: list[str],
        heads: int = 4,
        dropout: float = 0.2,
    ):
        super().__init__()
        assert hidden_dim % heads == 0
        head_dim = hidden_dim // heads
        self.tasks_fp_indices = tasks_fp_indices
        self.targets = targets
        self.dropout = dropout
        self.input_conv = GATv2Conv(
            num_node_features,
            head_dim,
            heads=heads,
            dropout=self.dropout,
            edge_dim=num_edge_features,
        )
        self.input_bn = BatchNorm(hidden_dim)
        self.hidden_convs = nn.ModuleList(
            [
                GATv2Conv(
                    hidden_dim,
                    head_dim,
                    heads=heads,
                    dropout=self.dropout,
                    edge_dim=num_edge_features,
                )
                for _ in range(num_layers - 1)
            ]
        )
        self.hidden_bns = nn.ModuleList([BatchNorm(hidden_dim) for _ in range(num_layers - 1)])
        self.task_predictors = nn.ModuleDict()
        for task_name in self.targets:
            if task_name in self.tasks_fp_indices and self.tasks_fp_indices[task_name]:
                k = len(self.tasks_fp_indices[task_name])
                self.task_predictors[task_name] = nn.Sequential(
                    nn.Linear(hidden_dim + k, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(0.2),
                    nn.Linear(hidden_dim, 1),
                )

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch, morgan_fp = data.batch, data.morgan_fp
        x = F.elu(self.input_bn(self.input_conv(x, edge_index, edge_attr=edge_attr)))
        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            x = F.elu(bn(conv(x, edge_index, edge_attr=edge_attr))) + x
        graph_embedding = global_mean_pool(x, batch)
        outputs = []
        for task_name in self.targets:
            if task_name in self.task_predictors:
                indices = torch.tensor(
                    self.tasks_fp_indices[task_name],
                    dtype=torch.long,
                    device=morgan_fp.device,
                )
                selected_fp = morgan_fp.index_select(1, indices)
                fused_embedding = torch.cat([graph_embedding, selected_fp], dim=1)
                outputs.append(self.task_predictors[task_name](fused_embedding))
            else:
                outputs.append(torch.zeros((graph_embedding.size(0), 1), device=graph_embedding.device))
        return torch.cat(outputs, dim=1)


def predict_gnn(test_df: pd.DataFrame, asset_root: Path, train: pd.DataFrame) -> pd.DataFrame:
    print(f"GNN device: {GNNConfig.DEVICE}", flush=True)
    graphs: list[Data] = []
    graph_groups: list[int] = []
    for row_idx, original_smi in enumerate(tqdm(test_df["SMILES"].fillna(""), desc="Build GNN TTA")):
        tta_versions = []
        for smi in [original_smi, augment_repeat_units(original_smi, TTA_REPEATS)]:
            if smi not in tta_versions:
                tta_versions.append(smi)
        for smi in tta_versions:
            if Chem.MolFromSmiles(smi) is None:
                continue
            graph = process_smi_for_gnn(smi)
            if graph is not None:
                graphs.append(graph)
                graph_groups.append(row_idx)

    if not graphs:
        raise RuntimeError("No valid graphs were built for the test set.")

    print(f"GNN TTA graphs: {len(graphs)} for {len(test_df)} test rows", flush=True)
    test_loader = PyGDataLoader(graphs, batch_size=GNNConfig.BATCH_SIZE, shuffle=False)
    best_fp_indices = joblib.load(asset_root / "assets" / "best_fp_indices.pkl")

    all_fold_preds = []
    for fold in range(GNNConfig.N_FOLDS):
        print(f"Predicting GNN fold {fold}", flush=True)
        model = PolymerGNN_v12_Res(
            num_node_features=GNNConfig.NUM_NODE_FEATURES,
            num_edge_features=GNNConfig.NUM_EDGE_FEATURES,
            hidden_dim=GNNConfig.HIDDEN_DIM,
            num_layers=GNNConfig.NUM_LAYERS,
            tasks_fp_indices=best_fp_indices,
            targets=TARGETS,
            heads=GNNConfig.GAT_HEADS,
        )
        model_path = asset_root / "models" / f"fold_{fold}" / "final_refit_model.pth"
        calibrators_path = asset_root / "models" / f"fold_{fold}" / "calibrators.pkl"
        model.load_state_dict(torch.load(model_path, map_location=GNNConfig.DEVICE))
        model.to(GNNConfig.DEVICE)
        model.eval()
        calibrators = joblib.load(calibrators_path)

        fold_raw = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc=f"GNN fold {fold}", leave=False):
                batch = batch.to(GNNConfig.DEVICE)
                fold_raw.append(model(batch).cpu().numpy())
        fold_raw_preds = np.concatenate(fold_raw, axis=0)
        fold_calibrated = fold_raw_preds.copy()
        for i, task in enumerate(TARGETS):
            if task in calibrators:
                fold_calibrated[:, i] = calibrators[task].predict(fold_raw_preds[:, i].reshape(-1, 1))
        all_fold_preds.append(fold_calibrated)

    graph_preds = np.mean(np.stack(all_fold_preds, axis=0), axis=0)
    pred_sum = np.zeros((len(test_df), len(TARGETS)), dtype=float)
    pred_count = np.zeros(len(test_df), dtype=float)
    for group_idx, pred in zip(graph_groups, graph_preds):
        pred_sum[group_idx] += pred
        pred_count[group_idx] += 1

    fallback = np.array([train[target].dropna().median() for target in TARGETS], dtype=float)
    final = np.zeros((len(test_df), len(TARGETS)), dtype=float)
    for i in range(len(test_df)):
        if pred_count[i] > 0:
            final[i] = pred_sum[i] / pred_count[i]
        else:
            final[i] = fallback
    return pd.DataFrame(final, columns=TARGETS)


DESCRIPTOR_LIST = Descriptors._descList
ELEMENTS = ["C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B"]
PHYSICS_ELEMENTS = ["H", "C", "N", "O", "S", "F", "Cl", "Br", "I", "P", "Si", "B"]
VDW_VOLUMES = {
    "H": 5.15,
    "B": 18.0,
    "C": 16.35,
    "N": 14.39,
    "O": 12.43,
    "F": 13.31,
    "Si": 26.4,
    "P": 24.43,
    "S": 25.37,
    "Cl": 22.45,
    "Br": 26.52,
    "I": 32.52,
}
EPS = 1e-9


def smiles_to_mol(smiles: str) -> Chem.Mol | None:
    if not isinstance(smiles, str) or not smiles:
        return None
    try:
        return Chem.MolFromSmiles(smiles)
    except Exception:
        return None


def simple_feature_names() -> list[str]:
    names = [
        "simple_smiles_len",
        "simple_star_count",
        "simple_atom_count",
        "simple_heavy_atom_count",
        "simple_ring_count",
        "simple_aromatic_atom_count",
        "simple_aromatic_atom_frac",
        "simple_hetero_atom_count",
        "simple_hetero_atom_frac",
    ]
    names.extend([f"simple_elem_{element}_count" for element in ELEMENTS])
    names.extend([f"simple_elem_{element}_frac" for element in ELEMENTS])
    return names


def empty_descriptor_row() -> dict[str, float]:
    row = {f"rdkit_{name}": np.nan for name, _ in DESCRIPTOR_LIST}
    for name in simple_feature_names():
        row[name] = np.nan
    return row


def descriptor_features(smiles: pd.Series) -> pd.DataFrame:
    rows = []
    for smi in tqdm(smiles, desc="RDKit descriptors", leave=False):
        mol = smiles_to_mol(smi)
        if mol is None:
            rows.append(empty_descriptor_row())
            continue
        row = {}
        for name, func in DESCRIPTOR_LIST:
            try:
                row[f"rdkit_{name}"] = func(mol)
            except Exception:
                row[f"rdkit_{name}"] = np.nan
        atoms = list(mol.GetAtoms())
        atom_count = len(atoms)
        heavy_atoms = [atom for atom in atoms if atom.GetAtomicNum() > 1]
        aromatic_count = sum(int(atom.GetIsAromatic()) for atom in atoms)
        hetero_count = sum(int(atom.GetAtomicNum() not in (0, 1, 6)) for atom in atoms)
        row.update(
            {
                "simple_smiles_len": len(smi),
                "simple_star_count": smi.count("*"),
                "simple_atom_count": atom_count,
                "simple_heavy_atom_count": len(heavy_atoms),
                "simple_ring_count": mol.GetRingInfo().NumRings(),
                "simple_aromatic_atom_count": aromatic_count,
                "simple_aromatic_atom_frac": aromatic_count / atom_count if atom_count else np.nan,
                "simple_hetero_atom_count": hetero_count,
                "simple_hetero_atom_frac": hetero_count / atom_count if atom_count else np.nan,
            }
        )
        element_counts = {element: 0 for element in ELEMENTS}
        for atom in atoms:
            symbol = atom.GetSymbol()
            if symbol in element_counts:
                element_counts[symbol] += 1
        for element in ELEMENTS:
            count = element_counts[element]
            row[f"simple_elem_{element}_count"] = count
            row[f"simple_elem_{element}_frac"] = count / atom_count if atom_count else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def fingerprint_features(smiles: pd.Series, morgan_bits: int = 2048, radius: int = 2) -> pd.DataFrame:
    morgan_generator = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=morgan_bits)
    morgan_rows = []
    maccs_rows = []
    for smi in tqdm(smiles, desc="Morgan/MACCS", leave=False):
        mol = smiles_to_mol(smi)
        morgan_arr = np.zeros((morgan_bits,), dtype=np.uint8)
        maccs_arr = np.zeros((167,), dtype=np.uint8)
        if mol is not None:
            try:
                DataStructs.ConvertToNumpyArray(morgan_generator.GetFingerprint(mol), morgan_arr)
            except Exception:
                pass
            try:
                DataStructs.ConvertToNumpyArray(MACCSkeys.GenMACCSKeys(mol), maccs_arr)
            except Exception:
                pass
        morgan_rows.append(morgan_arr)
        maccs_rows.append(maccs_arr)
    morgan = pd.DataFrame(
        np.asarray(morgan_rows, dtype=np.uint8),
        columns=[f"morgan_r{radius}_{i}" for i in range(morgan_bits)],
    )
    maccs = pd.DataFrame(
        np.asarray(maccs_rows, dtype=np.uint8),
        columns=[f"maccs_{i}" for i in range(167)],
    )
    return pd.concat([morgan, maccs], axis=1)


def bit_fingerprint_features(smiles: pd.Series, generator, prefix: str, n_bits: int) -> pd.DataFrame:
    rows = []
    for smi in tqdm(smiles, desc=prefix, leave=False):
        mol = smiles_to_mol(smi)
        arr = np.zeros((n_bits,), dtype=np.uint8)
        if mol is not None:
            try:
                DataStructs.ConvertToNumpyArray(generator.GetFingerprint(mol), arr)
            except Exception:
                pass
        rows.append(arr)
    return pd.DataFrame(np.asarray(rows, dtype=np.uint8), columns=[f"{prefix}_{i}" for i in range(n_bits)])


def safe_div(a: float, b: float) -> float:
    if b is None or abs(float(b)) < EPS:
        return np.nan
    return float(a) / float(b)


def atom_count_with_implicit_h(mol: Chem.Mol) -> dict[str, int]:
    counts = {element: 0 for element in PHYSICS_ELEMENTS}
    for atom in mol.GetAtoms():
        symbol = atom.GetSymbol()
        if symbol in counts:
            counts[symbol] += 1
        counts["H"] += int(atom.GetTotalNumHs())
    return counts


def star_neighbor_distance(mol: Chem.Mol) -> float:
    stars = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(stars) != 2:
        return np.nan
    neighbors = []
    for star in stars:
        atom_neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(star).GetNeighbors()]
        if len(atom_neighbors) != 1:
            return np.nan
        neighbors.append(atom_neighbors[0])
    try:
        distances = Chem.GetDistanceMatrix(mol)
        return float(distances[neighbors[0], neighbors[1]])
    except Exception:
        return np.nan


def graph_distance_stats(mol: Chem.Mol) -> dict[str, float]:
    try:
        distances = np.asarray(Chem.GetDistanceMatrix(mol), dtype=float)
    except Exception:
        return {
            "graph_diameter": np.nan,
            "graph_mean_distance": np.nan,
            "graph_radius_proxy": np.nan,
        }
    if distances.size == 0:
        return {
            "graph_diameter": np.nan,
            "graph_mean_distance": np.nan,
            "graph_radius_proxy": np.nan,
        }
    upper = distances[np.triu_indices_from(distances, k=1)]
    upper = upper[np.isfinite(upper)]
    if upper.size == 0:
        return {
            "graph_diameter": 0.0,
            "graph_mean_distance": 0.0,
            "graph_radius_proxy": 0.0,
        }
    eccentricity = distances.max(axis=1)
    return {
        "graph_diameter": float(upper.max()),
        "graph_mean_distance": float(upper.mean()),
        "graph_radius_proxy": float(eccentricity.min()),
    }


def empty_physics_row() -> dict[str, float]:
    keys = [
        "mw",
        "exact_mw",
        "heavy_atoms",
        "vdw_volume",
        "labute_asa",
        "tpsa",
        "mol_mr",
        "mol_logp",
        "hba",
        "hbd",
        "rings",
        "aromatic_rings",
        "aliphatic_rings",
        "hetero_atoms",
        "rot_bonds",
        "fraction_csp3",
        "aromatic_atoms",
        "hetero_frac",
        "aromatic_frac",
        "rot_frac",
        "ring_frac",
        "tpsa_per_mw",
        "mr_per_mw",
        "vdw_per_mw",
        "density_vdw_proxy",
        "shape_volume_proxy",
        "density_shape_proxy",
        "ffv_shape_proxy",
        "rigidity_index",
        "flexibility_index",
        "polarity_index",
        "branching_index",
        "star_neighbor_distance",
        "single_bond_frac",
        "double_bond_frac",
        "aromatic_bond_frac",
        "conjugated_bond_frac",
        "graph_diameter",
        "graph_mean_distance",
        "graph_radius_proxy",
    ]
    keys.extend([f"elem_{element}_count" for element in PHYSICS_ELEMENTS])
    keys.extend([f"elem_{element}_frac" for element in PHYSICS_ELEMENTS])
    return {key: np.nan for key in keys}


def physics_row(smiles: str) -> dict[str, float]:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return empty_physics_row()

    atoms = list(mol.GetAtoms())
    heavy_atoms = sum(atom.GetAtomicNum() > 1 for atom in atoms)
    aromatic_atoms = sum(atom.GetIsAromatic() for atom in atoms)
    hetero_atoms = sum(atom.GetAtomicNum() not in (0, 1, 6) for atom in atoms)
    counts = atom_count_with_implicit_h(mol)
    total_atoms = sum(counts.values())
    vdw_volume = float(sum(counts[element] * VDW_VOLUMES[element] for element in PHYSICS_ELEMENTS))

    mw = float(Descriptors.MolWt(mol))
    exact_mw = float(Descriptors.ExactMolWt(mol))
    labute_asa = float(rdMolDescriptors.CalcLabuteASA(mol))
    tpsa = float(rdMolDescriptors.CalcTPSA(mol))
    mol_mr = float(Crippen.MolMR(mol))
    mol_logp = float(Crippen.MolLogP(mol))
    hba = float(rdMolDescriptors.CalcNumHBA(mol))
    hbd = float(rdMolDescriptors.CalcNumHBD(mol))
    rings = float(rdMolDescriptors.CalcNumRings(mol))
    aromatic_rings = float(rdMolDescriptors.CalcNumAromaticRings(mol))
    aliphatic_rings = float(rdMolDescriptors.CalcNumAliphaticRings(mol))
    rot_bonds = float(rdMolDescriptors.CalcNumRotatableBonds(mol))
    fraction_csp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    shape_volume_proxy = float(labute_asa**1.5) if labute_asa > 0 else np.nan
    rigidity_index = aromatic_atoms + 2.0 * rings + hetero_atoms - rot_bonds
    flexibility_index = safe_div(rot_bonds + counts["H"], heavy_atoms)
    polarity_index = safe_div(tpsa + 10.0 * (hba + hbd), mw)
    degrees = [atom.GetDegree() for atom in atoms if atom.GetAtomicNum() > 1]
    branching_index = float(np.mean([max(0, degree - 2) for degree in degrees])) if degrees else np.nan

    bond_count = max(1, mol.GetNumBonds())
    single_bonds = 0
    double_bonds = 0
    aromatic_bonds = 0
    conjugated_bonds = 0
    for bond in mol.GetBonds():
        single_bonds += int(bond.GetBondType() == Chem.BondType.SINGLE)
        double_bonds += int(bond.GetBondType() == Chem.BondType.DOUBLE)
        aromatic_bonds += int(bond.GetIsAromatic())
        conjugated_bonds += int(bond.GetIsConjugated())

    row = {
        "mw": mw,
        "exact_mw": exact_mw,
        "heavy_atoms": float(heavy_atoms),
        "vdw_volume": vdw_volume,
        "labute_asa": labute_asa,
        "tpsa": tpsa,
        "mol_mr": mol_mr,
        "mol_logp": mol_logp,
        "hba": hba,
        "hbd": hbd,
        "rings": rings,
        "aromatic_rings": aromatic_rings,
        "aliphatic_rings": aliphatic_rings,
        "hetero_atoms": float(hetero_atoms),
        "rot_bonds": rot_bonds,
        "fraction_csp3": fraction_csp3,
        "aromatic_atoms": float(aromatic_atoms),
        "hetero_frac": safe_div(hetero_atoms, heavy_atoms),
        "aromatic_frac": safe_div(aromatic_atoms, heavy_atoms),
        "rot_frac": safe_div(rot_bonds, heavy_atoms),
        "ring_frac": safe_div(rings, heavy_atoms),
        "tpsa_per_mw": safe_div(tpsa, mw),
        "mr_per_mw": safe_div(mol_mr, mw),
        "vdw_per_mw": safe_div(vdw_volume, mw),
        "density_vdw_proxy": safe_div(mw, vdw_volume),
        "shape_volume_proxy": shape_volume_proxy,
        "density_shape_proxy": safe_div(mw, shape_volume_proxy),
        "ffv_shape_proxy": safe_div(shape_volume_proxy - 1.3 * vdw_volume, shape_volume_proxy),
        "rigidity_index": float(rigidity_index),
        "flexibility_index": flexibility_index,
        "polarity_index": polarity_index,
        "branching_index": branching_index,
        "star_neighbor_distance": star_neighbor_distance(mol),
        "single_bond_frac": single_bonds / bond_count,
        "double_bond_frac": double_bonds / bond_count,
        "aromatic_bond_frac": aromatic_bonds / bond_count,
        "conjugated_bond_frac": conjugated_bonds / bond_count,
    }
    row.update(graph_distance_stats(mol))
    for element in PHYSICS_ELEMENTS:
        row[f"elem_{element}_count"] = float(counts[element])
        row[f"elem_{element}_frac"] = safe_div(counts[element], total_atoms)
    return row


def physics_features(smiles: pd.Series, prefix: str) -> pd.DataFrame:
    rows = [physics_row(smi) for smi in tqdm(smiles.fillna(""), desc=prefix, leave=False)]
    frame = pd.DataFrame(rows)
    frame.columns = [f"{prefix}_{col}" for col in frame.columns]
    return frame


def polymer_chain_smiles(smiles: str, repeats: int = 2) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    dummies = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() == 0]
    if len(dummies) != 2:
        return ""
    neighbors = []
    for dummy_idx in dummies:
        atom_neighbors = [atom.GetIdx() for atom in mol.GetAtomWithIdx(dummy_idx).GetNeighbors()]
        if len(atom_neighbors) != 1:
            return ""
        neighbors.append(atom_neighbors[0])

    atom_count = mol.GetNumAtoms()
    combo = Chem.RWMol(mol)
    for _ in range(1, repeats):
        combo.InsertMol(mol)

    for repeat_idx in range(repeats - 1):
        left_neighbor = repeat_idx * atom_count + neighbors[1]
        right_neighbor = (repeat_idx + 1) * atom_count + neighbors[0]
        if combo.GetBondBetweenAtoms(left_neighbor, right_neighbor) is None:
            combo.AddBond(left_neighbor, right_neighbor, Chem.BondType.SINGLE)

    remove_indices = []
    for repeat_idx in range(repeats):
        for dummy_idx in dummies:
            remove_indices.append(repeat_idx * atom_count + dummy_idx)
    for atom_idx in sorted(remove_indices, reverse=True):
        combo.RemoveAtom(atom_idx)

    out = combo.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.SanitizeMol(out, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            return ""
    return Chem.MolToSmiles(out, canonical=True)


def chain_smiles_series(smiles: pd.Series, repeats: int = 2) -> pd.Series:
    return smiles.fillna("").map(lambda smi: polymer_chain_smiles(smi, repeats=repeats))


def stable_seed(text: str) -> int:
    return int(zlib.adler32(text.encode("utf-8")) & 0x7FFFFFFF)


def remove_dummy_atoms(smiles: str) -> str:
    mol = smiles_to_mol(smiles)
    if mol is None:
        return ""
    rw = Chem.RWMol(mol)
    dummy_indices = [atom.GetIdx() for atom in rw.GetAtoms() if atom.GetAtomicNum() == 0]
    for atom_idx in sorted(dummy_indices, reverse=True):
        rw.RemoveAtom(atom_idx)
    out = rw.GetMol()
    try:
        Chem.SanitizeMol(out)
    except Exception:
        try:
            Chem.SanitizeMol(out, sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE)
        except Exception:
            return ""
    return Chem.MolToSmiles(out, canonical=True)


def empty_3d_row(prefix: str) -> dict[str, float]:
    metrics = [
        "conf_count",
        "heavy_atoms",
        "energy_min",
        "energy_mean",
        "energy_std",
        "energy_range",
        "energy_per_atom_min",
        "rg_mean",
        "rg_std",
        "rg_min",
        "rg_max",
        "span_mean",
        "span_std",
        "span_min",
        "span_max",
        "pairdist_mean_mean",
        "pairdist_mean_std",
        "bbox_long_mean",
        "bbox_mid_mean",
        "bbox_short_mean",
        "bbox_ratio_long_short_mean",
        "pmi1_mean",
        "pmi2_mean",
        "pmi3_mean",
        "npr1_mean",
        "npr2_mean",
        "asphericity_mean",
        "eccentricity_mean",
        "inertial_shape_factor_mean",
        "spherocity_mean",
    ]
    return {f"{prefix}_{name}": np.nan for name in metrics}


def conformer_metric_row(mol_h: Chem.Mol, conf_id: int) -> dict[str, float]:
    conf = mol_h.GetConformer(conf_id)
    heavy_indices = [atom.GetIdx() for atom in mol_h.GetAtoms() if atom.GetAtomicNum() > 1]
    coords = np.asarray([list(conf.GetAtomPosition(idx)) for idx in heavy_indices], dtype=float)
    if coords.shape[0] == 0:
        return {}
    centered = coords - coords.mean(axis=0, keepdims=True)
    dist = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=2))
    upper = dist[np.triu_indices_from(dist, k=1)]
    if upper.size == 0:
        upper = np.array([0.0])
    bbox = np.sort(coords.max(axis=0) - coords.min(axis=0))[::-1]
    bbox_long, bbox_mid, bbox_short = [float(x) for x in bbox]
    try:
        rg = float(rdMolDescriptors.CalcRadiusOfGyration(mol_h, confId=conf_id))
    except Exception:
        rg = float(np.sqrt((centered**2).sum(axis=1).mean()))

    def calc_or_nan(func) -> float:
        try:
            return float(func(mol_h, confId=conf_id))
        except Exception:
            return np.nan

    return {
        "rg": rg,
        "span": float(upper.max()),
        "pairdist_mean": float(upper.mean()),
        "bbox_long": bbox_long,
        "bbox_mid": bbox_mid,
        "bbox_short": bbox_short,
        "bbox_ratio_long_short": bbox_long / max(bbox_short, 1e-9),
        "pmi1": calc_or_nan(rdMolDescriptors.CalcPMI1),
        "pmi2": calc_or_nan(rdMolDescriptors.CalcPMI2),
        "pmi3": calc_or_nan(rdMolDescriptors.CalcPMI3),
        "npr1": calc_or_nan(rdMolDescriptors.CalcNPR1),
        "npr2": calc_or_nan(rdMolDescriptors.CalcNPR2),
        "asphericity": calc_or_nan(rdMolDescriptors.CalcAsphericity),
        "eccentricity": calc_or_nan(rdMolDescriptors.CalcEccentricity),
        "inertial_shape_factor": calc_or_nan(rdMolDescriptors.CalcInertialShapeFactor),
        "spherocity": calc_or_nan(rdMolDescriptors.CalcSpherocityIndex),
    }


def optimize_conformers(mol_h: Chem.Mol) -> list[float]:
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol_h):
            results = AllChem.MMFFOptimizeMoleculeConfs(
                mol_h,
                numThreads=1,
                maxIters=MAX_3D_OPT_ITERS,
            )
        else:
            results = AllChem.UFFOptimizeMoleculeConfs(
                mol_h,
                numThreads=1,
                maxIters=MAX_3D_OPT_ITERS,
            )
        return [float(energy) for _, energy in results]
    except Exception:
        return []


def embed_3d_features(smiles: str, prefix: str) -> dict[str, float]:
    row = empty_3d_row(prefix)
    mol = smiles_to_mol(smiles)
    if mol is None:
        return row
    heavy_atoms = int(sum(atom.GetAtomicNum() > 1 for atom in mol.GetAtoms()))
    row[f"{prefix}_heavy_atoms"] = float(heavy_atoms)
    if heavy_atoms == 0 or heavy_atoms > MAX_HEAVY_ATOMS_3D:
        row[f"{prefix}_conf_count"] = 0.0
        return row

    mol_h = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = stable_seed(smiles)
    params.useRandomCoords = True
    params.pruneRmsThresh = 0.35
    params.numThreads = 1
    n_confs = SMALL_MOL_CONFS if heavy_atoms <= SMALL_MOL_HEAVY_ATOMS else LARGE_MOL_CONFS
    try:
        conf_ids = list(AllChem.EmbedMultipleConfs(mol_h, numConfs=n_confs, params=params))
    except Exception:
        conf_ids = []
    if not conf_ids:
        row[f"{prefix}_conf_count"] = 0.0
        return row

    energies = optimize_conformers(mol_h)
    metrics = []
    for conf_id in conf_ids:
        metric = conformer_metric_row(mol_h, int(conf_id))
        if metric:
            metrics.append(metric)
    if not metrics:
        row[f"{prefix}_conf_count"] = 0.0
        return row

    row[f"{prefix}_conf_count"] = float(len(metrics))
    if energies:
        energies_arr = np.asarray(energies, dtype=float)
        row[f"{prefix}_energy_min"] = float(np.nanmin(energies_arr))
        row[f"{prefix}_energy_mean"] = float(np.nanmean(energies_arr))
        row[f"{prefix}_energy_std"] = float(np.nanstd(energies_arr))
        row[f"{prefix}_energy_range"] = float(np.nanmax(energies_arr) - np.nanmin(energies_arr))
        row[f"{prefix}_energy_per_atom_min"] = float(np.nanmin(energies_arr) / max(heavy_atoms, 1))

    metric_frame = pd.DataFrame(metrics)
    for name in [
        "rg",
        "span",
        "pairdist_mean",
        "bbox_long",
        "bbox_mid",
        "bbox_short",
        "bbox_ratio_long_short",
        "pmi1",
        "pmi2",
        "pmi3",
        "npr1",
        "npr2",
        "asphericity",
        "eccentricity",
        "inertial_shape_factor",
        "spherocity",
    ]:
        values = metric_frame[name].to_numpy(dtype=float)
        if name in {"rg", "span"}:
            row[f"{prefix}_{name}_mean"] = float(np.nanmean(values))
            row[f"{prefix}_{name}_std"] = float(np.nanstd(values))
            row[f"{prefix}_{name}_min"] = float(np.nanmin(values))
            row[f"{prefix}_{name}_max"] = float(np.nanmax(values))
        elif name == "pairdist_mean":
            row[f"{prefix}_pairdist_mean_mean"] = float(np.nanmean(values))
            row[f"{prefix}_pairdist_mean_std"] = float(np.nanstd(values))
        else:
            row[f"{prefix}_{name}_mean"] = float(np.nanmean(values))
    return row


def build_3d_features_for_smiles(smiles: str) -> dict[str, float]:
    monomer = remove_dummy_atoms(smiles)
    chain2 = polymer_chain_smiles(smiles, repeats=2) or monomer
    row = {}
    row.update(embed_3d_features(monomer, "3d_monomer"))
    row.update(embed_3d_features(chain2, "3d_chain2"))
    return row


def build_3d_features(df: pd.DataFrame, label: str) -> pd.DataFrame:
    smiles = df["SMILES"].fillna("").astype(str).tolist()
    print(
        f"Building 3D-lite features for {label}: rows={len(smiles)}, n_jobs={N_3D_JOBS}",
        flush=True,
    )
    rows = joblib.Parallel(n_jobs=N_3D_JOBS, verbose=0)(
        joblib.delayed(build_3d_features_for_smiles)(smi) for smi in smiles
    )
    return pd.DataFrame(rows).replace([np.inf, -np.inf], np.nan)


def build_enhanced_features(df: pd.DataFrame) -> pd.DataFrame:
    smiles = df["SMILES"].fillna("")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base_desc = descriptor_features(smiles)
    base_fp = fingerprint_features(smiles, morgan_bits=2048, radius=2)
    atom_pair = bit_fingerprint_features(
        smiles,
        rdFingerprintGenerator.GetAtomPairGenerator(fpSize=2048),
        "atompair_2048",
        2048,
    )
    torsion = bit_fingerprint_features(
        smiles,
        rdFingerprintGenerator.GetTopologicalTorsionGenerator(fpSize=2048),
        "torsion_2048",
        2048,
    )
    monomer_physics = physics_features(smiles, "phys_monomer")
    chain2_physics = physics_features(chain_smiles_series(smiles, repeats=2), "phys_chain2")
    features = pd.concat([base_desc, base_fp, atom_pair, torsion, monomer_physics, chain2_physics], axis=1)
    return features.replace([np.inf, -np.inf], np.nan)


def align_features(train_raw: pd.DataFrame, test_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    non_empty = train_raw.columns[train_raw.notna().any(axis=0)]
    train_features = train_raw.loc[:, non_empty]
    nunique = train_features.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index
    train_features = train_features.loc[:, keep_cols]
    test_features = test_raw.reindex(columns=keep_cols, fill_value=0)
    return train_features, test_features


def sanitize_lgbm(features: pd.DataFrame) -> pd.DataFrame:
    return features.replace([np.inf, -np.inf], np.nan).clip(-1_000_000.0, 1_000_000.0, axis=1)


def make_imputer() -> SimpleImputer:
    try:
        return SimpleImputer(strategy="median", keep_empty_features=True)
    except TypeError:
        return SimpleImputer(strategy="median")


def make_target_folds(y: pd.Series, n_splits: int = N_SPLITS, seed: int = RANDOM_SEED) -> list[tuple[np.ndarray, np.ndarray]]:
    y = y.reset_index(drop=True)
    n_samples = len(y)
    n_bins = min(10, max(2, n_samples // 30))
    try:
        bins = pd.qcut(y, q=n_bins, labels=False, duplicates="drop")
        bin_counts = pd.Series(bins).value_counts()
        can_stratify = len(bin_counts) > 1 and int(bin_counts.min()) >= n_splits
    except ValueError:
        bins = None
        can_stratify = False
    if can_stratify:
        splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
        return list(splitter.split(np.zeros(n_samples), bins))
    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(splitter.split(np.zeros(n_samples)))


def lgbm_params(target: str) -> dict:
    params = {
        "objective": "regression_l1",
        "n_estimators": 3000,
        "learning_rate": 0.025,
        "num_leaves": 24,
        "max_depth": -1,
        "min_child_samples": 15,
        "subsample": 0.85,
        "subsample_freq": 1,
        "colsample_bytree": 0.45,
        "reg_alpha": 0.08,
        "reg_lambda": 0.8,
        "random_state": RANDOM_SEED,
        "n_jobs": -1,
        "verbosity": -1,
    }
    if target in {"Tg", "Density", "Rg"}:
        params.update({"num_leaves": 18, "min_child_samples": 8, "colsample_bytree": 0.55})
    if target == "FFV":
        params.update({"num_leaves": 32, "min_child_samples": 25, "colsample_bytree": 0.40})
    return params


def predict_enhanced_lgbm(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    print("Building physics-enhanced LGBM features", flush=True)
    train_base_raw = build_enhanced_features(train)
    test_base_raw = build_enhanced_features(test)
    train_base_features, test_base_features = align_features(train_base_raw, test_base_raw)
    train_base_features = sanitize_lgbm(train_base_features)
    test_base_features = sanitize_lgbm(test_base_features)
    print(f"Physics base feature count: {train_base_features.shape[1]}", flush=True)

    try:
        train_3d_raw = build_3d_features(train, "train")
        test_3d_raw = build_3d_features(test, "test")
        train_full_raw = pd.concat([train_base_raw, train_3d_raw], axis=1)
        test_full_raw = pd.concat([test_base_raw, test_3d_raw], axis=1)
        train_3d_features, test_3d_features = align_features(train_full_raw, test_full_raw)
        train_3d_features = sanitize_lgbm(train_3d_features)
        test_3d_features = sanitize_lgbm(test_3d_features)
        print(f"Physics + 3D feature count: {train_3d_features.shape[1]}", flush=True)
        has_3d = True
    except Exception as exc:
        print(f"3D-lite features failed; using base physics for all targets. Error: {exc}", flush=True)
        train_3d_features, test_3d_features = train_base_features, test_base_features
        has_3d = False

    predictions = pd.DataFrame(index=test.index)
    for target in TARGETS:
        use_3d = has_3d and target in PHYSICS_3D_TARGETS
        train_features = train_3d_features if use_3d else train_base_features
        test_features = test_3d_features if use_3d else test_base_features
        print(
            f"Training target {target} with {'physics+3D' if use_3d else 'physics-base'} features",
            flush=True,
        )
        mask = train[target].notna()
        y = train.loc[mask, target].reset_index(drop=True)
        x = train_features.loc[mask].reset_index(drop=True)
        folds = make_target_folds(y)
        fold_test_preds = []
        for fold, (tr_idx, va_idx) in enumerate(folds):
            x_train, y_train = x.iloc[tr_idx], y.iloc[tr_idx]
            x_valid, y_valid = x.iloc[va_idx], y.iloc[va_idx]
            imputer = make_imputer()
            x_train_imp = pd.DataFrame(imputer.fit_transform(x_train), columns=x.columns)
            x_valid_imp = pd.DataFrame(imputer.transform(x_valid), columns=x.columns)
            x_test_imp = pd.DataFrame(imputer.transform(test_features), columns=x.columns)
            model = lgb.LGBMRegressor(**lgbm_params(target))
            model.fit(
                x_train_imp,
                y_train,
                eval_set=[(x_valid_imp, y_valid)],
                eval_metric="l1",
                callbacks=[lgb.early_stopping(120, verbose=False), lgb.log_evaluation(0)],
            )
            valid_pred = model.predict(x_valid_imp, num_iteration=model.best_iteration_)
            fold_mae = float(np.mean(np.abs(valid_pred - y_valid.to_numpy())))
            print(f"{target} fold {fold} physics_lgbm_mae={fold_mae:.6f}", flush=True)
            fold_test_preds.append(model.predict(x_test_imp, num_iteration=model.best_iteration_))
        predictions[target] = np.mean(fold_test_preds, axis=0)
    return predictions[TARGETS]


def main() -> None:
    comp_dir = find_competition_dir()
    asset_root = find_asset_root()
    print(f"Competition dir: {comp_dir}", flush=True)
    print(f"Asset root: {asset_root}", flush=True)

    train = pd.read_csv(comp_dir / "train.csv")
    test = pd.read_csv(comp_dir / "test.csv")
    sample = pd.read_csv(comp_dir / "sample_submission.csv")
    print(f"train {train.shape} test {test.shape} sample {sample.shape}", flush=True)

    gnn_preds = predict_gnn(test, asset_root, train)
    print("GNN prediction ranges:", flush=True)
    for target in TARGETS:
        print(f"  {target}: {gnn_preds[target].min():.6f} to {gnn_preds[target].max():.6f}", flush=True)

    try:
        enhanced_preds = predict_enhanced_lgbm(train, test)
        use_lgbm = True
    except Exception as exc:
        print(f"Enhanced LGBM failed; falling back to GNN only. Error: {exc}", flush=True)
        enhanced_preds = pd.DataFrame(np.nan, index=test.index, columns=TARGETS)
        use_lgbm = False

    submission = sample.copy()
    for target in TARGETS:
        if use_lgbm:
            w = PHYSICS_LGBM_WEIGHTS[target]
            submission[target] = (1.0 - w) * gnn_preds[target].to_numpy() + w * enhanced_preds[target].to_numpy()
        else:
            submission[target] = gnn_preds[target].to_numpy()

    tg_shift = float(train["Tg"].dropna().std() * TG_STD_SHIFT_MULTIPLIER)
    submission["Tg"] = submission["Tg"] + tg_shift
    print(f"Applied Tg shift: +{tg_shift:.6f}", flush=True)

    # Conservative physical clipping. These bounds are intentionally broad.
    submission["FFV"] = submission["FFV"].clip(0.0, 1.0)
    submission["Tc"] = submission["Tc"].clip(0.0, 1.0)
    submission["Density"] = submission["Density"].clip(0.5, 2.5)
    submission["Rg"] = submission["Rg"].clip(0.0, 100.0)

    output_path = WORKING_DIR / "submission.csv"
    submission.to_csv(output_path, index=False)
    print(f"Saved {output_path}", flush=True)
    print(submission.head(), flush=True)


if __name__ == "__main__":
    main()
