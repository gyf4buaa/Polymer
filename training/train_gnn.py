"""Reconstructed GATv2 training pipeline.

Adapted from the MIT-licensed public third-place solution:
fresnellll/kaggle-NeurIPS-polymer-prediction-solution
Reference commit: f385d220d348283792c9f3dc8ed4ab0619e6f7c4

Outputs match the asset names consumed by the archived inference script.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from torch_geometric.data import InMemoryDataset
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, GATv2Conv, global_mean_pool
from tqdm import tqdm

TARGETS = ["Tg", "FFV", "Tc", "Density", "Rg"]


class Config:
    hidden_dim = 384
    num_layers = 6
    gat_heads = 8
    dropout = 0.2
    num_node_features = 7
    num_edge_features = 2
    epochs = 600
    batch_size = 64
    learning_rate = 1e-4
    early_stopping_patience = 40
    n_folds = 5
    seed = 42


class PolymerDataset(InMemoryDataset):
    def __init__(self, filepath: Path):
        super().__init__()
        self.data, self.slices = torch.load(filepath, weights_only=False)


class PolymerGNN(nn.Module):
    def __init__(self, tasks_fp_indices: dict[str, list[int]]):
        super().__init__()
        cfg = Config
        assert cfg.hidden_dim % cfg.gat_heads == 0
        head_dim = cfg.hidden_dim // cfg.gat_heads

        self.tasks_fp_indices = tasks_fp_indices
        self.input_conv = GATv2Conv(
            cfg.num_node_features,
            head_dim,
            heads=cfg.gat_heads,
            dropout=cfg.dropout,
            edge_dim=cfg.num_edge_features,
        )
        self.input_bn = BatchNorm(cfg.hidden_dim)
        self.hidden_convs = nn.ModuleList(
            [
                GATv2Conv(
                    cfg.hidden_dim,
                    head_dim,
                    heads=cfg.gat_heads,
                    dropout=cfg.dropout,
                    edge_dim=cfg.num_edge_features,
                )
                for _ in range(cfg.num_layers - 1)
            ]
        )
        self.hidden_bns = nn.ModuleList(
            [BatchNorm(cfg.hidden_dim) for _ in range(cfg.num_layers - 1)]
        )
        self.task_predictors = nn.ModuleDict()
        for task in TARGETS:
            indices = tasks_fp_indices.get(task, [])
            if indices:
                self.task_predictors[task] = nn.Sequential(
                    nn.Linear(cfg.hidden_dim + len(indices), cfg.hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.hidden_dim, 1),
                )

    def forward(self, data):
        x = F.elu(
            self.input_bn(
                self.input_conv(
                    data.x, data.edge_index, edge_attr=data.edge_attr
                )
            )
        )
        for conv, bn in zip(self.hidden_convs, self.hidden_bns):
            x = F.elu(bn(conv(x, data.edge_index, edge_attr=data.edge_attr))) + x

        graph_embedding = global_mean_pool(x, data.batch)
        outputs = []
        for task in TARGETS:
            if task not in self.task_predictors:
                outputs.append(
                    torch.zeros(
                        (graph_embedding.size(0), 1),
                        dtype=graph_embedding.dtype,
                        device=graph_embedding.device,
                    )
                )
                continue
            indices = torch.tensor(
                self.tasks_fp_indices[task],
                dtype=torch.long,
                device=data.morgan_fp.device,
            )
            selected_fp = data.morgan_fp.index_select(1, indices)
            fused = torch.cat([graph_embedding, selected_fp], dim=1)
            outputs.append(self.task_predictors[task](fused))
        return torch.cat(outputs, dim=1)


class WeightedMAELoss(nn.Module):
    def __init__(self, weights: dict[str, float], device: torch.device):
        super().__init__()
        tensor = torch.tensor(
            [weights.get(t, 0.0) for t in TARGETS],
            dtype=torch.float,
            device=device,
        ).unsqueeze(0)
        self.register_buffer("weights", tensor)

    def forward(self, predictions, targets):
        mask = ~torch.isnan(targets)
        if not torch.any(mask):
            return predictions.sum() * 0.0
        errors = torch.abs(predictions - targets) * self.weights
        return errors[mask].sum() / predictions.shape[0]


def competition_weights(train_stats: pd.DataFrame) -> dict[str, float]:
    k = len(TARGETS)
    n = {t: int(train_stats[t].notna().sum()) for t in TARGETS}
    ranges = {t: float(train_stats[t].max() - train_stats[t].min()) for t in TARGETS}
    sqrt_inv_n = {t: np.sqrt(1.0 / n[t]) if n[t] else 0.0 for t in TARGETS}
    denom = sum(sqrt_inv_n.values())
    return {
        t: (1.0 / ranges[t]) * (k * sqrt_inv_n[t]) / denom
        if denom > 0 and ranges[t] > 0
        else 0.0
        for t in TARGETS
    }


def weighted_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    weights: dict[str, float],
) -> float:
    values = []
    for i, target in enumerate(TARGETS):
        mask = ~np.isnan(y_true[:, i])
        if mask.any():
            values.append(
                np.nansum(np.abs(y_pred[mask, i] - y_true[mask, i]) * weights[target])
            )
    return float(sum(values) / len(y_true))


def train_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total = 0.0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        if torch.isfinite(loss) and loss.item() > 0:
            loss.backward()
            optimizer.step()
        if torch.isfinite(loss):
            total += float(loss.item()) * batch.num_graphs
    return total / max(len(loader.dataset), 1)


def predict(model, loader, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            preds.append(model(batch).cpu().numpy())
            trues.append(batch.y.cpu().numpy())
    return np.concatenate(preds), np.concatenate(trues)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, default=Path("training_artifacts"))
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    processed_path = args.artifact_root / "processed" / "all_augmented_master.pt"
    selected_path = args.artifact_root / "assets" / "best_fp_indices.pkl"
    model_root = args.artifact_root / "models"
    model_root.mkdir(parents=True, exist_ok=True)

    device = torch.device(
        args.device
        if args.device
        else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(f"device={device}")

    dataset = PolymerDataset(processed_path)
    all_y = np.asarray([d.y.numpy().reshape(-1) for d in dataset])
    stats = pd.DataFrame(all_y, columns=TARGETS)
    weights = competition_weights(stats)
    print("wMAE weights:", {k: round(v, 6) for k, v in weights.items()})

    fp_indices = joblib.load(selected_path)
    loss_fn = WeightedMAELoss(weights, device)

    splitter = KFold(
        n_splits=Config.n_folds, shuffle=True, random_state=Config.seed
    )

    for fold, (train_idx, val_idx) in enumerate(splitter.split(np.arange(len(dataset)))):
        print(f"\n=== fold {fold} ===")
        fold_dir = model_root / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        train_set = torch.utils.data.Subset(dataset, train_idx)
        val_set = torch.utils.data.Subset(dataset, val_idx)
        train_loader = DataLoader(
            train_set,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        val_loader = DataLoader(
            val_set,
            batch_size=Config.batch_size * 2,
            shuffle=False,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )

        model = PolymerGNN(fp_indices).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=Config.learning_rate)
        best_score = float("inf")
        best_epoch = 0
        patience = 0
        temp_path = fold_dir / "temp_best_model.pth"

        for epoch in range(1, Config.epochs + 1):
            train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device)
            val_pred, val_true = predict(model, val_loader, device)
            score = weighted_mae(val_true, val_pred, weights)

            if epoch == 1 or epoch % 10 == 0:
                print(
                    f"epoch={epoch:03d} train={train_loss:.6f} "
                    f"val_wMAE={score:.6f}"
                )

            if score < best_score:
                best_score = score
                best_epoch = epoch
                torch.save(model.state_dict(), temp_path)
                patience = 0
            else:
                patience += 1

            if patience >= Config.early_stopping_patience:
                break

        if best_epoch == 0:
            raise RuntimeError(f"Fold {fold} never produced a valid checkpoint")

        print(f"best_epoch={best_epoch} best_val_wMAE={best_score:.6f}")

        model.load_state_dict(torch.load(temp_path, map_location=device))
        val_pred, val_true = predict(model, val_loader, device)
        calibrators = {}
        for i, target in enumerate(TARGETS):
            mask = ~np.isnan(val_true[:, i])
            if mask.sum() > 1:
                calibrators[target] = LinearRegression().fit(
                    val_pred[mask, i].reshape(-1, 1),
                    val_true[mask, i],
                )
        joblib.dump(calibrators, fold_dir / "calibrators.pkl")

        full_fold = torch.utils.data.Subset(
            dataset, np.concatenate([train_idx, val_idx])
        )
        full_loader = DataLoader(
            full_fold,
            batch_size=Config.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=device.type == "cuda",
        )
        refit = PolymerGNN(fp_indices).to(device)
        refit_opt = torch.optim.AdamW(
            refit.parameters(), lr=Config.learning_rate
        )
        for _ in tqdm(range(best_epoch), desc=f"refit fold {fold}"):
            train_epoch(refit, full_loader, refit_opt, loss_fn, device)

        torch.save(refit.state_dict(), fold_dir / "final_refit_model.pth")
        temp_path.unlink(missing_ok=True)

    print("\nTraining complete.")


if __name__ == "__main__":
    main()
