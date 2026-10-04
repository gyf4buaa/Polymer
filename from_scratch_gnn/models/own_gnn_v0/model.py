"""The graph-only Own-GNN v0 model."""
from __future__ import annotations

from collections.abc import Mapping

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from ...src.metrics import TARGETS
from .graph import EDGE_FEATURE_DIM, NODE_CARDINALITIES


class OwnGNNv0(nn.Module):
    """Four edge-aware GINE blocks, mean+max pooling, and five task heads."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")

        self.targets = tuple(TARGETS)
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, hidden_dim) for cardinality in NODE_CARDINALITIES]
        )
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            update_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            self.convs.append(
                GINEConv(update_mlp, edge_dim=EDGE_FEATURE_DIM, train_eps=True)
            )
            self.norms.append(nn.LayerNorm(hidden_dim))

        readout_dim = 2 * hidden_dim
        self.heads = nn.ModuleDict(
            {
                target: nn.Sequential(
                    nn.Linear(readout_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, 1),
                )
                for target in TARGETS
            }
        )

    def _initial_node_embeddings(self, data: Data) -> torch.Tensor:
        return sum(
            embedding(data.x[:, feature_index])
            for feature_index, embedding in enumerate(self.atom_embeddings)
        )

    def encode_nodes(self, data: Data) -> torch.Tensor:
        if data.x.ndim != 2 or data.x.size(1) != len(NODE_CARDINALITIES):
            raise ValueError(
                f"Expected node features shaped [N, {len(NODE_CARDINALITIES)}], "
                f"got {tuple(data.x.shape)}"
            )
        if data.edge_attr.ndim != 2 or data.edge_attr.size(1) != EDGE_FEATURE_DIM:
            raise ValueError(
                f"Expected edge features shaped [E, {EDGE_FEATURE_DIM}], "
                f"got {tuple(data.edge_attr.shape)}"
            )

        hidden = self._initial_node_embeddings(data)
        edge_index = data.edge_index
        edge_attr = data.edge_attr
        for conv, norm in zip(self.convs, self.norms):
            update = conv(hidden, edge_index, edge_attr=edge_attr)
            hidden = norm(hidden + F.dropout(F.relu(update), p=self.dropout, training=self.training))
        return hidden

    def forward(self, data: Data) -> torch.Tensor:
        hidden = self.encode_nodes(data)

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)
        mean_pool = global_mean_pool(hidden, batch)
        max_pool = global_max_pool(hidden, batch)
        graph_embedding = torch.cat((mean_pool, max_pool), dim=1)
        outputs = [self.heads[target](graph_embedding) for target in TARGETS]
        return torch.cat(outputs, dim=1)


def masked_huber_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    delta: float = 1.0,
) -> torch.Tensor:
    """Average normalized Huber loss equally across observed property tasks."""
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError(
            f"prediction and target must share [batch, tasks] shape, got "
            f"{tuple(prediction.shape)} and {tuple(target.shape)}"
        )
    if delta <= 0:
        raise ValueError("delta must be positive")

    task_losses = []
    for task_index in range(prediction.size(1)):
        mask = torch.isfinite(target[:, task_index])
        if mask.any():
            task_losses.append(
                F.huber_loss(
                    prediction[mask, task_index],
                    target[mask, task_index],
                    delta=delta,
                    reduction="mean",
                )
            )
    if not task_losses:
        return prediction.sum() * 0.0
    return torch.stack(task_losses).mean()


def build_model(config: Mapping[str, object]) -> OwnGNNv0:
    model_config = config.get("model", config)
    if not isinstance(model_config, Mapping):
        raise ValueError("Model config must be a mapping")
    return OwnGNNv0(
        hidden_dim=int(model_config.get("hidden_dim", 256)),
        num_layers=int(model_config.get("num_layers", 4)),
        dropout=float(model_config.get("dropout", 0.1)),
    )
