"""Edge-aware message-passing operators with the frozen Own-GNN readout."""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv, PNAConv, global_max_pool, global_mean_pool

from ...src.metrics import TARGETS
from ..own_gnn_v0.graph import EDGE_FEATURE_DIM, NODE_CARDINALITIES


def build_operator_model(
    *,
    operator: str,
    hidden_dim: int,
    num_layers: int,
    dropout: float,
    operator_config: Mapping[str, object],
    degree_histogram: Sequence[int] | None = None,
) -> "OperatorGNN":
    return OperatorGNN(
        operator=operator,
        hidden_dim=hidden_dim,
        num_layers=num_layers,
        dropout=dropout,
        operator_config=operator_config,
        degree_histogram=degree_histogram,
    )


class OperatorGNN(nn.Module):
    """Shared input encoder, four hidden-width blocks, and frozen readout."""

    def __init__(
        self,
        *,
        operator: str,
        hidden_dim: int,
        num_layers: int,
        dropout: float,
        operator_config: Mapping[str, object],
        degree_histogram: Sequence[int] | None = None,
    ) -> None:
        super().__init__()
        if operator not in {"gatv2", "pna"}:
            raise ValueError(f"Unsupported Stage 3B operator: {operator}")
        if hidden_dim < 1 or num_layers < 1:
            raise ValueError("hidden_dim and num_layers must be positive")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in [0, 1)")
        self.operator = operator
        self.hidden_dim = int(hidden_dim)
        self.num_layers = int(num_layers)
        self.dropout = float(dropout)
        self.operator_config = dict(operator_config)
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(cardinality, hidden_dim) for cardinality in NODE_CARDINALITIES]
        )
        # Keep the frozen 16-column categorical bond representation. Each
        # edge-aware operator maps it internally to its 256-wide messages,
        # just as GINEConv does through its edge_dim projection.
        self.convs = nn.ModuleList()
        if operator == "gatv2":
            heads = int(operator_config["heads"])
            channels_per_head = int(operator_config["channels_per_head"])
            if heads * channels_per_head != hidden_dim:
                raise ValueError("GATv2 heads × channels_per_head must equal hidden_dim")
            if operator_config.get("concat") is not True or operator_config.get("edge_aware") is not True:
                raise ValueError("Stage 3B GATv2 must concatenate heads and consume edge_attr")
            for _ in range(num_layers):
                self.convs.append(
                    GATv2Conv(
                        in_channels=hidden_dim,
                        out_channels=channels_per_head,
                        heads=heads,
                        concat=True,
                        edge_dim=EDGE_FEATURE_DIM,
                        dropout=0.0,
                        add_self_loops=False,
                        residual=False,
                    )
                )
        else:
            if degree_histogram is None or not degree_histogram:
                raise ValueError("PNA requires the frozen benchmark degree histogram")
            aggregators = list(operator_config["aggregators"])
            scalers = list(operator_config["scalers"])
            degree = torch.as_tensor(degree_histogram, dtype=torch.long)
            for _ in range(num_layers):
                self.convs.append(
                    PNAConv(
                        in_channels=hidden_dim,
                        out_channels=hidden_dim,
                        aggregators=aggregators,
                        scalers=scalers,
                        deg=degree,
                        edge_dim=EDGE_FEATURE_DIM,
                        towers=int(operator_config.get("towers", 1)),
                        pre_layers=1,
                        post_layers=1,
                        divide_input=False,
                    )
                )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
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
        hidden = sum(
            embedding(data.x[:, feature_index])
            for feature_index, embedding in enumerate(self.atom_embeddings)
        )
        encoded_edges = data.edge_attr.to(dtype=hidden.dtype)
        for conv, norm in zip(self.convs, self.norms):
            update = conv(hidden, data.edge_index, edge_attr=encoded_edges)
            hidden = norm(
                hidden
                + F.dropout(F.relu(update), p=self.dropout, training=self.training)
            )
            if hidden.size(1) != self.hidden_dim:
                raise RuntimeError("Message-passing operator changed the frozen hidden width")
        return hidden

    def forward(self, data: Data) -> torch.Tensor:
        hidden = self.encode_nodes(data)
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)
        graph_embedding = torch.cat(
            (global_mean_pool(hidden, batch), global_max_pool(hidden, batch)), dim=1
        )
        return torch.cat([self.heads[target](graph_embedding) for target in TARGETS], dim=1)
