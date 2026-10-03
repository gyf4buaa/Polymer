"""Own-GNN architecture used identically by the Stage 3A variants."""
from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GINEConv, global_max_pool, global_mean_pool

from ...src.metrics import TARGETS
from ..own_gnn_v0.graph import EDGE_FEATURE_DIM, NODE_CARDINALITIES
from ..own_gnn_v0.model import OwnGNNv0


class OwnGNNRepresentation(OwnGNNv0):
    """Frozen v0 backbone plus one binary endpoint-identity node feature.

    Both variants use the same model and parameter shapes. Variant A supplies
    all-zero endpoint indicators; Variant B marks exactly its two valid
    endpoints. A zero-initialized feature vector leaves Variant A's forward
    function identical to v0 until the endpoint bit is present.
    """

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.polymer_endpoint_embedding = nn.Parameter(torch.zeros(hidden_dim))

    def forward(self, data: Data) -> torch.Tensor:
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
        endpoint = getattr(data, "polymer_endpoint", None)
        if endpoint is None or endpoint.ndim != 1 or endpoint.size(0) != data.x.size(0):
            raise ValueError("polymer_endpoint must contain one scalar per node")
        if not torch.all((endpoint == 0) | (endpoint == 1)):
            raise ValueError("polymer_endpoint must be a binary node feature")

        hidden = sum(
            embedding(data.x[:, feature_index])
            for feature_index, embedding in enumerate(self.atom_embeddings)
        )
        hidden = hidden + endpoint.to(hidden.dtype).unsqueeze(1) * self.polymer_endpoint_embedding
        for conv, norm in zip(self.convs, self.norms):
            update = conv(hidden, data.edge_index, edge_attr=data.edge_attr)
            hidden = norm(
                hidden
                + F.dropout(
                    F.relu(update), p=self.dropout, training=self.training
                )
            )

        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)
        graph_embedding = torch.cat(
            (global_mean_pool(hidden, batch), global_max_pool(hidden, batch)), dim=1
        )
        outputs = [self.heads[target](graph_embedding) for target in TARGETS]
        return torch.cat(outputs, dim=1)
