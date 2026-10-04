"""Own-GNN architecture used identically by the Stage 3A variants."""
from __future__ import annotations

import torch
from torch import nn
from torch_geometric.data import Data

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

    def _initial_node_embeddings(self, data: Data) -> torch.Tensor:
        endpoint = getattr(data, "polymer_endpoint", None)
        if endpoint is None or endpoint.ndim != 1 or endpoint.size(0) != data.x.size(0):
            raise ValueError("polymer_endpoint must contain one scalar per node")
        if not torch.all((endpoint == 0) | (endpoint == 1)):
            raise ValueError("polymer_endpoint must be a binary node feature")
        hidden = super()._initial_node_embeddings(data)
        return hidden + endpoint.to(hidden.dtype).unsqueeze(1) * self.polymer_endpoint_embedding
