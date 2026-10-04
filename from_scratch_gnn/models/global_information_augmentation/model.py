"""Minimal zero-initialized global-information residual over frozen G0."""
from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import global_max_pool, global_mean_pool

from ..polymer_representation_ablation.model import OwnGNNRepresentation

Variant = Literal["G0", "D", "M"]
GLOBAL_INPUT_DIMS = {"D": 20, "M": 2048}


class GlobalInformationGNN(OwnGNNRepresentation):
    """G0 trunk and heads, optionally plus a single linear residual projection."""

    def __init__(
        self,
        *,
        variant: Variant,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if variant not in ("G0", "D", "M"):
            raise ValueError(f"Unsupported Stage 4A variant: {variant}")
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.variant = variant
        # This is intentionally the last module constructed so G0 common
        # parameters retain their exact historical RNG initialization.
        self.global_projection: nn.Linear | None = None
        if variant in GLOBAL_INPUT_DIMS:
            self.global_projection = nn.Linear(
                GLOBAL_INPUT_DIMS[variant], 2 * hidden_dim, bias=False
            )
            nn.init.zeros_(self.global_projection.weight)

    def graph_embedding(self, data: Data, hidden: Tensor) -> Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            batch = torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)
        mean_pool = global_mean_pool(hidden, batch)
        max_pool = global_max_pool(hidden, batch)
        return torch.cat((mean_pool, max_pool), dim=1)

    def forward(self, data: Data) -> Tensor:
        if self.variant == "G0":
            # Keep the historical implementation byte-for-byte on G0.
            return super().forward(data)
        hidden = self.encode_nodes(data)
        embedding = self.graph_embedding(data, hidden)
        assert self.global_projection is not None
        global_features = getattr(data, "global_features", None)
        expected_dim = GLOBAL_INPUT_DIMS[self.variant]
        if global_features is None or global_features.ndim != 2:
            raise ValueError("Stage 4A variants require a batched global_features matrix")
        if global_features.size(1) != expected_dim:
            raise ValueError(
                f"Variant {self.variant} requires {expected_dim} global features, "
                f"received {global_features.size(1)}"
            )
        if global_features.size(0) != embedding.size(0):
            raise ValueError("Global features and graph readout batch sizes differ")
        if global_features.device != embedding.device:
            raise ValueError("Global features and graph readout must be on the same device")
        fused = embedding + self.global_projection(global_features.to(embedding.dtype))
        return torch.cat([self.heads[target](fused) for target in self.targets], dim=1)

    def projection_l2_norm(self) -> float:
        if self.global_projection is None:
            return 0.0
        return float(self.global_projection.weight.detach().norm().cpu())
