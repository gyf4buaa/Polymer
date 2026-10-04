"""Minimal zero-initialized physical-node residual over the frozen keep-dummy G0."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data

from ..polymer_representation_ablation.model import OwnGNNRepresentation
from .features import PHYSICAL_FEATURE_DIM


class ElementalPhysicalGNN(OwnGNNRepresentation):
    """Frozen G0 backbone plus Linear(5, hidden_dim, bias=False) on each atom."""

    def __init__(
        self,
        *,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        # Construct after every historical/common parameter to preserve G0 RNG.
        self.physical_projection = nn.Linear(PHYSICAL_FEATURE_DIM, hidden_dim, bias=False)
        nn.init.zeros_(self.physical_projection.weight)

    def _initial_node_embeddings(self, data: Data) -> Tensor:
        hidden = super()._initial_node_embeddings(data)
        physical = getattr(data, "physical_features", None)
        if physical is None or physical.ndim != 2:
            raise ValueError("physical_features must be a [N, 5] node matrix")
        if physical.shape != (data.x.size(0), PHYSICAL_FEATURE_DIM):
            raise ValueError(
                f"physical_features must have shape [N, {PHYSICAL_FEATURE_DIM}], "
                f"got {tuple(physical.shape)}"
            )
        if physical.device != hidden.device:
            raise ValueError("physical_features and categorical embeddings must share a device")
        physical = physical.to(dtype=hidden.dtype)
        if not torch.isfinite(physical).all():
            raise ValueError("physical_features must contain only finite values")
        return hidden + self.physical_projection(physical)

    def projection_l2_norm(self) -> float:
        return float(self.physical_projection.weight.detach().norm().cpu())

    def projection_column_l2_norms(self) -> list[float]:
        return self.physical_projection.weight.detach().norm(dim=0).cpu().tolist()
