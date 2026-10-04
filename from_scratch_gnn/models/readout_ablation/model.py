"""Historical and attentive graph readouts on the keep-dummy GINE trunk."""
from __future__ import annotations

import torch
from torch import Tensor, nn
from torch_geometric.data import Data
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_geometric.utils import softmax

from ...src.metrics import TARGETS
from ..polymer_representation_ablation.model import OwnGNNRepresentation


READOUT_VARIANTS = ("R0", "R1", "R2")


class ReadoutGNN(OwnGNNRepresentation):
    """Keep-dummy GINE with fixed or learned node-weighted mean pooling.

    The parent constructs the complete historical trunk, endpoint embedding,
    and five heads before any Stage 3C gate is allocated. This keeps every
    common parameter on the same RNG initialization path as the Stage 3A.1
    GINE model.
    """

    def __init__(
        self,
        *,
        readout_variant: str,
        hidden_dim: int = 256,
        num_layers: int = 4,
        dropout: float = 0.1,
    ) -> None:
        if readout_variant not in READOUT_VARIANTS:
            raise ValueError(f"Unsupported Stage 3C readout: {readout_variant}")
        super().__init__(hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
        self.readout_variant = readout_variant
        self.shared_gate: nn.Linear | None = None
        self.property_gates = nn.ModuleDict()
        if readout_variant == "R1":
            self.shared_gate = nn.Linear(hidden_dim, 1, bias=False)
            nn.init.zeros_(self.shared_gate.weight)
        elif readout_variant == "R2":
            self.property_gates = nn.ModuleDict(
                {
                    target: nn.Linear(hidden_dim, 1, bias=False)
                    for target in TARGETS
                }
            )
            for gate in self.property_gates.values():
                nn.init.zeros_(gate.weight)

    @staticmethod
    def _batch_index(data: Data, hidden: Tensor) -> Tensor:
        batch = getattr(data, "batch", None)
        if batch is None:
            return torch.zeros(hidden.size(0), dtype=torch.long, device=hidden.device)
        return batch

    def attention_weights(
        self, hidden: Tensor, batch: Tensor, target: str | None = None
    ) -> Tensor:
        """Return normalized node weights, softmaxed independently per graph."""
        if self.readout_variant == "R1":
            assert self.shared_gate is not None
            gate = self.shared_gate
        elif self.readout_variant == "R2":
            if target not in self.property_gates:
                raise ValueError(f"R2 requires one of the targets {tuple(TARGETS)}")
            gate = self.property_gates[target]
        else:
            raise ValueError("R0 has no learned attention weights")
        scores = gate(hidden).squeeze(-1)
        return softmax(scores, batch)

    def graph_embedding(
        self, hidden: Tensor, batch: Tensor, target: str | None = None
    ) -> Tensor:
        """Build the 512-wide readout used by one target head."""
        if self.readout_variant == "R0":
            mean = global_mean_pool(hidden, batch)
        else:
            weights = self.attention_weights(hidden, batch, target=target)
            mean = global_add_pool(hidden * weights.unsqueeze(-1), batch)
        maximum = global_max_pool(hidden, batch)
        return torch.cat((mean, maximum), dim=1)

    def forward(self, data: Data) -> Tensor:
        if self.readout_variant == "R0":
            return super().forward(data)
        hidden = self.encode_nodes(data)
        batch = self._batch_index(data, hidden)
        maximum = global_max_pool(hidden, batch)
        if self.readout_variant == "R1":
            weights = self.attention_weights(hidden, batch)
            mean = global_add_pool(hidden * weights.unsqueeze(-1), batch)
            embedding = torch.cat((mean, maximum), dim=1)
            return torch.cat(
                [self.heads[target](embedding) for target in TARGETS], dim=1
            )
        outputs = []
        for target in TARGETS:
            weights = self.attention_weights(hidden, batch, target=target)
            mean = global_add_pool(hidden * weights.unsqueeze(-1), batch)
            embedding = torch.cat((mean, maximum), dim=1)
            outputs.append(self.heads[target](embedding))
        return torch.cat(outputs, dim=1)

    def gate_l2_norms(self) -> dict[str, float]:
        if self.readout_variant == "R1":
            assert self.shared_gate is not None
            return {"shared": float(self.shared_gate.weight.detach().norm().cpu())}
        if self.readout_variant == "R2":
            return {
                target: float(gate.weight.detach().norm().cpu())
                for target, gate in self.property_gates.items()
            }
        return {}


def normalized_attention_entropy(weights: Tensor, batch: Tensor) -> Tensor:
    """Per-graph entropy divided by log(node count), with singleton entropy 0."""
    from torch_geometric.nn import global_add_pool

    tiny = torch.finfo(weights.dtype).tiny
    entropy = -global_add_pool(
        (weights * weights.clamp_min(tiny).log()).unsqueeze(-1), batch
    ).squeeze(-1)
    counts = global_add_pool(torch.ones_like(weights).unsqueeze(-1), batch).squeeze(-1)
    denominator = counts.clamp_min(2).log()
    return torch.where(counts > 1, entropy / denominator, 0.0)
