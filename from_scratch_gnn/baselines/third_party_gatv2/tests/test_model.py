from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

from from_scratch_gnn.baselines.third_party_gatv2.model import (
    PolymerGNNV12Res,
    fit_fold_fp_indices,
    process_smiles,
)
from from_scratch_gnn.baselines.third_party_gatv2.train_oof import (
    MaskedWeightedMAELoss,
    _score_validation,
)
from from_scratch_gnn.src.metrics import TARGETS, competition_weights


class ThirdPartyGATv2Tests(unittest.TestCase):
    def test_public_periodic_graph_features(self) -> None:
        graph = process_smiles("[*]CC[*]")
        self.assertEqual(tuple(graph.x.shape[1:]), (7,))
        self.assertEqual(tuple(graph.edge_attr.shape[1:]), (2,))
        self.assertEqual(tuple(graph.morgan_fp.shape), (1, 1024))
        self.assertEqual(int((graph.edge_attr[:, 0] == -1).sum()), 2)

    def test_masked_loss_and_gatv2_forward_backward(self) -> None:
        graphs = [process_smiles(smi) for smi in ("[*]CC[*]", "[*]CO[*]", "[*]CCO[*]")]
        rows = [
            {target: (float(i + j + 1) if (i + j) % 3 else None)
             for j, target in enumerate(TARGETS)}
            for i in range(len(graphs))
        ]
        data = [
            Data(
                x=graph.x,
                edge_index=graph.edge_index,
                edge_attr=graph.edge_attr,
                morgan_fp=graph.morgan_fp,
                y=torch.tensor(
                    [float(row[t]) if row[t] is not None else float("nan") for t in TARGETS]
                ).unsqueeze(0),
            )
            for graph, row in zip(graphs, rows)
        ]
        batch = Batch.from_data_list(data)
        bits = {target: list(range(50)) for target in TARGETS}
        model = PolymerGNNV12Res(bits, hidden_dim=48, num_layers=2, heads=4, dropout=0.0)
        output = model(batch)
        self.assertEqual(tuple(output.shape), (3, 5))
        loss = MaskedWeightedMAELoss(competition_weights(rows), torch.device("cpu"))(
            output, batch.y
        )
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))
        score, target_mae, predictions = _score_validation(
            model,
            DataLoader(data, batch_size=2, shuffle=False),
            torch.device("cpu"),
            rows,
            list(range(len(rows))),
            competition_weights(rows),
        )
        self.assertTrue(np.isfinite(score))
        self.assertEqual(predictions.shape, (3, 5))
        self.assertEqual(set(target_mae), set(TARGETS))

    def test_morgan_selection_ignores_validation_labels(self) -> None:
        rng = np.random.default_rng(2026)
        fingerprints = rng.binomial(1, 0.2, size=(40, 1024)).astype(np.float32)
        labels = [
            {
                target: float(values[target_index])
                for target_index, target in enumerate(TARGETS)
            }
            for values in rng.normal(size=(40, len(TARGETS)))
        ]
        train_indices = list(range(30))
        first = fit_fold_fp_indices(fingerprints, labels, train_indices)
        changed_validation = [dict(row) for row in labels]
        for row_index in (30, 31, 32, 33, 34, 35, 36, 37, 38, 39):
            for target_index, target in enumerate(TARGETS):
                changed_validation[row_index][target] = -1e6 * (target_index + 1)
        second = fit_fold_fp_indices(fingerprints, changed_validation, train_indices)
        self.assertEqual(first, second)


if __name__ == "__main__":
    unittest.main()
