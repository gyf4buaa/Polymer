# Stage 3B — Message-passing operator ablation

Formal source commit: `c57d016cc65c1b74e00b89e9ff1ba40acf59a0c9`.

GINE is the historical keep-dummy Variant A baseline from Stage 3A/3A.1; it was not retrained. Negative paired deltas favor the left-hand operator.

## Overall OOF wMAE

| Model | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Min–max |
|---|---:|---:|---:|---:|---:|---:|---:|
| GINE | 0.0227728722 | 0.0226486259 | 0.0228896667 | 0.0227330069 | 0.0228828622 | 0.0227854068 ± 0.0001024446 | 0.0226486259–0.0228896667 |
| GATv2 | 0.0230836920 | 0.0233582343 | 0.0234626269 | 0.0236539060 | 0.0233196560 | 0.0233756230 ± 0.0002083684 | 0.0230836920–0.0236539060 |
| PNA | 0.0310363356 | 0.0328282117 | 0.0314809960 | 0.0321503291 | 0.0314973000 | 0.0317986345 ± 0.0006993773 | 0.0310363356–0.0328282117 |

## Paired overall deltas

| Comparison | 42 | 43 | 44 | 45 | 46 | Mean ± sample SD | Left lower |
|---|---:|---:|---:|---:|---:|---:|---:|
| GATv2-GINE | +0.0003108197 | +0.0007096084 | +0.0005729602 | +0.0009208991 | +0.0004367939 | +0.0005902163 ± 0.0002374364 | 0/5 |
| PNA-GINE | +0.0082634634 | +0.0101795859 | +0.0085913293 | +0.0094173221 | +0.0086144378 | +0.0090132277 ± 0.0007782772 | 0/5 |
| GATv2-PNA | -0.0079526437 | -0.0094699774 | -0.0080183691 | -0.0084964231 | -0.0081776440 | -0.0084230114 ± 0.0006218347 | 5/5 |

## Per-target MAE

| Target | GINE mean ± SD | GATv2 mean ± SD | PNA mean ± SD | GATv2 − GINE | PNA − GINE |
|---|---:|---:|---:|---:|---:|
| Tg | 51.796823 ± 0.92 | 54.059867 ± 0.7203 | 55.05622 ± 0.5464 | +2.2630436 ± 0.8043 | +3.259397 ± 1.001 |
| FFV | 0.0058948628 ± 0.0001283 | 0.0059496203 ± 0.0001992 | 0.010847482 ± 0.0004631 | +5.4757473e-05 ± 0.0001804 | +0.0049526193 ± 0.000472 |
| Tc | 0.024661791 ± 0.0003661 | 0.025040287 ± 0.0005289 | 0.032492867 ± 0.001132 | +0.00037849646 ± 0.0004448 | +0.0078310757 ± 0.001102 |
| Density | 0.024601593 ± 0.0006658 | 0.028065642 ± 0.0009768 | 0.061492349 ± 0.001865 | +0.0034640495 ± 0.001156 | +0.036890756 ± 0.002295 |
| Rg | 1.5773881 ± 0.02164 | 1.5498713 ± 0.01849 | 1.9217767 ± 0.04707 | -0.027516782 ± 0.03439 | +0.34438863 ± 0.06316 |

## Architecture and parameter counts

| Model | Hidden width × layers | Trainable parameters | Message-passing parameters |
|---|---:|---:|---:|
| GINE | 256 × 4 | 1,243,657 | 543,748 |
| GATv2 | 256 × 4 | 1,244,421 | 544,768 |
| PNA | 256 × 4 | 5,176,581 | 4,476,928 |

GATv2 uses four 64-channel heads with concatenation and edge-aware attention. PNA uses mean/min/max/std aggregators, identity/amplification/attenuation scalers, one tower, and edge features.

## Frozen setup and provenance

- Train SHA256: `1f79c85c785698e8c3499d99721adfe3be9660a487f137a923dd34eb7ef845e1`
- Folds SHA256: `1bb066dd45d9b9a0f7861efbe7efd38c438522519ed36a745bf61f2a9191284a`
- Formal OOF runs: GATv2 5/5; PNA 5/5.
- Total formal runtime: 12360.8 seconds.
- CUDA device(s): NVIDIA GeForce RTX 4070.
- Mean sampled GPU utilization: GATv2 35.2%; PNA 61.7%.
- Maximum PyTorch allocated/reserved VRAM: 507.9/1286.0 MiB; maximum `nvidia-smi` observed memory: 3011 MiB.
- GATv2: {'heads': 4, 'channels_per_head': 64, 'concat': True, 'edge_aware': True, 'attention_dropout': 0.0, 'add_self_loops': False}.
- PNA: {'aggregators': ['mean', 'min', 'max', 'std'], 'scalers': ['identity', 'amplification', 'attenuation'], 'towers': 1, 'edge_aware': True}.
- PNA degree histogram on 7973 frozen graphs / 273433 nodes: `{'0': 0, '1': 48203, '2': 140996, '3': 77869, '4': 6364, '5': 0, '6': 1}`; graph fingerprint `a3107fc9257796215375e2cf8dc387e9edbabcff7fc2d37b8dd8507886572315`; generator SHA256 `cf55864c2c1aefbeca1d6df7042fab729dad83337ea0e5874089427549d881d4`.
- All 10 formal runs use the same source commit; GINE's existing five seeds were reused.
- The first GATv2 seed-46 attempt was interrupted by an SSH transport timeout and excluded; its same-config restart is the formal seed-46 run.
- No readout, global feature, descriptor, ensemble, or later-stage experiments were run.
