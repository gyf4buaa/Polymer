# Physics3D Blend Diagnostic

This is the light 3D conformer experiment for the NeurIPS OPP 2025 solution.

## 3D-lite settings

- Structures: capped monomer + chain2 dimer.
- Conformer generator: RDKit ETKDGv3.
- Optimization: MMFF when available, otherwise UFF.
- Heavy atom cutoff: `75`.
- Conformers:
  - `2` conformers when heavy atoms <= `45`.
  - `1` conformer otherwise.
- 3D target usage in the v2 script:
  - `Tc`: physics + 3D features.
  - `Rg`: physics + 3D features.
  - `Tg`, `FFV`, `Density`: base physics features.

## LGBM-only released diagnostic

| model | public | private |
|---|---:|---:|
| physics LGBM | 0.173886 | 0.090044 |
| physics + 3D LGBM | 0.174342 | 0.089175 |

The 3D features improve private mainly through `Rg` and `Tc`, while `Density` and `Tg` are not improved.

## Blend diagnostic

All rows include the same Tg shift used in v1.

| blend | public | private |
|---|---:|---:|
| v1: GNN + physics | 0.160564 | 0.080068 |
| fixed old weights, 3D only for Tc/Rg | 0.160504 | 0.079767 |
| v2 selected weights, 3D only for Tc/Rg | 0.160371 | 0.079016 |

## v2 blend weights

```text
Tg      0.00 LGBM / 1.00 GNN, base physics not used
FFV     0.11 base physics LGBM / 0.89 GNN
Tc      0.27 physics+3D LGBM / 0.73 GNN
Density 0.37 base physics LGBM / 0.63 GNN
Rg      0.38 physics+3D LGBM / 0.62 GNN
```

## Practical note

This is a heavier Kaggle Notebook than v1. The 3D branch is useful but modest; if Kaggle runtime becomes a problem, fall back to v1.
