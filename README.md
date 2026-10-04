# GNN + Physics3D v2（本地报告记载的最佳分数版本）

本目录整理了聚合物项目中报告分数最低的一条提交路线：GNN 与 physics/3D LightGBM 的目标级融合。它适合作为赛后复盘和 Kaggle Notebook 推理脚本归档。

## 分数记录

| 路线 | Public | Private | 证据口径 |
|---|---:|---:|---|
| GNN + Physics3D v2 | 0.06259 | 0.06830 | 2026-06-11 修订实验报告记载；本地没有对应的 Kaggle submission ref 或 API 运行摘要 |
| CodeBERTa adaptive TTA30 | 0.06297 | 0.08104 | 本地结构化 Kaggle 运行摘要，submission ref `53369928` |

分数越低越好。选中 v2 是因为它在本地修订报告中记录了最低的 Private 分数。成绩来源和限制见 [`SCORE_PROVENANCE.md`](SCORE_PROVENANCE.md)。

## 文件

- `kaggle_notebook_submission_gnn3_physics3d_blend.py`：完整 Kaggle Notebook 推理脚本。
- `evidence/physics3d_blend_released_diagnostic.md`：v1/v2 使用赛后释放标签的组件诊断，单独标记为 diagnostic。
- `evidence/codeberta_official_score_summary.json`：CodeBERTa 另一条路线的结构化 Kaggle 运行记录，用于成绩口径比较。
- `THIRD_PARTY_NOTICES.md`：GATv2 参考方案及 MIT 许可说明。

所选脚本 SHA-256：`eed5240025ec6686abb7ce07ebfcc10da4765b61de287cbb8de336a884497f25`。

## Kaggle 运行依赖

脚本需要比赛输入，以及一个附加的离线资产 Kaggle Dataset。原始 `gnn3_offline_assets.zip`（SHA-256 `4a01dbea2db931f45fd34634e542d90575aca5f9082b203f00600519d6ad419c`，105,938,993 字节）包含 5 折 GNN 权重、fingerprint 索引和离线 wheels；它超过 GitHub 单文件限制，且包含第三方模型资产，因此没有放进此 GitHub 发布目录。把确认可分发的资产包单独上传为 Kaggle Dataset 并附加到 Notebook 后，脚本会在 `/kaggle/input` 中查找并解压它。脚本面向 Internet Off 的 Kaggle 环境。

## 分数使用说明

该版本代码注释明示 Tg 偏移参考了赛后释放的 private 标签。它是赛后探索版本，分数不代表遵守竞赛盲测条件的有效名次。官方最终榜单和本地报告不一致；不要把报告中的阶段性“第 1 名”写成最终名次。

## Clean-room OOF 基准

从头训练的 Own-GNN 与传统基线使用独立的冻结 OOF 协议；它不复用本目录赛后 Kaggle blending 脚本里的 private-label 校准。Stage R 使用 `nopp2025_train_v1` 的同一 5 折与 competition-style wMAE，比较 fold median、Tanimoto kNN、SMILES-only LightGBM 和现有 C128 Own-GNN OOF。结果、相似度分层、逐样本误差和 Morgan 重复指纹审计见 [`Stage R aggregate report`](from_scratch_gnn/experiments/stageR/aggregate_summary.md)。
