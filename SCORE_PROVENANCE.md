# 成绩来源与口径

## 本地报告中的 v2 成绩

2026-06-11 的修订实验报告将 GNN + Physics3D 融合路线记为 Public `0.06259`、Private `0.06830`，并称其在报告时为阶段性第 1 / 2241。该报告没有保留对应的 Kaggle submission reference、逐次提交记录或 API 运行摘要，因此本仓库只能将它作为“本地报告记载的成绩”，不能独立核验该分数与当前脚本/模型资产完全对应。

同一 v2 包中的 [`physics3d_blend_released_diagnostic.md`](evidence/physics3d_blend_released_diagnostic.md) 记录的是另一套赛后释放标签诊断：Public `0.160371`、Private `0.079016`。这两个数不是上面的 Kaggle 提交分数，也不能与 OOF 分数混用。

## 公开最终榜单

截至 2026-10-02，Kaggle 竞赛页面标记竞赛已结束，并显示最终榜单冠军 Private 分数为 `0.07536`。页面没有把本地报告中的 `0.06830` 列为最终排名记录。因此这里保留报告原始说法，同时不宣称 v2 是最终榜单冠军。

来源：[Kaggle 官方最终榜单](https://www.kaggle.com/competitions/neurips-open-polymer-prediction-2025/leaderboard)。

## 可核验的另一条运行记录

本地归档的 CodeBERTa adaptive 运行摘要记载 submission ref `53369928`、状态 `COMPLETE`、Public `0.06297`、Private `0.08104`，并以当时榜单阈值估算为第 7 名水平。完整记录保存在 [`evidence/codeberta_official_score_summary.json`](evidence/codeberta_official_score_summary.json)。这是另一条模型路线，不是 v2 GNN + Physics3D 的运行记录。

## 赛后标签校准

v2 推理脚本顶部说明使用了“validated against the released private labels”的 Tg 偏移；脚本将 `Tg` 预测加上 `0.5644 * std(train.Tg)`。由于该偏移参考赛后释放的 private 标签，报告中的 `0.06830` 不能当作盲测期间可复现、无泄漏的竞赛名次。此仓库按赛后研究代码归档，并对这一限制作显式说明。
