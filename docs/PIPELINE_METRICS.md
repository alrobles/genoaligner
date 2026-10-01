# 管线产能指标（Before vs Now）

刷新：`python scripts/pipeline_metrics.py --out results/pipeline_metrics.tsv`
原始数据：`results/pipeline_metrics.tsv`（每条含来源，可追溯）

## A. 逐 locus 对齐（31 loci 全量）

| 方案 | 总耗时 | 中位/locus | 最大/locus | 备注 |
|---|---|---|---|---|
| MACSE（旧） | sacct 单次记录 ≈ 49h 累计（含多次 6h 超时重试） | — | — | `.macse.log` 无内建计时；`align_genes_qc` jobs 反复撞 6h 墙 |
| genomsa base | **2459s ≈ 41min** | 36.6s | 304s | GPU (MI210)，log 内建计时 |
| genomsa lf | **862s ≈ 14min** | 13.6s | 182s | GPU |

对齐阶段压缩量级：**天数级 → 分钟级**（即便按 MACSE 最保守估计也是 ~100× 以上）。

## B. 树推理（mini backbone, n≈100–150）

| 方案 | 单 cell 耗时 | 备注 |
|---|---|---|
| IQ-TREE (MFP+MERGE+UFBoot) | ~65–120min（v2 15 cells） | 全协议 |
| **CASTER** | **19–94s（中位 37s）** | quartet-by-site, 无 checkpoint |

## 进行中（暂无终值）

- `iqt_chain` ×4：全尺寸 4353-taxa backbone，chained links 累积中。
- `au_test`：`-z/-zb` 设计超时，待改 `-te` 快速 logL 方案。
- `ft_sm` ×3：kbs 排队。

## 读数口径注意

- MACSE 的 sacct 数含超时重试，**是上限不是单 locus 用时**；精确值需逐 locus 重测或查日志时间戳。
- CASTER minis 的 19–94s 为 8 线程下的完整运行（meta.tsv `elapsed_seconds`），可直接与 IQ-TREE 同 cell 对比。
- 所有数字随任务增加自动累积，重跑脚本即刷新。
