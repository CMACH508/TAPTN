# TAPTN 微调实验复现

对应论文中编码器微调与 GNN 对照表格：

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)

English README: [README.md](README.md).

在相同标签、划分与边上，分别在原始文本（**TA+LM**）、GraphICL 风格邻域文本（**GraphICL+LM**）或 TAPTN 增强文本（**TAPTN+LM**）上微调语言模型，并与各类 GNN 对照。每个格子为 5 次运行（run 1–5），报告 mean±std。

论文级总览：[../README_zh.md](../README_zh.md)。另外两章是同级目录：[`TAPTN_rewiring_repro`](../TAPTN_rewiring_repro)、[`TAPTN_icl_repro`](../TAPTN_icl_repro)。

## 环境

```bash
pip install -r requirements.txt
```

拼表与 LM 重新计分可在 CPU 上完成。`train` 需要 GPU。

## 数据

数据集、TAPTN/GraphICL 文本、预训练编码器与 LM 权重**不在本仓库内**。请从 [Google Drive](https://drive.google.com/drive/folders/1wIlc-r7HFd31YhQLGjc0uY0-Xy3qfUVZ?usp=sharing) 下载资源包并解压，再让代码指向该目录。

推荐布局（在论文总目录内）：

```
TAPTN_paper_repro/
  TAPTN_finetune_repro/           # 本仓库
  TAPTN_finetune_repro_assets/    # 解压后的资源包
```

```bash
export TAPTN_ASSETS=/path/to/TAPTN_finetune_repro_assets
```

未设置 `TAPTN_ASSETS` 时，`reproduce.py` 会依次查找并列目录 `TAPTN_finetune_repro_assets/` 或 `./assets/`。资源包内含 `dataset/`、`gpt_responses/`、`prt_lm/`、`pretrained/` 以及 WebKB 文件，细节见包内 README。

## 快速开始（不训练）

```bash
export TAPTN_ASSETS=/path/to/TAPTN_finetune_repro_assets
python reproduce.py check
python reproduce.py dry --table all --rescore-lm
```

`dry` 按论文表格顺序输出记录结果。`--rescore-lm` 用已保存的 `.pred` 重算 LM 测试精度（无需 GPU）。冻结 / 联合训练的 GNN 权重没有按 run 分文件（`output/<dataset>/<GNN>.pt` 会被后一次训练覆盖），这些格子使用记录值，而不是当前磁盘上的 `.pt`。

## 论文表格

PDF 编号与 TMLR 2026 相机就绪稿一致。`--table` 是命令行参数（沿用历史 TeX label 或短别名），顺序与论文一致：

| PDF | `--table` | 内容 |
|---|---|---|
| 表 11 | `gnn_summary` | 显著超过 TAPTN+LM 的流水线数量 |
| 表 12 | `p5_transfer` / `transfer` | TAPTN 嵌入 + 冻结 GNN（可迁移性计数） |
| 表 13 | `crosslm_encoder` | DeBERTa vs RoBERTa |
| 表 20 | `tb_6` | 冻结 TA+GNN、GraphICL+LM、TAPTN+LM（同配图） |
| 表 21 | `heterophilic` | 冻结 TA+GNN、TAPTN+LM（WebKB） |
| 表 22 | `joint` | 编码器与 GNN 联合训练 |
| 表 23 | `roberta` | RoBERTa-base 编码器 |

```bash
python reproduce.py dry --table gnn_summary
python reproduce.py dry --table transfer
python reproduce.py dry --table crosslm_encoder
python reproduce.py dry --table tb_6
python reproduce.py dry --table heterophilic
python reproduce.py dry --table joint
python reproduce.py dry --table roberta
```

表 12 的 caption 把 **P5** 定义为该表的**列缩写**：把 TAPTN 增强文本（而非原始文本）送给与 TA+GNN 相同的冻结 GNN。它不是主对照表中的行名（主表行是 TA+LM、GraphICL+LM、TAPTN+LM、TA+GNN、Joint）。

## 训练

使用 `--run 1` … `--run 5`。库所用 RNG 种子见 [`configs/run_seeds.yaml`](configs/run_seeds.yaml)。

```bash
python reproduce.py train --pipeline taptn_lm --dataset cora --run 1 --device 0
python reproduce.py train --pipeline grapicl_lm --dataset product --run 1 --device 0
python reproduce.py train --pipeline ta_lm --dataset cora --run 1 --device 0
python reproduce.py train --pipeline ta_gnn --dataset cora --run 1 --gnn APPNP --device 0
python reproduce.py train --pipeline taptn_gnn --dataset cora --run 1 --gnn GraphSAGE --device 0
python reproduce.py train --pipeline joint --dataset wisconsin --run 1 --gnn DirGNN --device 0
python reproduce.py train --pipeline taptn_lm --dataset cora --run 1 --encoder roberta --device 0
```

默认复用已有 LM checkpoint；从头微调请加 `--force-retrain`。GNN 训练会覆盖 `output/<dataset>/<GNN>.pt`（或联合训练对应文件）。

`--gnn` 同时接受论文名（`GraphSAGE`、`GCNII`、`ASDGN`、`ACM-GNN`、`GraphSAINT`）与库内名（`SAGE`、`GCN2`、`ASC`、`ACMGNN`、`Saint`）。

## 论文行与命令行

| 论文行 | `--pipeline` | 设定 |
|---|---|---|
| TA+LM | `ta_lm` | 在原始标题/摘要上微调 LM |
| GraphICL+LM | `grapicl_lm` | 在 GraphICL 风格邻域文本上微调 LM |
| TAPTN+LM | `taptn_lm` | 在 TAPTN 增强文本上微调 LM |
| TA+GNN | `ta_gnn` | 冻结 TA 嵌入上的 GNN |
| Joint | `joint` | 编码器与 GNN 端到端联合训练 |
| *（可迁移性对照）* | `taptn_gnn` | 与 TA+GNN 相同的冻结 GNN，输入换成 TAPTN 嵌入 |

WebKB 表没有 GraphICL+LM 行。DGI 与 GraphSAINT 没有联合训练变体。RoBERTa 全表含 RevGAT，不含 DGI、GraphSAINT、ASDGN。

## 记录结果

- [results/cells.md](results/cells.md) — 逐格、run 1–5、记录值与论文值
- [results/DISCREPANCIES.md](results/DISCREPANCIES.md) — 记录值与论文表不一致处
- [results/official_cells.json](results/official_cells.json) — `dry` 使用的机器可读格子（内部流水线代号为 `P1_LM` / `P2` / `P3` / `P4` / `P5`；本 README 与 `cells.md` 使用论文行名）

## 引用

```bibtex
@article{wang2026llms,
  title     = {{LLM}s Can Leverage Graph Structural Information in Text-Attributed Graphs},
  author    = {Hongyi Wang and Shikui Tu and Lei Xu},
  journal   = {Transactions on Machine Learning Research},
  issn      = {2835-8856},
  year      = {2026},
  url       = {https://openreview.net/forum?id=WhaVqEkkMY}
}
```

## 许可

本复现包供研究使用。使用请引用上述论文。
