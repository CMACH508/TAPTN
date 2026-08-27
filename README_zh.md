# TAPTN 论文复现总览

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)
> · [Code](https://github.com/CMACH508/LLMsCanLeverageGraphStructure)

English: [README.md](README.md).

本目录把 TMLR 2026 相机就绪稿的**实验表格**拆成三个互不混用的复现包。代码与资源分离：每个包旁边有独立的 `*_repro_assets/`。密钥只从环境变量读取，不写进 vendor。

| 包 | 论文章节 | 干运行做什么 | 代码 | 资源（约） |
|---|---|---|---|---:|
| [TAPTN_rewiring_repro](TAPTN_rewiring_repro/README_zh.md) | §2 邻域重连 + 附录无标签新生代 + TAPTN 内部结构通道 | pickle 重算准确率；重绘散点/热力图 | ~6 MB | 1.4 GB |
| [TAPTN_icl_repro](TAPTN_icl_repro/README_zh.md) | TAPTN / GraphICL 零样本 ICL | pickle 重算；拷贝方法示意图 | ~0.8 MB | 2.5 GB |
| [TAPTN_finetune_repro](TAPTN_finetune_repro/README_zh.md) | TAPTN+LM vs GNN | 记录格子拼表；LM `.pred` 重算 | ~1 MB | 92 GB |

不要把三个 `TAPTN_ASSETS` 指到同一个目录，也不要混用计分协议。

## 推荐布局

```
TAPTN_paper_repro/                 # 本目录（总览）
  README.md
  README_zh.md
  run_all_dry.sh
  TAPTN_rewiring_repro/            # §2 代码
  TAPTN_rewiring_repro_assets/
  TAPTN_icl_repro/                 # ICL 代码
  TAPTN_icl_repro_assets/
  TAPTN_finetune_repro/            # 微调代码
  TAPTN_finetune_repro_assets/
```

每个 `reproduce.py` 在未设置 `TAPTN_ASSETS` 时，会找**同级**的 `*_repro_assets/`。

## 论文表格 → 包

`--table` 与相机就绪稿 `\label{...}` 一致（ICL 包可省略 `tab:` 前缀）。

### 第 2 节与附录：重连（`TAPTN_rewiring_repro`）

| 论文 label | `--table` | 内容 |
|---|---|---|
| `tb_1` | `tb_1` | WebKB / 引文图同质性（引文图三格为论文记录值，图未随包） |
| `tb_2` | `tb_2` | 7 模型 × 4 WebKB × 翻转 / 极端 |
| `tb_3` | `tb_3` | GPT-3.5 + 逐步指令，仅翻转（Cornell / Texas 有 pickle；Washington / Wisconsin 无） |
| `tab:rewiring_stats` | `rewiring_stats` | 能力–敏感性回归 |
| `tab:nolabel_main` | `nolabel_main` | 无标签翻转，7 模型平均 |
| `tab:nolabel_stats` | `nolabel_stats` | 无标签能力–敏感性 |
| `tab:current_rewire` | `current_rewire` | 四个新生代模型的无标签翻转 |
| `tab:taptn_structcorrupt` | `taptn_structcorrupt` | TAPTN 内部去边 / 翻转 Δ（仅干运行） |

配图：`fig:hop1`、`fig:hop2`、附录 `fig:nolabel_*` 由 pickle 重绘；`fig_1`、`fig:case` 为相机就绪示意图拷贝。

重连表已相对未修订 PDF 做过勘误。对照旧数字：`python reproduce.py dry --table all --rescore --paper-compat`。

### TAPTN 方法章：零样本 ICL（`TAPTN_icl_repro`）

| 论文 label | `--table` | 内容 |
|---|---|---|
| `tab:main_5_datasets` | `main_5_datasets` | 0-hop / GraphICL+SAT / TAPTN，五数据集 |
| `tab:factorial` | `factorial` | 2-hop SAT × 指令 × 聚合 |
| `tab:product_70b` | `product_70b` | ogbn-products，400 节点，Llama-3.3-70B |
| `tab:cost` | `cost` | token / OpenRouter 费用 / 准确率 |
| `tb_52` | `tb_52` | 预算骨干（8B 首轮 + 70B 精炼） |
| `tb_4` | `tb_4` | GraphICL 自反思 vs TAPTN |
| `tab:decouple` | `decouple` | SAT × 指令，无聚合 |
| `tab:current_channel` | `current_channel` | Texas 结构通道，四模型 |
| `tab:current_taptn` | `current_taptn` | 2-hop TAPTN vs GraphICL+SAT（Texas + Cora） |

示意图：`fig:taptn_overview`（`TAPTN_overview.pdf`）、`fig_2`（`TAPTN_mp.pdf`）。Cora 2-hop TAPTN 的门控：`python reproduce.py gate`（不调 LLM）。

ogbn-products 全图（含 `Amazon-3M.raw`，约 7 GB）**未打包**。三张 products 表的干运行只用随包 pickle 与 `cost_probe/` JSON。

### 微调 vs GNN（`TAPTN_finetune_repro`）

| 论文 label | `--table` | 内容 |
|---|---|---|
| `tab:gnn_summary` | `gnn_summary` | 显著超过 TAPTN+LM 的流水线数量 |
| `tab:p5_transfer` | `p5_transfer` / `transfer` | TAPTN 嵌入 + 冻结 GNN（可迁移性计数） |
| `tab:crosslm_encoder` | `crosslm_encoder` | DeBERTa vs RoBERTa |
| `tb_6` | `tb_6` | 冻结 TA+GNN、GraphICL+LM、TAPTN+LM（同配图） |
| `tab:heterophilic_full` | `heterophilic` | 冻结 TA+GNN、TAPTN+LM（WebKB） |
| `tab:joint_gnn` | `joint` | 编码器与 GNN 联合训练 |
| `tab:roberta_full` | `roberta` | RoBERTa-base 编码器 |

每个格子为 5 次运行（run 1–5），mean±std。`tab:p5_transfer` 的 **P5** 是该表列缩写，不是主对照表行名。

### 不在三个实验包里干运行的论文对象

这些是正文/附录里的说明或超参表，不是 pickle 格子：

| label | 说明 |
|---|---|
| `tab:notation` | TAPTN 符号表 |
| `alg:taptn` | 伪代码，在 tex 中 |
| `tb_10a` | 数据集规模与划分（描述性） |
| `tb_7a` / `tb_8a` | GNN 结构与训练超参 |
| `tb_worked_nbr` | 附录 worked example 的邻域摘要 |
| `fig_3a` | `GNNPipe.pdf` 流水线示意图（未随微调资源包） |

## 环境

拼表与 pickle / `.pred` 重算需要 **PyTorch Geometric**（不少 pickle 含 `torch_geometric.data.Data`）。作者环境：

```bash
/home/wanghongyi/.conda/envs/gnn-llm/bin/python
```

各包 `requirements.txt` 略有不同：重连/ICL 需要 `openai`；微调还需要 `transformers`、`dgl`、`ogb`。实时 LLM 调用需要 `OPENAI_API_KEY` 或 `TAPTN_LLM_API_KEY`（可选 `OPENAI_BASE_URL`）。微调 `train` 需要 GPU。

## 一键干运行（不调用 LLM、不训练）

在本目录：

```bash
export PYTHON=/home/wanghongyi/.conda/envs/gnn-llm/bin/python   # 若默认 python 无 PyG
bash run_all_dry.sh
```

或逐包（每个包用**自己的**资源目录）：

```bash
export PYTHON=/home/wanghongyi/.conda/envs/gnn-llm/bin/python

export TAPTN_ASSETS="$PWD/TAPTN_rewiring_repro_assets"
"$PYTHON" TAPTN_rewiring_repro/reproduce.py check
"$PYTHON" TAPTN_rewiring_repro/reproduce.py dry --table all --rescore --figures

export TAPTN_ASSETS="$PWD/TAPTN_icl_repro_assets"
"$PYTHON" TAPTN_icl_repro/reproduce.py check
"$PYTHON" TAPTN_icl_repro/reproduce.py dry --table all --rescore --figures

export TAPTN_ASSETS="$PWD/TAPTN_finetune_repro_assets"
"$PYTHON" TAPTN_finetune_repro/reproduce.py check
"$PYTHON" TAPTN_finetune_repro/reproduce.py dry --table all --rescore-lm
```

## 计分（不要混用）

- **重连开源模型：** `results` 与未打乱 `test_mask` 对齐，跳过 `None`。
- **重连 GPT-3.5 与旧 ICL（`*2_*` / `arxiv2023_*`）：** `1 - |wrong_index|/|result|`。
- **新生代 ICL pickle：** 文件内 `accuracy`（Cora TAPTN 用 `acc_gate`，Cora GraphICL 用 `accuracy_reextract`）。
- **微调 LM：** 保存的 `.pred`；冻结/联合 GNN 的 `.pt` 会被后一次训练覆盖，干运行用记录日志而非当前磁盘权重。

逐格对照与无法复现项写在各包的 `results/DISCREPANCIES.md`。ICL 包中若干 Cora GPT-3.5 格子（0-hop 59.40、GraphICL 1/2-hop、TAPTN 1-hop、自反思）以及 Texas 0-hop 66.93、Cornell TAPTN 1-hop 85.08 **没有匹配 pickle**；TMLR 旧补充材料也不含这些结果文件。

## 实时重跑

命令与限制见各包 README。`tab:taptn_structcorrupt` 仅干运行。products 实时重跑需自行安装 OGB 官方数据。

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
