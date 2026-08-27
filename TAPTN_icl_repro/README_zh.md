# TAPTN ICL 复现

用于复现论文中的 **零样本 ICL / TAPTN** 表格：

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)

English: [README.md](README.md).

本包覆盖 TAPTN 方法章节：五数据集主表、析因/解耦消融、自反思对照、ogbn-products 可扩展性/成本/预算骨干，以及新生代 Texas 结构通道与 Cora/Texas TAPTN 面板。

论文级总览：[../README_zh.md](../README_zh.md)。

**不在本包**

- 第 2 节邻域重连，以及 TAPTN 内部 edge-blind/flip（`tab:taptn_structcorrupt`）→ [`TAPTN_rewiring_repro`](../TAPTN_rewiring_repro)
- 微调 vs GNN 表 → [`TAPTN_finetune_repro`](../TAPTN_finetune_repro)

## 环境

```bash
pip install -r requirements.txt
```

拼表与从 pickle 重算准确率只需 CPU，但需要安装 **PyTorch Geometric**（不少 pickle 内含 `torch_geometric.data.Data`）。`run` 实时推理需要 OpenAI 兼容 API。

建议使用带 PyG 的 conda 环境（作者环境：`gnn-llm`）。

## 数据

pickle、WebKB 网页、Cora / arXiv-2023 图、成本探测 JSON 和相机就绪图**不在**代码仓库里。请解压资源包并把路径指向该目录：

```
https://PLACEHOLDER.example/TAPTN_icl_repro_assets
```

推荐目录布局（在论文总目录内）：

```
TAPTN_paper_repro/
  TAPTN_icl_repro/           # 本仓库
  TAPTN_icl_repro_assets/    # 解压后的资源包
```

```bash
export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets
```

未设置时，`reproduce.py` 会查找同级 `TAPTN_icl_repro_assets/` 或 `./assets/`。

ogbn-products（含 `Amazon-3M.raw` 约 7 GB）**未打包**。`tab:product_70b` / `tab:cost` / `tb_52` 的干运行只需要已随包的 pickle 与 `cost_probe/` JSON。若要实时重跑 products，需自行安装 OGB 官方数据。

## 一键干运行（不调用 LLM）

```bash
export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets
python reproduce.py check
python reproduce.py dry --table all --rescore --figures
# 或
bash run_dry.sh
```

`dry` 打印相机就绪表格。`--rescore` 从保存的 pickle 重算每个格子（仍不调用 API）。`--figures` 把 `fig:taptn_overview` 与 `fig_2` 拷到 `output/figures/`。

## 论文表格

`--table` 使用相机就绪稿的 `\label{...}`（`tab:` 前缀可省略）：

| `--table` | 论文 label | 内容 |
|---|---|---|
| `main_5_datasets` | `tab:main_5_datasets` | 0-hop / GraphICL+SAT / TAPTN，五数据集 |
| `factorial` | `tab:factorial` | 2-hop SAT × 指令 × 聚合 |
| `product_70b` | `tab:product_70b` | ogbn-products，400 节点，Llama-3.3-70B |
| `cost` | `tab:cost` | token / OpenRouter 费用 / 准确率 |
| `tb_52` | `tb_52` | 预算骨干（8B 首轮 + 70B 精炼） |
| `tb_4` | `tb_4` | GraphICL 自反思 vs TAPTN |
| `decouple` | `tab:decouple` | SAT × 指令，无聚合 |
| `current_channel` | `tab:current_channel` | Texas 结构通道，四模型 |
| `current_taptn` | `tab:current_taptn` | 2-hop TAPTN vs GraphICL+SAT（Texas + Cora） |

计分协议**不能混用**：

- 旧 GPT-3.5 / WebKB `*2_*` / `arxiv2023_*`：`1 - |wrong_index|/|result|`
- 新式 `{ds}_hop{h}_{anon|noanon}_{guide|noguide}_{model}.pkl`：文件内 `accuracy`
- Cora 新生代 TAPTN：`acc_gate`（无参数邻居共识）
- Cora 新生代 GraphICL：`accuracy_reextract`
- products TAPTN 1-hop：hop-1 错误集与 400 测试节点求交（论文 83.75，不是 1430 节点上的 82.66）

找不到原始 pickle 的格子见 `results/DISCREPANCIES.md`。

## 实时重跑（LLM API）

密钥只从环境变量读取（源码中不含密钥）：

```bash
export OPENAI_API_KEY=...          # 或 TAPTN_LLM_API_KEY
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # 可选
export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets
```

新生代配置走 `vendor/run_taptn_expansion.py`：

```bash
python reproduce.py run --dataset texas --model gemma-4-31b-it --config taptn1
python reproduce.py run --dataset texas --model glm-5.1 --config taptn2 --wis_ver v2_2hop
python reproduce.py run --dataset texas --model llama-3.3-70b --config ego
```

配置：`ego`（0-hop）、`graphicl1`、`taptn1`、`graphicl2`、`taptn2`。`--anon` 去掉边方向。

论文里 Cora 2-hop TAPTN **不是**裸 iter-2，而是无参数全邻居多数票门控。可用已随包的源 pickle 重算（不调 LLM）：

```bash
python reproduce.py gate
```

## 记录结果

- [results/cells.md](results/cells.md) — 论文格子
- [results/DISCREPANCIES.md](results/DISCREPANCIES.md) — 干运行与论文的差异
- [results/official_cells.json](results/official_cells.json) — 机器可读的论文数字

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
