# TAPTN 重连实验复现

用于复现论文中的**邻域重连（rewiring）**表格与配图：

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)

English: [README.md](README.md).

本包覆盖第 2 节（翻转/极端重连、能力–敏感性回归、无标签对照）以及附录 G–H（新生代无标签重连、TAPTN 内部结构通道消融）。

论文级总览：[../README_zh.md](../README_zh.md)。另外两章是同级目录：[`TAPTN_icl_repro`](../TAPTN_icl_repro)、[`TAPTN_finetune_repro`](../TAPTN_finetune_repro)。

## 环境

```bash
pip install -r requirements.txt
```

拼表与从 pickle 重算准确率只需 CPU，但需要安装 **PyTorch Geometric**（pickle 内含 `torch_geometric.data.Data`）。`run` 实时推理需要 OpenAI 兼容 API。

## 数据

pickle、WebKB 网页、摘要、原始 CSV 和相机就绪图**不在**代码仓库里。请解压资源包并把路径指向该目录：

```
https://PLACEHOLDER.example/TAPTN_rewiring_repro_assets
```

推荐目录布局（在论文总目录内）：

```
TAPTN_paper_repro/
  TAPTN_rewiring_repro/           # 本仓库
  TAPTN_rewiring_repro_assets/    # 解压后的资源包
```

```bash
export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets
```

未设置时，`reproduce.py` 会查找同级 `TAPTN_rewiring_repro_assets/` 或 `./assets/`。

## 一键干运行（不调用 LLM）

```bash
export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets
python reproduce.py check
python reproduce.py dry --table all --rescore --figures
# 对照修订前 PDF：
python reproduce.py dry --table all --rescore --paper-compat
# 或
bash run_dry.sh
```

`dry` 打印**修订后**的相机就绪表格。`--rescore` 从保存的 pickle 重算每个格子（仍不调用 API）。`--figures` 把图写到 `output/figures/`。`--paper-compat` 对照修订前 PDF（含无标签 GPT-3.5 Wisconsin 原始 42.04）。示意图（图 1、图 4、图 11 / `former_rewire2.pdf`）按论文原件拷贝；散点/热力图由 pickle 重新绘制，与 PDF 像素不必一致。

## 论文表格

PDF 编号与 TMLR 2026 相机就绪稿一致。`--table` 是命令行参数（沿用历史 TeX label 的短别名）。

| PDF | `--table` | 内容 |
|---|---|---|
| 表 1 | `tb_1` | WebKB 与引文图同质性 |
| 表 2 | `tb_2` | 7 模型 × 4 WebKB × 翻转 / 极端 |
| 表 3 | `tb_3` | GPT-3.5-Turbo-0125 + 逐步指令，仅翻转 |
| 表 4 | `rewiring_stats` | 能力–敏感性回归 |
| 表 5 | `nolabel_main` | 无标签翻转，7 模型平均 |
| 表 6 | `nolabel_stats` | 无标签能力–敏感性 |
| 表 26 | `current_rewire` | 四个新生代模型的无标签翻转 |
| 表 29 | `taptn_structcorrupt` | TAPTN 内部去边 / 翻转 Δ |

配图：图 2、图 3、附录图 8–10，以及静态图 1 / 图 4。

## 实时重跑（LLM API）

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # 可选
export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets

python reproduce.py run --table tb_2 --hop 1 --model phi-4 --dataset cornell
python reproduce.py run --table tb_2 --hop 1 --model phi-4 --dataset cornell --rewired
python reproduce.py run --table tb_2 --hop 2 --model phi-4 --dataset texas --rewired
python reproduce.py run --table nolabel --model llama-3.3-70b-instruct --dataset wisconsin --rewired
python reproduce.py run --table tb_3 --dataset cornell
```

结果写到 `output/rerun/`（可用 `--output-dir` 覆盖）。实时推理有随机性，不会与保存的 pickle 逐字节一致。

表 29 在本包中仅支持干运行（完整 TAPTN 第二轮管线不在此包）。

## 计分约定

第 2 节开源模型 pickle 的 `results` 列表与未打乱的 `test_mask` 对齐；跳过 `None`，匹配规则与 `analyze_rewiring_lmarena.py` 相同。

GPT-3.5 的 Table 2 使用当年笔记中的 `wrong_index` / `result` 约定（`pkls/gpt35/` 下的旧文件名）。Table 3 对 `pkls/tb3/` 使用同一规则。

无标签表中的 Qwen 行是 **Qwen2.5-VL-72B**，不要与同分（LMArena 1302）的非 VL `qwen-2.5-72b-instruct` 文件混淆。

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
