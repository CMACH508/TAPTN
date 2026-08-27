# TAPTN paper reproduction

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)
> · [Code](https://github.com/CMACH508/LLMsCanLeverageGraphStructure)

中文总览：[README_zh.md](README_zh.md).

This directory groups the TMLR 2026 camera-ready **experimental tables** into three packages. Code and assets are separate: each package has a sibling `*_repro_assets/` bundle. API keys are read from the environment only and are not stored in vendor code.

| Package | Paper chapter | Dry-run | Code | Assets (approx.) |
|---|---|---|---|---:|
| [TAPTN_rewiring_repro](TAPTN_rewiring_repro/README.md) | §2 neighborhood rewiring + appendix label-free current models + TAPTN-internal structural channel | rescore pickles; redraw scatter/heatmaps | ~6 MB | 1.4 GB |
| [TAPTN_icl_repro](TAPTN_icl_repro/README.md) | TAPTN / GraphICL zero-shot ICL | rescore pickles; copy method figures | ~0.8 MB | 2.5 GB |
| [TAPTN_finetune_repro](TAPTN_finetune_repro/README.md) | TAPTN+LM vs GNN | reprint recorded cells; rescore LM `.pred` | ~1 MB | 92 GB |

Do not point all three `TAPTN_ASSETS` variables at the same folder. Do not mix scoring protocols.

## Layout

```
TAPTN_paper_repro/                 # this hub
  README.md
  README_zh.md
  run_all_dry.sh
  TAPTN_rewiring_repro/
  TAPTN_rewiring_repro_assets/
  TAPTN_icl_repro/
  TAPTN_icl_repro_assets/
  TAPTN_finetune_repro/
  TAPTN_finetune_repro_assets/
```

If `TAPTN_ASSETS` is unset, each `reproduce.py` looks for a **sibling** `*_repro_assets/` directory.

## Paper tables → package

`--table` follows the camera-ready `\label{...}` (the ICL package also accepts the name without a `tab:` prefix).

### Section 2 / appendix: rewiring (`TAPTN_rewiring_repro`)

| Paper label | `--table` | Contents |
|---|---|---|
| `tb_1` | `tb_1` | Homophily of WebKB and citation graphs (citation-graph cells are recorded paper values; those graphs are not shipped) |
| `tb_2` | `tb_2` | 7 models × 4 WebKB × flipping / extreme |
| `tb_3` | `tb_3` | GPT-3.5 + step-by-step instructions, flipping (Cornell / Texas pickles only; Washington / Wisconsin missing) |
| `tab:rewiring_stats` | `rewiring_stats` | Capability–sensitivity regression |
| `tab:nolabel_main` | `nolabel_main` | Label-free flipping, 7-model averages |
| `tab:nolabel_stats` | `nolabel_stats` | Label-free capability–sensitivity |
| `tab:current_rewire` | `current_rewire` | Four current-generation models, label-free flipping |
| `tab:taptn_structcorrupt` | `taptn_structcorrupt` | TAPTN-internal edge-blind / flip Δ (dry-run only) |

Figures: `fig:hop1`, `fig:hop2`, appendix `fig:nolabel_*` are regenerated from pickles; `fig_1` and `fig:case` are copied camera-ready illustrations.

Rewiring numbers include the camera-ready errata. To compare against the unrevised PDF: `python reproduce.py dry --table all --rescore --paper-compat`.

### TAPTN method chapter: zero-shot ICL (`TAPTN_icl_repro`)

| Paper label | `--table` | Contents |
|---|---|---|
| `tab:main_5_datasets` | `main_5_datasets` | 0-hop / GraphICL+SAT / TAPTN on five graphs |
| `tab:factorial` | `factorial` | 2-hop SAT × instructions × aggregation |
| `tab:product_70b` | `product_70b` | ogbn-products, 400 nodes, Llama-3.3-70B |
| `tab:cost` | `cost` | tokens / OpenRouter $ / accuracy |
| `tb_52` | `tb_52` | budget backbone (8B iter-1 + 70B refine) |
| `tb_4` | `tb_4` | GraphICL self-reflection vs TAPTN |
| `tab:decouple` | `decouple` | SAT × instructions, no aggregation |
| `tab:current_channel` | `current_channel` | Texas structural channels, four models |
| `tab:current_taptn` | `current_taptn` | 2-hop TAPTN vs GraphICL+SAT (Texas + Cora) |

Illustrations: `fig:taptn_overview` (`TAPTN_overview.pdf`), `fig_2` (`TAPTN_mp.pdf`). Cora 2-hop TAPTN gating: `python reproduce.py gate` (no LLM).

The full ogbn-products dump (~7 GB including `Amazon-3M.raw`) is **not** shipped. Dry-run of the three products tables uses the shipped pickles and `cost_probe/` JSON only.

### Fine-tuning vs GNN (`TAPTN_finetune_repro`)

| Paper label | `--table` | Contents |
|---|---|---|
| `tab:gnn_summary` | `gnn_summary` | How many pipelines significantly exceed TAPTN+LM |
| `tab:p5_transfer` | `p5_transfer` / `transfer` | TAPTN embeddings + frozen GNN (transferability counts) |
| `tab:crosslm_encoder` | `crosslm_encoder` | DeBERTa vs RoBERTa |
| `tb_6` | `tb_6` | Frozen TA+GNN, GraphICL+LM, TAPTN+LM (homophilic) |
| `tab:heterophilic_full` | `heterophilic` | Frozen TA+GNN, TAPTN+LM (WebKB) |
| `tab:joint_gnn` | `joint` | Jointly trained encoder+GNN |
| `tab:roberta_full` | `roberta` | RoBERTa-base encoder |

Each cell is five runs (run 1–5), mean±std. **P5** in `tab:p5_transfer` is a *column* abbreviation, not a row name in the main comparison tables.

### Paper objects that are not dry-run cells

| label | Note |
|---|---|
| `tab:notation` | TAPTN symbol table |
| `alg:taptn` | Pseudocode in the tex |
| `tb_10a` | Dataset sizes and splits (descriptive) |
| `tb_7a` / `tb_8a` | GNN architectures and training hyperparameters |
| `tb_worked_nbr` | Neighbourhood summary for the appendix worked example |
| `fig_3a` | `GNNPipe.pdf` pipeline illustration (not in the fine-tune asset bundle) |

## Environment

Assembling tables and re-scoring pickles / `.pred` files needs **PyTorch Geometric** (many pickles store `torch_geometric.data.Data`). Author env:

```bash
/home/wanghongyi/.conda/envs/gnn-llm/bin/python
```

`requirements.txt` differs slightly: rewiring/ICL need `openai`; fine-tuning also needs `transformers`, `dgl`, and `ogb`. Live LLM calls need `OPENAI_API_KEY` or `TAPTN_LLM_API_KEY` (optional `OPENAI_BASE_URL`). Fine-tune `train` needs a GPU.

## Dry-run everything (no LLM, no training)

From this directory:

```bash
export PYTHON=/home/wanghongyi/.conda/envs/gnn-llm/bin/python   # if default python lacks PyG
bash run_all_dry.sh
```

Or package by package (each with **its own** asset directory):

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

## Scoring (do not mix)

- **Rewiring open models:** `results` aligned to the unshuffled `test_mask`; skip `None`.
- **Rewiring GPT-3.5 and old ICL (`*2_*` / `arxiv2023_*`):** `1 - |wrong_index|/|result|`.
- **Current-generation ICL pickles:** stored `accuracy` (Cora TAPTN: `acc_gate`; Cora GraphICL: `accuracy_reextract`).
- **Fine-tune LM:** saved `.pred`. Frozen/joint GNN `.pt` files are overwritten by later jobs; dry-run uses recorded logs, not the file currently on disk.

Per-cell diffs and missing artifacts: each package’s `results/DISCREPANCIES.md`. Several Cora GPT-3.5 ICL cells (0-hop 59.40, GraphICL 1/2-hop, TAPTN 1-hop, self-reflection), Texas 0-hop 66.93, and Cornell TAPTN 1-hop 85.08 have **no matching pickle**; the old TMLR supplementary material does not contain those result files either.

## Live re-run

See each package README for commands and limits. `tab:taptn_structcorrupt` is dry-run only. A live products re-run needs the official OGB dump.

## Citation

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
