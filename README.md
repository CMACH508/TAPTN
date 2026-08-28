# TAPTN paper reproduction

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)
> · [Code](https://github.com/CMACH508/TAPTN)

中文总览：[README_zh.md](README_zh.md).

This directory groups the TMLR 2026 camera-ready **experimental tables** into three packages. Code and assets are separate: each package has a sibling `*_repro_assets/` bundle. API keys are read from the environment only and are not stored in vendor code.

| Package | Paper chapter | Dry-run | Code | Assets (approx.) | Download |
|---|---|---|---|---:|---|
| [TAPTN_rewiring_repro](TAPTN_rewiring_repro/README.md) | Section 2 neighborhood rewiring + Appendices G–H (label-free, current models, TAPTN-internal structural channel) | rescore pickles; redraw scatter/heatmaps | ~6 MB | 1.4 GB | [Google Drive](https://drive.google.com/drive/folders/1qQeo2Fy8_snk08b3jWFYgBmKyme_65oj?usp=sharing) |
| [TAPTN_icl_repro](TAPTN_icl_repro/README.md) | Section 3 TAPTN / GraphICL zero-shot ICL | rescore pickles; copy method figures | ~0.8 MB | 2.5 GB | [Google Drive](https://drive.google.com/drive/folders/1WGKDbH9sueIQat-UUJH7ZUlgdSwpJeu-?usp=sharing) |
| [TAPTN_finetune_repro](TAPTN_finetune_repro/README.md) | Section 4 TAPTN+LM vs GNN | reprint recorded cells; rescore LM `.pred` | ~1 MB | 92 GB | [Google Drive](https://drive.google.com/drive/folders/1wIlc-r7HFd31YhQLGjc0uY0-Xy3qfUVZ?usp=sharing) |

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

PDF numbers follow the TMLR 2026 camera-ready. `--table` is the CLI flag (kept as a short alias for the historical TeX label; the ICL package also accepts a `tab:` prefix).

### Section 2 and Appendices G–H: rewiring (`TAPTN_rewiring_repro`)

| PDF | `--table` | Contents |
|---|---|---|
| Table 1 | `tb_1` | Homophily of WebKB and citation graphs (citation-graph cells are recorded paper values; those graphs are not shipped) |
| Table 2 | `tb_2` | 7 models × 4 WebKB × flipping / extreme |
| Table 3 | `tb_3` | GPT-3.5 + step-by-step instructions, flipping (Cornell / Texas pickles only; Washington / Wisconsin missing) |
| Table 4 | `rewiring_stats` | Capability–sensitivity regression |
| Table 5 | `nolabel_main` | Label-free flipping, 7-model averages |
| Table 6 | `nolabel_stats` | Label-free capability–sensitivity |
| Table 26 | `current_rewire` | Four current-generation models, label-free flipping |
| Table 29 | `taptn_structcorrupt` | TAPTN-internal edge-blind / flip Δ (dry-run only) |

Figures: Figure 2, Figure 3, and Appendix Figures 8–10 are regenerated from pickles; Figure 1 and Figure 4 are copied camera-ready illustrations.

Rewiring numbers include the camera-ready errata. To compare against the unrevised PDF: `python reproduce.py dry --table all --rescore --paper-compat`.

### Section 3: zero-shot ICL (`TAPTN_icl_repro`)

| PDF | `--table` | Contents |
|---|---|---|
| Table 8 | `main_5_datasets` | 0-hop / GraphICL+SAT / TAPTN on five graphs |
| Table 9 | `factorial` | 2-hop SAT × instructions × aggregation |
| Table 10 | `product_70b` | ogbn-products, 400 nodes, Llama-3.3-70B |
| Table 14 | `cost` | tokens / OpenRouter $ / accuracy |
| Table 15 | `tb_52` | budget backbone (8B iter-1 + 70B refine) |
| Table 18 | `tb_4` | GraphICL self-reflection vs TAPTN |
| Table 19 | `decouple` | SAT × instructions, no aggregation |
| Table 27 | `current_channel` | Texas structural channels, four models |
| Table 28 | `current_taptn` | 2-hop TAPTN vs GraphICL+SAT (Texas + Cora) |

Illustrations: Figure 5 (`TAPTN_overview.pdf`), Figure 6 (`TAPTN_mp.pdf`). Cora 2-hop TAPTN gating: `python reproduce.py gate` (no LLM).

The full ogbn-products dump (~7 GB including `Amazon-3M.raw`) is **not** shipped. Dry-run of the three products tables uses the shipped pickles and `cost_probe/` JSON only.

### Section 4: fine-tuning vs GNN (`TAPTN_finetune_repro`)

| PDF | `--table` | Contents |
|---|---|---|
| Table 11 | `gnn_summary` | How many pipelines significantly exceed TAPTN+LM |
| Table 12 | `p5_transfer` / `transfer` | TAPTN embeddings + frozen GNN (transferability counts) |
| Table 13 | `crosslm_encoder` | DeBERTa vs RoBERTa |
| Table 20 | `tb_6` | Frozen TA+GNN, GraphICL+LM, TAPTN+LM (homophilic) |
| Table 21 | `heterophilic` | Frozen TA+GNN, TAPTN+LM (WebKB) |
| Table 22 | `joint` | Jointly trained encoder+GNN |
| Table 23 | `roberta` | RoBERTa-base encoder |

Each cell is five runs (run 1–5), mean±std. **P5** in Table 12 is a *column* abbreviation, not a row name in the main comparison tables.

### Paper objects that are not dry-run cells

| PDF | Note |
|---|---|
| Table 7 | TAPTN symbol table |
| Algorithm 1 | Pseudocode in the paper |
| Table 16 | Dataset sizes and splits (descriptive) |
| Tables 24 / 25 | GNN architectures and training hyperparameters |
| Table 17 | Neighbourhood summary for the appendix worked example |
| Figure 7 | `GNNPipe.pdf` pipeline illustration (not in the fine-tune asset bundle) |

## Environment

Assembling tables and re-scoring pickles / `.pred` files needs **PyTorch Geometric** (many pickles store `torch_geometric.data.Data`). Any Python with PyG on `PATH` is enough:

```bash
python -c "import torch_geometric"
```

`requirements.txt` differs slightly: rewiring/ICL need `openai`; fine-tuning also needs `transformers`, `dgl`, and `ogb`. Live LLM calls need `OPENAI_API_KEY` or `TAPTN_LLM_API_KEY` (optional `OPENAI_BASE_URL`). Fine-tune `train` needs a GPU.

## Dry-run everything (no LLM, no training)

From this directory (with a PyG-capable `python` on `PATH`):

```bash
bash run_all_dry.sh
```

If `python` is not that interpreter, set `PYTHON` first (`export PYTHON=$(which python)` after activating your env).

Or package by package (each with **its own** asset directory):

```bash
export TAPTN_ASSETS="$PWD/TAPTN_rewiring_repro_assets"
python TAPTN_rewiring_repro/reproduce.py check
python TAPTN_rewiring_repro/reproduce.py dry --table all --rescore --figures

export TAPTN_ASSETS="$PWD/TAPTN_icl_repro_assets"
python TAPTN_icl_repro/reproduce.py check
python TAPTN_icl_repro/reproduce.py dry --table all --rescore --figures

export TAPTN_ASSETS="$PWD/TAPTN_finetune_repro_assets"
python TAPTN_finetune_repro/reproduce.py check
python TAPTN_finetune_repro/reproduce.py dry --table all --rescore-lm
```

## Scoring (do not mix)

- **Rewiring open models:** `results` aligned to the unshuffled `test_mask`; skip `None`.
- **Rewiring GPT-3.5 and old ICL (`*2_*` / `arxiv2023_*`):** `1 - |wrong_index|/|result|`.
- **Current-generation ICL pickles:** stored `accuracy` (Cora TAPTN: `acc_gate`; Cora GraphICL: `accuracy_reextract`).
- **Fine-tune LM:** saved `.pred`. Frozen/joint GNN `.pt` files are overwritten by later jobs; dry-run uses recorded logs, not the file currently on disk.

Per-cell diffs and missing artifacts: each package’s `results/DISCREPANCIES.md`. Several Cora GPT-3.5 ICL cells (0-hop 59.40, GraphICL 1/2-hop, TAPTN 1-hop, self-reflection), Texas 0-hop 66.93, and Cornell TAPTN 1-hop 85.08 have **no matching pickle**; the old TMLR supplementary material does not contain those result files either.

## Live re-run

See each package README for commands and limits. Table 29 is dry-run only. A live products re-run needs the official OGB dump.

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
