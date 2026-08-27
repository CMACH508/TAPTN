# TAPTN ICL Reproduction

Code to reproduce the **zero-shot ICL / TAPTN** tables from

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)

中文说明：[README_zh.md](README_zh.md).

This package covers the TAPTN method chapter: main five-dataset table, factorial and decouple ablations, self-reflection control, ogbn-products scalability / cost / budget-backbone, and the current-generation Texas channel + Cora/Texas TAPTN panels.

Paper-level index: [../README.md](../README.md).

**Not in this bundle**

- Neighborhood rewiring (Section 2) and TAPTN-internal edge-blind/flip (`tab:taptn_structcorrupt`) → [`TAPTN_rewiring_repro`](../TAPTN_rewiring_repro)
- Fine-tuning vs GNN tables → [`TAPTN_finetune_repro`](../TAPTN_finetune_repro)

## Setup

```bash
pip install -r requirements.txt
```

Assembling tables and re-scoring pickles runs on CPU but needs **PyTorch Geometric** (many pickles store `torch_geometric.data.Data`). `run` needs an API key.

Recommended Python: a conda env with PyG (the author env is `gnn-llm`).

## Data

Pickles, WebKB HTML, Cora / arXiv-2023 graphs, cost-probe JSON, and camera-ready figures are **not** in the code repository. Download the asset bundle, unpack it, and point the code at that folder:

```
https://PLACEHOLDER.example/TAPTN_icl_repro_assets
```

Recommended layout (inside the paper hub):

```
TAPTN_paper_repro/
  TAPTN_icl_repro/           # this repository
  TAPTN_icl_repro_assets/    # unpacked bundle
```

```bash
export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets
```

If `TAPTN_ASSETS` is unset, `reproduce.py` looks for a sibling `TAPTN_icl_repro_assets/` directory or `./assets/`.

ogbn-products (~7 GB including `Amazon-3M.raw`) is **not** shipped. Dry-run of `tab:product_70b` / `tab:cost` / `tb_52` only needs the shipped pickles and `cost_probe/` JSON. A live products re-run requires the official OGB dump.

## Quick start (no LLM calls)

```bash
export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets
python reproduce.py check
python reproduce.py dry --table all --rescore --figures
# or
bash run_dry.sh
```

`dry` reprints the camera-ready tables. `--rescore` recomputes every cell from the saved pickles (still no GPU / no API). `--figures` copies `fig:taptn_overview` and `fig_2` under `output/figures/`.

## Paper tables

`--table` follows the camera-ready `\label{...}` (the `tab:` prefix is optional):

| `--table` | Paper label | Contents |
|---|---|---|
| `main_5_datasets` | `tab:main_5_datasets` | 0-hop / GraphICL+SAT / TAPTN on five graphs |
| `factorial` | `tab:factorial` | 2-hop SAT × instructions × aggregation |
| `product_70b` | `tab:product_70b` | ogbn-products, 400 nodes, Llama-3.3-70B |
| `cost` | `tab:cost` | tokens / OpenRouter $ / accuracy |
| `tb_52` | `tb_52` | budget backbone (8B iter-1 + 70B refine) |
| `tb_4` | `tb_4` | GraphICL self-reflection vs TAPTN |
| `decouple` | `tab:decouple` | SAT × instructions, no aggregation |
| `current_channel` | `tab:current_channel` | Texas structural channels, four models |
| `current_taptn` | `tab:current_taptn` | 2-hop TAPTN vs GraphICL+SAT (Texas + Cora) |

```bash
python reproduce.py dry --table main_5_datasets --rescore
python reproduce.py dry --table current_taptn --rescore --figures
```

Scoring is **not** uniform:

- Old GPT-3.5 / `*2_*` WebKB / `arxiv2023_*` pickles: `1 - |wrong_index|/|result|`
- New `{ds}_hop{h}_{anon|noanon}_{guide|noguide}_{model}.pkl`: stored `accuracy`
- Cora current TAPTN: `acc_gate` (parameter-free neighbor consensus)
- Cora current GraphICL: `accuracy_reextract`
- TAPTN 1-hop on products: hop-1 errors intersected with the 400 test ids of the 2-hop run (paper 83.75, not the 1430-node 82.66)

See `results/DISCREPANCIES.md` for cells whose pickles were never found.

## Live re-run (LLM API)

Credentials from the environment only (never from source):

```bash
export OPENAI_API_KEY=...          # or TAPTN_LLM_API_KEY
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional
export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets
```

Current-generation configs (Texas / Cora / WebKB) go through `vendor/run_taptn_expansion.py`:

```bash
python reproduce.py run --dataset texas --model gemma-4-31b-it --config taptn1
python reproduce.py run --dataset texas --model qwen3.5-27b --config graphicl2
python reproduce.py run --dataset texas --model glm-5.1 --config taptn2 --wis_ver v2_2hop
python reproduce.py run --dataset texas --model llama-3.3-70b --config ego
```

Configs: `ego` (0-hop), `graphicl1`, `taptn1`, `graphicl2`, `taptn2` (iter-2 on top of taptn1). `--anon` strips edge direction.

Cora 2-hop TAPTN in the paper is **not** raw iter-2: it applies a parameter-free all-neighbor majority-vote gate. Recompute it from the shipped source pickles (no LLM):

```bash
python reproduce.py gate
```

## Layout

```
TAPTN_icl_repro/
  reproduce.py
  core/          # paper numbers, scorers, table rendering, cell map
  vendor/        # sanitized ICL runner (keys stripped)
  results/       # official cells + discrepancy log
TAPTN_icl_repro_assets/
  pkls/          # paper pickles, split by table
  cost_probe/    # token measurements
  dataset/       # Cora + arXiv-2023 (not ogbn-products)
  webkb-data/    # WebKB HTML
  figures_paper/ # TAPTN_overview.pdf, TAPTN_mp.pdf
```

## Recorded results

- [results/cells.md](results/cells.md) — camera-ready cells
- [results/DISCREPANCIES.md](results/DISCREPANCIES.md) — dry-run vs paper
- [results/official_cells.json](results/official_cells.json) — machine-readable paper numbers

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
