# TAPTN Rewiring Reproduction

Code to reproduce the **neighborhood-rewiring** tables and figures from

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)

中文说明：[README_zh.md](README_zh.md).

This package covers Section 2 (flipping / extreme rewiring, capability–sensitivity statistics, label-free control) plus Appendix G–H (current-generation label-free extension and the TAPTN-internal structural-channel ablation).

Paper-level index: [../README.md](../README.md). The other two chapter packages are siblings: [`TAPTN_icl_repro`](../TAPTN_icl_repro), [`TAPTN_finetune_repro`](../TAPTN_finetune_repro).

## Setup

```bash
pip install -r requirements.txt
```

Assembling tables and re-scoring pickles runs on CPU but needs **PyTorch Geometric** (the pickles store `torch_geometric.data.Data`). `train`/`run` needs an API key.

## Data

Pickles, WebKB HTML, abstracts, recorded CSVs, and camera-ready figures are **not** in the code repository. Download the asset bundle from [Google Drive](https://drive.google.com/drive/folders/1qQeo2Fy8_snk08b3jWFYgBmKyme_65oj?usp=sharing), unpack it, and point the code at that folder.

Recommended layout (inside the paper hub):

```
TAPTN_paper_repro/
  TAPTN_rewiring_repro/           # this repository
  TAPTN_rewiring_repro_assets/    # unpacked bundle
```

```bash
export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets
```

If `TAPTN_ASSETS` is unset, `reproduce.py` looks for a sibling `TAPTN_rewiring_repro_assets/` directory or `./assets/`.

## Quick start (no LLM calls)

```bash
export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets
python reproduce.py check
python reproduce.py dry --table all --rescore --figures
# audit against the unrevised PDF:
python reproduce.py dry --table all --rescore --paper-compat
# or
bash run_dry.sh
```

`dry` reprints the **revised** camera-ready tables. `--rescore` recomputes every cell from the saved pickles (still no GPU / no API). `--figures` writes plots under `output/figures/`. `--paper-compat` compares against the *unrevised* PDF (pre-errata cells, including the 42.04 Wisconsin label-free GPT-3.5 original). Camera-ready illustrations (Figure 1, Figure 4, Figure 11 / `former_rewire2.pdf`) are copied as-is; scatter/heatmap panels are regenerated from the pickles and will not be pixel-identical to the PDF.

## Paper tables

PDF numbers follow the TMLR 2026 camera-ready. `--table` is the CLI flag (historical TeX-label alias).

| PDF | `--table` | Contents |
|---|---|---|
| Table 1 | `tb_1` | Homophily of WebKB and citation graphs |
| Table 2 | `tb_2` | 7 models × 4 WebKB × flipping / extreme |
| Table 3 | `tb_3` | GPT-3.5-Turbo-0125 + step-by-step instructions, flipping |
| Table 4 | `rewiring_stats` | Capability–sensitivity regression |
| Table 5 | `nolabel_main` | Label-free flipping, 7-model averages |
| Table 6 | `nolabel_stats` | Label-free capability–sensitivity |
| Table 26 | `current_rewire` | Four current-generation models, label-free flipping |
| Table 29 | `taptn_structcorrupt` | TAPTN-internal edge-blind / flip Δ |

Figures: Figure 2, Figure 3, Appendix Figures 8–10, plus static Figure 1 / Figure 4.

```bash
python reproduce.py dry --table tb_2 --rescore
python reproduce.py dry --table nolabel_main --rescore --figures
```

## Live re-run (LLM API)

```bash
export OPENAI_API_KEY=...
export OPENAI_BASE_URL=https://openrouter.ai/api/v1   # optional
export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets

# Table 2, flipping, one model / dataset
python reproduce.py run --table tb_2 --hop 1 --model phi-4 --dataset cornell
python reproduce.py run --table tb_2 --hop 1 --model phi-4 --dataset cornell --rewired

# Table 2 extreme (hop 2)
python reproduce.py run --table tb_2 --hop 2 --model phi-4 --dataset texas --rewired

# Label-free control (also used for Table 26)
python reproduce.py run --table nolabel --model llama-3.3-70b-instruct --dataset wisconsin --rewired

# Table 3 (instructions on)
python reproduce.py run --table tb_3 --dataset cornell
```

Outputs are written under `output/rerun/` (override with `--output-dir`). Live jobs are stochastic; they will not bit-match the saved pickles.

Table 29 is dry-run only in this bundle (complete TAPTN iteration-2 pipeline).

## Scoring

Section 2 open-model pickles store a `results` list aligned to the unshuffled `test_mask`. Accuracy skips `None` entries and matches labels with the same multi-stage rule as `analyze_rewiring_lmarena.py`.

GPT-3.5 Table 2 uses the original notebooks’ `wrong_index` / `result` convention (legacy filenames under `pkls/gpt35/`). Table 3 uses the same rule on `pkls/tb3/`.

The label-free Qwen row in the paper is **Qwen2.5-VL-72B**, not the non-VL `qwen-2.5-72b-instruct` files that share LMArena score 1302.

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
