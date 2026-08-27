# TAPTN Fine-tuning Reproduction

Code to reproduce the encoder fine-tuning and GNN comparison tables from

> Hongyi Wang, Shikui Tu, Lei Xu.
> **LLMs Can Leverage Graph Structural Information in Text-Attributed Graphs**.
> *Transactions on Machine Learning Research*, 2026.
> [OpenReview](https://openreview.net/forum?id=WhaVqEkkMY)

中文说明：[README_zh.md](README_zh.md).

An LM is fine-tuned on raw text (**TA+LM**), GraphICL-style neighbourhood text (**GraphICL+LM**), or TAPTN-enriched text (**TAPTN+LM**), and compared with GNNs under the same labels, splits, and edges. Each cell is five runs (run 1–5), reported as mean±std.

Paper-level index: [../README.md](../README.md). The other two chapter packages are siblings: [`TAPTN_rewiring_repro`](../TAPTN_rewiring_repro), [`TAPTN_icl_repro`](../TAPTN_icl_repro).

## Setup

```bash
pip install -r requirements.txt
```

Assembling tables and re-scoring LM predictions runs on CPU. `train` needs a GPU.

## Data

Datasets, TAPTN/GraphICL texts, pretrained encoders, and LM checkpoints are **not** in this repository. Download the asset bundle from [Google Drive](https://drive.google.com/drive/folders/1wIlc-r7HFd31YhQLGjc0uY0-Xy3qfUVZ?usp=sharing), unpack it, and point the code at that folder.

Recommended layout (inside the paper hub):

```
TAPTN_paper_repro/
  TAPTN_finetune_repro/           # this repository
  TAPTN_finetune_repro_assets/    # unpacked bundle
```

```bash
export TAPTN_ASSETS=/path/to/TAPTN_finetune_repro_assets
```

If `TAPTN_ASSETS` is unset, `reproduce.py` looks for a sibling `TAPTN_finetune_repro_assets/` directory or `./assets/`. The bundle contains `dataset/`, `gpt_responses/`, `prt_lm/`, `pretrained/`, and WebKB files; see the README inside the bundle.

## Quick start (no training)

```bash
export TAPTN_ASSETS=/path/to/TAPTN_finetune_repro_assets
python reproduce.py check
python reproduce.py dry --table all --rescore-lm
```

`dry` reprints the paper tables from recorded cells. `--rescore-lm` recomputes LM test accuracy from saved `.pred` files (no GPU). Frozen and jointly trained GNN checkpoints are a single file per architecture (`output/<dataset>/<GNN>.pt`) and are overwritten by later jobs, so those cells are reported from the recorded logs rather than from the file currently on disk.

## Paper tables

PDF numbers follow the TMLR 2026 camera-ready. `--table` is the CLI flag (historical TeX-label alias or a short name). Flags are listed in paper order:

| PDF | `--table` | Contents |
|---|---|---|
| Table 11 | `gnn_summary` | How many pipelines significantly exceed TAPTN+LM |
| Table 12 | `p5_transfer` / `transfer` | TAPTN embeddings + frozen GNN (transferability counts) |
| Table 13 | `crosslm_encoder` | DeBERTa vs RoBERTa |
| Table 20 | `tb_6` | Frozen TA+GNN, GraphICL+LM, TAPTN+LM (homophilic) |
| Table 21 | `heterophilic` | Frozen TA+GNN, TAPTN+LM (WebKB) |
| Table 22 | `joint` | Jointly trained encoder+GNN |
| Table 23 | `roberta` | RoBERTa-base encoder |

```bash
python reproduce.py dry --table gnn_summary
python reproduce.py dry --table transfer
python reproduce.py dry --table crosslm_encoder
python reproduce.py dry --table tb_6
python reproduce.py dry --table heterophilic
python reproduce.py dry --table joint
python reproduce.py dry --table roberta
```

The caption of Table 12 defines **P5** as a *column* abbreviation: TAPTN-enriched text fed to the same frozen GNN as TA+GNN. It is not a row name in the main comparison tables.

## Training

Use `--run 1` … `--run 5`. The library RNG seed is mapped in [`configs/run_seeds.yaml`](configs/run_seeds.yaml).

```bash
python reproduce.py train --pipeline taptn_lm --dataset cora --run 1 --device 0
python reproduce.py train --pipeline grapicl_lm --dataset product --run 1 --device 0
python reproduce.py train --pipeline ta_lm --dataset cora --run 1 --device 0
python reproduce.py train --pipeline ta_gnn --dataset cora --run 1 --gnn APPNP --device 0
python reproduce.py train --pipeline taptn_gnn --dataset cora --run 1 --gnn GraphSAGE --device 0
python reproduce.py train --pipeline joint --dataset wisconsin --run 1 --gnn DirGNN --device 0
python reproduce.py train --pipeline taptn_lm --dataset cora --run 1 --encoder roberta --device 0
```

Existing LM checkpoints are reused unless you pass `--force-retrain`. GNN training overwrites `output/<dataset>/<GNN>.pt` (or the joint checkpoint).

`--gnn` accepts paper names (`GraphSAGE`, `GCNII`, `ASDGN`, `ACM-GNN`, `GraphSAINT`) and library names (`SAGE`, `GCN2`, `ASC`, `ACMGNN`, `Saint`).

## Paper rows and CLI flags

| Paper row | `--pipeline` | Setting |
|---|---|---|
| TA+LM | `ta_lm` | LM fine-tuned on raw title/abstract |
| GraphICL+LM | `grapicl_lm` | LM fine-tuned on GraphICL-style neighbourhood text |
| TAPTN+LM | `taptn_lm` | LM fine-tuned on TAPTN-enriched text |
| TA+GNN | `ta_gnn` | GNN on frozen TA embeddings |
| Joint | `joint` | Encoder and GNN trained end-to-end |
| *(transferability control)* | `taptn_gnn` | Same frozen GNN as TA+GNN, TAPTN embeddings instead of raw-text embeddings |

WebKB tables have no GraphICL+LM row. DGI and GraphSAINT have no jointly trained variant. The RoBERTa table includes RevGAT and omits DGI, GraphSAINT, and ASDGN.

## Recorded results

- [results/cells.md](results/cells.md) — every cell, runs 1–5, recorded and paper mean±std
- [results/DISCREPANCIES.md](results/DISCREPANCIES.md) — where recorded cells differ from the paper
- [results/official_cells.json](results/official_cells.json) — machine-readable cells for `dry` (internal pipeline codes `P1_LM` / `P2` / `P3` / `P4` / `P5`; displayed names in this README and `cells.md` follow the paper)

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

## License

This reproduction package is provided for research use. Please cite the paper above.
