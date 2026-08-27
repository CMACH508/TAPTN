# Dry-run / recorded cells vs camera-ready tables

This file lists every official cell whose recorded mean or std differs from the TMLR 2026 camera-ready table by more than **0.06 percentage points**.
Dry-run LM re-score (`reproduce.py dry --rescore-lm`) is compared against **recorded** cells, not against the paper typesetting.

## How to read a difference

- **LM pipelines (TA+LM, GraphICL+LM, TAPTN+LM).** Saved `.pred` files can be re-scored without training. Small gaps vs the paper are typically later log re-evals or rounding. If dry-run matches the recorded cell, the artifact is consistent with this package.
- **Frozen GNN (TA+GNN, TAPTN embeddings + frozen GNN) and Joint.** Checkpoints are **not** indexed by run: a later training job writes `output/<dataset>/<GNN>.pt` (or the joint counterpart) and overwrites the previous run. Those numbers cannot be dry-run from weights. Train mode (`reproduce.py train --pipeline ta_gnn|taptn_gnn|joint ...`) re-trains from the saved LM `.emb` / `.ckpt`. Large joint gaps (especially GAT on arXiv-2023) come from this overwrite.
- **GraphICL+LM** is the GraphICL-style auxiliary-text pipeline (paper row GraphICL). ogbn-products matches the paper exactly; Cora / arXiv-2023 differ by a few tenths of a point.

## LM cells

| Encoder | Method | Dataset | Recorded | Paper | Δmean | Δstd |
|---|---|---|---|---|---:|---:|
| deberta | TA | Cora | 76.24±1.39 | 76.79±1.54 | -0.55 | -0.15 |
| deberta | GraphICL | arXiv-2023 | 92.83±1.38 | 93.18±1.59 | -0.35 | -0.21 |
| deberta | TAPTN+LM | Cora | 84.03±1.89 | 84.32±1.44 | -0.29 | +0.45 |
| deberta | TA | arXiv-2023 | 90.23±1.52 | 90.52±1.57 | -0.29 | -0.05 |
| deberta | TAPTN+LM | arXiv-2023 | 93.58±0.90 | 93.81±0.70 | -0.23 | +0.20 |
| deberta | GraphICL | Cora | 82.10±1.17 | 82.18±1.14 | -0.08 | +0.03 |

## Frozen TA+GNN cells (DeBERTa)

| Encoder | Method | Dataset | Recorded | Paper | Δmean | Δstd |
|---|---|---|---|---|---:|---:|
| deberta | DMP | Cora | 34.69±12.24 | 31.70±7.61 | +2.99 | +4.63 |
| deberta | GATv2 | arXiv-2023 | 90.87±0.53 | 91.96±1.03 | -1.09 | -0.50 |
| deberta | GAT | Cora | 83.99±2.08 | 84.91±0.93 | -0.92 | +1.15 |
| deberta | DMP | arXiv-2023 | 86.01±13.38 | 86.76±13.76 | -0.75 | -0.38 |
| deberta | GCNII | Cora | 77.68±2.05 | 78.30±1.56 | -0.62 | +0.49 |
| deberta | FSGNN | arXiv-2023 | 92.31±1.17 | 92.89±1.41 | -0.58 | -0.24 |
| deberta | GraphSAINT | arXiv-2023 | 90.58±0.53 | 91.15±1.09 | -0.57 | -0.56 |
| deberta | ASDGN | Cora | 77.08±1.72 | 77.60±1.31 | -0.52 | +0.41 |
| deberta | DirGNN | arXiv-2023 | 91.44±0.97 | 91.91±1.37 | -0.47 | -0.40 |
| deberta | APPNP | arXiv-2023 | 91.97±0.83 | 92.43±0.62 | -0.46 | +0.21 |
| deberta | ASDGN | arXiv-2023 | 91.62±1.10 | 91.97±0.85 | -0.35 | +0.25 |
| deberta | ChebNet | Cora | 83.84±1.03 | 84.13±0.95 | -0.29 | +0.08 |
| deberta | GraphSAINT | Cora | 83.95±1.94 | 84.21±1.77 | -0.26 | +0.17 |
| deberta | GraphSAGE | arXiv-2023 | 91.27±0.97 | 91.50±1.09 | -0.23 | -0.12 |
| deberta | ChebNet | arXiv-2023 | 90.35±1.37 | 90.12±1.11 | +0.23 | +0.26 |
| deberta | GraphTARIF | arXiv-2023 | 91.04±1.76 | 91.27±1.33 | -0.23 | +0.43 |
| deberta | DGI | Cora | 84.72±0.99 | 84.84±0.88 | -0.12 | +0.11 |
| deberta | DGI | arXiv-2023 | 92.77±0.98 | 92.89±0.93 | -0.12 | +0.05 |
| deberta | ACM-GNN | Cora | 82.29±2.91 | 82.18±2.85 | +0.11 | +0.06 |
| deberta | GraphSAGE | Cora | 84.32±1.03 | 84.24±1.11 | +0.08 | -0.08 |
| deberta | GraphTARIF | Cora | 81.81±1.97 | 81.88±2.00 | -0.07 | -0.03 |
| deberta | GAT | arXiv-2023 | 90.98±1.36 | 91.04±1.17 | -0.06 | +0.19 |
| deberta | ACM-GNN | arXiv-2023 | 91.73±1.37 | 91.68±0.38 | +0.05 | +0.99 |
| deberta | GATv2 | Cora | 83.14±2.13 | 83.17±2.06 | -0.03 | +0.07 |
| deberta | GCNII | arXiv-2023 | 91.33±1.10 | 91.33±0.85 | +0.00 | +0.25 |

## Joint encoder+GNN cells (DeBERTa)

| Encoder | Method | Dataset | Recorded | Paper | Δmean | Δstd |
|---|---|---|---|---|---:|---:|
| deberta | GAT | arXiv-2023 | 57.57±22.48 | 36.88±20.63 | +20.69 | +1.85 |
| deberta | GATv2 | Cora | 55.20±21.85 | 59.85±25.35 | -4.65 | -3.50 |
| deberta | ASDGN | arXiv-2023 | 83.41±10.57 | 80.35±18.24 | +3.06 | -7.67 |
| deberta | GCNII | Cora | 80.37±1.35 | 77.82±5.17 | +2.55 | -3.82 |
| deberta | DMP | Cora | 27.46±7.16 | 25.24±7.63 | +2.22 | -0.47 |
| deberta | DMP | arXiv-2023 | 91.62±1.79 | 93.29±1.15 | -1.67 | +0.64 |
| deberta | GCNII | arXiv-2023 | 90.69±3.39 | 92.02±0.48 | -1.33 | +2.91 |
| deberta | ChebNet | Cora | 85.53±1.48 | 84.24±2.92 | +1.29 | -1.44 |
| deberta | APPNP | arXiv-2023 | 83.99±7.39 | 85.20±5.39 | -1.21 | +2.00 |
| deberta | GAT | Cora | 67.60±21.45 | 68.56±22.15 | -0.96 | -0.70 |
| deberta | ACM-GNN | arXiv-2023 | 91.97±2.12 | 91.04±4.64 | +0.93 | -2.52 |
| deberta | SAGE | arXiv-2023 | 93.01±1.75 | 93.70±1.18 | -0.69 | +0.57 |
| deberta | DirGNN | Cora | 85.68±2.05 | 86.35±1.29 | -0.67 | +0.76 |
| deberta | APPNP | Cora | 86.87±2.36 | 87.49±2.54 | -0.62 | -0.18 |
| deberta | ASDGN | Cora | 77.79±1.37 | 77.23±2.26 | +0.56 | -0.89 |
| deberta | GraphTARIF | arXiv-2023 | 91.56±1.30 | 92.02±1.22 | -0.46 | +0.08 |
| deberta | FSGNN | Cora | 87.64±1.51 | 87.23±1.65 | +0.41 | -0.14 |
| deberta | FSGNN | arXiv-2023 | 93.30±1.20 | 93.70±0.75 | -0.40 | +0.45 |
| deberta | GraphTARIF | Cora | 82.73±1.24 | 82.40±1.04 | +0.33 | +0.20 |
| deberta | ACM-GNN | Cora | 81.62±2.72 | 81.33±3.07 | +0.29 | -0.35 |
| deberta | DirGNN | arXiv-2023 | 93.41±0.43 | 93.12±0.94 | +0.29 | -0.51 |
| deberta | GATv2 | arXiv-2023 | 91.68±1.84 | 91.50±1.92 | +0.18 | -0.08 |
| deberta | SAGE | Cora | 85.02±2.20 | 84.87±2.45 | +0.15 | -0.25 |
| deberta | ChebNet | arXiv-2023 | 92.37±1.20 | 92.48±1.24 | -0.11 | -0.04 |

## RoBERTa cells

All RoBERTa table cells agree with the paper within 0.06 pp.

## Summary

- Cells matching the paper within 0.06 pp: **197**
- LM cells with a larger gap: **6**
- Frozen GNN cells with a larger gap: **25**
- Joint cells with a larger gap: **24**
- RoBERTa cells with a larger gap: **0**

### Headline count tables

- **Transferability** (`p5_transfer`) and **encoder** (`crosslm_encoder`) counts from recorded cells match the paper exactly (Σ 23 / 5; DeBERTa 1/0/0/0/0/0 and RoBERTa 0/0/0/0/0/8).
- **tab:gnn_summary** significance counts match the paper (Cora 2, Wisconsin 1, others 0). Cora **#mean >TAPTN+LM** is 10 from recorded cells vs 9 in the paper, because recorded TAPTN+LM on Cora is 84.03 vs typeset 84.32, so one extra competitor has a higher mean.
- **tab:roberta_full** and **tab:heterophilic_full** recorded mean±std match the paper (within 0.06 pp).

### Notable gaps

- **Joint GAT / arXiv-2023**: recorded ≈57.57 vs paper 36.88. The on-disk joint checkpoint is not run-indexed and was overwritten after the camera-ready numbers were taken. Use `reproduce.py train --pipeline joint --dataset arxiv_2023 --run N --gnn GAT` to re-train that cell.
- **TAPTN+LM Cora**: recorded 84.03±1.89 vs paper 84.32±1.44. Dry-run should follow the recorded cell if `.pred` re-score matches.
- **GraphICL+LM ogbn-products**: recorded 81.50±3.47 equals the paper.

English summary: dry-run LM numbers track this package’s recorded cells. Frozen/joint GNN numbers are recorded from logs because run-specific GNN weights were overwritten; train mode rebuilds them from LM embeddings/checkpoints. Where recorded cells differ from the paper, the paper typesetting is listed above.

## Dry-run verification

`python reproduce.py dry --rescore-lm` re-scored **135 / 135** LM cells (TA+LM, GraphICL+LM, TAPTN+LM × DeBERTa/RoBERTa × five runs) from saved `.pred` files. **missing_pred=0, mismatch>0.001=0** versus `results/official_cells.json`.
WebKB node order is taken from `webkb_html_order_<dataset>.txt` so that copied HTML trees match the order used when the `.pred` files were written.
Frozen / joint GNN cells are **not** re-derived from `output/*.pt` (those files are not run-indexed).

