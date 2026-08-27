# Per-cell accuracy records

Accuracies are test-set node classification (%). Each cell is five runs numbered **1–5**.
The library RNG seed for run *k* on each dataset is listed in `configs/run_seeds.yaml` and is applied automatically by `reproduce.py`.

**Recorded** is the official cell stored with this package (dry-run LM re-score should match these numbers).
**Paper** is the camera-ready table entry when one exists. Differences are listed in `DISCREPANCIES.md`.

Names follow the paper: TA+LM, GraphICL+LM, TAPTN+LM, TA+GNN (frozen), Joint, and TAPTN embeddings + frozen GNN (transferability).
GraphICL+LM is DeBERTa fine-tuned on GraphICL-style auxiliary neighbourhood text (not TAPTN). WebKB tables have no GraphICL+LM row.

## Table tb_6 — Frozen TA+GNN / GraphICL+LM / TAPTN+LM (homophilic, DeBERTa)

| Encoder | Pipeline | Method | Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Recorded | Paper |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| deberta | TA+LM | TA | Cora | 74.91 | 78.41 | 75.65 | 76.75 | 75.46 | 76.24±1.39 | 76.79±1.54 |
| deberta | TA+LM | TA | arXiv-2023 | 90.17 | 88.73 | 92.77 | 89.60 | 89.88 | 90.23±1.52 | 90.52±1.57 |
| deberta | TA+LM | TA | ogbn-products | 70.00 | 56.25 | 48.75 | 75.00 | 66.25 | 63.25±10.63 | 63.25±10.63 |
| deberta | TA+GNN (frozen) | GAT | Cora | 85.61 | 83.39 | 85.61 | 84.69 | 80.63 | 83.99±2.08 | 84.91±0.93 |
| deberta | TA+GNN (frozen) | GAT | arXiv-2023 | 93.06 | 91.04 | 91.04 | 90.46 | 89.31 | 90.98±1.36 | 91.04±1.17 |
| deberta | TA+GNN (frozen) | GAT | ogbn-products | 71.25 | 71.25 | 58.75 | 73.75 | 66.25 | 68.25±5.97 | 68.25±5.97 |
| deberta | TA+GNN (frozen) | GraphSAGE | Cora | 83.76 | 83.39 | 85.24 | 85.61 | 83.58 | 84.32±1.03 | 84.24±1.11 |
| deberta | TA+GNN (frozen) | GraphSAGE | arXiv-2023 | 91.62 | 90.46 | 92.77 | 90.46 | 91.04 | 91.27±0.97 | 91.50±1.09 |
| deberta | TA+GNN (frozen) | GraphSAGE | ogbn-products | 70.00 | 67.50 | 58.75 | 77.50 | 72.50 | 69.25±6.94 | 69.25±6.94 |
| deberta | TA+GNN (frozen) | ChebNet | Cora | 85.24 | 82.66 | 84.32 | 83.95 | 83.03 | 83.84±1.03 | 84.13±0.95 |
| deberta | TA+GNN (frozen) | ChebNet | arXiv-2023 | 91.62 | 88.73 | 91.91 | 89.88 | 89.60 | 90.35±1.37 | 90.12±1.11 |
| deberta | TA+GNN (frozen) | ChebNet | ogbn-products | 72.50 | 68.75 | 57.50 | 72.50 | 68.75 | 68.00±6.16 | 68.00±6.16 |
| deberta | TA+GNN (frozen) | DGI | Cora | 84.32 | 84.32 | 84.87 | 86.35 | 83.76 | 84.72±0.99 | 84.84±0.88 |
| deberta | TA+GNN (frozen) | DGI | arXiv-2023 | 93.64 | 93.35 | 93.35 | 91.33 | 92.20 | 92.77±0.98 | 92.89±0.93 |
| deberta | TA+GNN (frozen) | DGI | ogbn-products | 73.75 | 71.25 | 66.25 | 78.75 | 73.75 | 72.75±4.54 | 72.75±4.54 |
| deberta | TA+GNN (frozen) | GATv2 | Cora | 82.29 | 83.21 | 85.61 | 84.50 | 80.07 | 83.14±2.13 | 83.17±2.06 |
| deberta | TA+GNN (frozen) | GATv2 | arXiv-2023 | 91.33 | 91.33 | 90.17 | 91.04 | 90.46 | 90.87±0.53 | 91.96±1.03 |
| deberta | TA+GNN (frozen) | GATv2 | ogbn-products | 66.25 | 68.75 | 65.00 | 77.50 | 68.75 | 69.25±4.89 | 69.25±4.89 |
| deberta | TA+GNN (frozen) | GCNII | Cora | 75.65 | 78.78 | 79.70 | 78.97 | 75.28 | 77.68±2.05 | 78.30±1.56 |
| deberta | TA+GNN (frozen) | GCNII | arXiv-2023 | 91.04 | 92.49 | 92.49 | 90.46 | 90.17 | 91.33±1.10 | 91.33±0.85 |
| deberta | TA+GNN (frozen) | GCNII | ogbn-products | 71.25 | 52.50 | 47.50 | 71.25 | 63.75 | 61.25±10.86 | 61.25±10.86 |
| deberta | TA+GNN (frozen) | ASDGN | Cora | 76.57 | 78.78 | 76.01 | 78.97 | 75.09 | 77.08±1.72 | 77.60±1.31 |
| deberta | TA+GNN (frozen) | ASDGN | arXiv-2023 | 92.20 | 92.20 | 92.77 | 90.75 | 90.17 | 91.62±1.10 | 91.97±0.85 |
| deberta | TA+GNN (frozen) | ASDGN | ogbn-products | 75.00 | 55.00 | 47.50 | 71.25 | 68.75 | 63.50±11.71 | 63.50±11.71 |
| deberta | TA+GNN (frozen) | DirGNN | Cora | 86.16 | 83.21 | 85.61 | 84.13 | 84.87 | 84.80±1.17 | 84.80±1.17 |
| deberta | TA+GNN (frozen) | DirGNN | arXiv-2023 | 91.62 | 91.04 | 93.06 | 90.75 | 90.75 | 91.44±0.97 | 91.91±1.37 |
| deberta | TA+GNN (frozen) | DirGNN | ogbn-products | 75.00 | 68.75 | 63.75 | 81.25 | 73.75 | 72.50±6.61 | 72.50±6.61 |
| deberta | TA+GNN (frozen) | ACM-GNN | Cora | 79.15 | 80.26 | 81.92 | 86.53 | 83.58 | 82.29±2.91 | 82.18±2.85 |
| deberta | TA+GNN (frozen) | ACM-GNN | arXiv-2023 | 91.62 | 92.20 | 93.64 | 91.33 | 89.88 | 91.73±1.37 | 91.68±0.38 |
| deberta | TA+GNN (frozen) | ACM-GNN | ogbn-products | 67.50 | 56.25 | 56.25 | 73.75 | 65.00 | 63.75±7.55 | 63.75±7.55 |
| deberta | TA+GNN (frozen) | DMP | Cora | 31.92 | 31.73 | 37.64 | 19.19 | 52.95 | 34.69±12.24 | 31.70±7.61 |
| deberta | TA+GNN (frozen) | DMP | arXiv-2023 | 93.06 | 92.77 | 90.46 | 62.14 | 91.62 | 86.01±13.38 | 86.76±13.76 |
| deberta | TA+GNN (frozen) | DMP | ogbn-products | 65.00 | 57.50 | 60.00 | 75.00 | 73.75 | 66.25±7.91 | 66.25±7.91 |
| deberta | TA+GNN (frozen) | GraphSAINT | Cora | 82.66 | 83.58 | 87.27 | 83.76 | 82.47 | 83.95±1.94 | 84.21±1.77 |
| deberta | TA+GNN (frozen) | GraphSAINT | arXiv-2023 | 89.88 | 90.46 | 90.46 | 91.33 | 90.75 | 90.58±0.53 | 91.15±1.09 |
| deberta | TA+GNN (frozen) | GraphSAINT | ogbn-products | 71.25 | 62.50 | 58.75 | 81.25 | 73.75 | 69.50±9.00 | 69.50±9.00 |
| deberta | TA+GNN (frozen) | FSGNN | Cora | 83.95 | 84.69 | 87.64 | 84.32 | 86.72 | 85.46±1.62 | 85.46±1.62 |
| deberta | TA+GNN (frozen) | FSGNN | arXiv-2023 | 93.06 | 91.91 | 93.93 | 91.04 | 91.62 | 92.31±1.17 | 92.89±1.41 |
| deberta | TA+GNN (frozen) | FSGNN | ogbn-products | 73.75 | 72.50 | 65.00 | 80.00 | 73.75 | 73.00±5.35 | 73.00±5.35 |
| deberta | TA+GNN (frozen) | APPNP | Cora | 88.38 | 87.45 | 90.04 | 89.11 | 88.38 | 88.67±0.97 | 88.63±0.98 |
| deberta | TA+GNN (frozen) | APPNP | arXiv-2023 | 93.35 | 91.91 | 91.33 | 91.91 | 91.33 | 91.97±0.83 | 92.43±0.62 |
| deberta | TA+GNN (frozen) | APPNP | ogbn-products | 76.25 | 78.75 | 66.25 | 85.00 | 72.50 | 75.75±6.99 | 75.75±6.99 |
| deberta | TA+GNN (frozen) | GraphTARIF | Cora | 81.92 | 80.07 | 79.89 | 84.69 | 82.47 | 81.81±1.97 | 81.88±2.00 |
| deberta | TA+GNN (frozen) | GraphTARIF | arXiv-2023 | 91.91 | 89.02 | 93.35 | 91.33 | 89.60 | 91.04±1.76 | 91.27±1.33 |
| deberta | TA+GNN (frozen) | GraphTARIF | ogbn-products | 68.75 | 61.25 | 52.50 | 72.50 | 63.75 | 63.75±7.65 | 63.75±7.65 |
| deberta | GraphICL+LM | GraphICL | Cora | 82.10 | 80.44 | 83.21 | 83.21 | 81.55 | 82.10±1.17 | 82.18±1.14 |
| deberta | GraphICL+LM | GraphICL | arXiv-2023 | 90.75 | 94.51 | 93.35 | 92.49 | 93.06 | 92.83±1.38 | 93.18±1.59 |
| deberta | GraphICL+LM | GraphICL | ogbn-products | 78.75 | 80.00 | 81.25 | 87.50 | 80.00 | 81.50±3.47 | 81.50±3.47 |
| deberta | TAPTN+LM | TAPTN+LM | Cora | 86.35 | 84.69 | 84.69 | 83.03 | 81.37 | 84.03±1.89 | 84.32±1.44 |
| deberta | TAPTN+LM | TAPTN+LM | arXiv-2023 | 94.22 | 94.80 | 92.77 | 93.35 | 92.77 | 93.58±0.90 | 93.81±0.70 |
| deberta | TAPTN+LM | TAPTN+LM | ogbn-products | 87.50 | 87.50 | 90.00 | 93.75 | 88.75 | 89.50±2.59 | 89.50±2.59 |

## Table heterophilic — Frozen TA+GNN / TAPTN+LM (WebKB, DeBERTa; no GraphICL+LM)

| Encoder | Pipeline | Method | Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Recorded | Paper |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| deberta | TA+LM | TA | Texas | 94.23 | 94.23 | 98.08 | 98.08 | 98.08 | 96.54±2.11 | 96.54±2.11 |
| deberta | TA+LM | TA | Wisconsin | 89.23 | 93.85 | 89.23 | 93.85 | 92.31 | 91.69±2.34 | 91.69±2.34 |
| deberta | TA+LM | TA | Cornell | 94.00 | 94.00 | 96.00 | 98.00 | 96.00 | 95.60±1.67 | 95.60±1.67 |
| deberta | TA+GNN (frozen) | GAT | Texas | 71.15 | 65.38 | 78.85 | 63.46 | 76.92 | 71.15±6.80 | 71.15±6.80 |
| deberta | TA+GNN (frozen) | GAT | Wisconsin | 56.92 | 58.46 | 55.38 | 70.77 | 56.92 | 59.69±6.29 | 59.69±6.29 |
| deberta | TA+GNN (frozen) | GAT | Cornell | 72.00 | 68.00 | 70.00 | 56.00 | 70.00 | 67.20±6.42 | 67.20±6.42 |
| deberta | TA+GNN (frozen) | GraphSAGE | Texas | 94.23 | 92.31 | 94.23 | 98.08 | 98.08 | 95.39±2.58 | — |
| deberta | TA+GNN (frozen) | GraphSAGE | Wisconsin | 86.15 | 90.77 | 92.31 | 93.85 | 92.31 | 91.08±2.96 | — |
| deberta | TA+GNN (frozen) | GraphSAGE | Cornell | 98.00 | 92.00 | 94.00 | 98.00 | 92.00 | 94.80±3.03 | — |
| deberta | TA+GNN (frozen) | ChebNet | Texas | 84.62 | 86.54 | 90.38 | 94.23 | 92.31 | 89.62±3.99 | 89.62±3.99 |
| deberta | TA+GNN (frozen) | ChebNet | Wisconsin | 76.92 | 86.15 | 84.62 | 87.69 | 86.15 | 84.31±4.27 | 84.31±4.27 |
| deberta | TA+GNN (frozen) | ChebNet | Cornell | 84.00 | 84.00 | 88.00 | 88.00 | 80.00 | 84.80±3.35 | 84.80±3.35 |
| deberta | TA+GNN (frozen) | DGI | Texas | 80.77 | 80.77 | 82.69 | 82.69 | 88.46 | 83.08±3.16 | 83.08±3.16 |
| deberta | TA+GNN (frozen) | DGI | Wisconsin | 75.38 | 76.92 | 80.00 | 80.00 | 75.38 | 77.54±2.34 | 77.54±2.34 |
| deberta | TA+GNN (frozen) | DGI | Cornell | 74.00 | 82.00 | 76.00 | 76.00 | 70.00 | 75.60±4.34 | 75.60±4.34 |
| deberta | TA+GNN (frozen) | GATv2 | Texas | 90.38 | 76.92 | 80.77 | 80.77 | 82.69 | 82.31±4.98 | 82.31±4.98 |
| deberta | TA+GNN (frozen) | GATv2 | Wisconsin | 72.31 | 90.77 | 80.00 | 93.85 | 90.77 | 85.54±9.08 | 85.54±9.08 |
| deberta | TA+GNN (frozen) | GATv2 | Cornell | 80.00 | 78.00 | 76.00 | 76.00 | 80.00 | 78.00±2.00 | 78.00±2.00 |
| deberta | TA+GNN (frozen) | GCNII | Texas | 94.23 | 92.31 | 92.31 | 96.15 | 98.08 | 94.62±2.51 | 94.62±2.51 |
| deberta | TA+GNN (frozen) | GCNII | Wisconsin | 89.23 | 93.85 | 87.69 | 92.31 | 90.77 | 90.77±2.43 | 90.77±2.43 |
| deberta | TA+GNN (frozen) | GCNII | Cornell | 92.00 | 94.00 | 96.00 | 98.00 | 92.00 | 94.40±2.61 | 94.40±2.61 |
| deberta | TA+GNN (frozen) | ASDGN | Texas | 94.23 | 90.38 | 90.38 | 96.15 | 98.08 | 93.84±3.44 | 93.84±3.44 |
| deberta | TA+GNN (frozen) | ASDGN | Wisconsin | 89.23 | 89.23 | 89.23 | 92.31 | 92.31 | 90.46±1.69 | 90.46±1.69 |
| deberta | TA+GNN (frozen) | ASDGN | Cornell | 88.00 | 92.00 | 94.00 | 98.00 | 96.00 | 93.60±3.85 | 93.60±3.85 |
| deberta | TA+GNN (frozen) | DirGNN | Texas | 96.15 | 96.15 | 98.08 | 98.08 | 100.00 | 97.69±1.61 | 97.69±1.61 |
| deberta | TA+GNN (frozen) | DirGNN | Wisconsin | 89.23 | 92.31 | 89.23 | 93.85 | 92.31 | 91.39±2.07 | 91.39±2.07 |
| deberta | TA+GNN (frozen) | DirGNN | Cornell | 100.00 | 98.00 | 98.00 | 100.00 | 96.00 | 98.40±1.67 | 98.40±1.67 |
| deberta | TA+GNN (frozen) | ACM-GNN | Texas | 94.23 | 94.23 | 94.23 | 98.08 | 98.08 | 95.77±2.11 | 95.77±2.11 |
| deberta | TA+GNN (frozen) | ACM-GNN | Wisconsin | 87.69 | 90.77 | 83.08 | 92.31 | 90.77 | 88.92±3.67 | 88.92±3.67 |
| deberta | TA+GNN (frozen) | ACM-GNN | Cornell | 86.00 | 94.00 | 94.00 | 100.00 | 96.00 | 94.00±5.10 | 94.00±5.10 |
| deberta | TA+GNN (frozen) | DMP | Texas | 11.54 | 67.31 | 38.46 | 65.38 | 57.69 | 48.08±23.39 | 48.08±23.39 |
| deberta | TA+GNN (frozen) | DMP | Wisconsin | 29.23 | 38.46 | 53.85 | 21.54 | 47.69 | 38.15±13.16 | 38.15±13.16 |
| deberta | TA+GNN (frozen) | DMP | Cornell | 60.00 | 32.00 | 60.00 | 54.00 | 48.00 | 50.80±11.63 | 50.80±11.63 |
| deberta | TA+GNN (frozen) | GraphSAINT | Texas | 94.23 | 86.54 | 92.31 | 78.85 | 78.85 | 86.16±7.25 | 86.16±7.25 |
| deberta | TA+GNN (frozen) | GraphSAINT | Wisconsin | 87.69 | 67.69 | 87.69 | 93.85 | 72.31 | 81.85±11.22 | 81.85±11.22 |
| deberta | TA+GNN (frozen) | GraphSAINT | Cornell | 78.00 | 90.00 | 88.00 | 56.00 | 76.00 | 77.60±13.52 | 77.60±13.52 |
| deberta | TA+GNN (frozen) | FSGNN | Texas | 92.31 | 94.23 | 98.08 | 96.15 | 98.08 | 95.77±2.51 | 95.77±2.51 |
| deberta | TA+GNN (frozen) | FSGNN | Wisconsin | 89.23 | 92.31 | 92.31 | 93.85 | 90.77 | 91.69±1.76 | 91.69±1.76 |
| deberta | TA+GNN (frozen) | FSGNN | Cornell | 90.00 | 94.00 | 94.00 | 96.00 | 94.00 | 93.60±2.19 | 93.60±2.19 |
| deberta | TA+GNN (frozen) | APPNP | Texas | 80.77 | 69.23 | 86.54 | 82.69 | 76.92 | 79.23±6.58 | 79.23±6.58 |
| deberta | TA+GNN (frozen) | APPNP | Wisconsin | 66.15 | 67.69 | 69.23 | 81.54 | 72.31 | 71.38±6.12 | 71.38±6.12 |
| deberta | TA+GNN (frozen) | APPNP | Cornell | 76.00 | 82.00 | 84.00 | 74.00 | 74.00 | 78.00±4.69 | 78.00±4.69 |
| deberta | TA+GNN (frozen) | GraphTARIF | Texas | 94.23 | 92.31 | 98.08 | 90.38 | 98.08 | 94.62±3.44 | 94.62±3.44 |
| deberta | TA+GNN (frozen) | GraphTARIF | Wisconsin | 87.69 | 93.85 | 87.69 | 89.23 | 92.31 | 90.15±2.80 | 90.15±2.80 |
| deberta | TA+GNN (frozen) | GraphTARIF | Cornell | 94.00 | 86.00 | 92.00 | 94.00 | 92.00 | 91.60±3.29 | 91.60±3.29 |
| deberta | TAPTN+LM | TAPTN+LM | Texas | 98.08 | 96.15 | 96.15 | 98.08 | 100.00 | 97.69±1.61 | 97.69±1.61 |
| deberta | TAPTN+LM | TAPTN+LM | Wisconsin | 86.15 | 87.69 | 87.69 | 83.08 | 90.77 | 87.08±2.79 | 87.08±2.79 |
| deberta | TAPTN+LM | TAPTN+LM | Cornell | 98.00 | 100.00 | 98.00 | 100.00 | 100.00 | 99.20±1.10 | 99.20±1.10 |

## Table joint — Jointly trained encoder+GNN (DeBERTa)

| Encoder | Pipeline | Method | Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Recorded | Paper |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| deberta | Joint encoder+GNN | GAT | Cora | 30.26 | 68.82 | 82.10 | 78.41 | 78.41 | 67.60±21.45 | 68.56±22.15 |
| deberta | Joint encoder+GNN | GAT | arXiv-2023 | 36.71 | 31.79 | 63.87 | 71.97 | 83.53 | 57.57±22.48 | 36.88±20.63 |
| deberta | Joint encoder+GNN | GAT | ogbn-products | 41.25 | 26.25 | 56.25 | 77.50 | 31.25 | 46.50±20.77 | 46.50±20.77 |
| deberta | Joint encoder+GNN | GAT | Texas | 57.69 | 15.38 | 73.08 | 76.92 | 57.69 | 56.15±24.42 | 56.15±24.42 |
| deberta | Joint encoder+GNN | GAT | Wisconsin | 26.15 | 27.69 | 66.15 | 50.77 | 66.15 | 47.38±19.71 | 47.38±19.71 |
| deberta | Joint encoder+GNN | GAT | Cornell | 56.00 | 46.00 | 66.00 | 66.00 | 40.00 | 54.80±11.71 | 54.80±11.71 |
| deberta | Joint encoder+GNN | GATv2 | Cora | 59.78 | 82.84 | 22.32 | 50.92 | 60.15 | 55.20±21.85 | 59.85±25.35 |
| deberta | Joint encoder+GNN | GATv2 | arXiv-2023 | 91.04 | 92.49 | 93.06 | 88.73 | 93.06 | 91.68±1.84 | 91.50±1.92 |
| deberta | Joint encoder+GNN | GATv2 | ogbn-products | 21.25 | 23.75 | 7.50 | 23.75 | 35.00 | 22.25±9.82 | 22.25±9.82 |
| deberta | Joint encoder+GNN | GATv2 | Texas | 15.38 | 69.23 | 92.31 | 98.08 | 82.69 | 71.54±33.24 | 71.54±33.24 |
| deberta | Joint encoder+GNN | GATv2 | Wisconsin | 49.23 | 52.31 | 87.69 | 87.69 | 53.85 | 66.15±19.73 | 66.15±19.73 |
| deberta | Joint encoder+GNN | GATv2 | Cornell | 88.00 | 76.00 | 70.00 | 20.00 | 78.00 | 66.40±26.74 | 66.40±26.74 |
| deberta | Joint encoder+GNN | SAGE | Cora | 85.24 | 83.76 | 87.82 | 86.16 | 82.10 | 85.02±2.20 | 84.87±2.45 |
| deberta | Joint encoder+GNN | SAGE | arXiv-2023 | 95.38 | 93.93 | 91.62 | 93.06 | 91.04 | 93.01±1.75 | 93.70±1.18 |
| deberta | Joint encoder+GNN | SAGE | ogbn-products | 73.75 | 65.00 | 42.50 | 80.00 | 66.25 | 65.50±14.21 | 65.50±14.21 |
| deberta | Joint encoder+GNN | SAGE | Texas | 94.23 | 92.31 | 96.15 | 90.38 | 98.08 | 94.23±3.04 | 94.23±3.04 |
| deberta | Joint encoder+GNN | SAGE | Wisconsin | 87.69 | 92.31 | 89.23 | 90.77 | 87.69 | 89.54±2.01 | 89.54±2.01 |
| deberta | Joint encoder+GNN | SAGE | Cornell | 94.00 | 94.00 | 96.00 | 94.00 | 82.00 | 92.00±5.66 | 92.00±5.66 |
| deberta | Joint encoder+GNN | ChebNet | Cora | 87.08 | 83.21 | 86.16 | 85.06 | 86.16 | 85.53±1.48 | 84.24±2.92 |
| deberta | Joint encoder+GNN | ChebNet | arXiv-2023 | 93.93 | 93.06 | 92.49 | 91.33 | 91.04 | 92.37±1.20 | 92.48±1.24 |
| deberta | Joint encoder+GNN | ChebNet | ogbn-products | 70.00 | 66.25 | 58.75 | 75.00 | 66.25 | 67.25±5.96 | 67.25±5.96 |
| deberta | Joint encoder+GNN | ChebNet | Texas | 86.54 | 92.31 | 90.38 | 88.46 | 88.46 | 89.23±2.19 | 89.23±2.19 |
| deberta | Joint encoder+GNN | ChebNet | Wisconsin | 83.08 | 90.77 | 84.62 | 89.23 | 80.00 | 85.54±4.43 | 85.54±4.43 |
| deberta | Joint encoder+GNN | ChebNet | Cornell | 82.00 | 92.00 | 90.00 | 90.00 | 94.00 | 89.60±4.56 | 89.60±4.56 |
| deberta | Joint encoder+GNN | GCNII | Cora | 79.15 | 78.78 | 81.73 | 80.63 | 81.55 | 80.37±1.35 | 77.82±5.17 |
| deberta | Joint encoder+GNN | GCNII | arXiv-2023 | 92.20 | 91.62 | 84.68 | 92.77 | 92.20 | 90.69±3.39 | 92.02±0.48 |
| deberta | Joint encoder+GNN | GCNII | ogbn-products | 71.25 | 61.25 | 58.75 | 72.50 | 66.25 | 66.00±6.02 | 66.00±6.02 |
| deberta | Joint encoder+GNN | GCNII | Texas | 94.23 | 86.54 | 90.38 | 94.23 | 98.08 | 92.69±4.39 | 92.69±4.39 |
| deberta | Joint encoder+GNN | GCNII | Wisconsin | 89.23 | 89.23 | 81.54 | 89.23 | 90.77 | 88.00±3.67 | 88.00±3.67 |
| deberta | Joint encoder+GNN | GCNII | Cornell | 94.00 | 90.00 | 94.00 | 90.00 | 86.00 | 90.80±3.35 | 90.80±3.35 |
| deberta | Joint encoder+GNN | APPNP | Cora | 88.01 | 85.06 | 90.41 | 84.69 | 86.16 | 86.87±2.36 | 87.49±2.54 |
| deberta | Joint encoder+GNN | APPNP | arXiv-2023 | 89.02 | 86.71 | 91.91 | 75.72 | 76.59 | 83.99±7.39 | 85.20±5.39 |
| deberta | Joint encoder+GNN | APPNP | ogbn-products | 73.75 | 72.50 | 62.50 | 83.75 | 73.75 | 73.25±7.53 | 73.25±7.53 |
| deberta | Joint encoder+GNN | APPNP | Texas | 75.00 | 59.62 | 42.31 | 75.00 | 65.38 | 63.46±13.53 | 63.46±13.53 |
| deberta | Joint encoder+GNN | APPNP | Wisconsin | 67.69 | 61.54 | 64.62 | 58.46 | 50.77 | 60.62±6.49 | 60.62±6.49 |
| deberta | Joint encoder+GNN | APPNP | Cornell | 34.00 | 46.00 | 82.00 | 66.00 | 32.00 | 52.00±21.54 | 52.00±21.54 |
| deberta | Joint encoder+GNN | ASDGN | Cora | 76.94 | 78.78 | 76.94 | 79.70 | 76.57 | 77.79±1.37 | 77.23±2.26 |
| deberta | Joint encoder+GNN | ASDGN | arXiv-2023 | 94.51 | 70.23 | 85.55 | 91.91 | 74.86 | 83.41±10.57 | 80.35±18.24 |
| deberta | Joint encoder+GNN | ASDGN | ogbn-products | 68.75 | 55.00 | 56.25 | 77.50 | 65.00 | 64.50±9.30 | 64.50±9.30 |
| deberta | Joint encoder+GNN | ASDGN | Texas | 90.38 | 92.31 | 96.15 | 90.38 | 98.08 | 93.46±3.50 | 93.46±3.50 |
| deberta | Joint encoder+GNN | ASDGN | Wisconsin | 81.54 | 90.77 | 87.69 | 89.23 | 90.77 | 88.00±3.83 | 88.00±3.83 |
| deberta | Joint encoder+GNN | ASDGN | Cornell | 86.00 | 92.00 | 94.00 | 96.00 | 92.00 | 92.00±3.74 | 92.00±3.74 |
| deberta | Joint encoder+GNN | DirGNN | Cora | 85.42 | 84.87 | 87.27 | 88.01 | 82.84 | 85.68±2.05 | 86.35±1.29 |
| deberta | Joint encoder+GNN | DirGNN | arXiv-2023 | 93.64 | 93.93 | 93.35 | 92.77 | 93.35 | 93.41±0.43 | 93.12±0.94 |
| deberta | Joint encoder+GNN | DirGNN | ogbn-products | 73.75 | 65.00 | 62.50 | 81.25 | 73.75 | 71.25±7.55 | 71.25±7.55 |
| deberta | Joint encoder+GNN | DirGNN | Texas | 96.15 | 94.23 | 96.15 | 96.15 | 98.08 | 96.15±1.36 | 96.15±1.36 |
| deberta | Joint encoder+GNN | DirGNN | Wisconsin | 90.77 | 92.31 | 90.77 | 92.31 | 93.85 | 92.00±1.29 | 92.00±1.29 |
| deberta | Joint encoder+GNN | DirGNN | Cornell | 96.00 | 96.00 | 98.00 | 100.00 | 96.00 | 97.20±1.79 | 97.20±1.79 |
| deberta | Joint encoder+GNN | ACM-GNN | Cora | 78.04 | 82.47 | 84.13 | 83.95 | 79.52 | 81.62±2.72 | 81.33±3.07 |
| deberta | Joint encoder+GNN | ACM-GNN | arXiv-2023 | 94.80 | 90.46 | 89.60 | 93.35 | 91.62 | 91.97±2.12 | 91.04±4.64 |
| deberta | Joint encoder+GNN | ACM-GNN | ogbn-products | 66.25 | 67.50 | 62.50 | 66.25 | 66.25 | 65.75±1.90 | 65.75±1.90 |
| deberta | Joint encoder+GNN | ACM-GNN | Texas | 92.31 | 94.23 | 96.15 | 98.08 | 98.08 | 95.77±2.51 | 95.77±2.51 |
| deberta | Joint encoder+GNN | ACM-GNN | Wisconsin | 76.92 | 93.85 | 86.15 | 90.77 | 90.77 | 87.69±6.62 | 87.69±6.62 |
| deberta | Joint encoder+GNN | ACM-GNN | Cornell | 96.00 | 92.00 | 94.00 | 92.00 | 90.00 | 92.80±2.28 | 92.80±2.28 |
| deberta | Joint encoder+GNN | DMP | Cora | 26.57 | 21.59 | 38.01 | 20.48 | 30.63 | 27.46±7.16 | 25.24±7.63 |
| deberta | Joint encoder+GNN | DMP | arXiv-2023 | 93.64 | 93.06 | 90.46 | 91.62 | 89.31 | 91.62±1.79 | 93.29±1.15 |
| deberta | Joint encoder+GNN | DMP | ogbn-products | 42.50 | 55.00 | 61.25 | 67.50 | 66.25 | 58.50±10.21 | 58.50±10.21 |
| deberta | Joint encoder+GNN | DMP | Texas | 59.62 | 30.77 | 46.15 | 65.38 | 42.31 | 48.85±13.84 | 48.85±13.84 |
| deberta | Joint encoder+GNN | DMP | Wisconsin | 13.85 | 30.77 | 1.54 | 40.00 | 26.15 | 22.46±15.02 | 22.46±15.02 |
| deberta | Joint encoder+GNN | DMP | Cornell | 24.00 | 44.00 | 54.00 | 40.00 | 44.00 | 41.20±10.92 | 41.20±10.92 |
| deberta | Joint encoder+GNN | FSGNN | Cora | 88.19 | 85.42 | 89.48 | 87.08 | 88.01 | 87.64±1.51 | 87.23±1.65 |
| deberta | Joint encoder+GNN | FSGNN | arXiv-2023 | 93.93 | 94.51 | 93.93 | 92.49 | 91.62 | 93.30±1.20 | 93.70±0.75 |
| deberta | Joint encoder+GNN | FSGNN | ogbn-products | 75.00 | 67.50 | 65.00 | 82.50 | 77.50 | 73.50±7.20 | 73.50±7.20 |
| deberta | Joint encoder+GNN | FSGNN | Texas | 94.23 | 94.23 | 98.08 | 98.08 | 98.08 | 96.54±2.11 | 96.54±2.11 |
| deberta | Joint encoder+GNN | FSGNN | Wisconsin | 89.23 | 89.23 | 90.77 | 92.31 | 89.23 | 90.15±1.38 | 90.15±1.38 |
| deberta | Joint encoder+GNN | FSGNN | Cornell | 88.00 | 92.00 | 96.00 | 96.00 | 90.00 | 92.40±3.58 | 92.40±3.58 |
| deberta | Joint encoder+GNN | GraphTARIF | Cora | 81.37 | 81.92 | 82.29 | 84.13 | 83.95 | 82.73±1.24 | 82.40±1.04 |
| deberta | Joint encoder+GNN | GraphTARIF | arXiv-2023 | 92.49 | 90.46 | 92.20 | 92.77 | 89.88 | 91.56±1.30 | 92.02±1.22 |
| deberta | Joint encoder+GNN | GraphTARIF | ogbn-products | 68.75 | 58.75 | 58.75 | 66.25 | 67.50 | 64.00±4.87 | 64.00±4.87 |
| deberta | Joint encoder+GNN | GraphTARIF | Texas | 82.69 | 92.31 | 17.31 | 92.31 | 98.08 | 76.54±33.57 | 76.54±33.60 |
| deberta | Joint encoder+GNN | GraphTARIF | Wisconsin | 90.77 | 92.31 | 84.62 | 92.31 | 81.54 | 88.31±4.94 | 88.31±4.94 |
| deberta | Joint encoder+GNN | GraphTARIF | Cornell | 92.00 | 84.00 | 96.00 | 74.00 | 88.00 | 86.80±8.44 | 86.80±8.44 |

## Table roberta — RoBERTa-base TA+LM / TA+GNN / TAPTN+LM

| Encoder | Pipeline | Method | Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Recorded | Paper |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| roberta | TA+LM | TA | Cora | 28.60 | 69.93 | 71.40 | 77.86 | 69.93 | 63.54±19.81 | 63.54±19.81 |
| roberta | TA+LM | TA | arXiv-2023 | 92.20 | 89.88 | 93.35 | 89.02 | 89.31 | 90.75±1.92 | 90.75±1.92 |
| roberta | TA+LM | TA | ogbn-products | 71.25 | 56.25 | 61.25 | 75.00 | 70.00 | 66.75±7.74 | 66.75±7.74 |
| roberta | TA+LM | TA | Cornell | 92.00 | 94.00 | 96.00 | 92.00 | 94.00 | 93.60±1.67 | 93.60±1.67 |
| roberta | TA+LM | TA | Texas | 94.23 | 94.23 | 96.15 | 94.23 | 96.15 | 95.00±1.05 | 95.00±1.05 |
| roberta | TA+LM | TA | Wisconsin | 90.77 | 93.85 | 89.23 | 95.38 | 93.85 | 92.62±2.53 | 92.62±2.53 |
| roberta | TA+GNN (frozen) | GAT | Cora | 28.60 | 79.70 | 83.95 | 83.76 | 81.55 | 71.51±24.05 | 71.51±24.05 |
| roberta | TA+GNN (frozen) | GAT | arXiv-2023 | 90.17 | 88.73 | 87.86 | 89.60 | 91.62 | 89.60±1.43 | 89.60±1.43 |
| roberta | TA+GNN (frozen) | GAT | ogbn-products | 70.00 | 57.50 | 53.75 | 67.50 | 63.75 | 62.50±6.79 | 62.50±6.79 |
| roberta | TA+GNN (frozen) | GAT | Cornell | 56.00 | 70.00 | 60.00 | 70.00 | 60.00 | 63.20±6.42 | 63.20±6.42 |
| roberta | TA+GNN (frozen) | GAT | Texas | 78.85 | 69.23 | 69.23 | 78.85 | 61.54 | 71.54±7.37 | 71.54±7.37 |
| roberta | TA+GNN (frozen) | GAT | Wisconsin | 60.00 | 66.15 | 72.31 | 63.08 | 66.15 | 65.54±4.56 | 65.54±4.56 |
| roberta | TA+GNN (frozen) | GATv2 | Cora | 28.60 | 81.73 | 82.29 | 82.10 | 82.29 | 71.40±23.93 | 71.40±23.93 |
| roberta | TA+GNN (frozen) | GATv2 | arXiv-2023 | 91.62 | 90.46 | 93.35 | 90.46 | 92.49 | 91.68±1.27 | 91.68±1.27 |
| roberta | TA+GNN (frozen) | GATv2 | ogbn-products | 75.00 | 68.75 | 66.25 | 83.75 | 73.75 | 73.50±6.75 | 73.50±6.75 |
| roberta | TA+GNN (frozen) | GATv2 | Cornell | 72.00 | 82.00 | 80.00 | 78.00 | 84.00 | 79.20±4.60 | 79.20±4.60 |
| roberta | TA+GNN (frozen) | GATv2 | Texas | 84.62 | 88.46 | 76.92 | 90.38 | 96.15 | 87.31±7.14 | 87.31±7.14 |
| roberta | TA+GNN (frozen) | GATv2 | Wisconsin | 67.69 | 75.38 | 87.69 | 78.46 | 69.23 | 75.69±8.02 | 75.69±8.02 |
| roberta | TA+GNN (frozen) | SAGE | Cora | 28.60 | 82.66 | 82.10 | 83.76 | 81.73 | 71.77±24.14 | 71.77±24.14 |
| roberta | TA+GNN (frozen) | SAGE | arXiv-2023 | 91.62 | 90.75 | 92.20 | 90.75 | 90.75 | 91.21±0.67 | 91.21±0.67 |
| roberta | TA+GNN (frozen) | SAGE | ogbn-products | 73.75 | 65.00 | 66.25 | 82.50 | 67.50 | 71.00±7.26 | 71.00±7.26 |
| roberta | TA+GNN (frozen) | SAGE | Cornell | 92.00 | 94.00 | 98.00 | 92.00 | 92.00 | 93.60±2.61 | 93.60±2.61 |
| roberta | TA+GNN (frozen) | SAGE | Texas | 94.23 | 94.23 | 96.15 | 94.23 | 96.15 | 95.00±1.05 | 95.00±1.05 |
| roberta | TA+GNN (frozen) | SAGE | Wisconsin | 92.31 | 93.85 | 89.23 | 93.85 | 93.85 | 92.62±2.01 | 92.62±2.01 |
| roberta | TA+GNN (frozen) | ChebNet | Cora | 26.94 | 81.55 | 81.37 | 85.61 | 81.92 | 71.48±24.96 | 71.48±24.96 |
| roberta | TA+GNN (frozen) | ChebNet | arXiv-2023 | 91.33 | 89.60 | 91.33 | 89.60 | 90.46 | 90.46±0.87 | 90.46±0.87 |
| roberta | TA+GNN (frozen) | ChebNet | ogbn-products | 72.50 | 67.50 | 58.75 | 77.50 | 68.75 | 69.00±6.93 | 69.00±6.93 |
| roberta | TA+GNN (frozen) | ChebNet | Cornell | 84.00 | 90.00 | 84.00 | 80.00 | 80.00 | 83.60±4.10 | 83.60±4.10 |
| roberta | TA+GNN (frozen) | ChebNet | Texas | 84.62 | 90.38 | 88.46 | 86.54 | 90.38 | 88.08±2.50 | 88.08±2.50 |
| roberta | TA+GNN (frozen) | ChebNet | Wisconsin | 87.69 | 90.77 | 87.69 | 84.62 | 84.62 | 87.08±2.57 | 87.08±2.57 |
| roberta | TA+GNN (frozen) | GCNII | Cora | 28.41 | 71.96 | 75.28 | 78.04 | 76.38 | 66.01±21.14 | 66.01±21.14 |
| roberta | TA+GNN (frozen) | GCNII | arXiv-2023 | 91.62 | 89.88 | 92.20 | 89.88 | 89.88 | 90.69±1.13 | 90.69±1.13 |
| roberta | TA+GNN (frozen) | GCNII | ogbn-products | 72.50 | 55.00 | 53.75 | 71.25 | 66.25 | 63.75±8.88 | 63.75±8.88 |
| roberta | TA+GNN (frozen) | GCNII | Cornell | 92.00 | 92.00 | 96.00 | 88.00 | 92.00 | 92.00±2.83 | 92.00±2.83 |
| roberta | TA+GNN (frozen) | GCNII | Texas | 94.23 | 94.23 | 90.38 | 92.31 | 96.15 | 93.46±2.19 | 93.46±2.19 |
| roberta | TA+GNN (frozen) | GCNII | Wisconsin | 92.31 | 93.85 | 86.15 | 92.31 | 93.85 | 91.69±3.19 | 91.69±3.19 |
| roberta | TA+GNN (frozen) | DirGNN | Cora | 27.12 | 80.07 | 82.84 | 85.42 | 82.10 | 71.51±24.89 | 71.51±24.89 |
| roberta | TA+GNN (frozen) | DirGNN | arXiv-2023 | 92.77 | 91.91 | 93.64 | 90.17 | 90.75 | 91.85±1.42 | 91.85±1.42 |
| roberta | TA+GNN (frozen) | DirGNN | ogbn-products | 72.50 | 71.25 | 66.25 | 76.25 | 72.50 | 71.75±3.60 | 71.75±3.60 |
| roberta | TA+GNN (frozen) | DirGNN | Cornell | 100.00 | 98.00 | 98.00 | 98.00 | 92.00 | 97.20±3.03 | 97.20±3.03 |
| roberta | TA+GNN (frozen) | DirGNN | Texas | 96.15 | 92.31 | 94.23 | 98.08 | 94.23 | 95.00±2.19 | 95.00±2.19 |
| roberta | TA+GNN (frozen) | DirGNN | Wisconsin | 92.31 | 92.31 | 92.31 | 95.38 | 93.85 | 93.23±1.37 | 93.23±1.37 |
| roberta | TA+GNN (frozen) | ACM-GNN | Cora | 28.60 | 81.37 | 79.15 | 82.10 | 82.47 | 70.74±23.59 | 70.74±23.59 |
| roberta | TA+GNN (frozen) | ACM-GNN | arXiv-2023 | 92.20 | 89.88 | 93.93 | 90.17 | 91.91 | 91.62±1.65 | 91.62±1.65 |
| roberta | TA+GNN (frozen) | ACM-GNN | ogbn-products | 70.00 | 60.00 | 61.25 | 73.75 | 68.75 | 66.75±5.90 | 66.75±5.90 |
| roberta | TA+GNN (frozen) | ACM-GNN | Cornell | 92.00 | 94.00 | 98.00 | 92.00 | 94.00 | 94.00±2.45 | 94.00±2.45 |
| roberta | TA+GNN (frozen) | ACM-GNN | Texas | 94.23 | 94.23 | 92.31 | 94.23 | 96.15 | 94.23±1.36 | 94.23±1.36 |
| roberta | TA+GNN (frozen) | ACM-GNN | Wisconsin | 92.31 | 93.85 | 89.23 | 93.85 | 93.85 | 92.62±2.01 | 92.62±2.01 |
| roberta | TA+GNN (frozen) | DMP | Cora | 23.06 | 18.82 | 38.38 | 29.52 | 47.97 | 31.55±11.77 | 31.55±11.77 |
| roberta | TA+GNN (frozen) | DMP | arXiv-2023 | 91.62 | 90.46 | 92.77 | 88.73 | 91.62 | 91.04±1.53 | 91.04±1.53 |
| roberta | TA+GNN (frozen) | DMP | ogbn-products | 67.50 | 57.50 | 51.25 | 76.25 | 66.25 | 63.75±9.64 | 63.75±9.64 |
| roberta | TA+GNN (frozen) | DMP | Cornell | 58.00 | 40.00 | 56.00 | 30.00 | 50.00 | 46.80±11.71 | 46.80±11.71 |
| roberta | TA+GNN (frozen) | DMP | Texas | 51.92 | 67.31 | 69.23 | 63.46 | 44.23 | 59.23±10.74 | 59.23±10.74 |
| roberta | TA+GNN (frozen) | DMP | Wisconsin | 50.77 | 38.46 | 36.92 | 46.15 | 27.69 | 40.00±8.91 | 40.00±8.91 |
| roberta | TA+GNN (frozen) | FSGNN | Cora | 32.47 | 80.26 | 87.64 | 86.35 | 84.13 | 74.17±23.48 | 74.17±23.48 |
| roberta | TA+GNN (frozen) | FSGNN | arXiv-2023 | 92.77 | 90.75 | 94.22 | 91.62 | 93.06 | 92.48±1.34 | 92.48±1.34 |
| roberta | TA+GNN (frozen) | FSGNN | ogbn-products | 75.00 | 68.75 | 75.00 | 81.25 | 75.00 | 75.00±4.42 | 75.00±4.42 |
| roberta | TA+GNN (frozen) | FSGNN | Cornell | 92.00 | 92.00 | 96.00 | 90.00 | 94.00 | 92.80±2.28 | 92.80±2.28 |
| roberta | TA+GNN (frozen) | FSGNN | Texas | 92.31 | 94.23 | 94.23 | 90.38 | 96.15 | 93.46±2.19 | 93.46±2.19 |
| roberta | TA+GNN (frozen) | FSGNN | Wisconsin | 92.31 | 93.85 | 87.69 | 92.31 | 93.85 | 92.00±2.53 | 92.00±2.53 |
| roberta | TA+GNN (frozen) | APPNP | Cora | 28.60 | 84.50 | 86.72 | 88.93 | 87.08 | 75.17±26.08 | 75.17±26.08 |
| roberta | TA+GNN (frozen) | APPNP | arXiv-2023 | 91.33 | 92.20 | 92.20 | 91.33 | 92.20 | 91.85±0.48 | 91.85±0.48 |
| roberta | TA+GNN (frozen) | APPNP | ogbn-products | 72.50 | 70.00 | 71.25 | 86.25 | 77.50 | 75.50±6.65 | 75.50±6.65 |
| roberta | TA+GNN (frozen) | APPNP | Cornell | 60.00 | 82.00 | 78.00 | 74.00 | 76.00 | 74.00±8.37 | 74.00±8.37 |
| roberta | TA+GNN (frozen) | APPNP | Texas | 76.92 | 71.15 | 78.85 | 84.62 | 71.15 | 76.54±5.68 | 76.54±5.68 |
| roberta | TA+GNN (frozen) | APPNP | Wisconsin | 67.69 | 69.23 | 70.77 | 75.38 | 80.00 | 72.61±5.03 | 72.61±5.03 |
| roberta | TA+GNN (frozen) | GraphTARIF | Cora | 28.60 | 81.37 | 81.92 | 79.34 | 79.70 | 70.19±23.27 | 70.19±23.27 |
| roberta | TA+GNN (frozen) | GraphTARIF | arXiv-2023 | 91.91 | 89.88 | 91.91 | 90.17 | 90.46 | 90.87±0.97 | 90.87±0.97 |
| roberta | TA+GNN (frozen) | GraphTARIF | ogbn-products | 67.50 | 62.50 | 56.25 | 75.00 | 70.00 | 66.25±7.18 | 66.25±7.18 |
| roberta | TA+GNN (frozen) | GraphTARIF | Cornell | 90.00 | 94.00 | 94.00 | 94.00 | 94.00 | 93.20±1.79 | 93.20±1.79 |
| roberta | TA+GNN (frozen) | GraphTARIF | Texas | 94.23 | 94.23 | 92.31 | 96.15 | 96.15 | 94.61±1.61 | 94.61±1.61 |
| roberta | TA+GNN (frozen) | GraphTARIF | Wisconsin | 92.31 | 93.85 | 89.23 | 95.38 | 93.85 | 92.92±2.33 | 92.92±2.33 |
| roberta | TA+GNN (frozen) | RevGAT | Cora | 52.40 | 83.58 | 79.89 | 81.00 | 80.63 | 75.50±12.99 | 75.50±12.99 |
| roberta | TA+GNN (frozen) | RevGAT | arXiv-2023 | 93.06 | 90.46 | 92.77 | 91.91 | 92.49 | 92.14±1.03 | 92.14±1.03 |
| roberta | TA+GNN (frozen) | RevGAT | ogbn-products | 77.50 | 66.25 | 65.00 | 78.75 | 71.25 | 71.75±6.29 | 71.75±6.29 |
| roberta | TA+GNN (frozen) | RevGAT | Cornell | 92.00 | 94.00 | 96.00 | 92.00 | 94.00 | 93.60±1.67 | 93.60±1.67 |
| roberta | TA+GNN (frozen) | RevGAT | Texas | 94.23 | 92.31 | 94.23 | 94.23 | 96.15 | 94.23±1.36 | 94.23±1.36 |
| roberta | TA+GNN (frozen) | RevGAT | Wisconsin | 90.77 | 93.85 | 92.31 | 95.38 | 93.85 | 93.23±1.75 | 93.23±1.75 |
| roberta | TAPTN+LM | TAPTN+LM | Cora | 82.66 | 79.89 | 78.78 | 81.55 | 82.66 | 81.11±1.73 | 81.11±1.73 |
| roberta | TAPTN+LM | TAPTN+LM | arXiv-2023 | 95.38 | 93.35 | 93.64 | 91.62 | 92.77 | 93.35±1.37 | 93.35±1.37 |
| roberta | TAPTN+LM | TAPTN+LM | ogbn-products | 87.50 | 87.50 | 90.00 | 93.75 | 87.50 | 89.25±2.74 | 89.25±2.74 |
| roberta | TAPTN+LM | TAPTN+LM | Cornell | 98.00 | 100.00 | 98.00 | 98.00 | 98.00 | 98.40±0.89 | 98.40±0.89 |
| roberta | TAPTN+LM | TAPTN+LM | Texas | 100.00 | 98.08 | 96.15 | 98.08 | 98.08 | 98.08±1.36 | 98.08±1.36 |
| roberta | TAPTN+LM | TAPTN+LM | Wisconsin | 86.15 | 89.23 | 86.15 | 84.62 | 92.31 | 87.69±3.08 | 87.69±3.08 |

## Transferability — TAPTN embeddings + frozen GNN (DeBERTa)

| Encoder | Pipeline | Method | Dataset | Run1 | Run2 | Run3 | Run4 | Run5 | Recorded | Paper |
|---|---|---|---|---:|---:|---:|---:|---:|---|---|
| deberta | TAPTN embeddings + frozen GNN | GAT | Cora | 87.08 | 83.76 | 84.69 | 86.72 | 81.37 | 84.72±2.33 | — |
| deberta | TAPTN embeddings + frozen GNN | GAT | arXiv-2023 | 92.49 | 92.49 | 91.91 | 90.75 | 94.22 | 92.37±1.25 | — |
| deberta | TAPTN embeddings + frozen GNN | GAT | ogbn-products | 52.50 | 76.25 | 77.50 | 88.75 | 80.00 | 75.00±13.49 | — |
| deberta | TAPTN embeddings + frozen GNN | GAT | Cornell | 76.00 | 76.00 | 68.00 | 64.00 | 70.00 | 70.80±5.22 | — |
| deberta | TAPTN embeddings + frozen GNN | GAT | Texas | 75.00 | 76.92 | 76.92 | 78.85 | 71.15 | 75.77±2.92 | — |
| deberta | TAPTN embeddings + frozen GNN | GAT | Wisconsin | 58.46 | 66.15 | 60.00 | 66.15 | 73.85 | 64.92±6.10 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAGE | Cora | 88.19 | 85.79 | 85.98 | 84.13 | 83.03 | 85.42±1.97 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAGE | arXiv-2023 | 93.93 | 94.22 | 92.20 | 92.49 | 93.64 | 93.30±0.90 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAGE | ogbn-products | 81.25 | 81.25 | 76.25 | 92.50 | 82.50 | 82.75±5.96 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAGE | Cornell | 96.00 | 100.00 | 98.00 | 94.00 | 98.00 | 97.20±2.28 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAGE | Texas | 96.15 | 96.15 | 94.23 | 98.08 | 98.08 | 96.54±1.61 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAGE | Wisconsin | 89.23 | 87.69 | 89.23 | 84.62 | 89.23 | 88.00±2.00 | — |
| deberta | TAPTN embeddings + frozen GNN | ChebNet | Cora | 88.01 | 84.87 | 87.64 | 84.13 | 81.37 | 85.20±2.73 | — |
| deberta | TAPTN embeddings + frozen GNN | ChebNet | arXiv-2023 | 92.49 | 93.06 | 91.04 | 93.35 | 92.77 | 92.54±0.90 | — |
| deberta | TAPTN embeddings + frozen GNN | ChebNet | ogbn-products | 76.25 | 78.75 | 77.50 | 78.75 | 77.50 | 77.75±1.05 | — |
| deberta | TAPTN embeddings + frozen GNN | ChebNet | Cornell | 84.00 | 98.00 | 86.00 | 90.00 | 96.00 | 90.80±6.10 | — |
| deberta | TAPTN embeddings + frozen GNN | ChebNet | Texas | 88.46 | 96.15 | 88.46 | 96.15 | 82.69 | 90.38±5.77 | — |
| deberta | TAPTN embeddings + frozen GNN | ChebNet | Wisconsin | 81.54 | 83.08 | 89.23 | 80.00 | 84.62 | 83.69±3.54 | — |
| deberta | TAPTN embeddings + frozen GNN | DGI | Cora | 86.72 | 85.42 | 86.53 | 85.24 | 84.50 | 85.68±0.93 | — |
| deberta | TAPTN embeddings + frozen GNN | DGI | arXiv-2023 | 93.64 | 94.51 | 92.77 | 92.77 | 94.22 | 93.58±0.80 | — |
| deberta | TAPTN embeddings + frozen GNN | DGI | ogbn-products | 83.75 | 82.50 | 77.50 | 87.50 | 85.00 | 83.25±3.71 | — |
| deberta | TAPTN embeddings + frozen GNN | DGI | Cornell | 80.00 | 88.00 | 80.00 | 88.00 | 86.00 | 84.40±4.10 | — |
| deberta | TAPTN embeddings + frozen GNN | DGI | Texas | 88.46 | 78.85 | 94.23 | 88.46 | 94.23 | 88.85±6.29 | — |
| deberta | TAPTN embeddings + frozen GNN | DGI | Wisconsin | 75.38 | 81.54 | 78.46 | 80.00 | 76.92 | 78.46±2.43 | — |
| deberta | TAPTN embeddings + frozen GNN | GATv2 | Cora | 87.27 | 83.95 | 85.42 | 86.16 | 83.03 | 85.17±1.70 | — |
| deberta | TAPTN embeddings + frozen GNN | GATv2 | arXiv-2023 | 92.49 | 94.51 | 92.49 | 93.06 | 94.22 | 93.35±0.96 | — |
| deberta | TAPTN embeddings + frozen GNN | GATv2 | ogbn-products | 65.00 | 78.75 | 76.25 | 85.00 | 81.25 | 77.25±7.57 | — |
| deberta | TAPTN embeddings + frozen GNN | GATv2 | Cornell | 84.00 | 80.00 | 88.00 | 82.00 | 96.00 | 86.00±6.32 | — |
| deberta | TAPTN embeddings + frozen GNN | GATv2 | Texas | 75.00 | 88.46 | 92.31 | 94.23 | 88.46 | 87.69±7.52 | — |
| deberta | TAPTN embeddings + frozen GNN | GATv2 | Wisconsin | 66.15 | 72.31 | 78.46 | 81.54 | 86.15 | 76.92±7.84 | — |
| deberta | TAPTN embeddings + frozen GNN | GCNII | Cora | 85.06 | 84.50 | 82.10 | 82.47 | 81.55 | 83.14±1.55 | — |
| deberta | TAPTN embeddings + frozen GNN | GCNII | arXiv-2023 | 94.22 | 92.77 | 90.17 | 92.77 | 91.04 | 92.19±1.60 | — |
| deberta | TAPTN embeddings + frozen GNN | GCNII | ogbn-products | 78.75 | 75.00 | 78.75 | 86.25 | 80.00 | 79.75±4.09 | — |
| deberta | TAPTN embeddings + frozen GNN | GCNII | Cornell | 96.00 | 94.00 | 96.00 | 90.00 | 94.00 | 94.00±2.45 | — |
| deberta | TAPTN embeddings + frozen GNN | GCNII | Texas | 96.15 | 96.15 | 94.23 | 98.08 | 100.00 | 96.92±2.19 | — |
| deberta | TAPTN embeddings + frozen GNN | GCNII | Wisconsin | 83.08 | 89.23 | 86.15 | 87.69 | 86.15 | 86.46±2.28 | — |
| deberta | TAPTN embeddings + frozen GNN | ASDGN | Cora | 84.32 | 84.32 | 84.32 | 82.29 | 80.26 | 83.10±1.82 | — |
| deberta | TAPTN embeddings + frozen GNN | ASDGN | arXiv-2023 | 94.80 | 93.64 | 92.77 | 93.64 | 92.49 | 93.47±0.91 | — |
| deberta | TAPTN embeddings + frozen GNN | ASDGN | ogbn-products | 87.50 | 80.00 | 82.50 | 88.75 | 80.00 | 83.75±4.15 | — |
| deberta | TAPTN embeddings + frozen GNN | ASDGN | Cornell | 98.00 | 100.00 | 92.00 | 100.00 | 98.00 | 97.60±3.29 | — |
| deberta | TAPTN embeddings + frozen GNN | ASDGN | Texas | 100.00 | 94.23 | 96.15 | 96.15 | 98.08 | 96.92±2.19 | — |
| deberta | TAPTN embeddings + frozen GNN | ASDGN | Wisconsin | 87.69 | 89.23 | 89.23 | 84.62 | 87.69 | 87.69±1.88 | — |
| deberta | TAPTN embeddings + frozen GNN | DirGNN | Cora | 88.01 | 85.42 | 86.16 | 85.06 | 83.39 | 85.61±1.68 | — |
| deberta | TAPTN embeddings + frozen GNN | DirGNN | arXiv-2023 | 93.93 | 95.09 | 92.77 | 93.93 | 93.93 | 93.93±0.82 | — |
| deberta | TAPTN embeddings + frozen GNN | DirGNN | ogbn-products | 81.25 | 82.50 | 78.75 | 90.00 | 80.00 | 82.50±4.42 | — |
| deberta | TAPTN embeddings + frozen GNN | DirGNN | Cornell | 98.00 | 100.00 | 92.00 | 100.00 | 96.00 | 97.20±3.35 | — |
| deberta | TAPTN embeddings + frozen GNN | DirGNN | Texas | 100.00 | 98.08 | 96.15 | 96.15 | 100.00 | 98.08±1.93 | — |
| deberta | TAPTN embeddings + frozen GNN | DirGNN | Wisconsin | 86.15 | 89.23 | 89.23 | 86.15 | 90.77 | 88.31±2.07 | — |
| deberta | TAPTN embeddings + frozen GNN | ACM-GNN | Cora | 85.98 | 84.13 | 85.79 | 81.37 | 79.70 | 83.39±2.77 | — |
| deberta | TAPTN embeddings + frozen GNN | ACM-GNN | arXiv-2023 | 93.35 | 94.80 | 93.06 | 92.49 | 93.35 | 93.41±0.85 | — |
| deberta | TAPTN embeddings + frozen GNN | ACM-GNN | ogbn-products | 82.50 | 83.75 | 76.25 | 90.00 | 82.50 | 83.00±4.89 | — |
| deberta | TAPTN embeddings + frozen GNN | ACM-GNN | Cornell | 94.00 | 100.00 | 98.00 | 100.00 | 98.00 | 98.00±2.45 | — |
| deberta | TAPTN embeddings + frozen GNN | ACM-GNN | Texas | 100.00 | 96.15 | 92.31 | 96.15 | 100.00 | 96.92±3.22 | — |
| deberta | TAPTN embeddings + frozen GNN | ACM-GNN | Wisconsin | 84.62 | 89.23 | 87.69 | 84.62 | 84.62 | 86.16±2.17 | — |
| deberta | TAPTN embeddings + frozen GNN | DMP | Cora | 46.49 | 30.07 | 29.15 | 8.49 | 37.27 | 30.29±14.03 | — |
| deberta | TAPTN embeddings + frozen GNN | DMP | arXiv-2023 | 92.49 | 94.51 | 91.04 | 92.77 | 92.49 | 92.66±1.24 | — |
| deberta | TAPTN embeddings + frozen GNN | DMP | ogbn-products | 77.50 | 72.50 | 58.75 | 82.50 | 72.50 | 72.75±8.86 | — |
| deberta | TAPTN embeddings + frozen GNN | DMP | Cornell | 56.00 | 46.00 | 56.00 | 58.00 | 70.00 | 57.20±8.56 | — |
| deberta | TAPTN embeddings + frozen GNN | DMP | Texas | 53.85 | 67.31 | 69.23 | 32.69 | 69.23 | 58.46±15.78 | — |
| deberta | TAPTN embeddings + frozen GNN | DMP | Wisconsin | 47.69 | 47.69 | 53.85 | 46.15 | 50.77 | 49.23±3.08 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAINT | Cora | 87.08 | 85.42 | 83.39 | 84.69 | 80.81 | 84.28±2.35 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAINT | arXiv-2023 | 93.35 | 93.93 | 91.33 | 91.33 | 92.77 | 92.54±1.18 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAINT | ogbn-products | 83.75 | 85.00 | 80.00 | 90.00 | 60.00 | 79.75±11.61 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAINT | Cornell | 84.00 | 82.00 | 74.00 | 78.00 | 82.00 | 80.00±4.00 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAINT | Texas | 92.31 | 82.69 | 90.38 | 90.38 | 90.38 | 89.23±3.75 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphSAINT | Wisconsin | 76.92 | 81.54 | 87.69 | 80.00 | 67.69 | 78.77±7.33 | — |
| deberta | TAPTN embeddings + frozen GNN | FSGNN | Cora | 89.11 | 84.32 | 88.01 | 86.35 | 85.06 | 86.57±1.99 | — |
| deberta | TAPTN embeddings + frozen GNN | FSGNN | arXiv-2023 | 93.93 | 95.09 | 93.06 | 93.35 | 93.93 | 93.87±0.78 | — |
| deberta | TAPTN embeddings + frozen GNN | FSGNN | ogbn-products | 85.00 | 83.75 | 78.75 | 91.25 | 82.50 | 84.25±4.56 | — |
| deberta | TAPTN embeddings + frozen GNN | FSGNN | Cornell | 94.00 | 98.00 | 94.00 | 100.00 | 100.00 | 97.20±3.03 | — |
| deberta | TAPTN embeddings + frozen GNN | FSGNN | Texas | 94.23 | 96.15 | 96.15 | 96.15 | 100.00 | 96.54±2.11 | — |
| deberta | TAPTN embeddings + frozen GNN | FSGNN | Wisconsin | 87.69 | 89.23 | 86.15 | 83.08 | 89.23 | 87.08±2.57 | — |
| deberta | TAPTN embeddings + frozen GNN | APPNP | Cora | 89.30 | 85.61 | 88.01 | 87.64 | 85.06 | 87.12±1.76 | — |
| deberta | TAPTN embeddings + frozen GNN | APPNP | arXiv-2023 | 93.06 | 93.64 | 92.20 | 91.91 | 93.06 | 92.77±0.71 | — |
| deberta | TAPTN embeddings + frozen GNN | APPNP | ogbn-products | 86.25 | 85.00 | 80.00 | 91.25 | 85.00 | 85.50±4.01 | — |
| deberta | TAPTN embeddings + frozen GNN | APPNP | Cornell | 74.00 | 98.00 | 88.00 | 84.00 | 86.00 | 86.00±8.60 | — |
| deberta | TAPTN embeddings + frozen GNN | APPNP | Texas | 82.69 | 82.69 | 86.54 | 84.62 | 86.54 | 84.62±1.93 | — |
| deberta | TAPTN embeddings + frozen GNN | APPNP | Wisconsin | 67.69 | 73.85 | 70.77 | 70.77 | 76.92 | 72.00±3.51 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphTARIF | Cora | 84.69 | 85.06 | 84.32 | 83.58 | 82.10 | 83.95±1.17 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphTARIF | arXiv-2023 | 93.64 | 94.80 | 93.06 | 93.35 | 92.49 | 93.47±0.86 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphTARIF | ogbn-products | 86.25 | 81.25 | 78.75 | 88.75 | 81.25 | 83.25±4.11 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphTARIF | Cornell | 96.00 | 100.00 | 92.00 | 100.00 | 98.00 | 97.20±3.35 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphTARIF | Texas | 100.00 | 92.31 | 96.15 | 96.15 | 98.08 | 96.54±2.85 | — |
| deberta | TAPTN embeddings + frozen GNN | GraphTARIF | Wisconsin | 86.15 | 90.77 | 87.69 | 83.08 | 86.15 | 86.77±2.79 | — |

