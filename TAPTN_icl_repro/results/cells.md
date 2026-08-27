# Official camera-ready TAPTN ICL cells

Numbers as typeset in the TMLR 2026 camera-ready. Dry-run reprint:

```bash
python reproduce.py dry --table all
```

Re-score from pickles (cells with no artifact print as —):

```bash
python reproduce.py dry --table all --rescore
```

## `tab:main_5_datasets`

| Method | Cora | arXiv-2023 | Texas | Wisconsin | Cornell |
|---|---:|---:|---:|---:|---:|
| 0-hop (zero-shot CoT) | 59.40 | 76.19 | 66.93 | 68.85 | 72.58 |
| GraphICL+SAT 1-hop | 62.55 | 83.49 | 65.23 | 76.01 | 76.21 |
| GraphICL+SAT 2-hop | 63.47 | 85.40 | 66.41 | 70.72 | 72.18 |
| TAPTN 1-hop | 72.69 | 87.30 | 89.84 | 85.67 | 85.08 |
| TAPTN 2-hop | 73.80 | 88.57 | 91.80 | 87.23 | 86.69 |

## `tab:factorial`

| Method | Cora | arXiv-2023 | Texas | Wisconsin | Cornell | Avg. |
|---|---:|---:|---:|---:|---:|---:|
| GraphICL+SAT (no instr., no aggr.) | 63.47 | 85.40 | 66.41 | 70.72 | 72.18 | 71.64 |
| + instructions (no aggr.) | 73.25 | 86.03 | 86.72 | 83.18 | 81.45 | 82.13 |
| + iterative aggregation (no instr.) | 70.85 | 87.30 | 66.79 | 73.21 | 77.82 | 75.19 |
| TAPTN (instr. + aggr.) | 73.80 | 88.57 | 91.80 | 87.23 | 86.69 | 85.62 |

## `tab:product_70b`

| Method | Accuracy (%) |
|---|---:|
| GraphICL+SAT 2-hop (no instr., no aggr.) | 76.50 |
| GraphICL+SAT 2-hop (dense full-graph nbhd.) | 76.75 |
| TAPTN 1-hop | 83.75 |
| TAPTN 2-hop | 86.75 |

## `tab:cost`

| Method | Reason calls/node | Tokens/node | Cost ($/1k) | Acc. (%) |
|---|---:|---:|---:|---:|
| GraphICL+SAT 2-hop | 1.0 | ≈2.8k | 0.31 | 76.50 |
| GraphICL+SAT 2-hop (dense nbhd.) | 1.0 | ≈8.1k | 0.91 | 76.75 |
| TAPTN 2-hop (uniform 70B) | 4.6 | ≈22k | 2.52 | 86.75 |

## `tb_52`

| Method | Hop | w/o instruction (%) | w/ instruction (%) |
|---|---:|---:|---:|
| GraphICL | 1 | 74.75 | / |
|  | 2 | 75.00 | / |
| TAPTN | 1 | 74.75 | 74.75 |
|  | 2 | 74.00 | 76.75 |

## `tb_4`

| Method | Order | RI | Cora | arXiv-2023 |
|---|---:|---:|---:|---:|
| GraphICL | 1 | 0 | 62.55 | 83.49 |
|  |  | 1 | 69.19 | 74.92 |
|  |  | 2 | 68.45 | 73.65 |
|  | 2 | 0 | 63.47 | 85.40 |
|  |  | 1 | 69.18 | 75.87 |
|  |  | 2 | 68.45 | 80.00 |
| TAPTN | 1 | / | 72.69 | 87.30 |
|  | 2 | / | 73.80 | 88.57 |

## `tab:decouple`

| Dataset | Hop | w/o Struct, w/o Instr. | w/o Struct, w/ Instr. | w/ Struct, w/o Instr. | w/ Struct, w/ Instr. |
|---|---|---:|---:|---:|---:|
| Cora | 1-hop | 69.37 | 71.40 | 62.55 | 72.69 |
|  | 2-hop | 62.55 | 72.14 | 63.47 | 73.25 |
| arXiv-2023 | 1-hop | 81.59 | 85.71 | 83.49 | 87.30 |
|  | 2-hop | 82.54 | 86.35 | 85.40 | 86.03 |
| Texas | 1-hop | 45.31 | 78.52 | 65.23 | 89.84 |
|  | 2-hop | 57.03 | 73.44 | 66.41 | 86.72 |
| Wisconsin | 1-hop | 55.45 | 81.31 | 76.01 | 85.67 |
|  | 2-hop | 60.75 | 80.37 | 70.72 | 83.18 |
| Cornell | 1-hop | 56.45 | 76.61 | 76.21 | 85.08 |
|  | 2-hop | 62.10 | 76.61 | 72.18 | 81.45 |

## `tab:current_channel`

| Model | ego 0-hop | GraphICL 1-hop anon. | GraphICL+SAT 1-hop | TAPTN 1-hop |
|---|---:|---:|---:|---:|
| Llama-3.3-70B (anchor) | 66.80 | 45.31 | 64.06 | 87.11 |
| Gemma-4-31B-it | 71.88 | 92.19 | 95.70 | 96.48 |
| Qwen3.5-27B | 68.75 | 69.92 | 92.19 | 94.92 |
| GLM-5.1 | 83.59 | 92.97 | 94.53 | 96.09 |

## `tab:current_taptn`

| Dataset | Model | GraphICL+SAT (2-hop) | TAPTN (2-hop) | Δ |
|---|---|---:|---:|---:|
| Texas | Gemma-4-31B-it | 96.09 | 97.66 | +1.57 |
|  | Qwen3.5-27B | 91.80 | 97.66 | +5.86 |
|  | GLM-5.1 | 94.92 | 96.88 | +1.96 |
| Cora | Gemma-4-31B-it | 78.04 | 79.52 | +1.48 |
|  | Qwen3.5-27B | 78.97 | 80.07 | +1.10 |
|  | GLM-5.1 | 77.49 | 81.00 | +3.51 |
