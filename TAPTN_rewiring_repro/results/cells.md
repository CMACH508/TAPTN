# Camera-ready rewiring cells (revised)

Numbers below are the **revised** TMLR 2026 camera-ready entries. Dry-run `--rescore` compares pickle-derived values against these; remaining gaps are in `DISCREPANCIES.md`. Use `--paper-compat` to compare against the unrevised PDF.

## Table 1 — Homophily

| Dataset | Cornell | Texas | Washington | Wisconsin | ogbn-arxiv | Pubmed | Cora |
|---|---:|---:|---:|---:|---:|---:|---:|
| Homophily | 0.1308 | 0.1448 | 0.1599 | 0.1869 | 0.6358 | 0.7924 | 0.8252 |

## Table 2 — Label-revealing flipping / extreme (%)

\(O_F,R_F\): original / flipped. \(O_E,R_E\): original / extreme. \(\Delta = R - O\).

| Model | Arena | Dataset | O_F | R_F | O_E | R_E | Δ_F | Δ_E |
|---|---:|---|---:|---:|---:|---:|---:|---:|
| GPT-3.5-Turbo-0125 | 1224 | Cornell | 49.39 | 55.06 | 80.57 | 78.95 | +5.67 | -1.62 |
| | | Texas | 43.87 | 39.53 | 69.57 | 64.82 | -4.34 | -4.75 |
| | | Washington | 51.36 | 49.81 | 69.65 | 67.70 | -1.55 | -1.95 |
| | | Wisconsin | 42.04 | 39.17 | 67.52 | 63.38 | -2.87 | -4.14 |
| Phi-4 | 1255 | Cornell | 51.21 | 43.15 | 63.31 | 58.07 | -8.06 | -5.24 |
| | | Texas | 41.01 | 39.45 | 60.94 | 55.08 | -1.56 | -5.86 |
| | | Washington | 53.01 | 52.63 | 69.17 | 60.53 | -0.38 | -8.68 |
| | | Wisconsin | 49.84 | 48.60 | 65.73 | 60.75 | -1.24 | -4.98 |
| Gemma-2-9B | 1265 | Cornell | 54.44 | 43.95 | 71.77 | 63.71 | -10.49 | -8.06 |
| | | Texas | 44.53 | 35.94 | 67.58 | 55.86 | -8.59 | -11.72 |
| | | Washington | 54.51 | 50.38 | 72.56 | 63.91 | -4.13 | -8.65 |
| | | Wisconsin | 60.44 | 51.41 | 76.01 | 71.34 | -9.03 | -4.67 |
| Llama-3-70B-Instruct | 1276 | Cornell | 73.79 | 52.02 | 82.66 | 83.06 | -21.77 | +0.40 |
| | | Texas | 64.06 | 50.00 | 83.21 | 78.52 | -14.06 | -4.69 |
| | | Washington | 67.67 | 58.27 | 84.96 | 77.82 | -9.40 | -7.14 |
| | | Wisconsin | 71.65 | 59.81 | 84.42 | 83.80 | -11.84 | -0.62 |
| Llama-3.1-70B-Instruct | 1293 | Cornell | 77.02 | 55.24 | 83.87 | 83.06 | -21.78 | -0.81 |
| | | Texas | 69.53 | 52.73 | 84.38 | 80.47 | -16.80 | -3.91 |
| | | Washington | 70.68 | 60.53 | 81.96 | 77.82 | -10.15 | -4.14 |
| | | Wisconsin | 74.45 | 57.94 | 85.05 | 83.80 | -16.51 | -1.25 |
| Qwen2.5-VL-72B-Instruct | 1302 | Cornell | 73.39 | 55.65 | 81.04 | 72.98 | -17.74 | -8.06 |
| | | Texas | 69.92 | 48.44 | 76.95 | 73.04 | -21.48 | -3.91 |
| | | Washington | 63.91 | 54.89 | 68.80 | 62.78 | -9.02 | -6.02 |
| | | Wisconsin | 77.88 | 59.50 | 83.49 | 80.69 | -18.38 | -2.80 |
| Llama-3.3-70B-Instruct | 1319 | Cornell | 75.40 | 51.61 | 78.63 | 77.02 | -23.79 | -1.61 |
| | | Texas | 70.70 | 52.73 | 85.54 | 83.98 | -17.97 | -1.56 |
| | | Washington | 69.55 | 62.03 | 77.07 | 74.06 | -7.52 | -3.01 |
| | | Wisconsin | 74.45 | 60.44 | 81.93 | 78.19 | -14.01 | -3.74 |

## Table 3 — GPT-3.5 + instructions, flipping (%)

| Dataset | Original | Rewired | Δ |
|---|---:|---:|---:|
| Cornell | 61.54 | 59.92 | -1.62 |
| Washington | 52.14 | 50.97 | -1.17 |
| Wisconsin | 50.96 | 49.04 | -1.92 |
| Texas | 54.55 | 49.80 | -4.75 |

## Table 4

Sensitivity \(S = O - R\). Slopes per 100 LMArena points.

| Target | Statistic | Flipping | Extreme |
|---|---|---:|---:|
| \(S\) | Pearson \(r\) (\(p\)) | +0.9160 (0.0038) | -0.2521 (0.5855) |
| \(S\) | Spearman \(\rho\) (\(p\)) | +0.8928 (0.0068) | -0.5714 (0.1802) |
| \(S\) | \(R^2\) | 0.8391 | 0.0635 |
| \(S\) | Slope /100 | +0.1943 | -0.0176 |
| Orig Acc | Pearson \(r\) (\(p\)) | +0.8999 (0.0058) | +0.6422 (0.1199) |
| Rewired Acc | Pearson \(r\) (\(p\)) | +0.8414 (0.0176) | +0.5858 (0.1669) |

## Table 5 (%)

\(\Delta = \bar R - \bar O\).

| Model | Arena | \(\bar O\) | \(\bar R\) | Δ | Rel. |
|---|---:|---:|---:|---:|---:|
| GPT-3.5-Turbo-0125 | 1224 | 70.10 | 74.09 | +3.99 | +6.12 |
| Phi-4 | 1255 | 71.50 | 71.43 | -0.06 | +0.08 |
| Gemma-2-9B | 1265 | 77.03 | 72.47 | -4.57 | -5.96 |
| Llama-3-70B-Instruct | 1276 | 83.50 | 79.17 | -4.34 | -5.18 |
| Llama-3.1-70B-Instruct | 1293 | 87.26 | 82.06 | -5.21 | -5.98 |
| Qwen2.5-VL-72B-Instruct | 1302 | 83.29 | 77.67 | -5.62 | -6.69 |
| Llama-3.3-70B-Instruct | 1319 | 85.31 | 79.95 | -5.37 | -6.21 |
| **Mean** | | 79.71 | 76.69 | -3.02 | -3.40 |

## Table 6

| Target | Pearson \(r\) (\(p\)) | Spearman \(\rho\) (\(p\)) | \(R^2\) | Slope /100 |
|---|---|---|---:|---:|
| \(S\) | +0.8875 (0.0077) | +0.9286 (0.0025) | 0.7876 | +0.1013 |
| \(\bar O\) | +0.8880 (0.0076) | +0.8214 (0.0234) | 0.7886 | +0.1915 |
| \(\bar R\) | +0.7081 (0.0750) | +0.7143 (0.0713) | 0.5014 | +0.0902 |

## Table 26 (%)

| Model | Arena | \(\bar O\) | \(\bar R\) | Δ (err. inc.) |
|---|---:|---:|---:|---|
| GPT-OSS-120B | 1353 | 83.84 | 79.67 | −4.17 (+26%) |
| Qwen3.5-27B | 1409 | 93.33 | 86.01 | −7.31 (+110%) |
| Gemma-4-31B-it | 1451 | 96.55 | 94.67 | −1.88 (+55%) |
| GLM-5.1 | 1475 | 95.17 | 93.48 | −1.69 (+35%) |

## Table 29 (Δ pp vs intact TAPTN)

| Backbone | Dataset | Edge-blind Δ | Flipped Δ |
|---|---|---:|---:|
| Llama-3.3-70B | Texas | -8.20 | -18.36 |
| Llama-3.3-70B | Cornell | -8.87 | -17.33 |
| Llama-3.3-70B | Wisconsin | -18.38 | -13.08 |
| Qwen3.5-27B | Texas | -3.90 | -25.00 |
