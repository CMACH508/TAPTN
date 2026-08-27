# Dry-run / pickle rescore vs revised camera-ready tables

Official cells are the **revised** TMLR 2026 camera-ready (errata: Gemma/Phi-4 Cornell \(O_F\), GPT-3.5 \(\Delta_F\), Table 3 row swap + Texas 49.80, label-free GPT-3.5 \(\bar O\)). Regenerated figures are compared by data, not by PNG checksum.

To compare pickles against the *unrevised* PDF instead:

```bash
python reproduce.py dry --table all --rescore --paper-compat
```

Default (revised paper):

```bash
export TAPTN_ASSETS=/path/to/TAPTN_rewiring_repro_assets
python reproduce.py dry --table all --rescore --figures
```

## How to read a difference

- **Open-model Table 2 cells** are re-scored from `pkls/hop1/` and `pkls/hop2/` with the same `test_mask` + multi-stage label match as `analyze_rewiring_lmarena.py`.
- **GPT-3.5 Table 2** is re-scored from the legacy notebooks in `pkls/gpt35/` via `1 - |wrong_index|/|result|` (the rule that produced the PDF).
- **Label-free** cells are re-scored from `pkls/full_abs/`. Wisconsin GPT-3.5 original uses the true pickle (77.39%), not the hop-1 42.04 substitution.
- **Table 3** uses `pkls/tb3/` (Cornell and Texas only).
- **Table 29** uses the stored `accuracy` field inside each TAPTN pickle.
- **Table 1 WebKB** uses the supplementary formula: undirected edges, isolated nodes remain in the denominator.

## Cannot reproduce (missing artifacts)

### Table 3 Washington and Wisconsin

Instruction-run pickles exist only for Cornell and Texas. After swapping the Cornell/Washington rows in the paper, Cornell 61.54 / 59.92 matches `pkls/tb3/`. Washington 52.14 / 50.97 and Wisconsin 50.96 / 49.04 have **no pickle** in the author trees and are reprinted from the (swapped) typesetting.

### Citation-graph homophily

Table 1 ogbn-arxiv / Pubmed / Cora are recorded paper values; those graphs are not in this bundle. WebKB four cells are recovered from the shipped pickles.

### Static figures

Figure 1 (`neighbor_rewiring.png`), Figure 4 (`rewiring_case.png`), and Figure 11 (`former_rewire2.pdf`) are hand illustrations. Dry-run copies the camera-ready files; they are not generated from pickles. Regenerated Figure 2 / Figure 3 / Figures 8–10 use the same data and layout as the analysis script but are not pixel-identical.

## Matches the revised paper

- **Table 1** WebKB homophily (undirected, isolated-in-denominator)
- **Table 2** all 56 accuracy cells (Gemma Cornell \(O_F=54.44\), Phi-4 Cornell \(O_F=51.21\), GPT-3.5 Cornell \(\Delta_F=+5.67\))
- **Table 3** Cornell and Texas (rewired 49.80)
- **Table 4** flipping Pearson \(r=+0.9160\), \(p=0.0038\), \(R^2=0.8391\), slope \(+0.1943\) / 100 LMArena points
- **Table 5** seven models including GPT-3.5 \(\bar O=70.10\), mean \(\Delta=-3.02\)
- **Table 6** sensitivity Pearson \(r=+0.8875\), slope \(+0.1013\) / 100 pts
- **Table 26** all four new models
- **Table 29** all four backbone–dataset pairs (Cornell flip \(\Delta=-17.34\) vs PDF \(-17.33\): rounding)

## Errata vs the unrevised PDF (now applied in the camera-ready source)

These were pickle/arithmetic/typesetting errors; they are **not** remaining dry-run failures.

| Location | Unrevised PDF | Revised |
|---|---|---|
| Gemma Cornell \(O_F\) / \(\Delta_F\) | 49.39 / −5.44 | 54.44 / −10.49 |
| Phi-4 Cornell \(O_F\) / \(\Delta_F\) | 51.52 / −8.37 | 51.21 / −8.06 |
| GPT-3.5 Cornell \(\Delta_F\) | +6.67 | +5.67 |
| Table 3 Cornell ↔ Washington | rows swapped | Cornell 61.54/59.92; Washington 52.14/50.97 |
| Table 3 Texas rewired | 50.20 / −4.35 | 49.80 / −4.75 |
| Label-free GPT-3.5 \(\bar O\) | 61.26 (Wisconsin original filled with hop-1 42.04) | 70.10 |
| Label-free seven-model mean \(\Delta\) | −1.76 | −3.02 |
