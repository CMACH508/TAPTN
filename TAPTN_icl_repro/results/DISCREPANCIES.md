# Dry-run / pickle rescore vs camera-ready TAPTN ICL tables

Official cells are the TMLR 2026 camera-ready as in `results/official_cells.json`.
These ICL tables were **not** part of the rewiring errata (no `--paper-compat`).

```bash
export TAPTN_ASSETS=/path/to/TAPTN_icl_repro_assets
python reproduce.py dry --table all --rescore --figures
```

Tolerance: **0.06 percentage points** on accuracies; cost $/1k within 0.03; reason-calls/node within 0.15. Paper token counts are typeset with ≈.

## How cells are scored

- **Old GPT-3.5 / `*2_*` WebKB / `arxiv2023_*`:** `1 - |wrong_index|/|result|` (same rule as Table 2 GPT-3.5).
- **New-style `{ds}_hop{h}_{anon|noanon}_{guide|noguide}_{model}.pkl`:** stored `accuracy` (the field used in `generate_tables.py` for non-hardcoded cells). Do not mix this with `wrong_index` on the same file: WebKB stored uses a 256-denominator including a few failed calls; `wrong_index` uses ~253 scored nodes.
- **Cora current TAPTN (Table 28):** `acc_gate` from `cora_2hop_gated_*.pkl` (parameter-free all-neighbor majority vote). `reproduce.py gate` rebuilds these from `pkls/current_taptn/gate_sources/`.
- **Cora current GraphICL:** `accuracy_reextract`.
- **Products TAPTN 1-hop 83.75:** hop-1 `wrong_index` treated as positions into the 1430-key result dict, intersected with the 400 test ids of `..._iter2_3.pkl`. The pickle's global stored accuracy is 82.66% on 1430 nodes.
- **Table 14:** `cost_probe/compute_cost.py` on `results.json` (OpenRouter Llama-3.3-70B $0.10/$0.32, Llama-3.1-8B $0.02/$0.05). Dense row uses `dense_input_tokens.json` + GraphICL output length + 8B extract.

Main-table WebKB / citation-graph TAPTN numbers come from the **old** `texas2_` / `wisconsin2_` / `cornell2_` / `arxiv2023_*` files, not from later `*_noanon_*_llama-3.3-70b-instruct.pkl` (those differ by 0.5–2 pp and are the Table 27 Llama anchors).

## Cannot reproduce (missing artifacts)

Searched home `/home/wanghongyi/LLM-Structured-Data-main`, data1 `LLM-Structured-Data-main`, and `TAPE-main`. The following paper cells have **no matching pickle**. `generate_tables.py` hardcodes several of them.

### Cora GPT-3.5 (citation-graph backbone)

| Cell | Paper | Note |
|---|---:|---|
| Table 8 / Table 18 / Table 19 0-hop | 59.40 | DATA1 ego runs are 63.65 and 52.40 — neither matches. |
| GraphICL+SAT 1-hop | 62.55 | Hardcoded. `cora_hop2_anon_noguide_gpt-3.5-turbo-0125.pkl` is **62.55** but is the decouple 2-hop *anonymized* cell, not SAT 1-hop. |
| GraphICL+SAT 2-hop | 63.47 | `main.py` comment: “original setting … 63.47% if abstracts are not used”. File not found. |
| TAPTN 1-hop | 72.69 | Hardcoded. A Llama-3.3 file happens to be 72.69; wrong backbone. |
| GraphICL 1-hop RI=1 / RI=2 | 69.19 / 68.45 | No `*reflect*` Cora pickle. |
| GraphICL 2-hop RI=1 / RI=2 | 69.18 / 68.45 | Same. |

Cora TAPTN **2-hop** 73.80 **is** recovered from `cot_cora_refining35_test.pkl`.

### Other missing

| Cell | Paper | Closest artifact |
|---|---:|---|
| Texas 0-hop | 66.93 | `texas_ego_noguide.pkl` = **66.80** (171/256). Used for Table 27 Llama ego, not substituted into the main table. |
| Cornell TAPTN 1-hop | 85.08 | `cornell2_hop1_guide_test.pkl` is 90.23 on n=256 (likely a mislabelled Texas-sized file). New `cornell_hop1_noanon_guide_llama-3.3-70b-instruct.pkl` is **85.89**. Neither is 85.08. |

These missing keys reappear in Table 9 (Cora GraphICL 2-hop), Table 19 (Cora w/ SAT 1-hop both instruction toggles, Cora 2-hop w/ SAT w/o instr., Cornell 1-hop w/ SAT w/ instr.), and Table 18.

## Matches the paper (rescore)

- **Table 8** all WebKB and arXiv-2023 cells that have pickles; Cora TAPTN 2-hop 73.80
- **Table 9** instruction row (new-style hop-2 guide), aggregation row (Cora neighbors 70.85, arXiv refine2 87.30, Wisconsin/Cornell `*_noguide_2`; Texas 66.80 vs typeset 66.79), TAPTN row
- **Table 10** all four rows, including 400-node TAPTN 1-hop 83.75
- **Table 14** calls, $/1k, accuracy. Dense tokens reconstruct at ≈8.7k vs typeset ≈8.1k (input mean 7.7k plus output/extract; cost still 0.91)
- **Table 15** all six numbers. TAPTN 1-hop w/ and w/o instructions share `amazon_hop1_guide_12.pkl` (74.75). TAPTN 2-hop w/ instr. is `amazon_hop1_guide24.pkl` (76.75) — **not** the dense GraphICL JSON, which is also 76.75 by coincidence
- **Table 18** entire arXiv-2023 column; Cora TAPTN 2-hop
- **Table 19** all anonymized columns; all WebKB/arXiv “w/ Struct” cells that reuse the main-table old files; hop-2 +instructions
- **Table 27** all 16 cells. Qwen TAPTN 1-hop 94.92 is the **pre-keyshift** snapshot (`*.pkl.keyshift_bak` in the home repo); the later file there is 96.09
- **Table 28** all 12 accuracy cells. Texas TAPTN uses `iter2_2hop` / `iter2_v2_2hop` (not the un-suffixed `iter2` that still has the key-shift bug). Displayed Δ can differ by 0.01 from the PDF because the PDF rounds the subtraction of already-rounded percents (Gemma +1.56 vs +1.57, GLM +1.95 vs +1.96, Qwen Cora +1.11 vs +1.10)

## Static figures

Figure 5 (`TAPTN_overview.pdf`) and Figure 6 (`TAPTN_mp.pdf`) are copied from the camera-ready directory. They are illustrations, not generated from pickles. Algorithm 1 is in the paper, not a separate asset.

## Out of scope (other packages)

- Table 26, Table 29, Section 2 flipping/extreme → `TAPTN_rewiring_repro`
- Table 11, Table 20, joint GNN, encoder controls → `TAPTN_finetune_repro`
