"""Revised camera-ready TAPTN ICL numbers from ijcai25_camera_ready.tex.

Accuracies are percentages as typeset. This package does not use --paper-compat:
the ICL tables were not part of the rewiring errata.
"""
from __future__ import annotations

# tab:main_5_datasets  rows × {cora, arxiv_2023, texas, wisconsin, cornell}
MAIN_5 = {
    "0hop":   {"cora": 59.40, "arxiv_2023": 76.19, "texas": 66.93, "wisconsin": 68.85, "cornell": 72.58},
    "gicl1":  {"cora": 62.55, "arxiv_2023": 83.49, "texas": 65.23, "wisconsin": 76.01, "cornell": 76.21},
    "gicl2":  {"cora": 63.47, "arxiv_2023": 85.40, "texas": 66.41, "wisconsin": 70.72, "cornell": 72.18},
    "taptn1": {"cora": 72.69, "arxiv_2023": 87.30, "texas": 89.84, "wisconsin": 85.67, "cornell": 85.08},
    "taptn2": {"cora": 73.80, "arxiv_2023": 88.57, "texas": 91.80, "wisconsin": 87.23, "cornell": 86.69},
}

# tab:factorial  2-hop, SAT on; last column is the five-dataset mean as typeset
FACTORIAL = {
    "gicl":  {"cora": 63.47, "arxiv_2023": 85.40, "texas": 66.41, "wisconsin": 70.72, "cornell": 72.18, "avg": 71.64},
    "instr": {"cora": 73.25, "arxiv_2023": 86.03, "texas": 86.72, "wisconsin": 83.18, "cornell": 81.45, "avg": 82.13},
    "aggr":  {"cora": 70.85, "arxiv_2023": 87.30, "texas": 66.79, "wisconsin": 73.21, "cornell": 77.82, "avg": 75.19},
    "taptn": {"cora": 73.80, "arxiv_2023": 88.57, "texas": 91.80, "wisconsin": 87.23, "cornell": 86.69, "avg": 85.62},
}

PRODUCT_70B = {
    "gicl2": 76.50,
    "dense": 76.75,
    "taptn1": 83.75,
    "taptn2": 86.75,
}

# tab:cost  (calls/node, tokens/node approx, $/1k nodes, acc)
COST = {
    "gicl2":  {"calls": 1.0, "tokens": 2800, "cost_per_1k": 0.31, "acc": 76.50},
    "dense":  {"calls": 1.0, "tokens": 8100, "cost_per_1k": 0.91, "acc": 76.75},
    "taptn2": {"calls": 4.6, "tokens": 22000, "cost_per_1k": 2.52, "acc": 86.75},
}

# tb_52  (hop, w/o instr, w/ instr)  "/" means not applicable
TB_52 = {
    ("gicl", 1):  (74.75, None),
    ("gicl", 2):  (75.00, None),
    ("taptn", 1): (74.75, 74.75),
    ("taptn", 2): (74.00, 76.75),
}

# tb_4  GraphICL RI and TAPTN on Cora / arXiv-2023
TB_4 = {
    ("gicl", 1, 0): {"cora": 62.55, "arxiv_2023": 83.49},
    ("gicl", 1, 1): {"cora": 69.19, "arxiv_2023": 74.92},
    ("gicl", 1, 2): {"cora": 68.45, "arxiv_2023": 73.65},
    ("gicl", 2, 0): {"cora": 63.47, "arxiv_2023": 85.40},
    ("gicl", 2, 1): {"cora": 69.18, "arxiv_2023": 75.87},
    ("gicl", 2, 2): {"cora": 68.45, "arxiv_2023": 80.00},
    ("taptn", 1, None): {"cora": 72.69, "arxiv_2023": 87.30},
    ("taptn", 2, None): {"cora": 73.80, "arxiv_2023": 88.57},
}

# tab:decouple  (dataset, hop) -> (woS_woI, woS_wI, wS_woI, wS_wI)
DECOUPLE = {
    ("cora", 1):       (69.37, 71.40, 62.55, 72.69),
    ("cora", 2):       (62.55, 72.14, 63.47, 73.25),
    ("arxiv_2023", 1): (81.59, 85.71, 83.49, 87.30),
    ("arxiv_2023", 2): (82.54, 86.35, 85.40, 86.03),
    ("texas", 1):      (45.31, 78.52, 65.23, 89.84),
    ("texas", 2):      (57.03, 73.44, 66.41, 86.72),
    ("wisconsin", 1):  (55.45, 81.31, 76.01, 85.67),
    ("wisconsin", 2):  (60.75, 80.37, 70.72, 83.18),
    ("cornell", 1):    (56.45, 76.61, 76.21, 85.08),
    ("cornell", 2):    (62.10, 76.61, 72.18, 81.45),
}

# tab:current_channel  Texas  ego / GraphICL-anon / GraphICL+SAT / TAPTN  (1-hop except ego)
CURRENT_CHANNEL = {
    "llama": (66.80, 45.31, 64.06, 87.11),
    "gemma": (71.88, 92.19, 95.70, 96.48),
    "qwen":  (68.75, 69.92, 92.19, 94.92),
    "glm":   (83.59, 92.97, 94.53, 96.09),
}

# tab:current_taptn  (dataset, model) -> (GraphICL+SAT 2-hop, TAPTN 2-hop, delta)
CURRENT_TAPTN = {
    ("texas", "gemma"): (96.09, 97.66, +1.57),
    ("texas", "qwen"):  (91.80, 97.66, +5.86),
    ("texas", "glm"):   (94.92, 96.88, +1.96),
    ("cora", "gemma"):  (78.04, 79.52, +1.48),
    ("cora", "qwen"):   (78.97, 80.07, +1.10),
    ("cora", "glm"):    (77.49, 81.00, +3.51),
}

STATIC_FIGURES = {
    "fig:taptn_overview": "TAPTN_overview.pdf",
    "fig_2": "TAPTN_mp.pdf",
}


def official_cells_dict():
    return {
        "tab:main_5_datasets": MAIN_5,
        "tab:factorial": FACTORIAL,
        "tab:product_70b": PRODUCT_70B,
        "tab:cost": {k: dict(v) for k, v in COST.items()},
        "tb_52": {f"{a}|{h}": list(v) for (a, h), v in TB_52.items()},
        "tb_4": {f"{a}|{h}|{ri}": dict(v) for (a, h, ri), v in TB_4.items()},
        "tab:decouple": {f"{ds}|{h}": list(v) for (ds, h), v in DECOUPLE.items()},
        "tab:current_channel": {m: list(v) for m, v in CURRENT_CHANNEL.items()},
        "tab:current_taptn": {
            f"{ds}|{m}": list(v) for (ds, m), v in CURRENT_TAPTN.items()
        },
    }
