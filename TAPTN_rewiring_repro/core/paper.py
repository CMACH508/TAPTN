"""Revised camera-ready numbers from ijcai25_camera_ready.tex (rewiring tables).

Accuracies are percentages as typeset. Stats keep the paper's sign convention:
sensitivity S = O - R (paper Table rewiring_stats / nolabel_stats).

`--paper-compat` loads PRE_REVISION_* so a dry-run can be compared against the
unrevised PDF (Gemma Cornell 49.39, GPT-3.5 Δ_F +6.67, Wisconsin label-free
original 42.04, Cornell/Washington tb_3 swap, Texas tb_3 50.20).
"""
from __future__ import annotations

import copy

TB_1 = {
    "cornell": 0.1308,
    "texas": 0.1448,
    "washington": 0.1599,
    "wisconsin": 0.1869,
    "ogbn-arxiv": 0.6358,
    "pubmed": 0.7924,
    "cora": 0.8252,
}

# Unrevised camera-ready (pre-errata).
PRE_REVISION_TB_2 = {
    "gpt_3.5_turbo_0125": {
        "cornell":    (49.39, 55.06, 80.57, 78.95, +6.67, -1.62),
        "texas":      (43.87, 39.53, 69.57, 64.82, -4.34, -4.75),
        "washington": (51.36, 49.81, 69.65, 67.70, -1.55, -1.95),
        "wisconsin":  (42.04, 39.17, 67.52, 63.38, -2.87, -4.14),
    },
    "phi_4": {
        "cornell":    (51.52, 43.15, 63.31, 58.07, -8.37, -5.24),
        "texas":      (41.01, 39.45, 60.94, 55.08, -1.56, -5.86),
        "washington": (53.01, 52.63, 69.17, 60.53, -0.38, -8.68),
        "wisconsin":  (49.84, 48.60, 65.73, 60.75, -1.24, -4.98),
    },
    "gemma_2_9b_it": {
        "cornell":    (49.39, 43.95, 71.77, 63.71, -5.44, -8.06),
        "texas":      (44.53, 35.94, 67.58, 55.86, -8.59, -11.72),
        "washington": (54.51, 50.38, 72.56, 63.91, -4.13, -8.65),
        "wisconsin":  (60.44, 51.41, 76.01, 71.34, -9.03, -4.67),
    },
    "llama_3_70b_ins": {
        "cornell":    (73.79, 52.02, 82.66, 83.06, -21.77, +0.40),
        "texas":      (64.06, 50.00, 83.21, 78.52, -14.06, -4.69),
        "washington": (67.67, 58.27, 84.96, 77.82, -9.40, -7.14),
        "wisconsin":  (71.65, 59.81, 84.42, 83.80, -11.84, -0.62),
    },
    "llama_3.1_70b_i": {
        "cornell":    (77.02, 55.24, 83.87, 83.06, -21.78, -0.81),
        "texas":      (69.53, 52.73, 84.38, 80.47, -16.80, -3.91),
        "washington": (70.68, 60.53, 81.96, 77.82, -10.15, -4.14),
        "wisconsin":  (74.45, 57.94, 85.05, 83.80, -16.51, -1.25),
    },
    "qwen2.5_vl_72b": {
        "cornell":    (73.39, 55.65, 81.04, 72.98, -17.74, -8.06),
        "texas":      (69.92, 48.44, 76.95, 73.04, -21.48, -3.91),
        "washington": (63.91, 54.89, 68.80, 62.78, -9.02, -6.02),
        "wisconsin":  (77.88, 59.50, 83.49, 80.69, -18.38, -2.80),
    },
    "llama_3.3_70b_i": {
        "cornell":    (75.40, 51.61, 78.63, 77.02, -23.79, -1.61),
        "texas":      (70.70, 52.73, 85.54, 83.98, -17.97, -1.56),
        "washington": (69.55, 62.03, 77.07, 74.06, -7.52, -3.01),
        "wisconsin":  (74.45, 60.44, 81.93, 78.19, -14.01, -3.74),
    },
}

# tb_2: O_F, R_F, O_E, R_E, dF, dE  (percentage points, revised typesetting)
TB_2 = copy.deepcopy(PRE_REVISION_TB_2)
TB_2["gpt_3.5_turbo_0125"]["cornell"] = (49.39, 55.06, 80.57, 78.95, +5.67, -1.62)
TB_2["phi_4"]["cornell"] = (51.21, 43.15, 63.31, 58.07, -8.06, -5.24)
TB_2["gemma_2_9b_it"]["cornell"] = (54.44, 43.95, 71.77, 63.71, -10.49, -8.06)

PRE_REVISION_TB_3 = {
    "cornell":    (52.14, 50.97, -1.17),
    "washington": (61.54, 59.92, -1.62),
    "wisconsin":  (50.96, 49.04, -1.92),
    "texas":      (54.55, 50.20, -4.35),
}

# Cornell/Washington rows were swapped in the unrevised PDF; Texas rewired 50.20→49.80.
TB_3 = {
    "cornell":    (61.54, 59.92, -1.62),
    "washington": (52.14, 50.97, -1.17),
    "wisconsin":  (50.96, 49.04, -1.92),
    "texas":      (54.55, 49.80, -4.75),
}

# tab:rewiring_stats — S = O-R; slopes per 100 LMArena points
# tuples: (pearson_r, pearson_p, spearman_rho, spearman_p, r2, intercept, slope, rmse)
REWIRING_STATS = {
    "flipping": {
        "S":     (+0.9160, 0.0038, +0.8928, 0.0068, 0.8391, -2.3737, +0.1943, 0.0251),
        "O":     (+0.8999, 0.0058, +0.8929, 0.0068, 0.8099, -3.6786, +0.3369, 0.0481),
        "R":     (+0.8414, 0.0176, +0.7857, 0.0362, 0.7080, -1.3049, +0.1426, 0.0270),
    },
    "extreme": {
        "S":     (-0.2521, 0.5855, -0.5714, 0.1802, 0.0635, 0.2685, -0.0176, 0.0199),
        "O":     (+0.6422, 0.1199, +0.6429, 0.1194, 0.4124, -1.0747, +0.1440, 0.0507),
        "R":     (+0.5858, 0.1669, +0.6071, 0.1482, 0.3432, -1.3433, +0.1616, 0.0659),
    },
}

# tab:nolabel_main  O, R, Delta=R-O, Rel%  (Rel = mean of per-dataset relative changes)
PRE_REVISION_NOLABEL_MAIN = {
    "gpt_3.5_turbo_0125": (61.26, 74.09, +12.83, +26.53),
    "phi_4":              (71.50, 71.43, -0.06, +0.08),
    "gemma_2_9b_it":      (77.03, 72.47, -4.57, -5.96),
    "llama_3_70b_ins":    (83.50, 79.17, -4.34, -5.18),
    "llama_3.1_70b_i":    (87.26, 82.06, -5.21, -5.98),
    "qwen2.5_vl_72b":     (83.29, 77.67, -5.62, -6.69),
    "llama_3.3_70b_i":    (85.31, 79.95, -5.37, -6.21),
    "_mean":              (78.45, 76.69, -1.76, -0.49),
}

NOLABEL_MAIN = copy.deepcopy(PRE_REVISION_NOLABEL_MAIN)
NOLABEL_MAIN["gpt_3.5_turbo_0125"] = (70.10, 74.09, +3.99, +6.12)
NOLABEL_MAIN["_mean"] = (79.71, 76.69, -3.02, -3.40)

# tab:nolabel_stats  S = O-R; slopes per 100 LMArena points
# S: (pearson_r, pearson_p, spearman_rho, spearman_p, r2, slope, rmse)
# O/R: (pearson_r, pearson_p, spearman_rho, spearman_p, r2, slope)
PRE_REVISION_NOLABEL_STATS = {
    "S": (+0.8409, 0.0178, +0.9286, 0.0025, 0.7071, +0.1773, 0.0336),
    "O": (+0.9150, 0.0039, +0.8214, 0.0234, 0.8373, +0.2675),
    "R": (+0.7081, 0.0750, +0.7143, 0.0713, 0.5014, +0.0902),
}

NOLABEL_STATS = {
    "S": (+0.8875, 0.0077, +0.9286, 0.0025, 0.7876, +0.1013, 0.0155),
    "O": (+0.8880, 0.0076, +0.8214, 0.0234, 0.7886, +0.1915),
    "R": (+0.7081, 0.0750, +0.7143, 0.0713, 0.5014, +0.0902),
}

# Unrevised PDF substituted Wisconsin original = 42.04 (label-revealing hop-1).
PAPER_COMPAT_NOLABEL_WISCONSIN_GPT_O = 0.4204

# tab:current_rewire  O, R, Delta=R-O, err_inc string as typeset
CURRENT_REWIRE = {
    "gpt_oss_120b":   (83.84, 79.67, -4.17, "+26%"),
    "qwen3.5_27b":    (93.33, 86.01, -7.31, "+110%"),
    "gemma_4_31b_it": (96.55, 94.67, -1.88, "+55%"),
    "glm_5.1":        (95.17, 93.48, -1.69, "+35%"),
}

# tab:taptn_structcorrupt  Edge-blind Δ, Flipped Δ (pp vs intact TAPTN)
TAPTN_STRUCTCORRUPT = {
    ("Llama-3.3-70B", "texas"):     (-8.20, -18.36),
    ("Llama-3.3-70B", "cornell"):   (-8.87, -17.33),
    ("Llama-3.3-70B", "wisconsin"): (-18.38, -13.08),
    ("Qwen3.5-27B", "texas"):       (-3.90, -25.00),
}

# Static figures that are illustrations, not generated from pkls.
STATIC_FIGURES = {
    "fig_1": "neighbor_rewiring.png",
    "fig:case": "rewiring_case.png",
    "former": "former_rewire2.pdf",
}

GENERATED_FIGURES = {
    "fig:hop1": "lmarena_correlation_analysis_hop1.png",
    "fig:hop2": "lmarena_correlation_analysis_hop2.png",
    "fig:nolabel_bar": "rewiring_nolabel_comparison.png",
    "fig:nolabel_corr": "rewiring_nolabel_correlation.png",
    "fig:nolabel_heat": "rewiring_nolabel_heatmap.png",
}


def apply_pre_revision():
    """Point module-level tables at the unrevised camera-ready numbers."""
    global TB_2, TB_3, NOLABEL_MAIN, NOLABEL_STATS
    TB_2 = copy.deepcopy(PRE_REVISION_TB_2)
    TB_3 = copy.deepcopy(PRE_REVISION_TB_3)
    NOLABEL_MAIN = copy.deepcopy(PRE_REVISION_NOLABEL_MAIN)
    NOLABEL_STATS = copy.deepcopy(PRE_REVISION_NOLABEL_STATS)


def official_cells_dict():
    return {
        "tb_1": TB_1,
        "tb_2": {m: {ds: list(v) for ds, v in dmap.items()} for m, dmap in TB_2.items()},
        "tb_3": {ds: list(v) for ds, v in TB_3.items()},
        "rewiring_stats": {
            k: {t: list(vv) for t, vv in inner.items()} for k, inner in REWIRING_STATS.items()
        },
        "nolabel_main": {m: list(v) for m, v in NOLABEL_MAIN.items()},
        "nolabel_stats": {t: list(v) for t, v in NOLABEL_STATS.items()},
        "current_rewire": {m: list(v) for m, v in CURRENT_REWIRE.items()},
        "taptn_structcorrupt": {
            f"{bb}|{ds}": list(v) for (bb, ds), v in TAPTN_STRUCTCORRUPT.items()
        },
    }
