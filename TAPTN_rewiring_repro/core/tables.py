"""Render paper-style markdown tables and compare cells."""
from __future__ import annotations

from .names import DATASETS, DS_PRINT, DISPLAY, LMARENA, PANEL_7, PARAMS, display
from . import paper as P


def fmt_pct(x, nd=2):
    return f"{x:.{nd}f}"


def fmt_signed(x, nd=2):
    return f"{x:+.{nd}f}"


def tb_1(h):
    order = ["cornell", "texas", "washington", "wisconsin", "ogbn-arxiv", "pubmed", "cora"]
    names = ["Cornell", "Texas", "Washington", "Wisconsin", "ogbn-arxiv", "Pubmed", "Cora"]
    lines = ["| Dataset | " + " | ".join(names) + " |",
             "|---|---" * 7]
    lines.append("| Homophily | " + " | ".join(fmt_pct(h[k], 4) for k in order) + " |")
    return "\n".join(lines)


def tb_2(cells):
    """cells[model][dataset] = (O_F, R_F, O_E, R_E) as percentages."""
    hdr = "| Model | Arena | #Params | Dataset | O_F | R_F | O_E | R_E | Δ_F | Δ_E |"
    lines = [hdr, "|---|---:|---|---|---:|---:|---:|---:|---:|---:|"]
    for m in PANEL_7:
        first = True
        for ds in DATASETS:
            o_f, r_f, o_e, r_e = cells[m][ds]
            d_f, d_e = r_f - o_f, r_e - o_e
            model = display(m) if first else ""
            arena = str(LMARENA[m]) if first else ""
            para = PARAMS.get(m, "") if first else ""
            first = False
            lines.append(
                f"| {model} | {arena} | {para} | {DS_PRINT[ds]} | "
                f"{fmt_pct(o_f)} | {fmt_pct(r_f)} | {fmt_pct(o_e)} | {fmt_pct(r_e)} | "
                f"{fmt_signed(d_f)} | {fmt_signed(d_e)} |"
            )
    return "\n".join(lines)


def tb_3(cells):
    lines = ["| Dataset | Original | Rewired | Δ(Acc.) |",
             "|---|---:|---:|---:|"]
    for ds in ["cornell", "washington", "wisconsin", "texas"]:
        o, r = cells[ds]
        lines.append(f"| {DS_PRINT[ds]} | {fmt_pct(o)} | {fmt_pct(r)} | {fmt_signed(r-o)} |")
    return "\n".join(lines)


def rewiring_stats(st_f, st_e):
    rows = [
        ("S", "Pearson r", "pearson_r", 4, True),
        ("S", "Pearson p", "pearson_p", 4, False),
        ("S", "Spearman ρ", "spearman_rho", 4, True),
        ("S", "Spearman p", "spearman_p", 4, False),
        ("S", "R²", "r2", 4, False),
        ("S", "Intercept", "intercept", 4, True),
        ("S", "Slope /100", "slope_per_100", 4, True),
        ("S", "RMSE", "rmse", 4, False),
        ("O", "Pearson r", "pearson_r", 4, True),
        ("O", "Pearson p", "pearson_p", 4, False),
        ("O", "Spearman ρ", "spearman_rho", 4, True),
        ("O", "Spearman p", "spearman_p", 4, False),
        ("O", "R²", "r2", 4, False),
        ("O", "Intercept", "intercept", 4, True),
        ("O", "Slope /100", "slope_per_100", 4, True),
        ("O", "RMSE", "rmse", 4, False),
        ("R", "Pearson r", "pearson_r", 4, True),
        ("R", "Pearson p", "pearson_p", 4, False),
        ("R", "Spearman ρ", "spearman_rho", 4, True),
        ("R", "Spearman p", "spearman_p", 4, False),
        ("R", "R²", "r2", 4, False),
        ("R", "Intercept", "intercept", 4, True),
        ("R", "Slope /100", "slope_per_100", 4, True),
        ("R", "RMSE", "rmse", 4, False),
    ]
    lines = ["| Target | Statistic | Flipping | Extreme |",
             "|---|---|---:|---:|"]
    keymap = {"S": "S", "O": "O", "R": "R"}
    for tgt, stat, key, nd, signed in rows:
        fv, ev = st_f[keymap[tgt]][key], st_e[keymap[tgt]][key]
        fmt = (lambda z: f"{z:+.{nd}f}") if signed else (lambda z: f"{z:.{nd}f}")
        lines.append(f"| {tgt} | {stat} | {fmt(fv)} | {fmt(ev)} |")
    return "\n".join(lines)


def nolabel_main(rows, mean_row=None):
    lines = ["| Model | Arena | O | R | Δ | Rel. |",
             "|---|---:|---:|---:|---:|---:|"]
    for r in rows:
        lines.append(
            f"| {display(r['model'])} | {LMARENA[r['model']]} | "
            f"{fmt_pct(100*r['O'])} | {fmt_pct(100*r['R'])} | "
            f"{fmt_signed(100*r['delta_R_minus_O'])} | {fmt_signed(r['rel'])} |"
        )
    if mean_row:
        lines.append(
            f"| **Mean** |  | {fmt_pct(100*mean_row['O'])} | {fmt_pct(100*mean_row['R'])} | "
            f"{fmt_signed(100*mean_row['delta_R_minus_O'])} | {fmt_signed(mean_row['rel'])} |"
        )
    return "\n".join(lines)


def nolabel_stats(st):
    lines = ["| Target | Statistic | Value |", "|---|---|---:|"]
    s, o, r = st["S"], st["O"], st["R"]
    lines += [
        f"| S | Pearson r (p) | {s['pearson_r']:+.4f} ({s['pearson_p']:.4f}) |",
        f"| S | Spearman ρ (p) | {s['spearman_rho']:+.4f} ({s['spearman_p']:.4f}) |",
        f"| S | R² | {s['r2']:.4f} |",
        f"| S | Slope /100 | {s['slope_per_100']:+.4f} |",
        f"| S | RMSE | {s['rmse']:.4f} |",
        f"| O | Pearson r (p) | {o['pearson_r']:+.4f} ({o['pearson_p']:.4f}) |",
        f"| O | Spearman ρ (p) | {o['spearman_rho']:+.4f} ({o['spearman_p']:.4f}) |",
        f"| O | R² | {o['r2']:.4f} |",
        f"| O | Slope /100 | {o['slope_per_100']:+.4f} |",
        f"| R | Pearson r (p) | {r['pearson_r']:+.4f} ({r['pearson_p']:.4f}) |",
        f"| R | Spearman ρ (p) | {r['spearman_rho']:+.4f} ({r['spearman_p']:.4f}) |",
        f"| R | R² | {r['r2']:.4f} |",
        f"| R | Slope /100 | {r['slope_per_100']:+.4f} |",
    ]
    return "\n".join(lines)


def current_rewire(rows):
    lines = ["| Model | Arena | O | R | Δ (Err. inc.) |",
             "|---|---:|---:|---:|---:|"]
    for r in rows:
        ei = r.get("err_inc")
        ei_s = f"{ei:+.0f}%" if ei == ei else "n/a"
        lines.append(
            f"| {display(r['model'])} | {LMARENA[r['model']]} | "
            f"{fmt_pct(100*r['O'])} | {fmt_pct(100*r['R'])} | "
            f"{fmt_signed(100*r['delta_R_minus_O'])} ({ei_s}) |"
        )
    return "\n".join(lines)


def taptn_structcorrupt(cells):
    lines = ["| Backbone | Dataset | Edge-blind Δ | Flipped Δ |",
             "|---|---|---:|---:|"]
    for (bb, ds), (blind, flip) in cells.items():
        lines.append(f"| {bb} | {DS_PRINT.get(ds, ds)} | {fmt_signed(blind)} | {fmt_signed(flip)} |")
    return "\n".join(lines)


def paper_tb2_as_cells():
    out = {}
    for m, dmap in P.TB_2.items():
        out[m] = {ds: (v[0], v[1], v[2], v[3]) for ds, v in dmap.items()}
    return out


def paper_tb3_as_cells():
    return {ds: (v[0], v[1]) for ds, v in P.TB_3.items()}
