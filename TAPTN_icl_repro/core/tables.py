"""Render TAPTN ICL tables as markdown."""
from __future__ import annotations

from .names import (
    CURRENT_MODELS,
    CURRENT_TAPTN_MODELS,
    DS_PRINT,
    FACTORIAL_PRINT,
    FACTORIAL_ROWS,
    MAIN_DATASETS,
    MAIN_METHODS,
    METHOD_PRINT,
    MODEL_PRINT,
)


def fmt(x, nd=2):
    if x is None:
        return "—"
    return f"{x:.{nd}f}"


def fmt_signed(x, nd=2):
    if x is None:
        return "—"
    return f"{x:+.{nd}f}"


def main_5(cells):
    """cells[method][ds] = percent or None."""
    hdr = "| Method | " + " | ".join(DS_PRINT[d] for d in MAIN_DATASETS) + " |"
    sep = "|---|" + "|".join("---:" for _ in MAIN_DATASETS) + "|"
    lines = [hdr, sep]
    for m in MAIN_METHODS:
        row = cells[m]
        lines.append(
            "| " + METHOD_PRINT[m] + " | "
            + " | ".join(fmt(row.get(d)) for d in MAIN_DATASETS) + " |"
        )
    return "\n".join(lines)


def factorial(cells):
    hdr = "| Method | " + " | ".join(DS_PRINT[d] for d in MAIN_DATASETS) + " | Avg. |"
    sep = "|---|" + "|".join("---:" for _ in MAIN_DATASETS) + "|---:|"
    lines = [hdr, sep]
    for m in FACTORIAL_ROWS:
        row = cells[m]
        vals = [row.get(d) for d in MAIN_DATASETS]
        present = [v for v in vals if v is not None]
        avg = sum(present) / len(present) if present else None
        lines.append(
            "| " + FACTORIAL_PRINT[m] + " | "
            + " | ".join(fmt(v) for v in vals) + " | " + fmt(avg) + " |"
        )
    return "\n".join(lines)


def product_70b(cells):
    labels = [
        ("gicl2", "GraphICL+SAT 2-hop (no instr., no aggr.)"),
        ("dense", "GraphICL+SAT 2-hop (dense full-graph nbhd.)"),
        ("taptn1", "TAPTN 1-hop"),
        ("taptn2", "TAPTN 2-hop"),
    ]
    lines = ["| Method | Accuracy (%) |", "|---|---:|"]
    for k, lab in labels:
        lines.append(f"| {lab} | {fmt(cells.get(k))} |")
    return "\n".join(lines)


def cost_table(rows):
    """rows[k] = dict(calls, tokens, cost_per_1k, acc)."""
    labels = [
        ("gicl2", "GraphICL+SAT 2-hop"),
        ("dense", "GraphICL+SAT 2-hop (dense nbhd.)"),
        ("taptn2", "TAPTN 2-hop (uniform 70B)"),
    ]
    lines = [
        "| Method | Reason calls/node | Tokens/node | Cost ($/1k) | Acc. (%) |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, lab in labels:
        r = rows[k]
        tok = r["tokens"]
        tok_s = f"≈{tok/1000:.1f}k" if tok >= 100 else fmt(tok, 0)
        lines.append(
            f"| {lab} | {fmt(r['calls'], 1)} | {tok_s} | "
            f"{fmt(r['cost_per_1k'], 2)} | {fmt(r['acc'])} |"
        )
    return "\n".join(lines)


def tb_52(cells):
    """cells[(method, hop)] = (no_instr, with_instr) percents; None = '/'."""
    lines = [
        "| Method | Hop | w/o instruction (%) | w/ instruction (%) |",
        "|---|---:|---:|---:|",
    ]
    for method, hops in (("gicl", (1, 2)), ("taptn", (1, 2))):
        for i, hop in enumerate(hops):
            no, yes = cells[(method, hop)]
            name = "GraphICL" if method == "gicl" else "TAPTN"
            lab = name if i == 0 else ""
            yes_s = "/" if yes is None else fmt(yes)
            lines.append(f"| {lab} | {hop} | {fmt(no)} | {yes_s} |")
    return "\n".join(lines)


def tb_4(cells):
    """cells[(method, hop, ri)][ds] = percent."""
    lines = [
        "| Method | Order | RI | Cora | arXiv-2023 |",
        "|---|---:|---:|---:|---:|",
    ]
    for hop in (1, 2):
        for ri in (0, 1, 2):
            row = cells[("gicl", hop, ri)]
            mlab = "GraphICL" if hop == 1 and ri == 0 else ""
            olab = str(hop) if ri == 0 else ""
            lines.append(
                f"| {mlab} | {olab} | {ri} | "
                f"{fmt(row.get('cora'))} | {fmt(row.get('arxiv_2023'))} |"
            )
    for hop in (1, 2):
        row = cells[("taptn", hop, None)]
        mlab = "TAPTN" if hop == 1 else ""
        lines.append(
            f"| {mlab} | {hop} | / | "
            f"{fmt(row.get('cora'))} | {fmt(row.get('arxiv_2023'))} |"
        )
    return "\n".join(lines)


def decouple(cells):
    """cells[(ds, hop)] = (woS_woI, woS_wI, wS_woI, wS_wI)."""
    lines = [
        "| Dataset | Hop | w/o Struct, w/o Instr. | w/o Struct, w/ Instr. | w/ Struct, w/o Instr. | w/ Struct, w/ Instr. |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for ds in MAIN_DATASETS:
        for hop in (1, 2):
            t = cells[(ds, hop)]
            name = DS_PRINT[ds] if hop == 1 else ""
            lines.append(
                f"| {name} | {hop}-hop | "
                + " | ".join(fmt(x) for x in t) + " |"
            )
    return "\n".join(lines)


def current_channel(cells):
    """cells[model] = (ego, anon, sat, taptn)."""
    lines = [
        "| Model | ego 0-hop | GraphICL 1-hop anon. | GraphICL+SAT 1-hop | TAPTN 1-hop |",
        "|---|---:|---:|---:|---:|",
    ]
    for m in CURRENT_MODELS:
        t = cells[m]
        lines.append(
            f"| {MODEL_PRINT[m]} | " + " | ".join(fmt(x) for x in t) + " |"
        )
    return "\n".join(lines)


def current_taptn(cells):
    """cells[(ds, model)] = (gicl, taptn)."""
    lines = [
        "| Dataset | Model | GraphICL+SAT (2-hop) | TAPTN (2-hop) | Δ |",
        "|---|---|---:|---:|---:|",
    ]
    for ds in ("texas", "cora"):
        first = True
        for m in CURRENT_TAPTN_MODELS:
            g, t = cells[(ds, m)]
            dlt = None if g is None or t is None else t - g
            dname = "Texas" if ds == "texas" else "Cora"
            lab = dname if first else ""
            first = False
            lines.append(
                f"| {lab} | {MODEL_PRINT[m]} | {fmt(g)} | {fmt(t)} | {fmt_signed(dlt)} |"
            )
    return "\n".join(lines)
