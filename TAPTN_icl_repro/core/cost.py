"""Recompute tab:cost from shipped cost_probe JSON (OpenRouter list prices)."""
from __future__ import annotations

import json
import math
from pathlib import Path

P70_IN, P70_OUT = 0.10, 0.32
P8_IN, P8_OUT = 0.02, 0.05
N_TEST = 400
EX_SYS, EX_OUT = 250, 20


def _cost(n, ti, to, pin, pout):
    return n * (ti * pin + to * pout) / 1e6


def compute_cost(assets: Path):
    r = json.loads((assets / "cost_probe" / "results.json").read_text())
    g_in = r["graphicl_2hop_input_recon"]["input_total"]["mean"]
    g_out = r["graphicl_2hop_output_pkl"]["mean"]
    t1_in = r["taptn_iter1_from_json"]["input_total"]["mean"]
    t1_out = r["taptn_iter1_output_pkl"]["mean"]
    t2_in = r["taptn_iter2_from_json"]["input_total"]["mean"]
    t2_out = r["taptn_iter2_output_pkl"]["mean"]
    n_iter1 = r["taptn_iter1_output_pkl"]["n_records"]

    g_reason = _cost(N_TEST, g_in, g_out, P70_IN, P70_OUT)
    g_extract = _cost(N_TEST, EX_SYS + g_out, EX_OUT, P8_IN, P8_OUT)
    g_tokens = N_TEST * (g_in + g_out + EX_SYS + g_out + EX_OUT)

    u_i1 = _cost(n_iter1, t1_in, t1_out, P70_IN, P70_OUT)
    u_i2 = _cost(N_TEST, t2_in, t2_out, P70_IN, P70_OUT)
    u_ex = (
        _cost(n_iter1, EX_SYS + t1_out, EX_OUT, P8_IN, P8_OUT)
        + _cost(N_TEST, EX_SYS + t2_out, EX_OUT, P8_IN, P8_OUT)
    )
    u_tokens = (
        n_iter1 * (t1_in + t1_out)
        + N_TEST * (t2_in + t2_out)
        + n_iter1 * (EX_SYS + t1_out + EX_OUT)
        + N_TEST * (EX_SYS + t2_out + EX_OUT)
    )

    dense_in = json.loads(
        (assets / "cost_probe" / "dense_graphicl" / "dense_input_tokens.json").read_text()
    )["mean"]
    d_reason = _cost(N_TEST, dense_in, g_out, P70_IN, P70_OUT)
    d_extract = g_extract
    d_tokens = N_TEST * (dense_in + g_out + EX_SYS + g_out + EX_OUT)

    dense_acc = json.loads(
        (assets / "cost_probe" / "dense_graphicl" / "dense_results.json").read_text()
    )["summary"]["accuracy"]

    return {
        "gicl2": {
            "calls": 1.0,
            "tokens": g_tokens / N_TEST,
            "cost_per_1k": (g_reason + g_extract) / N_TEST * 1000,
            "acc": 100.0 * r["graphicl_2hop_output_pkl"]["accuracy"],
        },
        "dense": {
            "calls": 1.0,
            "tokens": d_tokens / N_TEST,
            "cost_per_1k": (d_reason + d_extract) / N_TEST * 1000,
            "acc": 100.0 * dense_acc,
        },
        "taptn2": {
            "calls": (n_iter1 + N_TEST) / N_TEST,
            "tokens": u_tokens / N_TEST,
            "cost_per_1k": (u_i1 + u_i2 + u_ex) / N_TEST * 1000,
            "acc": 100.0 * r["taptn_iter2_output_pkl"]["accuracy"],
        },
        "latencies": {
            "70b": r["timing"]["llama33_70b_main_latency_s"]["mean"],
            "8b": r["timing"]["llama31_8b_extract_latency_s"]["mean"],
        },
    }
