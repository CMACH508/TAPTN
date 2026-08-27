#!/usr/bin/env python3
"""Reproduce TAPTN Section-2 rewiring tables (dry-run from pickles, or re-run LLMs).

Camera-ready source: ijcai25_camera_ready.tex
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

from core import figures as F
from core import homophily as H
from core import paper as P
from core import score as S
from core import stats as ST
from core import tables as T
from core.names import (
    CLI_MODELS,
    DATASETS,
    LMARENA,
    PANEL_4_NEW,
    PANEL_7,
    canon_model,
    display,
)

TABLES = [
    "tb_1",
    "tb_2",
    "tb_3",
    "rewiring_stats",
    "nolabel_main",
    "nolabel_stats",
    "current_rewire",
    "taptn_structcorrupt",
]


def resolve_assets(explicit=None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("TAPTN_ASSETS")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        PKG.parent / "TAPTN_rewiring_repro_assets",
        PKG / "assets",
        Path.cwd() / "TAPTN_rewiring_repro_assets",
        Path.cwd() / "assets",
    ]
    for c in candidates:
        if c.is_dir() and (c / "pkls").is_dir():
            return c.resolve()
    return candidates[0]


def setup_env(explicit=None) -> Path:
    assets = resolve_assets(explicit)
    os.environ["TAPTN_ASSETS"] = str(assets)
    vendor = PKG / "vendor"
    sys.path.insert(0, str(vendor))
    return assets


def pct(x):
    return 100.0 * float(x)


def scan_section2(folder: Path, hop: int, full_abs: bool):
    cells = {}
    missing = []
    for p in sorted(folder.glob("*.pkl")):
        ds, h, model, setting, fa = S.parse_section2_filename(p.stem)
        if h != hop or fa != full_abs:
            continue
        model = canon_model(model)
        try:
            out = S.score_path(p, prefer="section2")
        except Exception as e:
            missing.append(f"{p.name}: {e}")
            continue
        cells[(model, ds, setting)] = out
    return cells, missing


def scan_gpt35(folder: Path):
    """Old GPT-3.5 notebooks: *nofilp*_noguide / *filp*_noguide."""
    cells = {}
    for p in sorted(folder.glob("*.pkl")):
        name = p.name
        ds = name.split("_", 1)[0]
        if "2h" in name:
            hop = 2
        else:
            hop = 1
        if name.startswith(ds + "_filp") or "_filp_" in name or name.startswith(ds + "_flip"):
            setting = "rewired"
        else:
            setting = "no_rewiring"
        out = S.score_path(p, prefer="wrong_index")
        cells[(ds, hop, setting)] = out
        cells[(ds, hop, setting)]["path"] = str(p)
    return cells


def scan_tb3(folder: Path):
    cells = {}
    for p in sorted(folder.glob("*.pkl")):
        ds = p.name.split("_", 1)[0]
        if "noflip_link_pattern" in p.name:
            setting = "no_rewiring"
        elif "flip_link_pattern" in p.name:
            setting = "rewired"
        else:
            continue
        cells[(ds, setting)] = S.score_path(p, prefer="wrong_index")
    return cells


def scan_structcorrupt(folder: Path):
    mapping = {
        "texas_hop1_noanon_guide_llama-3.3-70b-instruct_iter2.pkl": ("Llama-3.3-70B", "texas", "intact"),
        "texas_hop1_noanon_guide_llama-3.3-70b-instruct_iter2_flip.pkl": ("Llama-3.3-70B", "texas", "flip"),
        "texas_hop1_anon_guide_llama-3.3-70b-instruct_iter2.pkl": ("Llama-3.3-70B", "texas", "blind"),
        "cornell_hop1_noanon_guide_llama-3.3-70b-instruct_iter2.pkl": ("Llama-3.3-70B", "cornell", "intact"),
        "cornell_hop1_noanon_guide_llama-3.3-70b-instruct_iter2_flip.pkl": ("Llama-3.3-70B", "cornell", "flip"),
        "cornell_hop1_anon_guide_llama-3.3-70b-instruct_iter2.pkl": ("Llama-3.3-70B", "cornell", "blind"),
        "wisconsin_hop1_noanon_guide_llama-3.3-70b-instruct_iter2.pkl": ("Llama-3.3-70B", "wisconsin", "intact"),
        "wisconsin_hop1_noanon_guide_llama-3.3-70b-instruct_iter2_flip.pkl": ("Llama-3.3-70B", "wisconsin", "flip"),
        "wisconsin_hop1_anon_guide_llama-3.3-70b-instruct_iter2.pkl": ("Llama-3.3-70B", "wisconsin", "blind"),
        "texas_hop1_noanon_guide_qwen3.5-27b_iter2.pkl": ("Qwen3.5-27B", "texas", "intact"),
        "texas_hop1_noanon_guide_qwen3.5-27b_iter2_flip.pkl": ("Qwen3.5-27B", "texas", "flip"),
        "texas_hop1_anon_guide_qwen3.5-27b_iter2.pkl": ("Qwen3.5-27B", "texas", "blind"),
    }
    acc = {}
    for fn, key in mapping.items():
        p = folder / fn
        if not p.exists():
            continue
        acc[key] = S.score_path(p, prefer="stored")["accuracy"]
    deltas = {}
    for bb, ds in [("Llama-3.3-70B", "texas"), ("Llama-3.3-70B", "cornell"),
                   ("Llama-3.3-70B", "wisconsin"), ("Qwen3.5-27B", "texas")]:
        intact = acc.get((bb, ds, "intact"))
        if intact is None:
            continue
        blind = acc.get((bb, ds, "blind"))
        flip = acc.get((bb, ds, "flip"))
        if blind is None or flip is None:
            continue
        deltas[(bb, ds)] = (100 * (blind - intact), 100 * (flip - intact))
    return acc, deltas


def tb2_from_scans(hop1, hop2, gpt35):
    """Return {model: {ds: (O_F,R_F,O_E,R_E)}} in percent."""
    out = {m: {} for m in PANEL_7}
    for m in PANEL_7:
        for ds in DATASETS:
            if m == "gpt_3.5_turbo_0125":
                of = gpt35[(ds, 1, "no_rewiring")]["accuracy"]
                rf = gpt35[(ds, 1, "rewired")]["accuracy"]
                oe = gpt35[(ds, 2, "no_rewiring")]["accuracy"]
                re = gpt35[(ds, 2, "rewired")]["accuracy"]
            else:
                of = hop1[(m, ds, "no_rewiring")]["accuracy"]
                rf = hop1[(m, ds, "rewired")]["accuracy"]
                oe = hop2[(m, ds, "no_rewiring")]["accuracy"]
                re = hop2[(m, ds, "rewired")]["accuracy"]
            out[m][ds] = (pct(of), pct(rf), pct(oe), pct(re))
    return out


def averages_from_pairs(hop_cells, models):
    pairs = {}
    for m in models:
        for ds in DATASETS:
            try:
                o = hop_cells[(m, ds, "no_rewiring")]["accuracy"]
                r = hop_cells[(m, ds, "rewired")]["accuracy"]
            except KeyError:
                continue
            pairs[(m, ds)] = (o, r)
    return ST.model_averages(pairs, models), pairs


def homophily_webkb(assets: Path):
    values = {}
    # any hop1 pkl per dataset
    for ds in DATASETS:
        cands = list((assets / "pkls" / "hop1").glob(f"{ds}_hop1_*.pkl"))
        if not cands:
            continue
        pkl = S.load_pkl(cands[0])
        values[ds] = H.homophily_from_pkl_data(pkl["data"], pkl["text"]["label"])
    return values


def cmd_check(assets: Path):
    problems = []
    if not assets.is_dir():
        print(f"ERROR: assets not found at {assets}")
        print("Unpack TAPTN_rewiring_repro_assets and export TAPTN_ASSETS=...")
        return 1
    need_dirs = [
        assets / "pkls" / "hop1",
        assets / "pkls" / "hop2",
        assets / "pkls" / "full_abs",
        assets / "pkls" / "gpt35",
        assets / "webkb-data",
        assets / "figures_paper",
    ]
    for d in need_dirs:
        if not d.exists():
            problems.append(f"missing {d}")
    n1 = len(list((assets / "pkls" / "hop1").glob("*.pkl")))
    n2 = len(list((assets / "pkls" / "hop2").glob("*.pkl")))
    nfa = len(list((assets / "pkls" / "full_abs").glob("*.pkl")))
    ng = len(list((assets / "pkls" / "gpt35").glob("*.pkl")))
    print(f"TAPTN_ASSETS = {assets}")
    print(f"  hop1 pkls:     {n1} (expect 48)")
    print(f"  hop2 pkls:     {n2} (expect 48)")
    print(f"  full_abs pkls: {nfa} (expect ≥88)")
    print(f"  gpt35 pkls:    {ng} (expect 16)")
    print(f"  tb3 pkls:      {len(list((assets/'pkls'/'tb3').glob('*.pkl')))} (cornell+texas; washington/wisconsin missing)")
    print(f"  structcorrupt: {len(list((assets/'pkls'/'structcorrupt').glob('*.pkl')))} (expect 12)")
    for ds in DATASETS:
        if not (assets / f"file_abs_{ds}.pkl").exists():
            problems.append(f"missing file_abs_{ds}.pkl")
    if n1 < 48:
        problems.append(f"hop1 incomplete ({n1}/48)")
    if n2 < 48:
        problems.append(f"hop2 incomplete ({n2}/48)")
    if ng < 16:
        problems.append(f"gpt35 incomplete ({ng}/16)")
    if problems:
        print("Problems:")
        for p in problems:
            print(" -", p)
        return 1
    print("check: ok")
    return 0


def compare_tb2(computed, thresh=0.06):
    diffs = []
    for m, dmap in P.TB_2.items():
        for ds, tup in dmap.items():
            names = ["O_F", "R_F", "O_E", "R_E"]
            for i, name in enumerate(names):
                paper_v = tup[i]
                got = computed[m][ds][i]
                d = got - paper_v
                if abs(d) > thresh:
                    diffs.append({
                        "table": "tb_2", "model": display(m), "dataset": ds,
                        "cell": name, "paper": paper_v, "got": round(got, 4),
                        "delta": round(d, 4),
                    })
    return diffs


def compare_scalar(table, cell, paper_v, got, thresh, diffs):
    d = got - paper_v
    if abs(d) > thresh:
        diffs.append({
            "table": table, "cell": cell, "paper": paper_v,
            "got": got, "delta": d,
        })


def cmd_dry(assets: Path, tables, rescore: bool, figures: bool, out_dir: Path,
            paper_compat: bool = False):
    out_dir.mkdir(parents=True, exist_ok=True)
    diffs = []
    notes = []

    want = TABLES if "all" in tables else tables

    if paper_compat:
        P.apply_pre_revision()
        notes.append(
            "--paper-compat: comparing against the unrevised camera-ready PDF "
            "(Gemma Cornell O_F 49.39, Phi-4 Cornell O_F 51.52, GPT-3.5 Δ_F +6.67, "
            "tb_3 Cornell/Washington swap, Texas 50.20, label-free GPT-3.5 O=61.26)."
        )

    print("=" * 72)
    tag = "unrevised PDF (--paper-compat)" if paper_compat else "revised camera-ready"
    print(f"TAPTN rewiring dry-run  ({tag})")
    print("=" * 72)

    hop1 = hop2 = full = gpt35 = None
    if rescore or figures or any(t in want for t in (
        "tb_2", "rewiring_stats", "nolabel_main", "nolabel_stats", "current_rewire", "tb_3",
        "taptn_structcorrupt", "tb_1",
    )):
        if rescore or figures:
            hop1, err1 = scan_section2(assets / "pkls" / "hop1", 1, False)
            hop2, err2 = scan_section2(assets / "pkls" / "hop2", 2, False)
            full, errf = scan_section2(assets / "pkls" / "full_abs", 1, True)
            gpt35 = scan_gpt35(assets / "pkls" / "gpt35")
            for e in err1 + err2 + errf:
                notes.append(e)

    if "tb_1" in want:
        print("\n## Table tb_1  Homophily (camera-ready numbers)")
        print(T.tb_1(P.TB_1))
        if rescore:
            hw = homophily_webkb(assets)
            print("\nRecomputed node-averaged undirected homophily on the shipped WebKB pickles")
            print("(supplementary formula: symmetrize edges; isolated nodes stay in the denominator):")
            for ds in DATASETS:
                v = hw.get(ds)
                if v is None:
                    continue
                print(f"  {ds}: {v:.4f}  (paper {P.TB_1[ds]:.4f})")
                compare_scalar("tb_1", ds, P.TB_1[ds], v, 5e-4, diffs)
            if any(d.get("table") == "tb_1" for d in diffs):
                notes.append(
                    "tb_1 WebKB values are not recovered from the supplementary homophily formula "
                    "on the shipped pickles. Citation-graph rows are recorded paper values; "
                    "those graphs are not in this bundle."
                )
            else:
                notes.append(
                    "tb_1 WebKB rows recovered. Citation-graph rows (ogbn-arxiv, Pubmed, Cora) "
                    "are recorded paper values; those graphs are not in this bundle."
                )

    if "tb_2" in want:
        print("\n## Table tb_2  Label-revealing flipping / extreme")
        if rescore:
            cells = tb2_from_scans(hop1, hop2, gpt35)
            diffs += compare_tb2(cells)
        else:
            cells = T.paper_tb2_as_cells()
        print(T.tb_2(cells))

    if "tb_3" in want:
        print("\n## Table tb_3  GPT-3.5 + step-by-step instructions (flipping)")
        cells = T.paper_tb3_as_cells()
        if rescore:
            tb3 = scan_tb3(assets / "pkls" / "tb3")
            got = dict(cells)
            for ds in DATASETS:
                if (ds, "no_rewiring") in tb3 and (ds, "rewired") in tb3:
                    got[ds] = (pct(tb3[(ds, "no_rewiring")]["accuracy"]),
                               pct(tb3[(ds, "rewired")]["accuracy"]))
                    compare_scalar("tb_3", f"{ds}.O", P.TB_3[ds][0], got[ds][0], 0.06, diffs)
                    compare_scalar("tb_3", f"{ds}.R", P.TB_3[ds][1], got[ds][1], 0.06, diffs)
                else:
                    notes.append(f"tb_3 {ds}: pickle missing; paper numbers reprinted.")
                    diffs.append({
                        "table": "tb_3", "cell": ds, "paper": P.TB_3[ds][:2],
                        "got": None, "delta": None,
                        "note": "pickle missing",
                    })
            cells = got
        print(T.tb_3(cells))

    st_f = st_e = None
    rows_f = rows_e = None
    if rescore and hop1 is not None:
        # open models + gpt35 as fractions
        pairs_f = {}
        pairs_e = {}
        for m in PANEL_7:
            for ds in DATASETS:
                if m == "gpt_3.5_turbo_0125":
                    pairs_f[(m, ds)] = (gpt35[(ds, 1, "no_rewiring")]["accuracy"],
                                        gpt35[(ds, 1, "rewired")]["accuracy"])
                    pairs_e[(m, ds)] = (gpt35[(ds, 2, "no_rewiring")]["accuracy"],
                                        gpt35[(ds, 2, "rewired")]["accuracy"])
                else:
                    pairs_f[(m, ds)] = (hop1[(m, ds, "no_rewiring")]["accuracy"],
                                        hop1[(m, ds, "rewired")]["accuracy"])
                    pairs_e[(m, ds)] = (hop2[(m, ds, "no_rewiring")]["accuracy"],
                                        hop2[(m, ds, "rewired")]["accuracy"])
        rows_f = ST.model_averages(pairs_f, PANEL_7)
        rows_e = ST.model_averages(pairs_e, PANEL_7)
        st_f = ST.capability_stats(rows_f, LMARENA)
        st_e = ST.capability_stats(rows_e, LMARENA)

    if "rewiring_stats" in want:
        print("\n## Table tab:rewiring_stats")
        if rescore:
            print(T.rewiring_stats(st_f, st_e))
            # compare S flipping pearson etc.
            paper_map = [
                ("flipping", "S", "pearson_r", 0, 5e-4),
                ("flipping", "S", "pearson_p", 1, 5e-4),
                ("flipping", "S", "r2", 4, 5e-4),
                ("flipping", "S", "slope_per_100", 6, 5e-4),
            ]
            key_alt = {"pearson_r": 0, "pearson_p": 1, "spearman_rho": 2, "spearman_p": 3,
                       "r2": 4, "intercept": 5, "slope": 6, "rmse": 7}
            for setting, tgt, key, idx, th in [
                ("flipping", "S", "pearson_r", 0, 6e-4),
                ("flipping", "S", "pearson_p", 1, 6e-4),
                ("flipping", "S", "spearman_rho", 2, 6e-4),
                ("flipping", "S", "r2", 4, 6e-4),
                ("flipping", "S", "slope_per_100", None, 6e-4),
                ("extreme", "S", "pearson_r", 0, 6e-4),
            ]:
                paper_t = P.REWIRING_STATS[setting][tgt]
                got_d = st_f if setting == "flipping" else st_e
                if key == "slope_per_100":
                    paper_v = paper_t[6]
                    got_v = got_d[tgt]["slope_per_100"]
                else:
                    paper_v = paper_t[idx]
                    got_v = got_d[tgt][key]
                compare_scalar("rewiring_stats", f"{setting}.{tgt}.{key}", paper_v, got_v, th, diffs)
        else:
            # reprint paper
            fake_f = {k: {"pearson_r": v[0], "pearson_p": v[1], "spearman_rho": v[2],
                          "spearman_p": v[3], "r2": v[4], "intercept": v[5],
                          "slope_per_100": v[6], "rmse": v[7]}
                      for k, v in P.REWIRING_STATS["flipping"].items()}
            fake_e = {k: {"pearson_r": v[0], "pearson_p": v[1], "spearman_rho": v[2],
                          "spearman_p": v[3], "r2": v[4], "intercept": v[5],
                          "slope_per_100": v[6], "rmse": v[7]}
                      for k, v in P.REWIRING_STATS["extreme"].items()}
            print(T.rewiring_stats(fake_f, fake_e))

    rows_nl = None
    st_nl = None
    pairs_nl = None
    if rescore and full is not None:
        pairs_nl = {}
        for m in PANEL_7 + PANEL_4_NEW:
            for ds in DATASETS:
                k0 = (m, ds, "no_rewiring")
                k1 = (m, ds, "rewired")
                if k0 in full and k1 in full:
                    pairs_nl[(m, ds)] = (full[k0]["accuracy"], full[k1]["accuracy"])
        if paper_compat:
            key = ("gpt_3.5_turbo_0125", "wisconsin")
            if key in pairs_nl:
                _o, r = pairs_nl[key]
                pairs_nl[key] = (P.PAPER_COMPAT_NOLABEL_WISCONSIN_GPT_O, r)
                notes.append(
                    "paper-compat: Wisconsin GPT-3.5 label-free original forced to 42.04 "
                    "(label-revealing hop-1 number used in the unrevised PDF)."
                )
        rows_nl = ST.model_averages(pairs_nl, PANEL_7)
        st_nl = ST.capability_stats(rows_nl, LMARENA)

    if "nolabel_main" in want:
        print("\n## Table tab:nolabel_main")
        if rescore:
            mean_row = {
                "O": float(np.mean([r["O"] for r in rows_nl])),
                "R": float(np.mean([r["R"] for r in rows_nl])),
                "delta_R_minus_O": float(np.mean([r["delta_R_minus_O"] for r in rows_nl])),
                "rel": float(np.mean([r["rel"] for r in rows_nl])),
            }
            print(T.nolabel_main(rows_nl, mean_row))
            for r in rows_nl:
                po, pr_, pd, prel = P.NOLABEL_MAIN[r["model"]]
                compare_scalar("nolabel_main", f"{r['model']}.O", po, pct(r["O"]), 0.06, diffs)
                compare_scalar("nolabel_main", f"{r['model']}.R", pr_, pct(r["R"]), 0.06, diffs)
            pm = P.NOLABEL_MAIN["_mean"]
            compare_scalar("nolabel_main", "mean.delta", pm[2], 100 * mean_row["delta_R_minus_O"], 0.06, diffs)
        else:
            rows = [{"model": m, "O": v[0] / 100, "R": v[1] / 100,
                     "delta_R_minus_O": v[2] / 100, "rel": v[3]}
                    for m, v in P.NOLABEL_MAIN.items() if m != "_mean"]
            mv = P.NOLABEL_MAIN["_mean"]
            mean_row = {"O": mv[0] / 100, "R": mv[1] / 100,
                        "delta_R_minus_O": mv[2] / 100, "rel": mv[3]}
            print(T.nolabel_main(rows, mean_row))

    if "nolabel_stats" in want:
        print("\n## Table tab:nolabel_stats")
        if rescore:
            print(T.nolabel_stats(st_nl))
            ps = P.NOLABEL_STATS["S"]
            compare_scalar("nolabel_stats", "S.pearson_r", ps[0], st_nl["S"]["pearson_r"], 6e-4, diffs)
            compare_scalar("nolabel_stats", "S.pearson_p", ps[1], st_nl["S"]["pearson_p"], 6e-4, diffs)
            compare_scalar("nolabel_stats", "S.slope_per_100", ps[5], st_nl["S"]["slope_per_100"], 6e-4, diffs)
        else:
            fake = {
                "S": {"pearson_r": P.NOLABEL_STATS["S"][0], "pearson_p": P.NOLABEL_STATS["S"][1],
                      "spearman_rho": P.NOLABEL_STATS["S"][2], "spearman_p": P.NOLABEL_STATS["S"][3],
                      "r2": P.NOLABEL_STATS["S"][4], "slope_per_100": P.NOLABEL_STATS["S"][5],
                      "rmse": P.NOLABEL_STATS["S"][6]},
                "O": {"pearson_r": P.NOLABEL_STATS["O"][0], "pearson_p": P.NOLABEL_STATS["O"][1],
                      "spearman_rho": P.NOLABEL_STATS["O"][2], "spearman_p": P.NOLABEL_STATS["O"][3],
                      "r2": P.NOLABEL_STATS["O"][4], "slope_per_100": P.NOLABEL_STATS["O"][5]},
                "R": {"pearson_r": P.NOLABEL_STATS["R"][0], "pearson_p": P.NOLABEL_STATS["R"][1],
                      "spearman_rho": P.NOLABEL_STATS["R"][2], "spearman_p": P.NOLABEL_STATS["R"][3],
                      "r2": P.NOLABEL_STATS["R"][4], "slope_per_100": P.NOLABEL_STATS["R"][5]},
            }
            print(T.nolabel_stats(fake))

    if "current_rewire" in want:
        print("\n## Table tab:current_rewire")
        if rescore:
            rows4 = ST.model_averages(pairs_nl, PANEL_4_NEW)
            print(T.current_rewire(rows4))
            for r in rows4:
                po, pr_, pd, _ = P.CURRENT_REWIRE[r["model"]]
                compare_scalar("current_rewire", f"{r['model']}.O", po, pct(r["O"]), 0.06, diffs)
                compare_scalar("current_rewire", f"{r['model']}.R", pr_, pct(r["R"]), 0.06, diffs)
        else:
            rows4 = [{"model": m, "O": v[0] / 100, "R": v[1] / 100,
                      "delta_R_minus_O": v[2] / 100, "err_inc": float(v[3].replace("%", "").replace("+", ""))}
                     for m, v in P.CURRENT_REWIRE.items()]
            print(T.current_rewire(rows4))

    if "taptn_structcorrupt" in want:
        print("\n## Table tab:taptn_structcorrupt")
        if rescore:
            _, deltas = scan_structcorrupt(assets / "pkls" / "structcorrupt")
            print(T.taptn_structcorrupt(deltas))
            for k, paper_v in P.TAPTN_STRUCTCORRUPT.items():
                got = deltas.get(k)
                if got is None:
                    diffs.append({"table": "taptn_structcorrupt", "cell": str(k),
                                  "paper": paper_v, "got": None, "note": "missing"})
                    continue
                compare_scalar("taptn_structcorrupt", f"{k}.blind", paper_v[0], got[0], 0.06, diffs)
                compare_scalar("taptn_structcorrupt", f"{k}.flip", paper_v[1], got[1], 0.06, diffs)
        else:
            print(T.taptn_structcorrupt(P.TAPTN_STRUCTCORRUPT))

    # figures
    fig_dir = out_dir / "figures"
    paper_fig = assets / "figures_paper"
    if figures or "all" in tables:
        fig_dir.mkdir(parents=True, exist_ok=True)
        # always copy static illustrations
        for lab, fn in {**P.STATIC_FIGURES, **P.GENERATED_FIGURES}.items():
            src = paper_fig / fn
            if src.exists():
                shutil.copy2(src, fig_dir / fn)
        if rescore and rows_f is not None:
            pivot_f = []
            pivot_e = []
            pivot_nl = []
            for m in PANEL_7:
                for ds in DATASETS:
                    if m == "gpt_3.5_turbo_0125":
                        of, rf = gpt35[(ds, 1, "no_rewiring")]["accuracy"], gpt35[(ds, 1, "rewired")]["accuracy"]
                        oe, re = gpt35[(ds, 2, "no_rewiring")]["accuracy"], gpt35[(ds, 2, "rewired")]["accuracy"]
                    else:
                        of = hop1[(m, ds, "no_rewiring")]["accuracy"]
                        rf = hop1[(m, ds, "rewired")]["accuracy"]
                        oe = hop2[(m, ds, "no_rewiring")]["accuracy"]
                        re = hop2[(m, ds, "rewired")]["accuracy"]
                    pivot_f.append({"dataset": ds, "model": m, "O": of, "R": rf})
                    pivot_e.append({"dataset": ds, "model": m, "O": oe, "R": re})
            for m in PANEL_7:
                for ds in DATASETS:
                    if (m, ds) in pairs_nl:
                        o, r = pairs_nl[(m, ds)]
                        pivot_nl.append({"dataset": ds, "model": m, "O": o, "R": r})
            F.correlation_panels(rows_f, LMARENA, fig_dir / "lmarena_correlation_analysis_hop1.png")
            F.correlation_panels(rows_e, LMARENA, fig_dir / "lmarena_correlation_analysis_hop2.png")
            F.comparison_by_dataset(pivot_nl, fig_dir / "rewiring_nolabel_comparison.png")
            F.correlation_panels(rows_nl, LMARENA, fig_dir / "rewiring_nolabel_correlation.png")
            F.heatmap(pivot_nl, fig_dir / "rewiring_nolabel_heatmap.png")
            F.comparison_by_dataset(pivot_f, fig_dir / "rewiring_comparison_by_dataset_hop1.png")
            F.heatmap(pivot_f, fig_dir / "rewiring_effect_heatmap_hop1.png")
            notes.append(
                "Generated figures are regenerated from scored pickles; they match the paper's data "
                "but are not pixel-identical to the camera-ready PNGs (matplotlib / adjustText). "
                "Illustrations fig_1 / fig:case / former_rewire2 are copied from the paper directory."
            )
            print(f"\nFigures written to {fig_dir}")
        elif figures:
            print(f"\nPaper figures copied to {fig_dir} (pass --rescore to regenerate scatter/heatmap).")

    # write discrepancy json
    disc_path = out_dir / "dry_discrepancies.json"
    payload = {"discrepancies": diffs, "notes": notes, "rescore": rescore}
    disc_path.write_text(json.dumps(payload, indent=2, default=str))
    print("\n" + "-" * 72)
    if not rescore:
        print("Reprinted camera-ready numbers (no pickle rescore). Re-run with --rescore to verify.")
    elif not diffs:
        print("Rescore matches the camera-ready tables within tolerance.")
    else:
        print(f"{len(diffs)} cell(s) differ from the camera-ready PDF:")
        for d in diffs:
            print(" -", d)
    for n in notes:
        print(" note:", n)
    print(f"Wrote {disc_path}")
    return 0 if rescore and not any(d.get("got") is None and d.get("note") == "missing" for d in diffs) else 0


def cmd_run(args, assets: Path):
    vendor = PKG / "vendor"
    python = sys.executable
    script = vendor / "batch_run_experiments.py"
    models = args.models or list(CLI_MODELS)
    datasets = args.datasets or DATASETS

    table = args.table
    hop = args.hop
    full_abs = False
    use_instructions = False
    include_neighbors = False
    if table in ("nolabel_main", "nolabel_stats", "current_rewire", "nolabel"):
        full_abs = True
        hop = 1
    elif table == "tb_3":
        use_instructions = True
        hop = 1
        models = ["gpt-3.5-turbo-0125"]
    elif table == "tb_2" and hop == 2:
        include_neighbors = False  # hop=2 already triggers extreme reconnect in utils
    elif table == "taptn_structcorrupt":
        print("Live re-run of tab:taptn_structcorrupt uses the TAPTN ICL pipeline "
              "(iteration 2 + instructions + optional --anonymize_edges / --rewiring). "
              "That runner ships in the TAPTN-ICL package, not this rewiring bundle.")
        return 1

    run_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (PKG / "output" / "rerun")
    run_dir.mkdir(parents=True, exist_ok=True)
    # file_abs and webkb must be visible from cwd
    for ds in DATASETS:
        src = assets / f"file_abs_{ds}.pkl"
        dst = run_dir / f"file_abs_{ds}.pkl"
        if src.exists() and not dst.exists():
            shutil.copy2(src, dst)

    os.environ["TAPTN_ASSETS"] = str(assets)
    os.environ["PYTHONPATH"] = str(vendor) + os.pathsep + os.environ.get("PYTHONPATH", "")

    cmd = [
        python, str(script),
        "--hop", str(hop),
        "--models", *models,
        "--datasets", *datasets,
        "--max_workers", str(args.max_workers),
        "--batch_workers", str(args.batch_workers),
    ]
    if args.rewired:
        cmd.append("--rewiring")
    else:
        cmd.append("--no-rewiring")
    if full_abs:
        cmd.append("--webkb_full_abs")
    if use_instructions:
        cmd.append("--use-instructions")
    if include_neighbors:
        cmd.append("--include_neighbors")

    print("cwd:", run_dir)
    print("cmd:", " ".join(cmd))
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("TAPTN_LLM_API_KEY"):
        print("WARNING: set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) before a live run.")
    return subprocess.call(cmd, cwd=str(run_dir), env=os.environ.copy())


def main():
    parser = argparse.ArgumentParser(description="Reproduce TAPTN rewiring tables")
    parser.add_argument("--assets", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check")

    p_dry = sub.add_parser("dry")
    p_dry.add_argument("--table", nargs="+", default=["all"],
                       choices=["all"] + TABLES)
    p_dry.add_argument("--rescore", action="store_true",
                       help="Recompute accuracies from pickles (no LLM calls)")
    p_dry.add_argument("--figures", action="store_true",
                       help="Write figures under output/figures/")
    p_dry.add_argument("--output", default=None)
    p_dry.add_argument(
        "--paper-compat", action="store_true",
        help="Compare against the unrevised camera-ready PDF (pre-errata cells, "
             "including the 42.04 Wisconsin label-free GPT-3.5 original).",
    )

    p_run = sub.add_parser("run", help="Live LLM re-run (needs API key)")
    p_run.add_argument("--table", default="tb_2",
                       choices=["tb_2", "tb_3", "nolabel", "nolabel_main", "current_rewire",
                                "taptn_structcorrupt"])
    p_run.add_argument("--hop", type=int, default=1, choices=[1, 2])
    p_run.add_argument("--models", nargs="+", choices=list(CLI_MODELS))
    p_run.add_argument("--datasets", nargs="+", choices=DATASETS)
    p_run.add_argument("--rewired", action="store_true")
    p_run.add_argument("--max_workers", type=int, default=20)
    p_run.add_argument("--batch_workers", type=int, default=4)
    p_run.add_argument("--output-dir", default=None)

    args = parser.parse_args()
    assets = setup_env(args.assets)
    if args.cmd == "check":
        return cmd_check(assets)
    if args.cmd == "dry":
        out = Path(args.output) if args.output else (PKG / "output")
        return cmd_dry(assets, args.table, args.rescore, args.figures, out,
                       paper_compat=args.paper_compat)
    if args.cmd == "run":
        return cmd_run(args, assets)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
