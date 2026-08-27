#!/usr/bin/env python3
"""Reproduce TAPTN fine-tuning tables (dry-run from saved artifacts, or train).

Public run indices are 1–5. The mapping onto library RNG seeds lives in
configs/run_seeds.yaml and is applied automatically.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import statistics as st
import subprocess
import sys
from pathlib import Path

PKG = Path(__file__).resolve().parent

TABLE_ORDER = [
    "gnn_summary",
    "p5_transfer",
    "crosslm_encoder",
    "tb_6",
    "heterophilic",
    "joint",
    "roberta",
]
TABLE_ALIASES = {"transfer": "p5_transfer"}

GNN_CODE = {
    "GAT": "GAT", "GraphSAGE": "SAGE", "SAGE": "SAGE", "ChebNet": "ChebNet",
    "DGI": "DGI", "GATv2": "GATv2", "GCNII": "GCN2", "GCN2": "GCN2",
    "ASDGN": "ASC", "ASC": "ASC", "DirGNN": "DirGNN",
    "ACM-GNN": "ACMGNN", "ACMGNN": "ACMGNN", "DMP": "DMP",
    "GraphSAINT": "Saint", "Saint": "Saint", "FSGNN": "FSGNN",
    "APPNP": "APPNP", "GraphTARIF": "GraphTARIF", "RevGAT": "RevGAT",
}

# Paper Table 3 (cross-encoder): TA+LM and 12 TA+GNN (13 raw-text pipelines).
CROSSLM_GNN = {
    "deberta": ["GAT", "GraphSAGE", "ChebNet", "GATv2", "GCNII", "ASDGN",
                "DirGNN", "ACM-GNN", "DMP", "FSGNN", "APPNP", "GraphTARIF"],
    "roberta": ["GAT", "GATv2", "SAGE", "ChebNet", "GCNII", "DirGNN",
                "ACM-GNN", "DMP", "FSGNN", "APPNP", "GraphTARIF", "RevGAT"],
}

P5_GNNS = ["GAT", "SAGE", "ChebNet", "DGI", "GATv2", "GCN2", "ASC", "DirGNN",
           "ACMGNN", "DMP", "Saint", "FSGNN", "APPNP", "GraphTARIF"]

PIPE_CLI = {
    "ta_lm": "P1_LM",
    "grapicl_lm": "P3",
    "taptn_lm": "P2",
    "ta_gnn": "P1_GNN",
    "taptn_gnn": "P5",
    "joint": "P4",
}

HOMOPHILY = {
    "cora": "homophilic", "arxiv_2023": "homophilic", "product": "homophilic",
    "cornell": "heterophilic", "texas": "heterophilic", "wisconsin": "heterophilic",
}

DS_PRINT = {
    "cora": "Cora", "arxiv_2023": "arXiv-2023", "product": "ogbn-products",
    "cornell": "Cornell", "texas": "Texas", "wisconsin": "Wisconsin",
}

_DATA_CACHE = {}


def load_bundle():
    with open(PKG / "results" / "official_cells.json") as f:
        return json.load(f)


def load_seeds(bundle=None):
    if bundle and "run_seeds" in bundle:
        return bundle["run_seeds"]
    path = PKG / "configs" / "run_seeds.yaml"
    text = path.read_text()
    seeds = {}
    cur = None
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("run_seeds"):
            continue
        if ":" in line and "[" in line:
            k, rest = line.split(":", 1)
            nums = [int(x.strip()) for x in rest.strip().strip("[]").split(",") if x.strip()]
            seeds[k.strip()] = nums
            cur = k.strip()
    return seeds


def resolve_assets(explicit=None):
    """Asset bundle: --assets, TAPTN_ASSETS, sibling directory, or ./assets."""
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("TAPTN_ASSETS")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        PKG.parent / "TAPTN_finetune_repro_assets",
        PKG / "assets",
        Path.cwd() / "TAPTN_finetune_repro_assets",
        Path.cwd() / "assets",
    ]
    for c in candidates:
        if c.is_dir() and (c / "prt_lm").is_dir():
            return c.resolve()
    return candidates[0]


def assets_root():
    p = os.environ.get("TAPTN_ASSETS")
    return Path(p) if p else resolve_assets()


def setup_env():
    assets = assets_root()
    os.environ["TAPTN_ASSETS"] = str(assets)
    os.environ.setdefault("TAPTN_PRETRAINED", str(assets / "pretrained"))
    sys.path.insert(0, str(PKG))
    os.chdir(assets)
    return assets


def rng_seed(run_seeds, dataset, run):
    return int(run_seeds[dataset][run - 1])


def fmt(m, s):
    return f"{m:.2f}±{s:.2f}"


def mean_std_pct(accs):
    xs = [100.0 * a for a in accs]
    return sum(xs) / len(xs), (st.stdev(xs) if len(xs) > 1 else 0.0)


def cell_runs(c):
    return [r["test_acc"] for r in c["runs"]]


def find_cell(cells, encoder, method, dataset, pipeline=None, gnn=None):
    for c in cells:
        if c["encoder"] != encoder or c["dataset"] != dataset:
            continue
        if method is not None and c["method"] != method:
            continue
        if pipeline is not None and c["pipeline"] != pipeline:
            continue
        if gnn is not None and (c.get("gnn") or "") != gnn:
            continue
        return c
    return None


def paired_p(a, b):
    if len(a) != len(b) or len(a) < 2:
        return None
    try:
        from scipy import stats
        _, p = stats.ttest_rel(a, b)
        return float(p)
    except Exception:
        d = [x - y for x, y in zip(a, b)]
        n = len(d)
        md = sum(d) / n
        var = sum((x - md) ** 2 for x in d) / (n - 1)
        if var == 0:
            return 0.0 if md != 0 else 1.0
        t = md / math.sqrt(var / n)
        # two-sided critical value ~2.776 for df=4, alpha=0.05
        return 0.01 if abs(t) > 2.776 else 0.20


def sig_better(a, b, alpha=0.05):
    """True iff mean(a)>mean(b) and two-sided paired t p<alpha."""
    if len(a) != len(b) or len(a) < 2:
        return False, False
    ma, mb = sum(a) / len(a), sum(b) / len(b)
    mean_better = ma > mb
    p = paired_p(a, b)
    sig = mean_better and p is not None and p < alpha
    return sig, mean_better


def print_table(title, headers, rows):
    print(f"\n## {title}")
    print("| " + " | ".join(headers) + " |")
    print("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        print("| " + " | ".join(str(x) for x in r) + " |")


def _cell_fmt(c, with_paper=True):
    if not c:
        return "—"
    mu, sd = mean_std_pct(cell_runs(c))
    s = fmt(mu, sd)
    if with_paper and c.get("paper_mean") is not None:
        pm, ps = c["paper_mean"], c["paper_std"]
        if abs(mu - pm) > 0.06 or abs(sd - ps) > 0.06:
            s += f"  [paper {fmt(pm, ps)}]"
    return s


def table_tb6(cells):
    methods = ["TA", "GAT", "GraphSAGE", "ChebNet", "DGI", "GATv2", "GCNII",
               "ASDGN", "DirGNN", "ACM-GNN", "DMP", "GraphSAINT", "FSGNN",
               "APPNP", "GraphTARIF", "GraphICL", "TAPTN+LM"]
    dss = ["cora", "arxiv_2023", "product"]
    rows = []
    for m in methods:
        pipe = "P3" if m == "GraphICL" else ("P2" if m == "TAPTN+LM" else
               ("P1_LM" if m == "TA" else "P1_GNN"))
        row = [m]
        for ds in dss:
            c = find_cell(cells, "deberta", m, ds, pipeline=pipe)
            row.append(_cell_fmt(c))
        rows.append(row)
    print_table(
        "tb_6 — Frozen-feature GNNs vs TAPTN+LM (homophilic). "
        "Bracketed paper values appear only when they differ from recorded cells.",
        ["Method", "Cora", "arXiv-2023", "ogbn-products"], rows)


def table_het(cells):
    methods = [
        ("TA", "P1_LM"), ("GAT", "P1_GNN"), ("GATv2", "P1_GNN"),
        ("GraphSAGE", "P1_GNN"), ("ChebNet", "P1_GNN"), ("DGI", "P1_GNN"),
        ("GCNII", "P1_GNN"), ("ASDGN", "P1_GNN"), ("DirGNN", "P1_GNN"),
        ("ACM-GNN", "P1_GNN"), ("DMP", "P1_GNN"), ("GraphSAINT", "P1_GNN"),
        ("FSGNN", "P1_GNN"), ("APPNP", "P1_GNN"), ("GraphTARIF", "P1_GNN"),
        ("TAPTN+LM", "P2"),
    ]
    dss = ["texas", "wisconsin", "cornell"]
    rows = []
    for disp, pipe in methods:
        row = [disp if disp != "GraphSAGE" else "SAGE"]
        accs = []
        ok = True
        for ds in dss:
            c = find_cell(cells, "deberta", disp, ds, pipeline=pipe)
            if not c:
                row.append("—")
                ok = False
            else:
                mu, sd = mean_std_pct(cell_runs(c))
                row.append(_cell_fmt(c))
                accs.append(mu)
        row.append(f"{sum(accs)/3:.2f}" if ok and len(accs) == 3 else "—")
        rows.append(row)
    print_table(
        "heterophilic — Frozen-feature GNNs vs TAPTN+LM (WebKB). No GraphICL+LM row.",
        ["Method", "Texas", "Wisconsin", "Cornell", "Avg"], rows)


def table_joint(cells):
    methods = ["GAT", "GATv2", "SAGE", "ChebNet", "GCNII", "APPNP", "ASDGN",
               "DirGNN", "ACM-GNN", "DMP", "FSGNN", "GraphTARIF", "TAPTN+LM"]
    dss = ["cora", "arxiv_2023", "product", "texas", "wisconsin", "cornell"]
    names = [DS_PRINT[d] for d in dss]
    rows = []
    for m in methods:
        row = [m]
        for ds in dss:
            if m == "TAPTN+LM":
                c = find_cell(cells, "deberta", "TAPTN+LM", ds, pipeline="P2")
            else:
                c = find_cell(cells, "deberta", m, ds, pipeline="P4",
                              gnn=GNN_CODE.get(m, m))
            row.append(_cell_fmt(c))
        rows.append(row)
    print_table(
        "joint — Encoder+GNN jointly trained (DGI / GraphSAINT omitted).",
        ["Method"] + names, rows)


def table_roberta(cells):
    methods = ["TA", "GAT", "GATv2", "SAGE", "ChebNet", "GCNII", "DirGNN",
               "ACM-GNN", "DMP", "FSGNN", "APPNP", "GraphTARIF", "RevGAT", "TAPTN+LM"]
    dss = ["cora", "arxiv_2023", "product", "cornell", "texas", "wisconsin"]
    names = [DS_PRINT[d] for d in dss]
    rows = []
    for m in methods:
        pipe = "P2" if m == "TAPTN+LM" else ("P1_LM" if m == "TA" else "P1_GNN")
        row = [m]
        for ds in dss:
            c = find_cell(cells, "roberta", m, ds, pipeline=pipe)
            row.append(_cell_fmt(c))
        rows.append(row)
    print_table("roberta — RoBERTa-base encoder (frozen TA+GNN + TAPTN+LM).",
                ["Method"] + names, rows)


def _competing_for_summary(cells, ds):
    """Pipelines counted in tab:gnn_summary (not TAPTN-embedding GNNs, not RevGAT)."""
    out = []
    for c in cells:
        if c["encoder"] != "deberta" or c["dataset"] != ds:
            continue
        if c["pipeline"] not in ("P1_LM", "P3", "P1_GNN", "P4"):
            continue
        if c["method"] == "TAPTN+LM":
            continue
        if c.get("gnn") == "RevGAT":
            continue
        out.append(c)
    return out


def table_gnn_summary(cells):
    dss = ["cora", "arxiv_2023", "product", "cornell", "texas", "wisconsin"]
    rows = []
    for ds in dss:
        taptn = find_cell(cells, "deberta", "TAPTN+LM", ds, pipeline="P2")
        if not taptn:
            continue
        tv = cell_runs(taptn)
        mu, sd = mean_std_pct(tv)
        n_sig = n_mb = 0
        for c in _competing_for_summary(cells, ds):
            ov = cell_runs(c)
            if len(ov) != len(tv):
                continue
            sig, mb = sig_better(ov, tv)
            n_sig += int(sig)
            n_mb += int(mb)
        hw = 1.24 * sd
        lo, hi = mu - hw, min(100.0, mu + hw)
        paper = ""
        if taptn.get("paper_mean") is not None:
            paper = fmt(taptn["paper_mean"], taptn["paper_std"])
        rows.append([
            DS_PRINT[ds], HOMOPHILY[ds], fmt(mu, sd),
            f"[{lo:.2f},{hi:.2f}]", n_sig, n_mb, paper or "—",
        ])
    print_table(
        "gnn_summary — How many competing pipelines exceed TAPTN+LM "
        "(frozen + joint + TA+LM + GraphICL+LM where available; not TAPTN-embedding GNNs).",
        ["Dataset", "Homophily", "TAPTN+LM (recorded)", "95% CI (approx.)",
         "#sig >TAPTN+LM", "#mean >TAPTN+LM", "TAPTN+LM (paper)"],
        rows)
    print("Paper headline counts: Cora 2/9, arXiv-2023 0/0, ogbn-products 0/0, "
          "Cornell 0/0, Texas 0/0, Wisconsin 1/15.")


def table_p5(cells):
    dss = ["cora", "arxiv_2023", "product", "cornell", "texas", "wisconsin"]
    paper = {"cora": (2, 5), "arxiv_2023": (5, 0), "product": (11, 0),
             "cornell": (4, 0), "texas": (1, 0), "wisconsin": (0, 0)}
    rows = []
    tot1 = tot2 = 0
    for ds in dss:
        taptn = find_cell(cells, "deberta", "TAPTN+LM", ds, pipeline="P2")
        tv = cell_runs(taptn) if taptn else None
        n1 = n2 = 0
        for g in P5_GNNS:
            p5 = find_cell(cells, "deberta", None, ds, pipeline="P5", gnn=g)
            p1 = find_cell(cells, "deberta", None, ds, pipeline="P1_GNN", gnn=g)
            if not p5 or not p1:
                continue
            sig, _ = sig_better(cell_runs(p5), cell_runs(p1))
            n1 += int(sig)
            if tv:
                sig2, _ = sig_better(cell_runs(p5), tv)
                n2 += int(sig2)
        tot1 += n1
        tot2 += n2
        pr = paper.get(ds, ("?", "?"))
        rows.append([DS_PRINT[ds], HOMOPHILY[ds], n1, n2, pr[0], pr[1]])
    rows.append(["Σ", "", tot1, tot2, 23, 5])
    print_table(
        "tab:p5_transfer — TAPTN embeddings + frozen GNN (14 architectures). "
        "The paper caption of this table defines “P5” as that setting "
        "(a column abbreviation, not a main-table row name).",
        ["Dataset", "Homophily",
         "#(TAPTN-emb GNN > raw GNN) recorded",
         "#(TAPTN-emb GNN > TAPTN+LM) recorded",
         "paper col. #(P5>raw GNN)", "paper col. #(P5>TAPTN+LM)"],
        rows)


def table_crosslm(cells):
    dss = ["cora", "arxiv_2023", "product", "cornell", "texas", "wisconsin"]
    paper = {"deberta": [1, 0, 0, 0, 0, 0], "roberta": [0, 0, 0, 0, 0, 8]}
    rows = []
    for i, ds in enumerate(dss):
        row = [DS_PRINT[ds], HOMOPHILY[ds]]
        for enc in ("deberta", "roberta"):
            taptn = find_cell(cells, enc, "TAPTN+LM", ds, pipeline="P2")
            if not taptn:
                row.append("—")
                continue
            tv = cell_runs(taptn)
            n_sig = 0
            ta = find_cell(cells, enc, "TA", ds, pipeline="P1_LM")
            if ta and len(cell_runs(ta)) == len(tv):
                sig, _ = sig_better(cell_runs(ta), tv)
                n_sig += int(sig)
            for m in CROSSLM_GNN[enc]:
                c = find_cell(cells, enc, m, ds, pipeline="P1_GNN")
                if not c or len(cell_runs(c)) != len(tv):
                    continue
                sig, _ = sig_better(cell_runs(c), tv)
                n_sig += int(sig)
            row.append(f"{n_sig}  (paper {paper[enc][i]})")
        rows.append(row)
    print_table(
        "crosslm_encoder — # raw-text pipelines (TA+LM + 12 TA+GNN) significantly > TAPTN+LM.",
        ["Dataset", "Homophily", "DeBERTa", "RoBERTa"], rows)


def lm_stems(encoder, pipeline, dataset, seed, assets):
    if encoder == "deberta":
        model = "microsoft/deberta-base"
        suf = {("P1_LM"): "", ("P2"): "3", ("P3"): "3nosem"}[pipeline]
        return assets / "prt_lm" / f"{dataset}{suf}" / f"{model}-seed{seed}"
    model = "FacebookAI/roberta-base"
    ar = "crosslm/FacebookAI_roberta-base_20260616_081354"
    suf = "" if pipeline == "P1_LM" else "3"
    return assets / "prt_lm" / ar / f"{dataset}{suf}" / f"{model}-seed{seed}"


def _load_split(dataset, seed):
    key = (dataset, seed)
    if key in _DATA_CACHE:
        return _DATA_CACHE[key]
    from core.data_utils.load import load_data
    data, ncls = load_data(dataset, use_text=False, seed=seed)
    _DATA_CACHE[key] = (data, ncls)
    return data, ncls


def score_pred(dataset, seed, stem):
    import numpy as np
    pred_path = str(stem) + ".pred"
    if not os.path.isfile(pred_path):
        return None, "missing_pred"
    data, _ncls = _load_split(dataset, seed)
    n = int(data.y.shape[0])
    nbytes = os.path.getsize(pred_path)
    dim = nbytes // (n * 2)
    if dim <= 0:
        return None, f"bad_pred_size({nbytes} bytes, n={n})"
    pred = np.memmap(pred_path, dtype=np.float16, mode="r", shape=(n, dim))
    y = data.y.view(-1).detach().cpu().numpy()
    mask = data.test_mask.detach().cpu().numpy().astype(bool)
    yhat = np.argmax(pred[mask], axis=-1)
    acc = float((yhat == y[mask]).mean())
    return acc, "ok"


def dry_rescore_lm(cells, assets):
    n_ok = n_miss = n_mismatch = 0
    mismatches = []
    for c in cells:
        if c["pipeline"] not in ("P1_LM", "P2", "P3"):
            continue
        for r in c["runs"]:
            stem = lm_stems(c["encoder"], c["pipeline"], c["dataset"],
                            r["rng_seed"], assets)
            acc, stt = score_pred(c["dataset"], r["rng_seed"], stem)
            if acc is None:
                n_miss += 1
                mismatches.append((c, r, stt, None))
                continue
            n_ok += 1
            if abs(acc - r["test_acc"]) > 1e-3:
                n_mismatch += 1
                mismatches.append((c, r, "acc_diff", acc))
    print(f"\nLM .pred dry-run: scored={n_ok}  missing_pred={n_miss}  "
          f"mismatch>0.001={n_mismatch}")
    for c, r, why, acc in mismatches[:30]:
        rec = r["test_acc"]
        dry = "NA" if acc is None else f"{acc:.4f}"
        print(f"  {c['encoder']} {c['pipeline']} {c['method']} {c['dataset']} "
              f"run{r['run']} recorded={rec:.4f} dry={dry} ({why})")
    if len(mismatches) > 30:
        print(f"  ... {len(mismatches) - 30} more")
    return n_ok, n_miss, n_mismatch


def cmd_check(bundle):
    assets = setup_env()
    miss = []
    n = 0
    for c in bundle["cells"]:
        if c["pipeline"] not in ("P1_LM", "P2", "P3"):
            continue
        for r in c["runs"]:
            stem = lm_stems(c["encoder"], c["pipeline"], c["dataset"],
                            r["rng_seed"], assets)
            for ext in (".pred", ".emb", ".ckpt"):
                n += 1
                p = Path(str(stem) + ext)
                if not p.is_file():
                    miss.append(str(p))
    print(f"assets: {assets}")
    print(f"LM artifacts expected={n}  missing={len(miss)}")
    for p in miss[:20]:
        print("  missing", p)
    pre = assets / "pretrained"
    for name in ("microsoft/deberta-base", "FacebookAI/roberta-base"):
        ok = (pre / name).is_dir()
        print(f"  pretrained {name}: {'ok' if ok else 'MISSING'}")
    for rel in ("dataset/cora", "dataset/arxiv_2023", "dataset/product_cache",
                "gpt_responses/cora", "webkb-data",
                "webkb_html_order_texas.txt", "webkb_html_order_wisconsin.txt",
                "webkb_html_order_cornell.txt"):
        ok = (assets / rel).exists()
        print(f"  data {rel}: {'ok' if ok else 'MISSING'}")


def _train_cmd(args, run_seeds):
    ds = args.dataset
    seed = rng_seed(run_seeds, ds, args.run)
    enc = args.encoder
    model = "microsoft/deberta-base" if enc == "deberta" else "FacebookAI/roberta-base"
    art = "" if enc == "deberta" else "crosslm/FacebookAI_roberta-base_20260616_081354"
    py = sys.executable
    extra = ["lm.model.name", model, "lm.model.root", os.environ["TAPTN_PRETRAINED"],
             "lm.model.artifact_root", art, "dataset", ds, "seed", str(seed),
             "device", str(args.device)]
    pipe = args.pipeline
    if pipe == "ta_lm":
        cmd = [py, "-m", "core.trainLM", *extra, "lm.train.use_gpt", "False"]
    elif pipe == "taptn_lm":
        cmd = [py, "-m", "core.trainLM", *extra, "lm.train.use_gpt", "True",
               "lm.train.sem", "True"]
    elif pipe == "grapicl_lm":
        cmd = [py, "-m", "core.trainLM", *extra, "lm.train.use_gpt", "True",
               "lm.train.sem", "False"]
    elif pipe == "ta_gnn":
        gnn = GNN_CODE[args.gnn]
        cmd = [py, "-m", "core.trainEnsemble", *extra, "gnn.model.name", gnn,
               "gnn.train.feature_type", "TA"]
    elif pipe == "taptn_gnn":
        gnn = GNN_CODE[args.gnn]
        cmd = [py, "-m", "core.trainEnsemble", *extra, "gnn.model.name", gnn,
               "gnn.train.feature_type", "E"]
    elif pipe == "joint":
        gnn = GNN_CODE[args.gnn]
        cmd = [py, "-m", "core.trainJoint", *extra, "gnn.model.name", gnn]
    else:
        raise SystemExit(f"unknown pipeline {pipe}")
    if args.force_retrain and pipe in ("ta_lm", "taptn_lm", "grapicl_lm"):
        cmd += ["lm.train.force_retrain", "True"]
    return cmd, seed


def cmd_train(args, run_seeds):
    setup_env()
    env = os.environ.copy()
    env["PYTHONPATH"] = str(PKG) + os.pathsep + env.get("PYTHONPATH", "")
    cmd, seed = _train_cmd(args, run_seeds)
    print(f"run={args.run}  mapped_rng_seed={seed}")
    print(" ".join(cmd))
    if args.pipeline in ("ta_gnn", "taptn_gnn", "joint"):
        gnn = GNN_CODE[args.gnn]
        print("Note: GNN checkpoints are not run-indexed. This command overwrites "
              f"output/{args.dataset}/{gnn}.pt (or the joint counterpart).")
    rc = subprocess.call(cmd, cwd=str(assets_root()), env=env)
    sys.exit(rc)


def cmd_dry(args, bundle):
    assets = setup_env()
    cells = bundle["cells"]
    print("TAPTN fine-tuning reproduction (dry-run)")
    print(f"  code:    {PKG}")
    print(f"  assets:  {assets}  exists={assets.is_dir()}")
    print("  Run index 1–5 is mapped to a library RNG seed via configs/run_seeds.yaml.")
    print("  GNN *.pt files on disk are not run-indexed (later runs overwrite).")
    print("  Frozen/joint GNN numbers below are the recorded official cells.")
    print("  GraphICL+LM is the GraphICL-style auxiliary-text pipeline (not TAPTN).")
    which = TABLE_ALIASES.get(args.table, args.table)
    fn = {
        "gnn_summary": table_gnn_summary,
        "p5_transfer": table_p5,
        "crosslm_encoder": table_crosslm,
        "tb_6": table_tb6,
        "heterophilic": table_het,
        "joint": table_joint,
        "roberta": table_roberta,
    }
    order = TABLE_ORDER if which == "all" else [which]
    for t in order:
        fn[t](cells)
    if args.rescore_lm:
        print("\n--- Re-score LM classifiers from saved .pred (no training) ---")
        dry_rescore_lm(cells, assets)
    else:
        print("\n(LM .pred re-score skipped; pass --rescore-lm to verify against recorded cells.)")
    print("\nSee results/DISCREPANCIES.md for paper-table vs recorded-cell diffs.")
    print("See results/cells.md for per-run (run index 1–5) accuracies.")


def main():
    ap = argparse.ArgumentParser(
        description="Reproduce TAPTN fine-tuning tables.")
    ap.add_argument(
        "--assets", default=None,
        help="Unpacked data/weight bundle. Default: $TAPTN_ASSETS, else "
             "sibling TAPTN_finetune_repro_assets/ or ./assets/")
    sub = ap.add_subparsers(dest="cmd", required=True)

    d = sub.add_parser("dry", help="Assemble paper tables from recorded cells; optionally re-score LM .pred")
    d.add_argument("--table", default="all",
                   choices=TABLE_ORDER + list(TABLE_ALIASES) + ["all"])
    d.add_argument("--rescore-lm", action="store_true",
                   help="Recompute LM test acc from saved .pred files (no training)")
    d.add_argument("--skip-rescore", action="store_true", help=argparse.SUPPRESS)

    t = sub.add_parser("train", help="Train one cell (needed when GNN weights were overwritten)")
    t.add_argument("--pipeline", required=True, choices=list(PIPE_CLI))
    t.add_argument("--dataset", required=True,
                   choices=["cora", "arxiv_2023", "product", "cornell", "texas", "wisconsin"])
    t.add_argument("--run", type=int, required=True, choices=[1, 2, 3, 4, 5])
    t.add_argument("--encoder", default="deberta", choices=["deberta", "roberta"])
    t.add_argument("--gnn", default="SAGE",
                   help="Paper or library name: GAT, GraphSAGE, GCNII, ASDGN, ACM-GNN, ...")
    t.add_argument("--device", type=int, default=0)
    t.add_argument("--force-retrain", action="store_true",
                   help="Ignore an existing LM checkpoint and fine-tune from scratch")

    sub.add_parser("check", help="Verify that assets (data, LM weights) are present")

    args = ap.parse_args()
    assets = resolve_assets(args.assets)
    os.environ["TAPTN_ASSETS"] = str(assets)
    os.environ.setdefault("TAPTN_PRETRAINED", str(assets / "pretrained"))
    bundle = load_bundle()
    run_seeds = load_seeds(bundle)
    if args.cmd == "dry":
        cmd_dry(args, bundle)
    elif args.cmd == "train":
        cmd_train(args, run_seeds)
    else:
        cmd_check(bundle)


if __name__ == "__main__":
    main()
