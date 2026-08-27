#!/usr/bin/env python3
"""Reproduce TAPTN ICL tables (dry-run from pickles, or re-run LLMs).

Camera-ready source: ijcai25_camera_ready.tex
Rewiring / fine-tuning tables live in TAPTN_rewiring_repro and TAPTN_finetune_repro.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

PKG = Path(__file__).resolve().parent
sys.path.insert(0, str(PKG))

from core import cells as C
from core import cost as COST
from core import paper as P
from core import score as S
from core import tables as T
from core.names import (
    CURRENT_MODELS,
    CURRENT_TAPTN_MODELS,
    FACTORIAL_ROWS,
    MAIN_DATASETS,
    MAIN_METHODS,
)

TABLES = [
    "tab:main_5_datasets",
    "tab:factorial",
    "tab:product_70b",
    "tab:cost",
    "tb_52",
    "tb_4",
    "tab:decouple",
    "tab:current_channel",
    "tab:current_taptn",
]
# CLI aliases without the tab: prefix
TABLE_ALIAS = {t.split(":", 1)[-1] if t.startswith("tab:") else t: t for t in TABLES}
TABLE_ALIAS.update({t: t for t in TABLES})


def resolve_assets(explicit=None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    env = os.environ.get("TAPTN_ASSETS")
    if env:
        return Path(env).expanduser().resolve()
    candidates = [
        PKG.parent / "TAPTN_icl_repro_assets",
        PKG / "assets",
        Path.cwd() / "TAPTN_icl_repro_assets",
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


def canon_table(name: str) -> str:
    if name == "all":
        return "all"
    if name not in TABLE_ALIAS:
        raise SystemExit(f"unknown table {name!r}; choose from {list(TABLE_ALIAS)}")
    return TABLE_ALIAS[name]


def score_all(assets: Path, tables):
    want = set(TABLES if "all" in tables else tables)
    by_table = defaultdict(dict)
    diffs = []
    notes = []
    for cell in C.CELLS:
        if cell["table"] not in want and "all" not in tables:
            continue
        # cost table is computed from JSON, not CELLS
        if cell["table"] == "tab:cost":
            continue
        out = S.score_cell(cell, assets)
        by_table[cell["table"]][cell["key"]] = out
        paper_v = cell["paper"]
        got = S.pct(out)
        if out.get("missing"):
            diffs.append({
                "table": cell["table"], "cell": cell["key"],
                "paper": paper_v, "got": None, "delta": None,
                "note": out.get("note") or cell.get("note") or "pickle missing",
            })
            if cell.get("note"):
                notes.append(f"{cell['table']} {cell['key']}: {cell['note']}")
            continue
        d = got - paper_v
        if abs(d) > 0.06:
            diffs.append({
                "table": cell["table"], "cell": cell["key"],
                "paper": paper_v, "got": round(got, 4), "delta": round(d, 4),
                "note": cell.get("note"),
            })
    return by_table, diffs, notes


def _get(scored, key):
    out = scored.get(key)
    return S.pct(out)


def build_main(scored, rescore):
    if not rescore:
        return {m: dict(P.MAIN_5[m]) for m in MAIN_METHODS}
    out = {}
    for m in MAIN_METHODS:
        out[m] = {ds: _get(scored, f"{m}.{ds}") for ds in MAIN_DATASETS}
    return out


def build_factorial(scored, rescore):
    if not rescore:
        return {m: {k: v for k, v in P.FACTORIAL[m].items() if k != "avg"}
                for m in FACTORIAL_ROWS}
    out = {}
    for m in FACTORIAL_ROWS:
        out[m] = {ds: _get(scored, f"{m}.{ds}") for ds in MAIN_DATASETS}
    return out


def build_product(scored, rescore):
    if not rescore:
        return dict(P.PRODUCT_70B)
    return {k: _get(scored, k) for k in ("gicl2", "dense", "taptn1", "taptn2")}


def build_tb52(scored, rescore):
    if not rescore:
        return dict(P.TB_52)
    return {
        ("gicl", 1): (_get(scored, "gicl.1.no"), None),
        ("gicl", 2): (_get(scored, "gicl.2.no"), None),
        ("taptn", 1): (_get(scored, "taptn.1.no"), _get(scored, "taptn.1.yes")),
        ("taptn", 2): (_get(scored, "taptn.2.no"), _get(scored, "taptn.2.yes")),
    }


def build_tb4(scored, rescore):
    if not rescore:
        return {k: dict(v) for k, v in P.TB_4.items()}
    out = {}
    for hop in (1, 2):
        for ri in (0, 1, 2):
            out[("gicl", hop, ri)] = {
                "cora": _get(scored, f"gicl.{hop}.{ri}.cora"),
                "arxiv_2023": _get(scored, f"gicl.{hop}.{ri}.arxiv_2023"),
            }
        out[("taptn", hop, None)] = {
            "cora": _get(scored, f"taptn.{hop}.cora"),
            "arxiv_2023": _get(scored, f"taptn.{hop}.arxiv_2023"),
        }
    return out


def build_decouple(scored, rescore):
    if not rescore:
        return dict(P.DECOUPLE)
    out = {}
    cols = ("woS_woI", "woS_wI", "wS_woI", "wS_wI")
    for ds in MAIN_DATASETS:
        for hop in (1, 2):
            out[(ds, hop)] = tuple(_get(scored, f"{ds}.{hop}.{c}") for c in cols)
    return out


def build_channel(scored, rescore):
    if not rescore:
        return dict(P.CURRENT_CHANNEL)
    cols = ("ego", "anon", "sat", "taptn")
    return {m: tuple(_get(scored, f"{m}.{c}") for c in cols) for m in CURRENT_MODELS}


def build_current_taptn(scored, rescore):
    if not rescore:
        return {(ds, m): (a, b) for (ds, m), (a, b, _) in P.CURRENT_TAPTN.items()}
    out = {}
    for ds in ("texas", "cora"):
        for m in CURRENT_TAPTN_MODELS:
            out[(ds, m)] = (
                _get(scored, f"{ds}.{m}.gicl"),
                _get(scored, f"{ds}.{m}.taptn"),
            )
    return out


def cmd_check(assets: Path):
    if not assets.is_dir():
        print(f"ERROR: assets not found at {assets}")
        print("Unpack TAPTN_icl_repro_assets and export TAPTN_ASSETS=...")
        return 1
    copies = C.unique_copies()
    missing = []
    present = 0
    for dest in sorted(copies):
        p = assets / dest
        if p.exists():
            present += 1
        else:
            missing.append(dest)
    n_cells = len(C.CELLS)
    n_miss_map = sum(1 for c in C.CELLS if c["method"] == "missing")
    print(f"TAPTN_ASSETS = {assets}")
    print(f"  mapped cells:     {n_cells} ({n_miss_map} have no pickle in any author tree)")
    print(f"  shipped artifacts: {present}/{len(copies)}")
    print(f"  figures_paper:     {list((assets/'figures_paper').glob('*'))}")
    print(f"  dataset/cora:      {(assets/'dataset'/'cora').is_dir()}")
    print(f"  dataset/arxiv_2023:{(assets/'dataset'/'arxiv_2023').is_dir()}")
    print(f"  webkb-data:        {(assets/'webkb-data').is_dir()}")
    if missing:
        print("Missing from assets:")
        for m in missing:
            print(" -", m)
        return 1
    print("check: ok")
    return 0


def cmd_dry(assets: Path, tables, rescore: bool, figures: bool, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    want = TABLES if "all" in tables else tables
    scored = {}
    diffs, notes = [], []
    if rescore:
        scored, diffs, notes = score_all(assets, want)

    print("=" * 72)
    print("TAPTN ICL dry-run  (revised camera-ready)")
    print("=" * 72)

    if "tab:main_5_datasets" in want:
        print("\n## Table tab:main_5_datasets")
        print(T.main_5(build_main(scored.get("tab:main_5_datasets", {}), rescore)))

    if "tab:factorial" in want:
        print("\n## Table tab:factorial")
        print(T.factorial(build_factorial(scored.get("tab:factorial", {}), rescore)))

    if "tab:product_70b" in want:
        print("\n## Table tab:product_70b")
        print(T.product_70b(build_product(scored.get("tab:product_70b", {}), rescore)))

    if "tab:cost" in want:
        print("\n## Table tab:cost")
        if rescore:
            rows = COST.compute_cost(assets)
            print(T.cost_table(rows))
            for k, paper in P.COST.items():
                got = rows[k]
                for field, th in (("calls", 0.15), ("tokens", 800), ("cost_per_1k", 0.03), ("acc", 0.06)):
                    d = got[field] - paper[field]
                    if abs(d) > th:
                        diffs.append({
                            "table": "tab:cost", "cell": f"{k}.{field}",
                            "paper": paper[field], "got": round(got[field], 4),
                            "delta": round(d, 4),
                        })
        else:
            print(T.cost_table(P.COST))

    if "tb_52" in want:
        print("\n## Table tb_52")
        print(T.tb_52(build_tb52(scored.get("tb_52", {}), rescore)))

    if "tb_4" in want:
        print("\n## Table tb_4")
        print(T.tb_4(build_tb4(scored.get("tb_4", {}), rescore)))

    if "tab:decouple" in want:
        print("\n## Table tab:decouple")
        print(T.decouple(build_decouple(scored.get("tab:decouple", {}), rescore)))

    if "tab:current_channel" in want:
        print("\n## Table tab:current_channel")
        print(T.current_channel(build_channel(scored.get("tab:current_channel", {}), rescore)))

    if "tab:current_taptn" in want:
        print("\n## Table tab:current_taptn")
        print(T.current_taptn(build_current_taptn(scored.get("tab:current_taptn", {}), rescore)))

    fig_dir = out_dir / "figures"
    if figures or "all" in tables:
        fig_dir.mkdir(parents=True, exist_ok=True)
        paper_fig = assets / "figures_paper"
        for lab, fn in P.STATIC_FIGURES.items():
            src = paper_fig / fn
            if src.exists():
                shutil.copy2(src, fig_dir / fn)
                print(f"copied {lab} -> {fig_dir / fn}")
            else:
                notes.append(f"missing figure {src}")

    disc_path = out_dir / "dry_discrepancies.json"
    payload = {"discrepancies": diffs, "notes": notes, "rescore": rescore}
    disc_path.write_text(json.dumps(payload, indent=2, default=str))
    print("\n" + "-" * 72)
    if not rescore:
        print("Reprinted camera-ready numbers (no pickle rescore). Re-run with --rescore to verify.")
    else:
        missing = [d for d in diffs if d.get("got") is None]
        numeric = [d for d in diffs if d.get("got") is not None]
        print(f"{len(missing)} cell(s) have no pickle; {len(numeric)} numeric mismatch(es) > 0.06 pp.")
        for d in diffs:
            print(" -", d)
    for n in notes:
        if n not in {d.get("note") for d in diffs}:
            print(" note:", n)
    print(f"Wrote {disc_path}")
    return 0


def cmd_run(args, assets: Path):
    vendor = PKG / "vendor"
    python = sys.executable
    script = vendor / "run_taptn_expansion.py"
    if not script.exists():
        print("vendor/run_taptn_expansion.py missing")
        return 1

    run_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else (PKG / "output" / "rerun")
    run_dir.mkdir(parents=True, exist_ok=True)

    # Make dataset/ and file_abs visible from cwd.
    for name in ("dataset", "webkb-data"):
        src = assets / name
        dst = run_dir / name
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    for p in assets.glob("file_abs_*.pkl"):
        dst = run_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)
    for p in assets.glob("webkb_html_order_*.txt"):
        dst = run_dir / p.name
        if not dst.exists():
            shutil.copy2(p, dst)

    os.environ["TAPTN_ASSETS"] = str(assets)
    os.environ["PYTHONPATH"] = str(vendor) + os.pathsep + os.environ.get("PYTHONPATH", "")

    cmd = [
        python, str(script),
        "--dataset", args.dataset,
        "--model", args.model,
        "--config", args.config,
        "--max_workers", str(args.max_workers),
    ]
    if args.anon:
        cmd.append("--anon")
    if args.instr_v2:
        cmd.append("--instr_v2")
    if args.wis_ver:
        cmd.extend(["--wis_ver", args.wis_ver])
    print("cwd:", run_dir)
    print("cmd:", " ".join(cmd))
    if not os.environ.get("OPENAI_API_KEY") and not os.environ.get("TAPTN_LLM_API_KEY"):
        print("WARNING: set OPENAI_API_KEY (and optionally OPENAI_BASE_URL) before a live run.")
    return subprocess.call(cmd, cwd=str(run_dir), env=os.environ.copy())


def cmd_gate(assets: Path, out_dir: Path):
    """Recompute Cora neighbor-consensus gate from shipped source pickles (no LLM)."""
    vendor = PKG / "vendor"
    src_dir = assets / "pkls" / "current_taptn" / "gate_sources"
    if not src_dir.is_dir():
        print("gate sources missing under assets/pkls/current_taptn/gate_sources")
        return 1
    run_dir = out_dir / "gate"
    run_dir.mkdir(parents=True, exist_ok=True)
    for p in src_dir.glob("*.pkl"):
        shutil.copy2(p, run_dir / p.name)
    os.environ["TAPTN_ASSETS"] = str(assets)
    os.environ["PYTHONPATH"] = str(vendor) + os.pathsep + os.environ.get("PYTHONPATH", "")
    # cora_2hop_gate.py expects files in cwd and dataset via TAPTN_ASSETS
    for name in ("dataset",):
        src = assets / name
        dst = run_dir / name
        if src.exists() and not dst.exists():
            os.symlink(src, dst)
    script = vendor / "cora_2hop_gate.py"
    print("recomputing gate in", run_dir)
    return subprocess.call([sys.executable, str(script)], cwd=str(run_dir), env=os.environ.copy())


def main():
    parser = argparse.ArgumentParser(description="Reproduce TAPTN ICL tables")
    parser.add_argument("--assets", default=None)
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check")

    p_dry = sub.add_parser("dry")
    p_dry.add_argument("--table", nargs="+", default=["all"])
    p_dry.add_argument("--rescore", action="store_true",
                       help="Recompute accuracies from pickles (no LLM calls)")
    p_dry.add_argument("--figures", action="store_true")
    p_dry.add_argument("--output", default=None)

    p_run = sub.add_parser("run", help="Live LLM re-run via run_taptn_expansion.py")
    p_run.add_argument("--dataset", required=True,
                       choices=["texas", "cornell", "washington", "wisconsin", "cora"])
    p_run.add_argument("--model", required=True,
                       choices=["gemma-4-31b-it", "qwen3.5-27b", "gpt-oss-120b",
                                "glm-5.1", "llama-3.3-70b"])
    p_run.add_argument("--config", required=True,
                       choices=["graphicl1", "taptn1", "graphicl2", "taptn2", "ego"])
    p_run.add_argument("--anon", action="store_true")
    p_run.add_argument("--instr_v2", action="store_true")
    p_run.add_argument("--wis_ver", default="",
                       choices=["", "v2", "v3", "2hop", "v2_2hop"])
    p_run.add_argument("--max_workers", type=int, default=20)
    p_run.add_argument("--output-dir", default=None)

    p_gate = sub.add_parser("gate", help="Recompute Cora 2-hop neighbor-consensus (no LLM)")
    p_gate.add_argument("--output", default=None)

    args = parser.parse_args()
    assets = setup_env(args.assets)
    if args.cmd == "check":
        return cmd_check(assets)
    if args.cmd == "dry":
        tables = [canon_table(t) for t in args.table]
        out = Path(args.output) if args.output else (PKG / "output")
        return cmd_dry(assets, tables, args.rescore, args.figures, out)
    if args.cmd == "run":
        return cmd_run(args, assets)
    if args.cmd == "gate":
        out = Path(args.output) if args.output else (PKG / "output")
        return cmd_gate(assets, out)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
