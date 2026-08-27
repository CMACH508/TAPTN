"""Score TAPTN ICL pickles under the protocol that produced each paper cell."""
from __future__ import annotations

import json
import pickle
from pathlib import Path


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def as_percent(acc_frac: float) -> float:
    return 100.0 * float(acc_frac)


def score_wrong_index(pkl):
    results = pkl.get("result") or pkl.get("results")
    wrong = pkl.get("wrong_index")
    if wrong is None:
        wrong = pkl.get("wrong_indexes") or []
    n = len(results)
    w = len(wrong)
    acc = (n - w) / n if n else 0.0
    return {"accuracy": acc, "correct": n - w, "total": n, "method": "wrong_index"}


def score_stored(pkl):
    if "accuracy" not in pkl:
        raise KeyError("no stored accuracy")
    acc = float(pkl["accuracy"])
    return {"accuracy": acc, "correct": None, "total": None, "method": "stored"}


def score_acc_gate(pkl):
    if "acc_gate" not in pkl:
        raise KeyError("no acc_gate")
    # already stored as percent
    pct = float(pkl["acc_gate"])
    return {"accuracy": pct / 100.0, "correct": None, "total": None, "method": "acc_gate"}


def score_reextract(pkl):
    if "accuracy_reextract" not in pkl:
        raise KeyError("no accuracy_reextract")
    acc = float(pkl["accuracy_reextract"])
    return {"accuracy": acc, "correct": None, "total": None, "method": "reextract"}


def score_product_400(hop1_pkl, ref400_pkl):
    """Map hop1 wrong_index as positions into result keys, intersect with 400 test ids."""
    res1 = hop1_pkl.get("result") or hop1_pkl.get("results")
    res2 = ref400_pkl.get("result") or ref400_pkl.get("results")
    w1 = list(hop1_pkl.get("wrong_index") or hop1_pkl.get("wrong_indexes") or [])
    keys1 = list(res1.keys())
    ids400 = set(res2.keys())
    wrong_keys = {keys1[i] for i in w1 if 0 <= i < len(keys1)}
    n_wrong = len(wrong_keys & ids400)
    n = len(ids400)
    acc = (n - n_wrong) / n if n else 0.0
    return {
        "accuracy": acc, "correct": n - n_wrong, "total": n,
        "method": "product_400", "n_wrong_400": n_wrong,
    }


def score_dense_json(path):
    data = json.loads(Path(path).read_text())
    acc = float(data["summary"]["accuracy"])
    n = int(data["summary"]["n_scored"])
    c = int(data["summary"]["n_correct"])
    return {"accuracy": acc, "correct": c, "total": n, "method": "dense_json"}


def score_cell(cell, assets: Path):
    """Return {accuracy: fraction, ...} or {missing: True}."""
    if cell["method"] == "missing" or not cell["dest"]:
        return {"missing": True, "accuracy": None, "method": "missing",
                "note": cell.get("note")}
    path = assets / cell["dest"]
    if not path.exists():
        return {"missing": True, "accuracy": None, "method": cell["method"],
                "note": f"file not in assets: {cell['dest']}"}

    if cell["method"] == "dense_json":
        out = score_dense_json(path)
        out["path"] = str(path)
        return out

    pkl = load_pkl(path)
    if cell["method"] == "wrong_index":
        out = score_wrong_index(pkl)
    elif cell["method"] == "stored":
        out = score_stored(pkl)
    elif cell["method"] == "acc_gate":
        out = score_acc_gate(pkl)
    elif cell["method"] == "reextract":
        out = score_reextract(pkl)
    elif cell["method"] == "product_400":
        extra = assets / cell["extra"]
        out = score_product_400(pkl, load_pkl(extra))
    else:
        raise ValueError(f"unknown method {cell['method']}")
    out["path"] = str(path)
    return out


def pct(out) -> float | None:
    if not out or out.get("missing") or out.get("accuracy") is None:
        return None
    return 100.0 * float(out["accuracy"])
