"""Score rewiring pickles the same way as analyze_rewiring_lmarena.py / the GPT-3.5 notebooks."""
from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np

WEBKB_OPTIONS = ["faculty", "staff", "department", "course", "project", "student"]


def normalize_label(label):
    if label is None:
        return ""
    normalized = str(label).strip()
    normalized = re.sub(r"[.,!?;:]+$", "", normalized)
    return normalized.lower()


def get_matched_option(prediction, valid_options):
    if not prediction:
        return ""
    prediction = str(prediction).lower()
    matched_option = ""
    earliest_position = len(prediction)
    for option in valid_options:
        position = prediction.find(option.lower())
        if position != -1 and position < earliest_position:
            matched_option = option
            earliest_position = position
    return matched_option


def is_correct(predicted, true_label, valid_options=None):
    if predicted is None:
        return False
    valid_options = valid_options or WEBKB_OPTIONS
    true_label = str(true_label)
    if predicted == true_label:
        return True
    extracted = get_matched_option(predicted, valid_options)
    if extracted:
        if extracted == true_label:
            return True
        if normalize_label(extracted) == normalize_label(true_label):
            return True
    if normalize_label(predicted) == normalize_label(true_label):
        return True
    return False


def _test_indices(data):
    mask = data.test_mask
    if hasattr(mask, "numpy"):
        mask = mask.numpy()
    return np.where(mask)[0]


def score_section2(pkl, valid_options=None):
    """List `results` aligned to unshuffled test_mask (Section-2 refine pkls)."""
    valid_options = valid_options or WEBKB_OPTIONS
    data, text, results = pkl["data"], pkl["text"], pkl["results"]
    test_indices = _test_indices(data)
    if len(results) == len(test_indices):
        node_index_list = test_indices.tolist()
    else:
        node_index_list = test_indices[: len(results)].tolist()
    correct = total = 0
    for i, node_idx in enumerate(node_index_list):
        if results[i] is None:
            continue
        total += 1
        if is_correct(results[i], text["label"][node_idx], valid_options):
            correct += 1
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "correct": correct, "total": total, "method": "section2"}


def score_wrong_index(pkl):
    """GPT-3.5 / tb_3 notebooks: acc = 1 - len(wrong_index)/len(result)."""
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


def score_legacy_robust(pkl, valid_options=None):
    """Robust-match the old `result` list against test labels (first n test nodes)."""
    valid_options = valid_options or WEBKB_OPTIONS
    results = pkl.get("result") or pkl.get("results")
    data, text = pkl["data"], pkl["text"]
    test_indices = _test_indices(data)
    n = min(len(results), len(test_indices))
    correct = total = 0
    for i in range(n):
        pred = results[i]
        if pred is None:
            continue
        total += 1
        if is_correct(pred, text["label"][test_indices[i]], valid_options):
            correct += 1
    acc = correct / total if total else 0.0
    return {"accuracy": acc, "correct": correct, "total": total, "method": "legacy_robust"}


def load_pkl(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def score_path(path, prefer="auto"):
    """Score one pickle. prefer: auto|section2|wrong_index|stored|legacy_robust."""
    path = Path(path)
    pkl = load_pkl(path)
    if prefer != "auto":
        fn = {
            "section2": score_section2,
            "wrong_index": score_wrong_index,
            "stored": score_stored,
            "legacy_robust": score_legacy_robust,
        }[prefer]
        out = fn(pkl)
        out["path"] = str(path)
        return out

    if isinstance(pkl.get("results"), list) and pkl.get("data") is not None:
        out = score_section2(pkl)
    elif pkl.get("result") is not None and (
        pkl.get("wrong_index") is not None or pkl.get("wrong_indexes") is not None
    ):
        out = score_wrong_index(pkl)
    elif "accuracy" in pkl:
        out = score_stored(pkl)
    elif isinstance(pkl.get("result"), list) and pkl.get("data") is not None:
        out = score_legacy_robust(pkl)
    else:
        raise ValueError(f"unrecognised pickle schema: {path} keys={list(pkl)[:12]}")
    out["path"] = str(path)
    return out


def parse_section2_filename(stem: str):
    """Return (dataset, hop, model, setting, full_abs) from a refine filename."""
    parts = stem.split("_")
    dataset = parts[0]
    hop = int(parts[1].replace("hop", "")) if parts[1].startswith("hop") else None
    setting = "rewired" if stem.endswith("_rewired") else "no_rewiring"
    full_abs = "full_abs" in stem
    flag_tokens = {"instr", "refl", "refine", "anon"}
    try:
        refine_idx = parts.index("refine")
        if "no" in parts:
            no_idx = parts.index("no")
            model_parts = parts[refine_idx + 1 : no_idx]
        else:
            rewired_idx = parts.index("rewired")
            model_parts = parts[refine_idx + 1 : rewired_idx]
        while model_parts:
            tok = model_parts[0]
            if tok in flag_tokens:
                model_parts = model_parts[1:]
            elif tok == "full" and len(model_parts) > 1 and model_parts[1] == "abs":
                model_parts = model_parts[2:]
            else:
                break
        model = "_".join(p for p in model_parts if p).strip("_")
    except Exception:
        model = "unknown"
    return dataset, hop, model, setting, full_abs
