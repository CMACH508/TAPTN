"""Capability–sensitivity regression (paper sign: S = O − R)."""
from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


def linreg(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    lr = LinearRegression()
    X = x.reshape(-1, 1)
    lr.fit(X, y)
    pred = lr.predict(X)
    return {
        "pearson_r": float(pr),
        "pearson_p": float(pp),
        "spearman_rho": float(sr),
        "spearman_p": float(sp),
        "r2": float(r2_score(y, pred)),
        "intercept": float(lr.intercept_),
        "slope": float(lr.coef_[0]),
        "slope_per_100": float(lr.coef_[0] * 100.0),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "model": lr,
    }


def model_averages(cells, models):
    """cells[(model, dataset)] = (O, R) as fractions 0–1. Return list of dicts."""
    rows = []
    for m in models:
        os_, rs = [], []
        for ds in ("cornell", "texas", "washington", "wisconsin"):
            if (m, ds) not in cells:
                os_, rs = [], []
                break
            o, r = cells[(m, ds)]
            os_.append(o)
            rs.append(r)
        if not os_:
            continue
        o = float(np.mean(os_))
        r = float(np.mean(rs))
        # Paper Rel. column = mean of per-dataset relative changes, not
        # (mean R − mean O) / mean O. Phi-4 is the row where these differ in sign.
        rels = [(rr - oo) / oo * 100.0 if oo else float("nan") for oo, rr in zip(os_, rs)]
        rel = float(np.mean(rels))
        err_inc = ((1.0 - r) / (1.0 - o) - 1.0) * 100.0 if o < 1 else float("nan")
        rows.append({
            "model": m,
            "O": o,
            "R": r,
            "delta_R_minus_O": r - o,
            "S": o - r,
            "rel": rel,
            "err_inc": err_inc,
        })
    return rows


def capability_stats(rows, scores):
    x = [scores[r["model"]] for r in rows]
    out = {
        "S": linreg(x, [r["S"] for r in rows]),
        "O": linreg(x, [r["O"] for r in rows]),
        "R": linreg(x, [r["R"] for r in rows]),
    }
    # drop sklearn model objects for JSON
    for k in out:
        out[k] = {kk: vv for kk, vv in out[k].items() if kk != "model"}
    return out
