"""Regenerate paper figures from scored cells (not bitwise copies of the PNGs)."""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

from .names import DATASETS, DS_PRINT, display

try:
    from adjustText import adjust_text
    ADJUSTTEXT = True
except ImportError:
    ADJUSTTEXT = False


def _style():
    try:
        plt.style.use("seaborn-v0_8-darkgrid")
    except Exception:
        try:
            plt.style.use("seaborn-darkgrid")
        except Exception:
            pass
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Arial", "sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False


def comparison_by_dataset(pivot_rows, out_path: Path, title_suffix=""):
    """pivot_rows: list of dicts with dataset, model, O, R (fractions)."""
    _style()
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    for idx, dataset in enumerate(DATASETS):
        ax = axes[idx]
        rows = [r for r in pivot_rows if r["dataset"] == dataset]
        rows = sorted(rows, key=lambda r: r["R"] - r["O"])
        x = np.arange(len(rows))
        width = 0.35
        ax.bar(x - width / 2, [r["O"] for r in rows], width, label="Original", alpha=0.8)
        ax.bar(x + width / 2, [r["R"] for r in rows], width, label="Rewired", alpha=0.8)
        ax.set_xlabel("Model", fontweight="bold")
        ax.set_ylabel("Accuracy", fontweight="bold")
        ax.set_title(f"{DS_PRINT[dataset]} Dataset{title_suffix}", fontsize=14, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([display(r["model"]) for r in rows], rotation=45, ha="right", fontsize=9)
        ax.legend()
        ax.grid(True, alpha=0.3)
        for i, r in enumerate(rows):
            delta = r["R"] - r["O"]
            y_pos = max(r["O"], r["R"]) + 0.02
            color = "green" if delta > 0 else "red" if delta < 0 else "black"
            ax.text(i, y_pos, f"{delta:+.3f}", ha="center", va="bottom",
                    fontsize=8, color=color, fontweight="bold")
    plt.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def correlation_panels(rows, scores, out_path: Path, caption_note=""):
    """rows: model averages with O, R, S, rel. scores: model -> LMArena."""
    _style()
    x = np.array([scores[r["model"]] for r in rows], dtype=float)
    o = np.array([r["O"] for r in rows])
    r = np.array([r["R"] for r in rows])
    s = np.array([r["S"] for r in rows])
    rel = np.array([r["rel"] for r in rows])  # (R-O)/O * 100; relative sensitivity = -rel
    rel_sens = -rel

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    acc_y_min = min(o.min(), r.min()) - 0.05
    acc_y_max = max(o.max(), r.max()) + 0.05
    x_line = np.linspace(x.min(), x.max(), 100)

    def _scatter(ax, y, color, ylabel, title_prefix, y0=None):
        from scipy import stats as spstats
        pr, pp = spstats.pearsonr(x, y)
        lr = LinearRegression().fit(x.reshape(-1, 1), y)
        y_line = lr.predict(x_line.reshape(-1, 1))
        r2 = r2_score(y, lr.predict(x.reshape(-1, 1)))
        ax.scatter(x, y, s=150, alpha=0.6, color=color, edgecolors="black")
        ax.plot(x_line, y_line, "r--", linewidth=2, label=f"Regression (R²={r2:.3f})")
        texts = []
        y_range = y.max() - y.min() if y.max() != y.min() else 0.05
        for xi, yi, row in zip(x, y, rows):
            texts.append(ax.text(xi, yi + 0.04 * y_range, display(row["model"]),
                                 fontsize=8, ha="center", va="bottom"))
        if ADJUSTTEXT:
            adjust_text(texts, ax=ax,
                        arrowprops=dict(arrowstyle="-", color="gray", lw=0.5, alpha=0.5),
                        expand_points=(2.0, 2.0), expand_text=(1.3, 1.3),
                        force_points=(0.3, 0.3), force_text=(0.6, 0.6),
                        only_move={"points": "y", "text": "xy"})
        ax.set_xlabel("LMArena Score", fontweight="bold")
        ax.set_ylabel(ylabel, fontweight="bold")
        ax.set_title(f"{title_prefix}\nr={pr:.3f}, p={pp:.4f}", fontsize=12, fontweight="bold")
        ax.legend()
        ax.grid(True, alpha=0.3)
        if y0 is not None:
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        return pr, pp

    _scatter(axes[0, 0], s, "C0", "Structural Sensitivity",
             "Structural Sensitivity vs Model Capability", y0=0)
    _scatter(axes[0, 1], o, "orange", "Accuracy (Original)", "Original Performance")
    axes[0, 1].set_ylim(acc_y_min, acc_y_max)
    _scatter(axes[1, 0], r, "green", "Accuracy (Rewired)", "Rewired Performance")
    axes[1, 0].set_ylim(acc_y_min, acc_y_max)
    _scatter(axes[1, 1], rel_sens, "purple", "Relative Sensitivity (%)",
             "Relative Sensitivity", y0=0)
    plt.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close(fig)


def heatmap(pivot_rows, out_path: Path):
    _style()
    models = []
    seen = set()
    for r in pivot_rows:
        if r["model"] not in seen:
            seen.add(r["model"])
            models.append(r["model"])
    mat = np.full((len(models), len(DATASETS)), np.nan)
    mi = {m: i for i, m in enumerate(models)}
    for r in pivot_rows:
        mat[mi[r["model"]], DATASETS.index(r["dataset"])] = r["O"] - r["R"]  # sensitivity
    plt.figure(figsize=(12, 8))
    sns.heatmap(mat, annot=True, fmt=".4f", cmap="RdYlGn", center=0,
                xticklabels=[DS_PRINT[d] for d in DATASETS],
                yticklabels=[display(m) for m in models],
                cbar_kws={"label": "Structural Sensitivity"}, linewidths=0.5)
    plt.title("Structural Sensitivity Heatmap Across Datasets", fontsize=14, fontweight="bold", pad=20)
    plt.xlabel("Dataset", fontweight="bold")
    plt.ylabel("Model", fontweight="bold")
    plt.tight_layout(pad=2.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.3)
    plt.close()
