"""
Modality Ablation Study — Expression vs. Mutation vs. CNV vs. Joint MOFA+ Factors.

For each of four modality conditions we:
  1. Train MOFA+ on that subset of views (or reuse the existing joint result)
  2. Run 5-fold CV with FTT, XGBoost, LightGBM on the resulting factor scores
  3. Record mean ± std C-index per model per condition

The joint condition reuses ``results/mofa_factors.csv`` (already trained) to
avoid redundant computation.  Single-modality MOFA+ trains in ~3–6 min each.

Outputs
-------
  results/mofa_factors_expr_only.csv
  results/mofa_factors_mut_only.csv
  results/mofa_factors_cnv_only.csv
  results/mofa_variance_expr_only.csv  (and _mut_only, _cnv_only)
  results/ablation_cindex_summary.csv  — 4 conditions × 3 models, mean ± std
  results/figures/ablation_cindex_comparison.png
  results/figures/ablation_variance_heatmap.png

Usage
-----
    python -m src.analysis.modality_ablation
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch

RESULTS_DIR  = Path("results")
FIGURES_DIR  = Path("results/figures")
RANDOM_STATE = 42

# Ablation conditions: label → modalities to pass to MOFA+
# "joint" is handled separately (reuse existing mofa_factors.csv)
ABLATION_SETS: dict[str, list[str]] = {
    "expr_only": ["expression"],
    "mut_only":  ["mutations"],
    "cnv_only":  ["cnv"],
}

# Display-friendly names for the four conditions
CONDITION_LABELS = {
    "joint":     "Joint (3-modal)",
    "expr_only": "Expression only",
    "mut_only":  "Mutation only",
    "cnv_only":  "CNV only",
}


# ── helpers ───────────────────────────────────────────────────────────────────

def _load_joint_cv_scores() -> dict[str, list[float]] | None:
    """
    Load per-fold C-indices from the existing joint model_cv_scores.csv.
    Returns dict {"ftt": [...], "xgb": [...], "lgb": [...]} or None if missing.
    """
    p = RESULTS_DIR / "model_cv_scores.csv"
    if not p.exists():
        print(f"  [warn] {p} not found — joint scores will be re-computed")
        return None
    df = pd.read_csv(p, index_col=0)
    # Columns are model names; rows are folds
    result: dict[str, list[float]] = {}
    for col in df.columns:
        key = col.lower().replace("-", "").replace("_", "").replace(" ", "")
        # Exclude baseline/raw columns (e.g. xgb_raw_cindex)
        if "raw" in key or "baseline" in key or "pca" in key:
            continue
        if "ftt" in key or "transformer" in key:
            result["ftt"] = df[col].tolist()
        elif "xgb" in key or "xgboost" in key:
            result["xgb"] = df[col].tolist()
        elif "lgb" in key or "lightgbm" in key:
            result["lgb"] = df[col].tolist()
    if not result:
        return None
    print(f"  Loaded joint CV scores from {p.name}: "
          f"FTT={np.mean(result.get('ftt',[])):.4f}, "
          f"XGB={np.mean(result.get('xgb',[])):.4f}, "
          f"LGB={np.mean(result.get('lgb',[])):.4f}")
    return result


def _run_single_condition(
    label: str,
    mofa_path: Path,
    dataset,
    device: torch.device,
) -> dict[str, list[float]]:
    """
    Load factor scores from ``mofa_path`` and run 5-fold CV.
    Returns {"ftt": [fold_scores], "xgb": [...], "lgb": [...]}.
    """
    from src.supervised.model_comparison import (
        build_mofa_feature_matrix,
        cross_validate_models,
    )

    print(f"\n{'='*60}")
    print(f"  Running survival CV for: {CONDITION_LABELS[label]}")
    print(f"  Factor file: {mofa_path.name}")
    print(f"{'='*60}")

    X, y = build_mofa_feature_matrix(dataset, mofa_path=mofa_path)
    cv   = cross_validate_models(X, y, device, label=label)

    return {"ftt": cv["ftt"], "xgb": cv["xgb"], "lgb": cv["lgb"]}


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_cindex_comparison(summary: pd.DataFrame, out_path: Path) -> None:
    """
    Grouped bar chart: 4 modality conditions × 3 model bars.
    summary has columns [ftt_mean, ftt_std, xgb_mean, xgb_std, lgb_mean, lgb_std]
    and index = condition labels.
    """
    conditions = list(summary.index)
    n_cond = len(conditions)
    models = ["ftt", "xgb", "lgb"]
    model_labels = {"ftt": "FT-Transformer", "xgb": "XGBoost", "lgb": "LightGBM"}
    model_colors = {"ftt": "#2c5f8a", "xgb": "#8a4a2c", "lgb": "#2c7a3a"}

    fig, ax = plt.subplots(figsize=(10, 5.5))
    bar_w   = 0.22
    x       = np.arange(n_cond)

    for mi, m in enumerate(models):
        means = summary[f"{m}_mean"].values
        stds  = summary[f"{m}_std"].values
        offset = (mi - 1) * bar_w
        bars = ax.bar(
            x + offset, means, bar_w,
            yerr=stds, capsize=4,
            color=model_colors[m], alpha=0.88,
            label=model_labels[m],
            error_kw=dict(linewidth=1.2, ecolor="#444"),
        )
        # Annotate bar tops
        for rect, mean in zip(bars, means):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.003,
                f"{mean:.4f}",
                ha="center", va="bottom",
                fontsize=7, color="#222",
            )

    # Reference line at joint best
    joint_best = max(
        summary.loc["Joint (3-modal)", f"{m}_mean"] for m in models
    )
    ax.axhline(joint_best, color="#c8b96e", linewidth=1.2,
               linestyle="--", label=f"Joint best ({joint_best:.4f})")

    ax.set_xticks(x)
    ax.set_xticklabels(
        [CONDITION_LABELS.get(c, c) for c in conditions],
        fontsize=10,
    )
    ax.set_ylabel("Concordance Index (C-index)", fontsize=11)
    ax.set_title(
        "Modality Ablation — Survival Prediction (5-fold CV)",
        fontsize=13, fontweight="bold",
    )
    ax.set_ylim(
        max(0.45, summary[[f"{m}_mean" for m in models]].min().min() - 0.05),
        min(1.00, summary[[f"{m}_mean" for m in models]].max().max() + 0.04),
    )
    ax.legend(fontsize=9, loc="lower right")
    ax.yaxis.grid(True, linestyle=":", alpha=0.5)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_variance_heatmap(out_path: Path) -> None:
    """
    Heatmap: factor × condition — total variance explained.
    Reads mofa_variance.csv, mofa_variance_expr_only.csv, etc.
    """
    var_files = {
        "Joint":      RESULTS_DIR / "mofa_variance.csv",
        "Expr only":  RESULTS_DIR / "mofa_variance_expr_only.csv",
        "Mut only":   RESULTS_DIR / "mofa_variance_mut_only.csv",
        "CNV only":   RESULTS_DIR / "mofa_variance_cnv_only.csv",
    }

    frames = {}
    for label, path in var_files.items():
        if path.exists():
            df = pd.read_csv(path, index_col=0)
            frames[label] = df.sum(axis=1)  # total var per factor

    if not frames:
        print("  [skip] No variance CSVs found — skipping heatmap")
        return

    heat = pd.DataFrame(frames)  # factors × conditions

    fig, ax = plt.subplots(figsize=(max(6, len(frames) * 1.8), max(5, len(heat) * 0.4)))
    im = ax.imshow(heat.values * 100, aspect="auto", cmap="YlOrRd",
                   interpolation="nearest")

    ax.set_xticks(range(len(heat.columns)))
    ax.set_xticklabels(heat.columns, fontsize=10)
    ax.set_yticks(range(len(heat.index)))
    ax.set_yticklabels(heat.index, fontsize=8)

    # Annotate cells
    for i in range(len(heat.index)):
        for j in range(len(heat.columns)):
            val = heat.values[i, j] * 100
            ax.text(j, i, f"{val:.1f}", ha="center", va="center",
                    fontsize=7, color="black" if val < 5 else "white")

    plt.colorbar(im, ax=ax, label="Total variance explained (%)")
    ax.set_title("MOFA+ Variance Explained per Factor — Modality Conditions",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")

    # Load dataset (shared across all conditions)
    print("\n=== Loading dataset ===")
    from src.preprocess import load_multiomics
    dataset = load_multiomics()

    from src.unsupervised.mofa_analysis import run_mofa_pipeline

    # Collect per-fold C-indices for all conditions
    all_cv: dict[str, dict[str, list[float]]] = {}

    # ── Joint condition (reuse existing factors if available) ──────────────────
    joint_factors = RESULTS_DIR / "mofa_factors.csv"
    joint_cv_loaded = _load_joint_cv_scores()

    if joint_cv_loaded is not None:
        print("\n=== Joint condition: using existing mofa_factors.csv + model_cv_scores.csv ===")
        all_cv["Joint (3-modal)"] = joint_cv_loaded
    else:
        # Fallback: re-run CV with existing factors (still skip re-training MOFA+)
        if joint_factors.exists():
            print("\n=== Joint condition: re-running CV on existing mofa_factors.csv ===")
        else:
            print("\n=== Joint condition: no existing factors — training joint MOFA+ ===")
            run_mofa_pipeline(dataset, out_suffix="")
        all_cv["Joint (3-modal)"] = _run_single_condition(
            "joint", joint_factors, dataset, device
        )

    # ── Single-modality conditions ─────────────────────────────────────────────
    for label, modalities in ABLATION_SETS.items():
        out_suffix  = f"_{label}"
        mofa_path   = RESULTS_DIR / f"mofa_factors{out_suffix}.csv"
        display_lbl = CONDITION_LABELS[label]

        print(f"\n{'='*60}")
        print(f"  MOFA+ training for: {display_lbl}")
        print(f"{'='*60}")

        run_mofa_pipeline(
            dataset,
            modalities=modalities,
            out_suffix=out_suffix,
        )

        all_cv[display_lbl] = _run_single_condition(
            label, mofa_path, dataset, device
        )

    # ── Summarise ─────────────────────────────────────────────────────────────
    print("\n=== Ablation Summary ===")
    rows = []
    for cond, cv in all_cv.items():
        row = {"condition": cond}
        for m in ["ftt", "xgb", "lgb"]:
            scores = cv.get(m, [])
            row[f"{m}_mean"] = float(np.mean(scores)) if scores else np.nan
            row[f"{m}_std"]  = float(np.std(scores))  if scores else np.nan
        rows.append(row)

    summary = pd.DataFrame(rows).set_index("condition")
    print(summary.to_string())

    out_csv = RESULTS_DIR / "ablation_cindex_summary.csv"
    summary.to_csv(out_csv)
    print(f"\n  Saved: {out_csv.name}")

    # ── Figures ───────────────────────────────────────────────────────────────
    print("\n=== Generating figures ===")
    plot_cindex_comparison(summary, FIGURES_DIR / "ablation_cindex_comparison.png")
    plot_variance_heatmap(FIGURES_DIR / "ablation_variance_heatmap.png")

    # Print interpretation
    print("\n=== Interpretation ===")
    joint_row = summary.loc["Joint (3-modal)"]
    for cond, row in summary.iterrows():
        if cond == "Joint (3-modal)":
            continue
        for m in ["ftt", "xgb", "lgb"]:
            delta = row[f"{m}_mean"] - joint_row[f"{m}_mean"]
            print(f"  {cond:<20} {m.upper()}: "
                  f"{row[f'{m}_mean']:.4f} (delta {delta:+.4f} vs joint)")

    print("\nModality ablation complete.")


if __name__ == "__main__":
    main()
