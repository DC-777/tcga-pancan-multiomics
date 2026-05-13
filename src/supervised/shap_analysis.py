"""
Phase 3C: SHAP interpretation of Cox Ridge and DeepSurv survival models.

Methods:
  Cox Ridge  — linear model, so SHAP values are exact:
               shap_i(x) = beta_i * (x_i - E_train[x_i])
               (equivalent to LinearExplainer output)

  DeepSurv   — shap.GradientExplainer on the PyTorch risk head

Outputs (results/figures/):
  shap_cox_beeswarm.png         SHAP beeswarm (top 20 features)
  shap_cox_bar.png              Mean |SHAP| bar chart
  shap_deepsurv_beeswarm.png    DeepSurv SHAP beeswarm
  shap_deepsurv_bar.png         DeepSurv mean |SHAP| bar
  shap_model_comparison.png     Top-20 importance per model side-by-side
  shap_values_cox.csv           Per-sample SHAP matrix (Cox)
  shap_values_deepsurv.csv      Per-sample SHAP matrix (DeepSurv)

Usage:
    python -m src.supervised.shap_analysis
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as _mpl

import torch
import shap

from sklearn.preprocessing import StandardScaler

FIGURES_DIR  = Path("results/figures")
RESULTS_DIR  = Path("results")
RANDOM_STATE = 42
TOP_N        = 20   # features shown in beeswarm / bar charts


# ── colour helpers ────────────────────────────────────────────────────────────

def _feature_label(name: str) -> str:
    return (name.replace("EXPR_PC", "ExprPC")
                .replace("CNV_PC",  "CNVPC")
                .replace("MUT_",    ""))


# ── Cox SHAP (exact linear) ──────────────────────────────────────────────────

def compute_cox_shap(cox_model, X_scaled: np.ndarray,
                     feature_names: list[str]) -> np.ndarray:
    """
    Exact SHAP for a linear Cox model.
    shap_i(x) = beta_i * (x_i - E[x_i])
    X_scaled must already be StandardScaler-transformed (mean~0, std~1).
    E[x_i] ~ 0, so shap_i(x) ~ beta_i * x_i.
    We centre per-feature to be exact.
    """
    betas = cox_model.params_.values          # shape (n_features,)
    X_mean = X_scaled.mean(axis=0)
    shap_vals = (X_scaled - X_mean) * betas   # (n_samples, n_features)
    return shap_vals


# ── DeepSurv SHAP (GradientExplainer) ────────────────────────────────────────

class _DeepSurv2D(torch.nn.Module):
    """Thin wrapper that keeps output 2D (n, 1) for shap.GradientExplainer."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return out.unsqueeze(1) if out.dim() == 1 else out


def compute_deepsurv_shap(model, X_scaled: np.ndarray,
                           device, n_background: int = 100) -> np.ndarray:
    """
    GradientExplainer SHAP for DeepSurv.
    Uses a random subsample as the background distribution.
    """
    rng = np.random.default_rng(RANDOM_STATE)
    bg_idx = rng.choice(len(X_scaled), min(n_background, len(X_scaled)), replace=False)
    background  = torch.FloatTensor(X_scaled[bg_idx]).to(device)
    data_tensor = torch.FloatTensor(X_scaled).to(device)

    wrapper  = _DeepSurv2D(model).to(device)
    wrapper.eval()
    explainer = shap.GradientExplainer(wrapper, background)
    shap_vals = explainer.shap_values(data_tensor)   # list[(n_samples, n_features)]
    if isinstance(shap_vals, list):
        shap_vals = shap_vals[0]
    shap_vals = np.array(shap_vals)
    if shap_vals.ndim == 3:
        shap_vals = shap_vals.squeeze(-1)
    return shap_vals


# ── plots ─────────────────────────────────────────────────────────────────────

def _beeswarm(shap_vals: np.ndarray, X_scaled: np.ndarray,
               feature_names: list[str], title: str, out_path: Path,
               top_n: int = TOP_N) -> None:
    """
    Manual beeswarm-style SHAP plot (no shap.plots dependency issues).
    Rows = top features by mean|SHAP|; points coloured by feature value.
    """
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = np.argsort(mean_abs)[::-1][:top_n][::-1]   # ascending for horizontal

    order   = list(order)
    labels  = [_feature_label(feature_names[i]) for i in order]
    sv_sub  = shap_vals[:, order]          # (n_samples, top_n)
    xv_sub  = X_scaled[:, order]           # feature values for colouring

    rng = np.random.default_rng(RANDOM_STATE)
    jitter = rng.uniform(-0.3, 0.3, sv_sub.shape)

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)))
    cmap = _mpl.colormaps["coolwarm"]

    for fi in range(top_n):
        y_pos  = fi + jitter[:, fi]
        colour = cmap((xv_sub[:, fi] - xv_sub[:, fi].min()) /
                      (xv_sub[:, fi].max() - xv_sub[:, fi].min() + 1e-9))
        ax.scatter(sv_sub[:, fi], y_pos, c=colour, s=6, alpha=0.5, linewidths=0)

    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("SHAP value  (impact on log risk)")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("Feature value (low → high)", fontsize=8)
    cb.set_ticks([0, 1])
    cb.set_ticklabels(["Low", "High"])

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _bar_chart(shap_vals: np.ndarray, feature_names: list[str],
               title: str, out_path: Path, top_n: int = TOP_N) -> None:
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = list(np.argsort(mean_abs)[::-1][:top_n][::-1])
    labels   = [_feature_label(feature_names[i]) for i in order]
    vals     = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh(range(top_n), vals, color="steelblue", alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_model_comparison(cox_shap: np.ndarray, ds_shap: np.ndarray,
                           feature_names: list[str], out_path: Path,
                           top_n: int = TOP_N) -> None:
    cox_imp = np.abs(cox_shap).mean(axis=0)
    ds_imp  = np.abs(ds_shap).mean(axis=0)

    # Union of top features from each model
    top_cox = set(np.argsort(cox_imp)[::-1][:top_n])
    top_ds  = set(np.argsort(ds_imp)[::-1][:top_n])
    union   = sorted(top_cox | top_ds, key=lambda i: -(cox_imp[i] + ds_imp[i]))[:top_n]
    union   = union[::-1]   # ascending for barh

    labels = [_feature_label(feature_names[i]) for i in union]
    x      = np.arange(len(union))
    width  = 0.38

    fig, ax = plt.subplots(figsize=(9, max(5, len(union) * 0.38)))
    ax.barh(x - width/2, cox_imp[union], width,
            label=f"Cox Ridge",  color="steelblue",  alpha=0.85)
    ax.barh(x + width/2, ds_imp[union],  width,
            label=f"DeepSurv",   color="darkorange", alpha=0.85)
    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("SHAP Feature Importance — Cox Ridge vs. DeepSurv")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_shap_pipeline(dataset=None) -> dict:
    if dataset is None:
        from src.preprocess import load_multiomics
        dataset = load_multiomics()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    from src.supervised.cox_models import build_feature_matrix, elasticnet_select
    from src.supervised.deepsurv import train_deepsurv, DeepSurv

    print("\n=== Building feature matrix ===")
    X_surv, y_surv = build_feature_matrix(dataset)

    print("\n=== ElasticNet feature selection ===")
    features = elasticnet_select(X_surv, y_surv)
    X_raw    = X_surv[features].values
    n_feat   = len(features)

    # ── Fit Cox on full data ──────────────────────────────────────────────────
    print("\n=== Fitting Cox Ridge (full data) for SHAP ===")
    from lifelines import CoxPHFitter

    sc_cox  = StandardScaler()
    X_cox_s = sc_cox.fit_transform(X_raw)
    df_full = pd.concat([
        pd.DataFrame(X_cox_s, columns=features),
        y_surv.reset_index(drop=True)
    ], axis=1)
    cox_model = CoxPHFitter(penalizer=0.1)
    cox_model.fit(df_full, duration_col="OS_time", event_col="OS_event")
    print("  Cox fitted.")

    print("\n=== Computing Cox SHAP values ===")
    cox_shap_vals = compute_cox_shap(cox_model, X_cox_s, features)
    print(f"  Cox SHAP shape: {cox_shap_vals.shape}")

    _beeswarm(cox_shap_vals, X_cox_s, features,
              "Cox Ridge SHAP — top features (beeswarm)",
              FIGURES_DIR / "shap_cox_beeswarm.png")
    _bar_chart(cox_shap_vals, features,
               "Cox Ridge — Mean |SHAP value| per feature",
               FIGURES_DIR / "shap_cox_bar.png")

    pd.DataFrame(cox_shap_vals, columns=features).to_csv(
        RESULTS_DIR / "shap_values_cox.csv", index=False)
    print(f"  SHAP values saved: shap_values_cox.csv")

    # ── Fit DeepSurv on full data ─────────────────────────────────────────────
    print("\n=== Fitting DeepSurv (full data) for SHAP ===")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    sc_ds  = StandardScaler()
    X_ds_s = sc_ds.fit_transform(X_raw)

    n_val = max(1, int(0.1 * len(X_ds_s)))
    rng   = np.random.default_rng(RANDOM_STATE)
    val_idx = rng.choice(len(X_ds_s), n_val, replace=False)
    val_mask = np.zeros(len(X_ds_s), dtype=bool)
    val_mask[val_idx] = True

    ds_model, _, _ = train_deepsurv(
        X_ds_s[~val_mask], y_surv.iloc[~val_mask],
        X_ds_s[val_mask],  y_surv.iloc[val_mask],
        device,
    )
    ds_model.eval()
    print("  DeepSurv fitted.")

    print("\n=== Computing DeepSurv SHAP values (GradientExplainer) ===")
    ds_shap_vals = compute_deepsurv_shap(ds_model, X_ds_s, device)
    print(f"  DeepSurv SHAP shape: {ds_shap_vals.shape}")

    _beeswarm(ds_shap_vals, X_ds_s, features,
              "DeepSurv SHAP — top features (beeswarm)",
              FIGURES_DIR / "shap_deepsurv_beeswarm.png")
    _bar_chart(ds_shap_vals, features,
               "DeepSurv — Mean |SHAP value| per feature",
               FIGURES_DIR / "shap_deepsurv_bar.png")

    pd.DataFrame(ds_shap_vals, columns=features).to_csv(
        RESULTS_DIR / "shap_values_deepsurv.csv", index=False)
    print(f"  SHAP values saved: shap_values_deepsurv.csv")

    # ── Comparison ────────────────────────────────────────────────────────────
    print("\n=== Model comparison plot ===")
    plot_model_comparison(cox_shap_vals, ds_shap_vals, features,
                          FIGURES_DIR / "shap_model_comparison.png")

    # Summary table: rank features by mean importance across both models
    cox_imp = np.abs(cox_shap_vals).mean(axis=0)
    ds_imp  = np.abs(ds_shap_vals).mean(axis=0)
    summary = pd.DataFrame({
        "feature":          features,
        "cox_mean_abs_shap": cox_imp,
        "ds_mean_abs_shap":  ds_imp,
        "avg_importance":   (cox_imp + ds_imp) / 2,
    }).sort_values("avg_importance", ascending=False).reset_index(drop=True)
    summary.to_csv(RESULTS_DIR / "shap_importance_summary.csv", index=False)
    print(f"  Importance summary saved: shap_importance_summary.csv")
    print(f"\n  Top 10 features by average SHAP importance:")
    print(summary.head(10)[["feature", "cox_mean_abs_shap", "ds_mean_abs_shap"]].to_string(index=False))

    return {
        "cox_shap": cox_shap_vals,
        "ds_shap":  ds_shap_vals,
        "features": features,
        "summary":  summary,
    }


if __name__ == "__main__":
    results = run_shap_pipeline()
    print("\nSHAP analysis complete.")
