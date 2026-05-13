"""
Phase 3A: Cox proportional hazards survival model.

Pipeline:
  1. Build multi-omics feature matrix for survival-annotated samples
       - Expression: PCA(50) fitted on full cohort
       - Mutations:  top-K most frequent binary genes
       - CNV:        PCA(30) fitted on full cohort
  2. ElasticNet feature pre-selection (sklearn) to find survival-associated features
  3. Cox Ridge (lifelines CoxPHFitter, L2 penalizer) with 5-fold CV
  4. Evaluate: C-index per fold + mean ± std
  5. Refit on full survival cohort; plot hazard ratios (top 30 features)
  6. Kaplan-Meier: split cohort into high/low predicted risk and compare survival

Outputs (results/figures/):
  cox_cv_cindex.png        C-index per fold + mean
  cox_hazard_ratios.png    Forest plot of top 30 hazard ratios
  cox_risk_km.png          KM curves: predicted high vs. low risk
  cox_feature_importance.csv  All features with log(HR) and p-value

Usage:
    python -m src.supervised.cox_models
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import StratifiedKFold

from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

FIGURES_DIR = Path("results/figures")
RESULTS_DIR = Path("results")
RANDOM_STATE = 42

N_EXPR_PCA  = 50
N_CNV_PCA   = 30
N_TOP_MUTS  = 100   # top-frequency mutation genes to include
N_CV_FOLDS  = 5


# ── feature engineering ────────────────────────────────────────────────────

def build_feature_matrix(dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Construct a (samples x features) matrix for all samples and for the
    survival-annotated subset.

    Returns:
        X_all   : features for all 7902 samples (for PCA fitting)
        X_surv  : features for samples with OS_time + OS_event
        y_surv  : DataFrame with columns OS_time, OS_event
    """
    clin = dataset.clinical
    surv_mask = clin["OS_time"].notna() & clin["OS_event"].notna() & (clin["OS_time"] > 0)
    surv_idx  = clin.index[surv_mask]

    print(f"  Survival-annotated samples: {len(surv_idx)}")

    # --- Expression PCA (fit on all, transform survival subset) ---
    print(f"  Expression PCA({N_EXPR_PCA}) ...")
    expr_scaler = StandardScaler()
    expr_scaled = expr_scaler.fit_transform(dataset.expression.values)
    expr_pca    = PCA(n_components=N_EXPR_PCA, random_state=RANDOM_STATE)
    expr_pca.fit(expr_scaled)

    expr_all  = pd.DataFrame(
        expr_pca.transform(expr_scaled),
        index=dataset.expression.index,
        columns=[f"EXPR_PC{i+1}" for i in range(N_EXPR_PCA)],
    )

    # --- Mutation features (top-frequency binary genes) ---
    print(f"  Mutation features (top {N_TOP_MUTS} genes) ...")
    mut_freq  = dataset.mutations.mean(axis=0)
    top_muts  = mut_freq.nlargest(N_TOP_MUTS).index
    mut_all   = dataset.mutations[top_muts].copy()
    mut_all.columns = [f"MUT_{g}" for g in top_muts]

    # --- CNV PCA (fit on all) ---
    print(f"  CNV PCA({N_CNV_PCA}) ...")
    cnv_scaler = StandardScaler()
    cnv_scaled = cnv_scaler.fit_transform(dataset.cnv.values)
    cnv_pca    = PCA(n_components=N_CNV_PCA, random_state=RANDOM_STATE)
    cnv_pca.fit(cnv_scaled)

    cnv_all = pd.DataFrame(
        cnv_pca.transform(cnv_scaled),
        index=dataset.cnv.index,
        columns=[f"CNV_PC{i+1}" for i in range(N_CNV_PCA)],
    )

    # --- Concatenate ---
    X_all  = pd.concat([expr_all, mut_all, cnv_all], axis=1)
    X_surv = X_all.loc[surv_idx]
    y_surv = clin.loc[surv_idx, ["OS_time", "OS_event"]].copy()

    print(f"  Feature matrix: {X_surv.shape}  (samples x features)")
    return X_surv, y_surv


def elasticnet_select(X: pd.DataFrame, y: pd.DataFrame,
                       max_features: int = 80) -> list[str]:
    """
    Use ElasticNetCV on OS_time (ignoring censoring) to select the most
    survival-associated features. Returns selected column names.
    """
    print(f"  ElasticNet feature selection (target: ~{max_features} features) ...")
    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X.values)

    en = ElasticNetCV(
        l1_ratio=[0.5, 0.7, 0.9, 1.0],
        cv=5,
        max_iter=5000,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    en.fit(X_s, y["OS_time"].values)

    coef_mask = en.coef_ != 0
    selected  = X.columns[coef_mask].tolist()

    # If too few selected, fall back to top-|coef| features
    if len(selected) < 10:
        order    = np.argsort(np.abs(en.coef_))[::-1]
        selected = X.columns[order[:max_features]].tolist()
    elif len(selected) > max_features:
        order    = np.argsort(np.abs(en.coef_[coef_mask]))[::-1]
        selected = [selected[i] for i in order[:max_features]]

    print(f"  Selected {len(selected)} features")
    return selected


# ── Cox cross-validation ────────────────────────────────────────────────────

def cox_cross_validate(
    X: pd.DataFrame,
    y: pd.DataFrame,
    features: list[str],
    penalizer: float = 0.1,
    n_folds: int = N_CV_FOLDS,
) -> tuple[list[float], CoxPHFitter]:
    """
    5-fold CV of CoxPHFitter; returns per-fold C-indices and a model
    refit on the full dataset.
    """
    X_sel = X[features].copy()
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    # stratify by event to balance folds
    strat = y["OS_event"].values.astype(int)

    cindex_scores = []
    for fold, (train_idx, test_idx) in enumerate(skf.split(X_sel, strat)):
        X_tr  = X_sel.iloc[train_idx]
        X_te  = X_sel.iloc[test_idx]
        y_tr  = y.iloc[train_idx]
        y_te  = y.iloc[test_idx]

        # Standardise inside fold
        sc    = StandardScaler()
        X_tr_s = pd.DataFrame(sc.fit_transform(X_tr), columns=features)
        X_te_s = pd.DataFrame(sc.transform(X_te),     columns=features)

        df_tr = pd.concat([X_tr_s.reset_index(drop=True),
                           y_tr.reset_index(drop=True)], axis=1)

        cpf = CoxPHFitter(penalizer=penalizer)
        try:
            cpf.fit(df_tr, duration_col="OS_time", event_col="OS_event")
            risk = cpf.predict_partial_hazard(X_te_s)
            ci   = concordance_index(y_te["OS_time"], -risk, y_te["OS_event"])
        except Exception as e:
            print(f"    Fold {fold+1} error: {e}")
            ci = 0.5
        cindex_scores.append(ci)
        print(f"    Fold {fold+1}/{n_folds}  C-index={ci:.3f}")

    # Refit on full data
    sc_full = StandardScaler()
    X_full_s = pd.DataFrame(sc_full.fit_transform(X_sel), columns=features)
    df_full  = pd.concat([X_full_s.reset_index(drop=True),
                          y.reset_index(drop=True)], axis=1)
    cox_full = CoxPHFitter(penalizer=penalizer)
    cox_full.fit(df_full, duration_col="OS_time", event_col="OS_event")

    return cindex_scores, cox_full


# ── plots ──────────────────────────────────────────────────────────────────

def plot_cv_cindex(scores: list[float], out_path: Path) -> None:
    mean_ci = np.mean(scores)
    std_ci  = np.std(scores)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(scores) + 1), scores, color="steelblue", alpha=0.8)
    ax.axhline(mean_ci, color="red", linestyle="--",
               label=f"Mean C-index = {mean_ci:.3f} ± {std_ci:.3f}")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="Random (0.5)")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("C-index")
    ax.set_title("Cox Ridge — 5-fold CV C-index (multi-omics input)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_hazard_ratios(cox_model: CoxPHFitter, top_n: int = 30,
                        out_path: Path = None) -> pd.DataFrame:
    summary = cox_model.summary[["coef", "exp(coef)", "p"]].copy()
    summary.columns = ["log_HR", "HR", "p_value"]
    summary = summary.sort_values("log_HR", key=abs, ascending=False).head(top_n)

    # Shorten feature names for display
    labels = [
        n.replace("EXPR_PC", "ExprPC")
         .replace("CNV_PC", "CNVPC")
         .replace("MUT_", "")
        for n in summary.index
    ]

    colours = ["#d62728" if v > 0 else "#1f77b4" for v in summary["log_HR"]]

    fig, ax = plt.subplots(figsize=(8, max(6, top_n * 0.28)))
    y_pos = range(len(summary))
    ax.barh(list(y_pos), summary["log_HR"].values, color=colours, alpha=0.8)
    ax.set_yticks(list(y_pos))
    ax.set_yticklabels(labels, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("log(Hazard Ratio)  [red = worse prognosis, blue = protective]")
    ax.set_title(f"Cox Ridge — Top {top_n} features by |log HR|")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")
    return cox_model.summary


def plot_risk_km(cox_model: CoxPHFitter, X: pd.DataFrame,
                  y: pd.DataFrame, features: list[str],
                  out_path: Path) -> None:
    sc = StandardScaler()
    X_s = pd.DataFrame(sc.fit_transform(X[features]), columns=features)

    risk_scores = cox_model.predict_partial_hazard(X_s).values
    median_risk = np.median(risk_scores)

    high_mask = risk_scores >= median_risk
    low_mask  = ~high_mask

    fig, ax = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()

    for mask, label, colour in [
        (high_mask, f"High risk  (n={high_mask.sum()})", "#d62728"),
        (low_mask,  f"Low risk   (n={low_mask.sum()})",  "#1f77b4"),
    ]:
        kmf.fit(y["OS_time"].values[mask], y["OS_event"].values[mask], label=label)
        kmf.plot_survival_function(ax=ax, color=colour, linewidth=2)

    # Log-rank p-value
    lr = logrank_test(
        y["OS_time"].values[high_mask],  y["OS_time"].values[low_mask],
        y["OS_event"].values[high_mask], y["OS_event"].values[low_mask],
    )
    ax.set_title(f"Cox Ridge — Predicted high vs. low risk\n(log-rank p={lr.p_value:.2e})")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ───────────────────────────────────────────────────────────────────

def run_cox_pipeline(dataset=None, penalizer: float = 0.1) -> dict:
    if dataset is None:
        from src.preprocess import load_multiomics
        dataset = load_multiomics()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("\n=== Building feature matrix ===")
    X_surv, y_surv = build_feature_matrix(dataset)

    print("\n=== ElasticNet feature selection ===")
    features = elasticnet_select(X_surv, y_surv)

    print(f"\n=== Cox Ridge CV (penalizer={penalizer}) ===")
    cv_scores, cox_model = cox_cross_validate(X_surv, y_surv, features, penalizer)

    mean_ci = np.mean(cv_scores)
    print(f"\n  Mean C-index: {mean_ci:.3f} ± {np.std(cv_scores):.3f}")

    print("\n=== Generating plots ===")
    plot_cv_cindex(cv_scores, FIGURES_DIR / "cox_cv_cindex.png")
    feature_summary = plot_hazard_ratios(cox_model, out_path=FIGURES_DIR / "cox_hazard_ratios.png")
    plot_risk_km(cox_model, X_surv, y_surv, features, FIGURES_DIR / "cox_risk_km.png")

    feature_summary.to_csv(RESULTS_DIR / "cox_feature_importance.csv")
    print(f"  Feature importance saved to results/cox_feature_importance.csv")

    return {"cv_scores": cv_scores, "mean_cindex": mean_ci, "model": cox_model,
            "features": features, "X_surv": X_surv, "y_surv": y_surv}


if __name__ == "__main__":
    results = run_cox_pipeline()
    print(f"\nCox pipeline complete. Mean C-index = {results['mean_cindex']:.3f}")
