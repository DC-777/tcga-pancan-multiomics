"""
Phase 4: MOFA+ (Multi-Omics Factor Analysis) via mofapy2.

MOFA+ discovers latent factors that jointly explain variance across multiple
omics views. Each factor captures a coordinated signal (e.g. a cancer-type
program, a mutational signature) that may span expression, mutations, and CNV.

Pipeline:
  1. Prepare three views for all 7,902 samples:
       - Expression: top-2000 variable genes (log2 TPM, StandardScaled)
       - Mutations:  top-500 most-frequently-mutated genes (binary)
       - CNV:        top-2000 most-variable genes (StandardScaled)
  2. Train MOFA+ with 20 factors
  3. Extract:
       - Factor scores per sample (Z matrix)
       - Feature weights per view per factor (W matrices)
       - Variance explained per factor per view
  4. Plots:
       - mofa_variance_explained.png   stacked bar: R² per view per factor
       - mofa_factor_umap.png          UMAP of samples, coloured by top factors
       - mofa_factor_cancer_type.png   violin of Factor1 per cancer type
       - mofa_factor_survival.png      KM curves binned by top factor scores
       - mofa_top_weights_factor*.png  Top feature weights (one panel per factor)

Outputs:
  results/mofa_factors.csv            sample x factor scores
  results/mofa_variance.csv           factor x view variance explained
  results/mofa_weights_expression.csv factor x gene weights (expression view)
  results/mofa_weights_mutations.csv
  results/mofa_weights_cnv.csv
  results/figures/mofa_*.png

Usage:
    python -m src.unsupervised.mofa_analysis
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from sklearn.preprocessing import StandardScaler

FIGURES_DIR  = Path("results/figures")
RESULTS_DIR  = Path("results")
RANDOM_STATE = 42

N_FACTORS      = 20
N_EXPR_GENES   = 1000  # increased — full 7,902-sample cohort supports richer views
N_MUT_GENES    = 300
N_CNV_GENES    = 1000
N_ITER         = 400
CONVERGENCE_TOL = 1e-6


# ── data preparation ──────────────────────────────────────────────────────────

def _dedup_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop duplicate column names (can arise after HGNC symbol mapping)."""
    return df.loc[:, ~df.columns.duplicated()]


def prepare_views(
    dataset,
    modalities: list[str] | None = None,
) -> dict[str, np.ndarray]:
    """
    Returns dict of view_name -> (mat, feat_names) where mat is (samples x features).
    Views are already aligned to dataset.samples.

    Parameters
    ----------
    modalities : list of str, optional
        Subset of {"expression", "mutations", "cnv"} to include.
        ``None`` (default) includes all three — backward-compatible.
    """
    if modalities is None:
        modalities = ["expression", "mutations", "cnv"]

    all_views: dict[str, tuple] = {}

    if "expression" in modalities:
        print("  Preparing expression view ...")
        expr = _dedup_columns(dataset.expression)
        gene_var  = expr.var(axis=0)
        top_expr  = gene_var.nlargest(N_EXPR_GENES).index
        expr_sub  = expr[top_expr]
        expr_mat  = StandardScaler().fit_transform(expr_sub.values).astype(np.float32)
        all_views["expression"] = (expr_mat, list(top_expr))

    if "mutations" in modalities:
        print("  Preparing mutation view ...")
        mut = _dedup_columns(dataset.mutations)
        mut_freq = mut.mean(axis=0)
        top_muts = mut_freq.nlargest(N_MUT_GENES).index
        mut_mat  = mut[top_muts].values.astype(np.float32)
        all_views["mutations"] = (mut_mat, list(top_muts))

    if "cnv" in modalities:
        print("  Preparing CNV view ...")
        cnv = _dedup_columns(dataset.cnv)
        cnv_var  = cnv.var(axis=0)
        top_cnv  = cnv_var.nlargest(N_CNV_GENES).index
        cnv_sub  = cnv[top_cnv]
        cnv_mat  = StandardScaler().fit_transform(cnv_sub.values).astype(np.float32)
        all_views["cnv"] = (cnv_mat, list(top_cnv))

    shapes = {v: all_views[v][0].shape for v in all_views}
    print(f"  View shapes: {shapes}")
    return all_views


# ── MOFA+ training ────────────────────────────────────────────────────────────

def train_mofa(views: dict, samples: pd.Index, n_factors: int = N_FACTORS) -> object:
    """
    Train MOFA+ using mofapy2 matrix API (faster than long-format DF).
    data structure: data[group_idx][view_idx] = (samples x features) array
    Returns the trained model object.
    """
    from mofapy2.run.entry_point import entry_point

    view_names  = list(views.keys())
    _likelihood_map = {"expression": "gaussian", "mutations": "bernoulli", "cnv": "gaussian"}
    likelihoods = [_likelihood_map[v] for v in view_names]

    # mofapy2 requires globally unique feature names across all views
    # prefix with view abbreviation to guarantee uniqueness
    _prefix = {"expression": "E", "mutations": "M", "cnv": "C"}

    # mofapy2 matrix API: data[view][group] = (samples x features) array
    data_matrix  = [[views[v][0]] for v in view_names]
    feat_names   = [
        [f"{_prefix.get(v, v[:2])}_{f}" for f in views[v][1]]
        for v in view_names
    ]
    sample_names = [list(samples)]   # 1 group

    print(f"  Building MOFA+ model ...")
    ent = entry_point()
    ent.set_data_options(scale_groups=False, scale_views=False)
    ent.set_data_matrix(
        data_matrix,
        likelihoods=likelihoods,
        views_names=view_names,
        groups_names=["group1"],
        samples_names=sample_names,
        features_names=feat_names,
    )
    ent.set_model_options(
        factors=n_factors,
        spikeslab_weights=True,
        ard_factors=True,
        ard_weights=True,
    )
    ent.set_train_options(
        iter=N_ITER,
        convergence_mode="fast",
        startELBO=1,
        freqELBO=5,
        dropR2=0.001,
        seed=RANDOM_STATE,
        verbose=False,
    )
    print(f"  Training MOFA+ ({n_factors} factors, up to {N_ITER} iterations) ...")
    ent.build()
    ent.run()
    return ent.model


def extract_results(model, views: dict, samples: pd.Index) -> dict:
    """Extract factors, weights, and variance from trained MOFA+ model."""
    K = model.dim["K"]
    factor_cols = [f"Factor{i+1}" for i in range(K)]
    view_names  = list(views.keys())

    # Factor scores — shape may be (n_samples, K) or dict/list wrapping it
    Z_raw = model.nodes["Z"].getExpectation()
    if isinstance(Z_raw, dict):
        Z = list(Z_raw.values())[0]
    elif isinstance(Z_raw, list):
        Z = Z_raw[0]
    else:
        Z = Z_raw
    factors_df = pd.DataFrame(Z, index=samples, columns=factor_cols)

    # Weights per view — shape (n_features, K)
    # Use original (un-prefixed) feature names for readability
    weights = {}
    W_raw = model.nodes["W"].getExpectation()
    for vi, vname in enumerate(view_names):
        if isinstance(W_raw, dict):
            W = W_raw[vname]
        elif isinstance(W_raw, list):
            W = W_raw[vi]
        else:
            W = W_raw[vi]
        orig_feats = list(views[vname][1])   # unprefixed names
        weights[vname] = pd.DataFrame(W, index=orig_feats, columns=factor_cols)

    # Variance explained per factor per view
    r2 = model.calculate_variance_explained()
    var_rows = {}
    for vname in view_names:
        v = r2[vname] if isinstance(r2, dict) else r2
        if isinstance(v, list):
            v = v[0]
        var_rows[vname] = np.array(v).flatten()[:K]

    var_df = pd.DataFrame(var_rows, index=factor_cols)

    return {"factors": factors_df, "weights": weights, "variance": var_df}


# ── fallback: manual MOFA-like NMF if mofapy2 unavailable ───────────────────

def _nmf_fallback(views: dict, samples: pd.Index,
                  n_factors: int = N_FACTORS) -> dict:
    """
    Concatenate all views and run sklearn NMF as a lightweight MOFA proxy.
    Used only when mofapy2 is not installed.
    """
    print("  [fallback] mofapy2 not found — using NMF on concatenated views")
    from sklearn.decomposition import NMF

    mats = []
    feat_names_all = []
    view_slices = {}
    start = 0
    for vname, (mat, feat_names) in views.items():
        # Shift to non-negative for NMF
        mat_nn = mat - mat.min()
        mats.append(mat_nn)
        feat_names_all.extend(feat_names)
        view_slices[vname] = (start, start + len(feat_names))
        start += len(feat_names)

    X_cat = np.hstack(mats)
    nmf = NMF(n_components=n_factors, max_iter=500, random_state=RANDOM_STATE)
    H = nmf.fit_transform(X_cat)   # samples x factors
    W = nmf.components_            # factors x features

    factor_cols = [f"Factor{i+1}" for i in range(n_factors)]
    factors_df  = pd.DataFrame(H, index=samples, columns=factor_cols)

    weights = {}
    for vname, (s, e) in view_slices.items():
        feat_names = list(views[vname][1])
        weights[vname] = pd.DataFrame(
            W[:, s:e].T, index=feat_names, columns=factor_cols
        )

    # Variance explained: fraction of variance in each view captured per factor
    var_rows = {}
    for vname, (mat, _) in views.items():
        s, e = view_slices[vname]
        recon = H @ W[:, s:e]
        total_var = mat.var()
        factor_vars = []
        for k in range(n_factors):
            recon_k = H[:, k:k+1] @ W[k:k+1, s:e]
            factor_vars.append(1 - ((mat - mat.min() - recon_k).var() /
                                    (total_var + 1e-9)))
        var_rows[vname] = np.clip(factor_vars, 0, None)

    var_df = pd.DataFrame(var_rows, index=factor_cols)
    return {"factors": factors_df, "weights": weights, "variance": var_df}


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_variance_explained(var_df: pd.DataFrame, out_path: Path) -> None:
    """Stacked bar: variance explained per factor, split by view."""
    view_names  = var_df.columns.tolist()
    factor_names = var_df.index.tolist()
    n_factors   = len(factor_names)

    colours = plt.cm.Set2(np.linspace(0, 1, len(view_names)))
    fig, ax = plt.subplots(figsize=(max(10, n_factors * 0.55), 5))
    bottom  = np.zeros(n_factors)

    for vi, vname in enumerate(view_names):
        vals = var_df[vname].values * 100   # percent
        ax.bar(range(n_factors), vals, bottom=bottom, color=colours[vi],
               label=vname, alpha=0.9)
        bottom += vals

    ax.set_xticks(range(n_factors))
    ax.set_xticklabels(factor_names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Variance explained (%)")
    ax.set_title("MOFA+ — Variance Explained per Factor per View")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_factor_umap(factors_df: pd.DataFrame, samples: pd.Index,
                     n_top: int = 6, out_path: Path = None) -> None:
    """UMAP of samples coloured by top factor scores (if umap_coords.csv exists)."""
    umap_path = RESULTS_DIR / "umap_coords.csv"
    if not umap_path.exists():
        print("  [skip] umap_coords.csv not found — skipping factor UMAP")
        return

    umap = pd.read_csv(umap_path, index_col=0).reindex(samples)
    coords = umap[["UMAP1", "UMAP2"]].values

    # Pick top factors by total variance
    top_factors = factors_df.columns[:n_top]
    nrows = 2
    ncols = (n_top + 1) // 2

    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 5, nrows * 4))
    axes = axes.flatten()

    for i, fac in enumerate(top_factors):
        vals = factors_df[fac].values
        sc = axes[i].scatter(coords[:, 0], coords[:, 1],
                             c=vals, cmap="RdBu_r", s=3, alpha=0.6,
                             linewidths=0, vmin=np.percentile(vals, 5),
                             vmax=np.percentile(vals, 95))
        axes[i].set_title(fac, fontsize=10)
        axes[i].set_xlabel("UMAP 1", fontsize=8)
        axes[i].set_ylabel("UMAP 2", fontsize=8)
        plt.colorbar(sc, ax=axes[i], shrink=0.7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("MOFA+ Factor Scores on UMAP", fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_factor_by_cancer_type(factors_df: pd.DataFrame,
                                cancer_types: pd.Series,
                                factor: str, out_path: Path) -> None:
    """Box plot of a factor across cancer types."""
    df = pd.DataFrame({
        "factor":      factors_df[factor].values,
        "cancer_type": cancer_types.reindex(factors_df.index).values,
    }).dropna()

    ct_order = (df.groupby("cancer_type")["factor"]
                  .median()
                  .sort_values(ascending=False)
                  .index.tolist())

    fig, ax = plt.subplots(figsize=(max(14, len(ct_order) * 0.5), 5))
    data_by_ct = [df[df["cancer_type"] == ct]["factor"].values for ct in ct_order]
    bp = ax.boxplot(data_by_ct, patch_artist=True, notch=False,
                    medianprops=dict(color="red", linewidth=1.5),
                    whiskerprops=dict(linewidth=0.8),
                    flierprops=dict(markersize=2, alpha=0.4))

    cmap = plt.cm.tab20(np.linspace(0, 1, len(ct_order)))
    for patch, colour in zip(bp["boxes"], cmap):
        patch.set_facecolor(colour)
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(ct_order) + 1))
    ax.set_xticklabels(ct_order, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel(f"{factor} score")
    ax.set_title(f"MOFA+ {factor} by Cancer Type")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_factor_survival(factors_df: pd.DataFrame, clinical: pd.DataFrame,
                          factor: str, out_path: Path) -> None:
    """KM curves: high vs. low factor score groups."""
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import logrank_test
    except ImportError:
        print("  [skip] lifelines not available")
        return

    df = pd.DataFrame({
        "factor":   factors_df[factor].reindex(clinical.index).values,
        "OS_time":  clinical["OS_time"].values,
        "OS_event": clinical["OS_event"].values,
    }).dropna()

    if len(df) < 30:
        print(f"  [skip] Insufficient survival data for {factor}")
        return

    median = df["factor"].median()
    high   = df["factor"] >= median
    low    = ~high

    kmf = KaplanMeierFitter()
    fig, ax = plt.subplots(figsize=(9, 6))
    for mask, label, colour in [
        (high, f"High {factor}  (n={high.sum()})", "#d62728"),
        (low,  f"Low {factor}   (n={low.sum()})",  "#1f77b4"),
    ]:
        kmf.fit(df["OS_time"][mask], df["OS_event"][mask], label=label)
        kmf.plot_survival_function(ax=ax, color=colour, linewidth=2)

    lr = logrank_test(
        df["OS_time"][high], df["OS_time"][low],
        df["OS_event"][high], df["OS_event"][low],
    )
    ax.set_title(f"MOFA+ {factor} — High vs. Low score\n(log-rank p={lr.p_value:.2e})")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_top_weights(weights: dict[str, pd.DataFrame], factor: str,
                     out_path: Path, top_n: int = 15) -> None:
    """Horizontal bar chart of top feature weights for one factor, per view."""
    views     = list(weights.keys())
    n_views   = len(views)
    fig, axes = plt.subplots(1, n_views, figsize=(6 * n_views, 5))
    if n_views == 1:
        axes = [axes]

    for ax, vname in zip(axes, views):
        w = weights[vname][factor].sort_values(key=abs, ascending=False).head(top_n)
        colours = ["#d62728" if v > 0 else "#1f77b4" for v in w.values]
        ax.barh(range(len(w)), w.values[::-1], color=colours[::-1], alpha=0.8)
        ax.set_yticks(range(len(w)))
        ax.set_yticklabels(w.index[::-1], fontsize=7)
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{vname}", fontsize=9)
        ax.set_xlabel("Weight")

    fig.suptitle(f"MOFA+ {factor} — Top {top_n} weights per view", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ──────────────────────────────────────────────────────────────────────

def run_mofa_pipeline(
    dataset=None,
    n_factors: int = N_FACTORS,
    modalities: list[str] | None = None,
    out_suffix: str = "",
) -> dict:
    """
    Parameters
    ----------
    modalities : list of str, optional
        Subset of {"expression", "mutations", "cnv"}.
        ``None`` (default) uses all three — backward-compatible.
    out_suffix : str
        Appended to output filenames, e.g. ``"_expr_only"`` produces
        ``mofa_factors_expr_only.csv``.  Default ``""`` preserves original names.
    """
    if dataset is None:
        from src.preprocess import load_multiomics
        dataset = load_multiomics()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    samples = dataset.expression.index

    print("\n=== Preparing MOFA+ views ===")
    views = prepare_views(dataset, modalities=modalities)

    # Try mofapy2; fall back to NMF proxy
    print("\n=== Training MOFA+ ===")
    try:
        import mofapy2
        model = train_mofa(views, samples, n_factors=n_factors)
        results = extract_results(model, views, samples)
        print("  MOFA+ training complete.")
    except ImportError:
        print("  [warning] mofapy2 not installed — using NMF fallback")
        results = _nmf_fallback(views, samples, n_factors=n_factors)

    factors_df = results["factors"]
    weights    = results["weights"]
    var_df     = results["variance"]

    print("\n=== Saving results ===")
    factors_df.to_csv(RESULTS_DIR / f"mofa_factors{out_suffix}.csv")
    var_df.to_csv(RESULTS_DIR / f"mofa_variance{out_suffix}.csv")
    for vname, w_df in weights.items():
        w_df.to_csv(RESULTS_DIR / f"mofa_weights_{vname}{out_suffix}.csv")
    print("  CSVs saved.")

    print("\n=== Generating plots ===")
    sfx = out_suffix  # short alias for filenames
    plot_variance_explained(var_df, FIGURES_DIR / f"mofa_variance_explained{sfx}.png")
    plot_factor_umap(factors_df, samples, n_top=6,
                     out_path=FIGURES_DIR / f"mofa_factor_umap{sfx}.png")

    # Top factor by total variance explained
    total_var  = var_df.sum(axis=1)
    top_factor = total_var.idxmax()
    print(f"  Top factor by total variance: {top_factor}")

    plot_factor_by_cancer_type(factors_df, dataset.cancer_types, top_factor,
                               FIGURES_DIR / f"mofa_factor_cancer_type{sfx}.png")
    plot_factor_survival(factors_df, dataset.clinical, top_factor,
                         FIGURES_DIR / f"mofa_factor_survival{sfx}.png")

    # Weight plots for top 5 factors (skip for ablation runs to save time)
    if not out_suffix:
        for fac in total_var.nlargest(5).index:
            plot_top_weights(weights, fac,
                             FIGURES_DIR / f"mofa_top_weights_{fac}.png")

        # Survival KM for each of top 3 factors
        for fac in total_var.nlargest(3).index:
            if fac != top_factor:
                plot_factor_survival(factors_df, dataset.clinical, fac,
                                     FIGURES_DIR / f"mofa_factor_survival_{fac}.png")

    print(f"\n  Variance explained summary (top factors):")
    print(var_df.head(10).to_string())

    return results


if __name__ == "__main__":
    results = run_mofa_pipeline()
    print("\nMOFA+ pipeline complete.")
