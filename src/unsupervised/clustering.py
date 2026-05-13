"""
Phase 2B: Molecular subtype discovery via clustering in PCA space.

Two complementary methods:
  1. K-means  — scans k=2..30, selects optimal k by silhouette + elbow
  2. Leiden   — graph-based, resolution-free cluster count; more sensitive
                to local density differences between cancer types

Outputs (all written to results/figures/ and results/):
  - kmeans_selection.png      silhouette + inertia curves to guide k choice
  - umap_kmeans.png           UMAP coloured by k-means cluster labels
  - umap_leiden.png           UMAP coloured by Leiden cluster labels
  - cluster_cancer_type.png   stacked bar: cancer-type composition per cluster
  - cluster_survival.png      Kaplan-Meier curves per cluster (log-rank p-value)
  - cluster_top_genes.png     heatmap of mean expression per cluster (top genes)
  - cluster_labels.csv        sample -> kmeans_label, leiden_label

Usage:
    python -m src.unsupervised.clustering
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib as _mpl

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors

import igraph as ig
import leidenalg

FIGURES_DIR = Path("results/figures")
RESULTS_DIR = Path("results")
N_PCA       = 50
RANDOM_STATE = 42


# ── helpers ────────────────────────────────────────────────────────────────

def _get_pca_embedding(X: np.ndarray, n_components: int = N_PCA) -> np.ndarray:
    print(f"  StandardScaler + PCA({n_components}) ...")
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=n_components, random_state=RANDOM_STATE)
    return pca.fit_transform(X_scaled)


def _load_umap_coords(samples: pd.Index) -> np.ndarray | None:
    path = RESULTS_DIR / "umap_coords.csv"
    if path.exists():
        df = pd.read_csv(path, index_col=0)
        df = df.reindex(samples)
        return df[["UMAP1", "UMAP2"]].values
    return None


# ── K-means ────────────────────────────────────────────────────────────────

def _kmeans_scan(X_pca: np.ndarray, k_range: range) -> tuple[list, list, list]:
    inertias, sil_scores, labels_list = [], [], []
    for k in k_range:
        print(f"    k={k} ...", end=" ", flush=True)
        km = MiniBatchKMeans(
            n_clusters=k, random_state=RANDOM_STATE,
            n_init=10, batch_size=2048,
        )
        labels = km.fit_predict(X_pca)
        inertias.append(km.inertia_)
        # silhouette on a random subsample (expensive on full dataset)
        idx = np.random.default_rng(RANDOM_STATE).choice(len(X_pca), min(3000, len(X_pca)), replace=False)
        sil = silhouette_score(X_pca[idx], labels[idx])
        sil_scores.append(sil)
        labels_list.append(labels)
        print(f"sil={sil:.3f}")
    return inertias, sil_scores, labels_list


def _plot_kmeans_selection(k_range: range, inertias: list, sil_scores: list,
                            best_k: int, out_path: Path) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(list(k_range), inertias, marker="o", markersize=4)
    ax1.axvline(best_k, color="red", linestyle="--", label=f"Selected k={best_k}")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia (within-cluster SSE)")
    ax1.set_title("K-means Elbow"); ax1.legend()

    ax2.plot(list(k_range), sil_scores, marker="o", markersize=4, color="darkorange")
    ax2.axvline(best_k, color="red", linestyle="--", label=f"Selected k={best_k}")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette score")
    ax2.set_title("Silhouette vs k"); ax2.legend()

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── Leiden ─────────────────────────────────────────────────────────────────

def _leiden_clustering(X_pca: np.ndarray, n_neighbors: int = 15,
                        resolution: float = 0.5) -> np.ndarray:
    print(f"  Building {n_neighbors}-NN graph ...")
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, metric="euclidean", n_jobs=-1)
    nn.fit(X_pca)
    distances, indices = nn.kneighbors(X_pca)

    # Build igraph from KNN edges
    edges, weights = [], []
    n = len(X_pca)
    for i in range(n):
        for j_idx, d in zip(indices[i, 1:], distances[i, 1:]):
            edges.append((i, int(j_idx)))
            weights.append(float(1.0 / (d + 1e-9)))

    g = ig.Graph(n=n, edges=edges, directed=False)
    g.es["weight"] = weights
    g = g.simplify(combine_edges="mean")

    print(f"  Leiden clustering (resolution={resolution}) ...")
    partition = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights="weight",
        resolution_parameter=resolution,
        seed=RANDOM_STATE,
    )
    labels = np.array(partition.membership)
    n_clusters = labels.max() + 1
    print(f"  Leiden found {n_clusters} clusters")
    return labels


# ── Visualisation helpers ──────────────────────────────────────────────────

def _discrete_palette(n: int) -> list:
    cmap = _mpl.colormaps["tab20" if n <= 20 else "hsv"]
    return [cmap(i / max(n - 1, 1)) for i in range(n)]


def _plot_umap_clusters(umap_coords: np.ndarray, labels: np.ndarray,
                         title: str, out_path: Path) -> None:
    unique = sorted(set(labels))
    colours = _discrete_palette(len(unique))
    cmap = {c: colours[i] for i, c in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(10, 8))
    for c in unique:
        mask = labels == c
        ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
                   c=[cmap[c]], s=4, alpha=0.6, linewidths=0, label=str(c))
    ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
    ax.set_title(title)
    handles = [mpatches.Patch(color=cmap[c], label=f"C{c}") for c in unique]
    ncol = max(1, len(unique) // 20)
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=7, ncol=ncol, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_cluster_cancer_composition(labels: np.ndarray, cancer_types: pd.Series,
                                      method: str, out_path: Path) -> None:
    df = pd.DataFrame({"cluster": labels, "cancer_type": cancer_types.values})
    ct_counts = df.groupby(["cluster", "cancer_type"]).size().unstack(fill_value=0)
    ct_frac = ct_counts.div(ct_counts.sum(axis=1), axis=0)

    cancer_list = ct_frac.columns.tolist()
    n_ct = len(cancer_list)
    colours = _discrete_palette(n_ct)

    fig, ax = plt.subplots(figsize=(max(12, len(ct_frac) * 0.5), 6))
    bottom = np.zeros(len(ct_frac))
    for i, ct in enumerate(cancer_list):
        vals = ct_frac[ct].values
        ax.bar(ct_frac.index, vals, bottom=bottom, color=colours[i], label=ct, width=0.8)
        bottom += vals

    ax.set_xlabel(f"{method} Cluster")
    ax.set_ylabel("Fraction of samples")
    ax.set_title(f"Cancer type composition per {method} cluster")
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=6,
              ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_survival_km(labels: np.ndarray, clinical: pd.DataFrame,
                       method: str, out_path: Path) -> None:
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.statistics import multivariate_logrank_test
    except ImportError:
        print("  [skip] lifelines not installed — skipping KM plot")
        return

    df = pd.DataFrame({
        "cluster":  labels,
        "OS_time":  clinical["OS_time"].values,
        "OS_event": clinical["OS_event"].values,
    })
    df = df.dropna(subset=["OS_time", "OS_event"])
    if len(df) < 20:
        print("  [skip] Too few samples with survival data for KM plot")
        return

    unique_clusters = sorted(df["cluster"].unique())
    n = len(unique_clusters)
    colours = _discrete_palette(n)

    fig, ax = plt.subplots(figsize=(10, 6))
    kmf = KaplanMeierFitter()
    for i, c in enumerate(unique_clusters):
        sub = df[df["cluster"] == c]
        kmf.fit(sub["OS_time"], sub["OS_event"], label=f"C{c} (n={len(sub)})")
        kmf.plot_survival_function(ax=ax, color=colours[i], ci_show=False, linewidth=1.2)

    # Log-rank test
    try:
        result = multivariate_logrank_test(df["OS_time"], df["cluster"], df["OS_event"])
        p = result.p_value
        ax.set_title(f"KM Survival — {method} clusters  (log-rank p={p:.2e})")
    except Exception:
        ax.set_title(f"KM Survival — {method} clusters")

    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=7, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_top_gene_heatmap(labels: np.ndarray, expression: pd.DataFrame,
                            method: str, out_path: Path, n_genes: int = 50) -> None:
    df = expression.copy()
    df["cluster"] = labels

    # Mean expression per cluster
    cluster_means = df.groupby("cluster").mean()

    # Top genes: highest variance across cluster means
    gene_var = cluster_means.var(axis=0)
    top_genes = gene_var.nlargest(n_genes).index
    heatmap_data = cluster_means[top_genes]

    # Normalise each gene 0..1 for display
    heatmap_norm = (heatmap_data - heatmap_data.min()) / (heatmap_data.max() - heatmap_data.min() + 1e-9)

    fig, ax = plt.subplots(figsize=(max(16, n_genes * 0.28), max(6, len(cluster_means) * 0.4)))
    im = ax.imshow(heatmap_norm.values, aspect="auto", cmap="RdYlBu_r", vmin=0, vmax=1)
    ax.set_xticks(range(n_genes))
    # Gene labels are now HGNC symbols; rotate 45 degrees for readability
    ax.set_xticklabels(top_genes.tolist(), rotation=45, ha="right", fontsize=6)
    ax.set_yticks(range(len(cluster_means)))
    ax.set_yticklabels([f"C{c}" for c in cluster_means.index], fontsize=9)
    ax.set_title(f"Mean expression of top {n_genes} cluster-discriminating genes ({method})")
    plt.colorbar(im, ax=ax, label="Normalised mean expression", shrink=0.6)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main pipeline ──────────────────────────────────────────────────────────

def run_clustering_pipeline(
    dataset=None,
    k_range: range = range(2, 26),
    leiden_resolution: float = 0.4,
    leiden_neighbors: int = 15,
    figures_dir: Path = FIGURES_DIR,
) -> pd.DataFrame:
    """
    Full clustering pipeline: K-means + Leiden, with characterisation plots.

    Returns DataFrame indexed by sample barcode with columns:
        kmeans_label, leiden_label
    """
    if dataset is None:
        from src.preprocess import load_multiomics
        dataset = load_multiomics()

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X = dataset.expression.values
    samples = dataset.expression.index

    print("\n=== PCA embedding ===")
    X_pca = _get_pca_embedding(X, N_PCA)

    umap_coords = _load_umap_coords(samples)

    # ── K-means ──
    print(f"\n=== K-means scan k={k_range.start}..{k_range.stop - 1} ===")
    inertias, sil_scores, labels_list = _kmeans_scan(X_pca, k_range)

    best_idx = int(np.argmax(sil_scores))
    best_k   = list(k_range)[best_idx]
    km_labels = labels_list[best_idx]
    print(f"\n  Best k={best_k}  (silhouette={sil_scores[best_idx]:.3f})")

    _plot_kmeans_selection(k_range, inertias, sil_scores, best_k,
                           figures_dir / "kmeans_selection.png")

    if umap_coords is not None:
        _plot_umap_clusters(umap_coords, km_labels,
                            f"UMAP — K-means (k={best_k})",
                            figures_dir / "umap_kmeans.png")

    _plot_cluster_cancer_composition(km_labels, dataset.cancer_types,
                                     f"K-means (k={best_k})",
                                     figures_dir / "cluster_cancer_type_kmeans.png")
    _plot_survival_km(km_labels, dataset.clinical,
                      f"K-means k={best_k}",
                      figures_dir / "cluster_survival_kmeans.png")
    _plot_top_gene_heatmap(km_labels, dataset.expression,
                           f"K-means k={best_k}",
                           figures_dir / "cluster_genes_kmeans.png")

    # ── Leiden ──
    print(f"\n=== Leiden clustering ===")
    leiden_labels = _leiden_clustering(X_pca, n_neighbors=leiden_neighbors,
                                        resolution=leiden_resolution)

    if umap_coords is not None:
        _plot_umap_clusters(umap_coords, leiden_labels,
                            f"UMAP — Leiden (resolution={leiden_resolution})",
                            figures_dir / "umap_leiden.png")

    _plot_cluster_cancer_composition(leiden_labels, dataset.cancer_types,
                                     f"Leiden (res={leiden_resolution})",
                                     figures_dir / "cluster_cancer_type_leiden.png")
    _plot_survival_km(leiden_labels, dataset.clinical,
                      f"Leiden res={leiden_resolution}",
                      figures_dir / "cluster_survival_leiden.png")
    _plot_top_gene_heatmap(leiden_labels, dataset.expression,
                           f"Leiden res={leiden_resolution}",
                           figures_dir / "cluster_genes_leiden.png")

    # ── Save labels ──
    out_df = pd.DataFrame({
        "kmeans_label":  km_labels,
        "leiden_label":  leiden_labels,
        "cancer_type":   dataset.cancer_types.values,
        "OS_time":       dataset.clinical["OS_time"].values,
        "OS_event":      dataset.clinical["OS_event"].values,
    }, index=samples)

    out_path = RESULTS_DIR / "cluster_labels.csv"
    out_df.to_csv(out_path)
    print(f"\n  Cluster labels saved to {out_path}")
    return out_df


if __name__ == "__main__":
    labels = run_clustering_pipeline()
    print("\nK-means distribution:")
    print(labels["kmeans_label"].value_counts().sort_index().to_string())
    print("\nLeiden distribution:")
    print(labels["leiden_label"].value_counts().sort_index().to_string())
