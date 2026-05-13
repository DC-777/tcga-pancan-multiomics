"""
Phase 2A: Dimensionality reduction and UMAP visualization.

Pipeline:
  1. StandardScaler on expression matrix
  2. PCA to n_pca components (captures major axes of variance)
  3. UMAP on PCA embedding -> 2D coordinates
  4. Generate diagnostic plots coloured by:
       - Cancer type (33 TCGA labels)
       - OS status (alive / deceased)
       - OS time quartile
       - Age at diagnosis

Usage:
    python -m src.unsupervised.dim_reduction
    # or from another script:
    from src.unsupervised.dim_reduction import run_umap_pipeline
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend; safe on headless / Windows
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import umap

FIGURES_DIR = Path("results/figures")
RESULTS_DIR = Path("results")


def _pca_then_umap(
    X: np.ndarray,
    n_pca: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Scales X, runs PCA, then UMAP.
    Returns (pca_embedding, umap_embedding) as (N, n_pca) and (N, 2) arrays.
    """
    print("  StandardScaler ...")
    X_scaled = StandardScaler().fit_transform(X)

    print(f"  PCA to {n_pca} components ...")
    pca = PCA(n_components=n_pca, random_state=random_state)
    X_pca = pca.fit_transform(X_scaled)
    var_explained = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"  PCA variance explained ({n_pca} PCs): {var_explained:.1%}")

    print(f"  UMAP (n_neighbors={n_neighbors}, min_dist={min_dist}) ...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=2,
        random_state=random_state,
        verbose=False,
    )
    X_umap = reducer.fit_transform(X_pca)
    return X_pca, X_umap


def _plot_cancer_type(
    umap_coords: np.ndarray,
    cancer_types: pd.Series,
    out_path: Path,
) -> None:
    labels = cancer_types.values
    unique = sorted(set(labels[~pd.isnull(labels)]))
    cmap = plt.colormaps["tab20"].resampled(max(len(unique), 1))
    colour_map = {ct: cmap(i / max(len(unique) - 1, 1)) for i, ct in enumerate(unique)}

    fig, ax = plt.subplots(figsize=(14, 10))
    for ct in unique:
        mask = labels == ct
        ax.scatter(
            umap_coords[mask, 0], umap_coords[mask, 1],
            c=[colour_map[ct]], s=4, alpha=0.6, linewidths=0, label=ct,
        )
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of TCGA Pan-Cancer Expression\n(coloured by cancer type)")
    # Legend outside
    handles = [mpatches.Patch(color=colour_map[ct], label=ct) for ct in unique]
    ax.legend(handles=handles, bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=6, ncol=2, frameon=False)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_survival(
    umap_coords: np.ndarray,
    clinical: pd.DataFrame,
    out_path: Path,
) -> None:
    os_event = clinical["OS_event"].values
    os_time  = clinical["OS_time"].values

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Left: vital status
    colours = np.where(os_event == 1, "#d62728", "#aec7e8")
    valid = ~pd.isnull(os_event)
    axes[0].scatter(umap_coords[valid, 0], umap_coords[valid, 1],
                    c=colours[valid], s=4, alpha=0.6, linewidths=0)
    axes[0].scatter([], [], c="#d62728", label="Deceased", s=20)
    axes[0].scatter([], [], c="#aec7e8", label="Alive", s=20)
    axes[0].legend(fontsize=9, frameon=False)
    axes[0].set_title("Vital status")
    axes[0].set_xlabel("UMAP 1"); axes[0].set_ylabel("UMAP 2")

    # Right: OS time quartile (continuous colormap)
    valid_t = ~pd.isnull(os_time)
    sc = axes[1].scatter(
        umap_coords[valid_t, 0], umap_coords[valid_t, 1],
        c=os_time[valid_t], cmap="viridis_r", s=4, alpha=0.6,
        linewidths=0, vmin=0, vmax=np.nanpercentile(os_time, 95),
    )
    plt.colorbar(sc, ax=axes[1], label="OS time (days)")
    axes[1].set_title("Overall survival time")
    axes[1].set_xlabel("UMAP 1"); axes[1].set_ylabel("UMAP 2")

    fig.suptitle("UMAP of TCGA Pan-Cancer Expression — Survival", y=1.01)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def _plot_pca_variance(pca_embedding: np.ndarray, X: np.ndarray, out_path: Path) -> None:
    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)
    pca = PCA(n_components=min(100, X.shape[1], X.shape[0]))
    pca.fit(X_s)
    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(range(1, len(cumvar) + 1), cumvar, marker=".", markersize=4)
    ax.axhline(80, color="red", linestyle="--", linewidth=1, label="80% threshold")
    ax.axhline(90, color="orange", linestyle="--", linewidth=1, label="90% threshold")
    ax.set_xlabel("Number of PCs")
    ax.set_ylabel("Cumulative variance explained (%)")
    ax.set_title("PCA Scree — Expression (top 5000 variable genes)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def run_umap_pipeline(
    dataset=None,
    n_pca: int = 50,
    n_neighbors: int = 30,
    min_dist: float = 0.3,
    figures_dir: Path = FIGURES_DIR,
    save_coords: bool = True,
) -> pd.DataFrame:
    """
    Full PCA -> UMAP pipeline on expression data.

    Args:
        dataset      : MultiomicsDataset; loaded if None
        n_pca        : PCA components before UMAP
        n_neighbors  : UMAP n_neighbors
        min_dist     : UMAP min_dist
        figures_dir  : where to write PNG files
        save_coords  : if True, saves umap_coords.csv to results/

    Returns:
        DataFrame with columns [UMAP1, UMAP2, cancer_type, OS_event, OS_time]
        indexed by sample barcode.
    """
    if dataset is None:
        from src.preprocess import load_multiomics
        dataset = load_multiomics()

    figures_dir = Path(figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    X = dataset.expression.values
    samples = dataset.expression.index

    print("\n--- PCA scree plot ---")
    _plot_pca_variance(None, X, figures_dir / "pca_scree.png")

    print("\n--- PCA + UMAP ---")
    pca_coords, umap_coords = _pca_then_umap(X, n_pca=n_pca,
                                              n_neighbors=n_neighbors,
                                              min_dist=min_dist)

    coords_df = pd.DataFrame(
        umap_coords, index=samples, columns=["UMAP1", "UMAP2"]
    )
    coords_df["cancer_type"] = dataset.cancer_types.values
    coords_df["OS_event"]    = dataset.clinical["OS_event"].values
    coords_df["OS_time"]     = dataset.clinical["OS_time"].values

    print("\n--- Generating figures ---")
    _plot_cancer_type(umap_coords, dataset.cancer_types, figures_dir / "umap_cancer_type.png")
    _plot_survival(umap_coords, dataset.clinical, figures_dir / "umap_survival.png")

    if save_coords:
        out = Path(RESULTS_DIR) / "umap_coords.csv"
        coords_df.to_csv(out)
        print(f"  UMAP coordinates saved to {out}")

    return coords_df


if __name__ == "__main__":
    coords = run_umap_pipeline()
    print("\nUMAP done. Figures in results/figures/")
    print(coords.head())
