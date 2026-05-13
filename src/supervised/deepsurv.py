"""
Phase 3B: DeepSurv — deep neural network with Cox partial log-likelihood loss.

Architecture:
  Input  -> BN -> [Linear -> BN -> ReLU -> Dropout] x L -> Linear(1) -> risk score

Pipeline:
  1. Same multi-omics feature matrix as Cox (Expression PCA(50) + top-100 mutations + CNV PCA(30))
  2. ElasticNet feature pre-selection (same 80-feature budget, for fair comparison)
  3. 5-fold CV: train DeepSurv per fold, evaluate C-index on held-out fold
  4. Refit on full survival cohort
  5. Plots: training loss curve, CV C-index bars, KM high vs. low risk

Outputs (results/figures/):
  deepsurv_loss_curve.png      Training + val loss per epoch (last fold shown)
  deepsurv_cv_cindex.png       C-index per fold + mean
  deepsurv_risk_km.png         KM curves: predicted high vs. low risk
  deepsurv_feature_importance.csv  Gradient-based feature importance scores

Usage:
    python -m src.supervised.deepsurv
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

FIGURES_DIR  = Path("results/figures")
RESULTS_DIR  = Path("results")
RANDOM_STATE = 42

# ── hyper-parameters ────────────────────────────────────────────────────────
HIDDEN_DIMS  = [256, 128, 64]
DROPOUT      = 0.4
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 150
BATCH_SIZE   = 128
PATIENCE     = 20          # early-stopping patience (validation loss)
N_CV_FOLDS   = 5


# ── dataset ─────────────────────────────────────────────────────────────────

class SurvivalDataset(Dataset):
    def __init__(self, X: np.ndarray, times: np.ndarray, events: np.ndarray):
        self.X      = torch.FloatTensor(X)
        self.times  = torch.FloatTensor(times)
        self.events = torch.FloatTensor(events)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.times[idx], self.events[idx]


# ── model ────────────────────────────────────────────────────────────────────

class DeepSurv(nn.Module):
    def __init__(self, in_features: int,
                 hidden_dims: list[int] = HIDDEN_DIMS,
                 dropout: float = DROPOUT):
        super().__init__()
        layers = [nn.BatchNorm1d(in_features)]
        prev = in_features
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Cox partial log-likelihood loss ─────────────────────────────────────────

def cox_partial_loglikelihood(risk: torch.Tensor,
                               times: torch.Tensor,
                               events: torch.Tensor) -> torch.Tensor:
    """
    Breslow approximation of Cox partial log-likelihood.
    Negative value returned (minimise).
    """
    # Sort by descending time
    order = torch.argsort(times, descending=True)
    risk   = risk[order]
    events = events[order]

    # log-sum-exp of risks in risk set
    log_risk      = torch.logcumsumexp(risk, dim=0)
    uncensored    = events == 1
    partial_ll    = (risk[uncensored] - log_risk[uncensored]).sum()
    n_events      = uncensored.sum().clamp(min=1)
    return -partial_ll / n_events


# ── training ─────────────────────────────────────────────────────────────────

def _train_epoch(model, loader, optimiser, device):
    model.train()
    total_loss = 0.0
    for X_b, t_b, e_b in loader:
        X_b, t_b, e_b = X_b.to(device), t_b.to(device), e_b.to(device)
        optimiser.zero_grad()
        risk = model(X_b)
        loss = cox_partial_loglikelihood(risk, t_b, e_b)
        loss.backward()
        optimiser.step()
        total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


def _eval_loss(model, loader, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_b, t_b, e_b in loader:
            X_b, t_b, e_b = X_b.to(device), t_b.to(device), e_b.to(device)
            risk = model(X_b)
            loss = cox_partial_loglikelihood(risk, t_b, e_b)
            total_loss += loss.item() * len(X_b)
    return total_loss / len(loader.dataset)


def _predict_risk(model, X: np.ndarray, device) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        t = torch.FloatTensor(X).to(device)
        return model(t).cpu().numpy()


def train_deepsurv(
    X_tr: np.ndarray, y_tr: pd.DataFrame,
    X_val: np.ndarray, y_val: pd.DataFrame,
    device,
    hidden_dims: list[int] = HIDDEN_DIMS,
    dropout: float = DROPOUT,
    lr: float = LR,
    weight_decay: float = WEIGHT_DECAY,
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    patience: int = PATIENCE,
) -> tuple[DeepSurv, list[float], list[float]]:
    """Train one fold; returns (model, train_losses, val_losses)."""
    ds_tr  = SurvivalDataset(X_tr, y_tr["OS_time"].values, y_tr["OS_event"].values)
    ds_val = SurvivalDataset(X_val, y_val["OS_time"].values, y_val["OS_event"].values)
    dl_tr  = DataLoader(ds_tr,  batch_size=batch_size, shuffle=True,  drop_last=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False)

    model = DeepSurv(in_features=X_tr.shape[1], hidden_dims=hidden_dims, dropout=dropout).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)

    train_losses, val_losses = [], []
    best_val  = float("inf")
    best_state = None
    wait = 0

    for epoch in range(epochs):
        tl = _train_epoch(model, dl_tr, opt, device)
        vl = _eval_loss(model, dl_val, device)
        sched.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, train_losses, val_losses


# ── cross-validation ─────────────────────────────────────────────────────────

def deepsurv_cross_validate(
    X: np.ndarray,
    y: pd.DataFrame,
    device,
    n_folds: int = N_CV_FOLDS,
) -> tuple[list[float], list[float], list[float], DeepSurv]:
    """
    5-fold CV; returns (cv_cindex, last_train_losses, last_val_losses, model_on_full).
    """
    skf   = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    strat = y["OS_event"].values.astype(int)

    cv_scores     = []
    last_tr_loss  = []
    last_val_loss = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, strat)):
        X_tr_raw, X_te_raw = X[tr_idx], X[te_idx]
        y_tr, y_te         = y.iloc[tr_idx], y.iloc[te_idx]

        sc = StandardScaler()
        X_tr_s = sc.fit_transform(X_tr_raw)
        X_te_s = sc.transform(X_te_raw)

        # Use 10 % of training fold as internal validation for early stopping
        n_val = max(1, int(0.1 * len(X_tr_s)))
        val_mask = np.zeros(len(X_tr_s), dtype=bool)
        rng = np.random.default_rng(RANDOM_STATE + fold)
        val_idx = rng.choice(len(X_tr_s), n_val, replace=False)
        val_mask[val_idx] = True

        X_inner_tr = X_tr_s[~val_mask]
        y_inner_tr = y_tr.iloc[~val_mask]
        X_inner_val = X_tr_s[val_mask]
        y_inner_val = y_tr.iloc[val_mask]

        model, tr_losses, val_losses = train_deepsurv(
            X_inner_tr, y_inner_tr,
            X_inner_val, y_inner_val,
            device,
        )

        risk = _predict_risk(model, X_te_s, device)
        ci   = concordance_index(y_te["OS_time"], -risk, y_te["OS_event"])
        cv_scores.append(ci)

        if fold == n_folds - 1:
            last_tr_loss  = tr_losses
            last_val_loss = val_losses

        print(f"    Fold {fold+1}/{n_folds}  C-index={ci:.3f}  epochs={len(tr_losses)}")

    # Refit on full data
    sc_full = StandardScaler()
    X_full_s = sc_full.fit_transform(X)
    n_val_full = max(1, int(0.1 * len(X_full_s)))
    rng_full = np.random.default_rng(RANDOM_STATE)
    val_idx_full = rng_full.choice(len(X_full_s), n_val_full, replace=False)
    val_mask_full = np.zeros(len(X_full_s), dtype=bool)
    val_mask_full[val_idx_full] = True

    model_full, _, _ = train_deepsurv(
        X_full_s[~val_mask_full], y.iloc[~val_mask_full],
        X_full_s[val_mask_full],  y.iloc[val_mask_full],
        device,
    )
    # store scaler inside model for later inference
    model_full._scaler = sc_full

    return cv_scores, last_tr_loss, last_val_loss, model_full, sc_full


# ── gradient-based feature importance ────────────────────────────────────────

def compute_gradient_importance(model: DeepSurv, X: np.ndarray,
                                 feature_names: list[str], device) -> pd.DataFrame:
    model.eval()
    t = torch.FloatTensor(X).to(device)
    t.requires_grad_(True)
    risk = model(t)
    risk.sum().backward()
    grads = t.grad.detach().cpu().numpy()
    importance = np.abs(grads).mean(axis=0)
    return pd.DataFrame({"feature": feature_names, "importance": importance}).sort_values(
        "importance", ascending=False
    ).reset_index(drop=True)


# ── plots ────────────────────────────────────────────────────────────────────

def plot_loss_curve(tr_losses: list[float], val_losses: list[float],
                    out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(tr_losses,  label="Train loss", linewidth=1.5)
    ax.plot(val_losses, label="Val loss",   linewidth=1.5, linestyle="--")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cox partial neg-LL")
    ax.set_title("DeepSurv — Training & Validation Loss (last CV fold)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_cv_cindex(scores: list[float], out_path: Path, label: str = "DeepSurv") -> None:
    mean_ci = np.mean(scores)
    std_ci  = np.std(scores)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(range(1, len(scores) + 1), scores, color="darkorange", alpha=0.8)
    ax.axhline(mean_ci, color="red", linestyle="--",
               label=f"Mean C-index = {mean_ci:.3f} ± {std_ci:.3f}")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1, label="Random (0.5)")
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("CV Fold")
    ax.set_ylabel("C-index")
    ax.set_title(f"{label} — 5-fold CV C-index (multi-omics input)")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_risk_km(model: DeepSurv, X: np.ndarray, y: pd.DataFrame,
                  sc: StandardScaler, out_path: Path) -> None:
    X_s = sc.transform(X)
    risk_scores = _predict_risk(model, X_s, next(model.parameters()).device)
    median_risk = np.median(risk_scores)
    high_mask   = risk_scores >= median_risk
    low_mask    = ~high_mask

    fig, ax = plt.subplots(figsize=(9, 6))
    kmf = KaplanMeierFitter()
    for mask, label, colour in [
        (high_mask, f"High risk  (n={high_mask.sum()})", "#d62728"),
        (low_mask,  f"Low risk   (n={low_mask.sum()})",  "#1f77b4"),
    ]:
        kmf.fit(y["OS_time"].values[mask], y["OS_event"].values[mask], label=label)
        kmf.plot_survival_function(ax=ax, color=colour, linewidth=2)

    lr = logrank_test(
        y["OS_time"].values[high_mask],  y["OS_time"].values[low_mask],
        y["OS_event"].values[high_mask], y["OS_event"].values[low_mask],
    )
    ax.set_title(f"DeepSurv — Predicted high vs. low risk\n(log-rank p={lr.p_value:.2e})")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── comparison bar chart ────────────────────────────────────────────────────

def plot_model_comparison(cox_scores: list[float], ds_scores: list[float],
                           out_path: Path) -> None:
    labels = [f"Fold {i+1}" for i in range(len(cox_scores))]
    x      = np.arange(len(labels))
    width  = 0.35

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(x - width/2, cox_scores, width, label=f"Cox Ridge  (mean={np.mean(cox_scores):.3f})",
           color="steelblue", alpha=0.85)
    ax.bar(x + width/2, ds_scores,  width, label=f"DeepSurv   (mean={np.mean(ds_scores):.3f})",
           color="darkorange", alpha=0.85)
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0.4, 1.0)
    ax.set_ylabel("C-index")
    ax.set_title("Cox Ridge vs. DeepSurv — 5-fold CV C-index")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main ─────────────────────────────────────────────────────────────────────

def run_deepsurv_pipeline(dataset=None, cox_cv_scores: list[float] | None = None) -> dict:
    if dataset is None:
        from src.preprocess import load_multiomics
        dataset = load_multiomics()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  Device: {device}")

    # Reuse the same feature matrix builder as Cox
    from src.supervised.cox_models import build_feature_matrix, elasticnet_select
    print("\n=== Building feature matrix ===")
    X_surv, y_surv = build_feature_matrix(dataset)

    print("\n=== ElasticNet feature selection ===")
    features = elasticnet_select(X_surv, y_surv)
    X_np = X_surv[features].values

    print(f"\n=== DeepSurv 5-fold CV  [device={device}] ===")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)

    cv_scores, tr_losses, val_losses, model_full, sc_full = deepsurv_cross_validate(
        X_np, y_surv, device,
    )
    mean_ci = np.mean(cv_scores)
    print(f"\n  Mean C-index: {mean_ci:.3f} ± {np.std(cv_scores):.3f}")

    print("\n=== Generating plots ===")
    plot_loss_curve(tr_losses, val_losses, FIGURES_DIR / "deepsurv_loss_curve.png")
    plot_cv_cindex(cv_scores, FIGURES_DIR / "deepsurv_cv_cindex.png")
    plot_risk_km(model_full, X_np, y_surv, sc_full, FIGURES_DIR / "deepsurv_risk_km.png")

    if cox_cv_scores is not None:
        plot_model_comparison(cox_cv_scores, cv_scores,
                              FIGURES_DIR / "model_comparison_cindex.png")

    # Gradient importance
    X_scaled = sc_full.transform(X_np)
    imp_df = compute_gradient_importance(model_full, X_scaled, features, device)
    imp_path = RESULTS_DIR / "deepsurv_feature_importance.csv"
    imp_df.to_csv(imp_path, index=False)
    print(f"  Feature importance saved to {imp_path.name}")

    return {
        "cv_scores": cv_scores,
        "mean_cindex": mean_ci,
        "model": model_full,
        "features": features,
        "X_surv": X_surv,
        "y_surv": y_surv,
    }


if __name__ == "__main__":
    results = run_deepsurv_pipeline()
    print(f"\nDeepSurv pipeline complete. Mean C-index = {results['mean_cindex']:.3f}")
