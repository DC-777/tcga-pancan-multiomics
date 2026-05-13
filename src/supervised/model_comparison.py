"""
Multi-model survival comparison: FT-Transformer vs. XGBoost vs. LightGBM.

Feature sets
------------
  A (primary)  : MOFA+ factors + cancer-type one-hot + age_at_dx
  B (baseline) : Raw PCA feature matrix (ExprPC50 + top-100 mut + CNVPC30,
                 ElasticNet-selected ~80 features) — same as cox_models.py

Models
------
  FT-Transformer  PyTorch, Feature Tokeniser + self-attention, Breslow Cox loss
  XGBoost         survival:cox objective, TreeSHAP
  LightGBM        cox objective, TreeSHAP

Evaluation
----------
  5-fold StratifiedKFold (stratified by OS_event)
  Metrics: Harrell C-index per fold (mean ± std)
  Risk-group KM curves (tertile split) for winning model

SHAP
----
  XGBoost / LightGBM : shap.TreeExplainer  (exact, fast)
  FT-Transformer      : shap.GradientExplainer (100-sample background)

Outputs  results/figures/
  model_comparison_cindex.png
  shap_ftt_beeswarm.png  shap_ftt_bar.png
  shap_xgb_beeswarm.png  shap_xgb_bar.png
  shap_lgb_beeswarm.png  shap_lgb_bar.png
  shap_three_model_comparison.png
  risk_km_best_model.png

Outputs  results/
  model_cv_scores.csv
  shap_values_ftt.csv
  shap_values_xgb.csv
  shap_values_lgb.csv
  shap_importance_summary_models.csv

Usage
-----
    python -m src.supervised.model_comparison
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

import xgboost as xgb
import lightgbm as lgb
import shap

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from lifelines import KaplanMeierFitter
from lifelines.utils import concordance_index
from lifelines.statistics import logrank_test

FIGURES_DIR  = Path("results/figures")
RESULTS_DIR  = Path("results")
RANDOM_STATE = 42
N_FOLDS      = 5
TOP_N        = 20       # features shown in SHAP plots
DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── helpers ───────────────────────────────────────────────────────────────────

def _feat_label(name: str) -> str:
    return (name.replace("MOFA_Factor", "Factor")
                .replace("EXPR_PC", "ExprPC")
                .replace("CNV_PC", "CNVPC")
                .replace("MUT_", "mut:"))


def _surv_label(y: pd.DataFrame) -> np.ndarray:
    """
    XGBoost / LightGBM survival Cox label convention:
      positive  = event time   (OS_event == 1)
      negative  = censoring time (OS_event == 0)
    """
    return np.where(y["OS_event"] == 1, y["OS_time"], -y["OS_time"]).astype(np.float32)


def _lgb_cox_objective(preds: np.ndarray, dataset: "lgb.Dataset"):
    """
    Custom Cox partial log-likelihood objective for LightGBM 4.x (Breslow approximation).

    LightGBM 4.x removed the built-in 'cox' objective and changed the custom
    objective API: function signature is now (preds, dataset) not (labels, preds),
    and it is passed via params['objective'] = callable rather than fobj=.

    Gradient:  g_k = -d_k + exp(η_k) * Σ_{i ≤ k, d_i=1} 1/S_i
    Hessian:   h_k ≈ exp(η_k) * Σ_{i ≤ k, d_i=1} 1/S_i  (diagonal approx.)

    where k indexes ascending-time sorted samples,
          d_k = event indicator, S_i = Σ_{j ≥ i} exp(η_j) (risk-set denominator).
    """
    labels = dataset.get_label()
    times  = np.abs(labels).astype(np.float64)
    events = (labels > 0).astype(np.float64)

    order     = np.argsort(times)          # ascending by time
    inv_order = np.argsort(order)

    preds_s  = preds[order].astype(np.float64)
    events_s = events[order]

    # Numerically stable: gradient is invariant to an additive constant in preds
    exp_s = np.exp(preds_s - preds_s.max())

    # S_i = Σ_{j ≥ i} exp(η_j) = reverse cumulative sum along ascending-sorted array
    S = np.cumsum(exp_s[::-1])[::-1] + 1e-12

    # weight_k = Σ_{i ≤ k, d_i=1} 1/S_i
    weight = np.cumsum(np.where(events_s == 1, 1.0 / S, 0.0))

    grad_s = -events_s + exp_s * weight
    hess_s = np.maximum(exp_s * weight, 1e-6)

    return grad_s[inv_order].astype(np.float32), hess_s[inv_order].astype(np.float32)


def _lgb_cox_metric(preds: np.ndarray, dataset: "lgb.Dataset"):
    """C-index evaluation metric for LightGBM 4.x custom Cox objective."""
    labels = dataset.get_label()
    times  = np.abs(labels)
    events = (labels > 0).astype(int)
    ci     = concordance_index(times, -preds, events)
    return "c_index", ci, True   # (name, value, higher_is_better)


# ── feature matrix builders ───────────────────────────────────────────────────

def build_mofa_feature_matrix(
    dataset,
    mofa_path: Path = RESULTS_DIR / "mofa_factors.csv",
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Feature Set A: MOFA+ factors + cancer-type one-hot + age.

    Returns
    -------
    X      : DataFrame (survival-annotated samples × features)
    y_surv : DataFrame with OS_time, OS_event columns
    """
    # Load MOFA+ factors
    factors = pd.read_csv(mofa_path, index_col=0)
    factors.columns = [f"MOFA_Factor{i+1}" for i in range(factors.shape[1])]

    # Survival-annotated samples
    clin = dataset.clinical
    has_surv = clin["OS_time"].notna() & clin["OS_event"].notna() & (clin["OS_time"] > 0)
    clin_s = clin[has_surv].copy()

    # Align with MOFA+ factors (MOFA+ ran on all 7,902 samples)
    common = factors.index.intersection(clin_s.index)
    factors_s = factors.loc[common]
    clin_s    = clin_s.loc[common]

    # Cancer type one-hot (drop first to avoid perfect collinearity)
    ct = pd.get_dummies(clin_s["cancer_type"], prefix="CT", drop_first=False)
    ct = ct.astype(np.float32)

    # Age (continuous)
    age = clin_s["age_at_dx"].fillna(clin_s["age_at_dx"].median()).rename("age_at_dx")

    X = pd.concat([factors_s.astype(np.float32), ct, age.astype(np.float32)], axis=1)
    y = clin_s[["OS_time", "OS_event"]].copy()
    y["OS_event"] = y["OS_event"].astype(int)

    print(f"  Feature set A (MOFA+): {X.shape[0]:,} samples × {X.shape[1]} features")
    print(f"    {factors_s.shape[1]} MOFA factors | {ct.shape[1]} cancer-type dummies | 1 age")
    print(f"    Events: {y['OS_event'].sum()} / {len(y)}")
    return X, y


def build_raw_feature_matrix(dataset) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Feature Set B: ElasticNet-selected PCA features (same as cox_models.py).
    Used as a baseline comparison.

    Returns
    -------
    X      : DataFrame
    y_surv : DataFrame with OS_time, OS_event
    """
    from src.supervised.cox_models import build_feature_matrix, elasticnet_select

    X_raw, y_surv = build_feature_matrix(dataset)
    feats = elasticnet_select(X_raw, y_surv)
    X = X_raw[feats]
    print(f"  Feature set B (raw PCA): {X.shape[0]:,} samples × {X.shape[1]} features")
    return X, y_surv


# ── FT-Transformer ────────────────────────────────────────────────────────────

class FeatureTokenizer(nn.Module):
    """
    Each numerical feature j gets its own linear projection:
      token_j = W_j * x_j + b_j   (W_j, b_j ∈ R^d_token)

    Output shape: (batch, n_features, d_token)
    """
    def __init__(self, n_features: int, d_token: int):
        super().__init__()
        self.W = nn.Parameter(torch.empty(n_features, d_token))
        self.b = nn.Parameter(torch.zeros(n_features, d_token))
        nn.init.kaiming_uniform_(self.W, a=np.sqrt(5))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)  →  (B, F, d_token)
        return x.unsqueeze(-1) * self.W.unsqueeze(0) + self.b.unsqueeze(0)


class _PreNormBlock(nn.Module):
    """Pre-LayerNorm transformer block (more stable than post-norm)."""
    def __init__(self, d_token: int, n_heads: int, ffn_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_token)
        self.attn  = nn.MultiheadAttention(d_token, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_token)
        self.ffn   = nn.Sequential(
            nn.Linear(d_token, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_token),
        )
        self.drop  = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(h)
        x = x + self.drop(self.ffn(self.norm2(x)))
        return x


class FTTransformer(nn.Module):
    """
    Feature Tokeniser Transformer for survival (Gorishniy et al. 2021).

    Architecture:
      FeatureTokenizer → prepend CLS → L × PreNormBlock
      → CLS token → LayerNorm → Linear(1)  [log-risk score]
    """
    def __init__(
        self,
        n_features: int,
        d_token:    int  = 64,
        n_heads:    int  = 4,
        n_layers:   int  = 2,
        ffn_factor: int  = 4,
        dropout:    float = 0.2,
    ):
        super().__init__()
        ffn_dim = d_token * ffn_factor
        self.tokenizer  = FeatureTokenizer(n_features, d_token)
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, d_token))
        self.blocks     = nn.ModuleList([
            _PreNormBlock(d_token, n_heads, ffn_dim, dropout)
            for _ in range(n_layers)
        ])
        self.norm_out   = nn.LayerNorm(d_token)
        self.head       = nn.Linear(d_token, 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, F)
        tokens = self.tokenizer(x)                          # (B, F, d_token)
        cls    = self.cls_token.expand(x.size(0), -1, -1)  # (B, 1, d_token)
        tokens = torch.cat([cls, tokens], dim=1)            # (B, F+1, d_token)
        for block in self.blocks:
            tokens = block(tokens)
        cls_out = self.norm_out(tokens[:, 0])               # (B, d_token)
        return self.head(cls_out).squeeze(-1)               # (B,)


class _FTT2D(nn.Module):
    """2-D output wrapper for shap.GradientExplainer."""
    def __init__(self, model: FTTransformer):
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.model(x)
        return out.unsqueeze(1) if out.dim() == 1 else out


# ── Cox loss (Breslow) ────────────────────────────────────────────────────────

def _cox_loss(risk: torch.Tensor, times: torch.Tensor, events: torch.Tensor) -> torch.Tensor:
    """Negative mean Cox partial log-likelihood (Breslow approximation)."""
    order   = torch.argsort(times, descending=True)
    risk    = risk[order]
    events  = events[order]
    log_cum = torch.logcumsumexp(risk, dim=0)
    ll      = risk - log_cum
    n_ev    = events.sum()
    if n_ev == 0:
        return torch.tensor(0.0, requires_grad=True)
    return -(ll * events).sum() / n_ev


# ── FT-Transformer training ───────────────────────────────────────────────────

def train_fttransformer(
    X_tr:   np.ndarray,
    y_tr:   pd.DataFrame,
    X_val:  np.ndarray,
    y_val:  pd.DataFrame,
    device: torch.device,
    n_epochs:   int   = 150,
    batch_size: int   = 256,
    lr:         float = 1e-3,
    patience:   int   = 20,
) -> FTTransformer:
    n_feat = X_tr.shape[1]
    model  = FTTransformer(n_feat).to(device)
    opt    = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=8, factor=0.5)

    X_tr_t  = torch.FloatTensor(X_tr).to(device)
    t_tr_t  = torch.FloatTensor(y_tr["OS_time"].values).to(device)
    e_tr_t  = torch.FloatTensor(y_tr["OS_event"].values).to(device)

    X_val_t = torch.FloatTensor(X_val).to(device)
    t_val_t = torch.FloatTensor(y_val["OS_time"].values).to(device)
    e_val_t = torch.FloatTensor(y_val["OS_event"].values).to(device)

    # DataLoader for mini-batch training
    ds      = TensorDataset(X_tr_t, t_tr_t, e_tr_t)
    loader  = DataLoader(ds, batch_size=batch_size, shuffle=True)

    best_val_loss = float("inf")
    best_state    = None
    no_improve    = 0

    for epoch in range(n_epochs):
        model.train()
        for xb, tb, eb in loader:
            opt.zero_grad()
            risk = model(xb)
            loss = _cox_loss(risk, tb, eb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

        # Validation loss
        model.eval()
        with torch.no_grad():
            val_risk = model(X_val_t)
            val_loss = _cox_loss(val_risk, t_val_t, e_val_t)

        sched.step(val_loss)
        if val_loss.item() < best_val_loss - 1e-5:
            best_val_loss = val_loss.item()
            best_state    = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve    = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


# ── cross-validation ──────────────────────────────────────────────────────────

def cross_validate_models(
    X:      pd.DataFrame,
    y:      pd.DataFrame,
    device: torch.device,
    label:  str = "MOFA+",
) -> dict:
    """
    5-fold CV for FT-Transformer, XGBoost, and LightGBM.
    Returns dict with per-fold C-indices and fitted full-data models.
    """
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    X_np = X.values.astype(np.float32)
    strat = y["OS_event"].values

    cv = {
        "ftt": [], "xgb": [], "lgb": [],
        "models_ftt": [], "models_xgb": [], "models_lgb": [],
        "scalers": [],
    }

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_np, strat)):
        print(f"\n  Fold {fold+1}/{N_FOLDS} —", end=" ")
        X_tr_raw, X_te_raw = X_np[tr_idx], X_np[te_idx]
        y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]

        sc = StandardScaler()
        X_tr = sc.fit_transform(X_tr_raw)
        X_te = sc.transform(X_te_raw)
        cv["scalers"].append(sc)

        t_te = y_te["OS_time"].values
        e_te = y_te["OS_event"].values

        # ── FT-Transformer ──────────────────────────────────────────────────
        n_val = max(1, int(0.1 * len(X_tr)))
        rng   = np.random.default_rng(RANDOM_STATE + fold)
        vi    = rng.choice(len(X_tr), n_val, replace=False)
        vm    = np.zeros(len(X_tr), dtype=bool); vm[vi] = True

        torch.manual_seed(RANDOM_STATE + fold)
        ftt = train_fttransformer(
            X_tr[~vm], y_tr.iloc[~vm],
            X_tr[vm],  y_tr.iloc[vm],
            device,
        )
        ftt.eval()
        with torch.no_grad():
            ftt_risk = ftt(torch.FloatTensor(X_te).to(device)).cpu().numpy()
        c_ftt = concordance_index(t_te, -ftt_risk, e_te)
        cv["ftt"].append(c_ftt)
        cv["models_ftt"].append(ftt)
        print(f"FTT={c_ftt:.3f}", end=" | ")

        # ── XGBoost ─────────────────────────────────────────────────────────
        xgb_params = dict(
            objective        = "survival:cox",
            eval_metric      = "cox-nloglik",
            max_depth        = 5,
            learning_rate    = 0.05,
            n_estimators     = 600,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_weight = 5,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            random_state     = RANDOM_STATE,
            n_jobs           = -1,
            verbosity        = 0,
        )
        xgb_lbl_tr = _surv_label(y_tr)
        xgb_lbl_te = _surv_label(y_te)
        dtrain  = xgb.DMatrix(X_tr, label=xgb_lbl_tr)
        dval    = xgb.DMatrix(X_tr[vm], label=xgb_lbl_tr[vm])
        dtest   = xgb.DMatrix(X_te, label=xgb_lbl_te)
        bst_xgb = xgb.train(
            {k: v for k, v in xgb_params.items()
             if k not in ("n_estimators", "random_state", "n_jobs")},
            dtrain,
            num_boost_round    = xgb_params["n_estimators"],
            evals              = [(dval, "val")],
            early_stopping_rounds = 30,
            verbose_eval       = False,
        )
        xgb_risk = bst_xgb.predict(dtest)
        c_xgb = concordance_index(t_te, -xgb_risk, e_te)
        cv["xgb"].append(c_xgb)
        cv["models_xgb"].append(bst_xgb)
        print(f"XGB={c_xgb:.3f}", end=" | ")

        # ── LightGBM (custom Cox objective) ──────────────────────────────────
        # LightGBM 4.x dropped the built-in 'cox' objective; we use a custom
        # fobj (gradient + hessian from Breslow approximation) and feval (C-index).
        lgb_params = dict(
            objective        = _lgb_cox_objective,   # callable via params (LGB 4.x API)
            num_leaves       = 31,
            learning_rate    = 0.05,
            n_estimators     = 600,
            subsample        = 0.8,
            colsample_bytree = 0.8,
            min_child_samples= 10,
            reg_alpha        = 0.1,
            reg_lambda       = 1.0,
            verbose          = -1,
        )
        lgb_lbl     = _surv_label(y_tr)
        lgb_lbl_val = _surv_label(y_tr.iloc[vm])
        dtrain_lgb  = lgb.Dataset(X_tr,      label=lgb_lbl)
        dval_lgb    = lgb.Dataset(X_tr[vm],  label=lgb_lbl_val, reference=dtrain_lgb)
        bst_lgb = lgb.train(
            lgb_params,
            dtrain_lgb,
            num_boost_round = lgb_params["n_estimators"],
            feval           = _lgb_cox_metric,
            valid_sets      = [dval_lgb],
            callbacks       = [lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(period=-1)],
        )
        lgb_risk = bst_lgb.predict(X_te, raw_score=True)
        c_lgb = concordance_index(t_te, -lgb_risk, e_te)
        cv["lgb"].append(c_lgb)
        cv["models_lgb"].append(bst_lgb)
        print(f"LGB={c_lgb:.3f}")

    for key in ("ftt", "xgb", "lgb"):
        arr = cv[key]
        print(f"  {key.upper():3s}  mean={np.mean(arr):.3f}  std={np.std(arr):.3f}  "
              f"folds={[f'{v:.3f}' for v in arr]}")

    return cv


# ── SHAP analysis ─────────────────────────────────────────────────────────────

def compute_shap_values(
    cv_results: dict,
    X:          pd.DataFrame,
    y:          pd.DataFrame,
    device:     torch.device,
) -> dict:
    """
    Compute SHAP values using full-data models (last fold's models, refitted on full data).
    Returns dict: model_key -> np.ndarray (n_samples, n_features)
    """
    feature_names = list(X.columns)
    X_np  = X.values.astype(np.float32)

    # Refit scaler on full data
    sc    = StandardScaler()
    X_sc  = sc.fit_transform(X_np)
    shap_out = {}

    # ── FT-Transformer SHAP (GradientExplainer) ──────────────────────────────
    print("\n  Computing FTT SHAP (GradientExplainer) ...")
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    n_val = max(1, int(0.1 * len(X_sc)))
    rng   = np.random.default_rng(RANDOM_STATE)
    vi    = rng.choice(len(X_sc), n_val, replace=False)
    vm    = np.zeros(len(X_sc), dtype=bool); vm[vi] = True

    ftt_full = train_fttransformer(
        X_sc[~vm], y.iloc[~vm],
        X_sc[vm],  y.iloc[vm],
        device,
    )
    ftt_full.eval()

    bg_idx  = rng.choice(len(X_sc), min(100, len(X_sc)), replace=False)
    bg_t    = torch.FloatTensor(X_sc[bg_idx]).to(device)
    data_t  = torch.FloatTensor(X_sc).to(device)
    wrapper = _FTT2D(ftt_full).to(device)
    wrapper.eval()
    exp_ftt = shap.GradientExplainer(wrapper, bg_t)
    sv_ftt  = exp_ftt.shap_values(data_t)
    if isinstance(sv_ftt, list):
        sv_ftt = sv_ftt[0]
    sv_ftt  = np.array(sv_ftt)
    if sv_ftt.ndim == 3:
        sv_ftt = sv_ftt.squeeze(-1)
    shap_out["ftt"] = sv_ftt
    print(f"    FTT SHAP shape: {sv_ftt.shape}")

    # ── XGBoost SHAP (TreeExplainer) ─────────────────────────────────────────
    print("  Computing XGBoost SHAP (TreeExplainer) ...")
    xgb_params = dict(
        objective="survival:cox", max_depth=5, learning_rate=0.05,
        n_estimators=600, subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, reg_alpha=0.1, reg_lambda=1.0, verbosity=0,
    )
    n_val = max(1, int(0.1 * len(X_sc)))
    vi2   = rng.choice(len(X_sc), n_val, replace=False)
    vm2   = np.zeros(len(X_sc), dtype=bool); vm2[vi2] = True
    lbl_full  = _surv_label(y)
    dtrain_xf = xgb.DMatrix(X_sc, label=lbl_full)
    dval_xf   = xgb.DMatrix(X_sc[vm2], label=lbl_full[vm2])
    bst_xgb_full = xgb.train(
        xgb_params, dtrain_xf,
        num_boost_round=xgb_params["n_estimators"],
        evals=[(dval_xf, "val")], early_stopping_rounds=30, verbose_eval=False,
    )
    exp_xgb  = shap.TreeExplainer(bst_xgb_full)
    sv_xgb   = exp_xgb.shap_values(X_sc)
    if isinstance(sv_xgb, list):
        sv_xgb = sv_xgb[0]
    shap_out["xgb"] = sv_xgb
    print(f"    XGB SHAP shape: {sv_xgb.shape}")

    # ── LightGBM SHAP (TreeExplainer, custom Cox objective) ──────────────────
    print("  Computing LightGBM SHAP (TreeExplainer) ...")
    lgb_params = dict(
        objective=_lgb_cox_objective,
        num_leaves=31, learning_rate=0.05,
        n_estimators=600, subsample=0.8, colsample_bytree=0.8,
        min_child_samples=10, reg_alpha=0.1, reg_lambda=1.0, verbose=-1,
    )
    lbl_lgb_full = _surv_label(y)
    dtrain_lgf   = lgb.Dataset(X_sc, label=lbl_lgb_full)
    dval_lgf     = lgb.Dataset(X_sc[vm2], label=lbl_lgb_full[vm2], reference=dtrain_lgf)
    bst_lgb_full = lgb.train(
        lgb_params, dtrain_lgf,
        num_boost_round=lgb_params["n_estimators"],
        feval=_lgb_cox_metric,
        valid_sets=[dval_lgf],
        callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
    )
    exp_lgb  = shap.TreeExplainer(bst_lgb_full)
    sv_lgb   = exp_lgb.shap_values(X_sc)
    if isinstance(sv_lgb, list):
        sv_lgb = sv_lgb[0]
    shap_out["lgb"] = sv_lgb
    print(f"    LGB SHAP shape: {sv_lgb.shape}")

    # Save SHAP matrices
    for key, sv in shap_out.items():
        out = pd.DataFrame(sv, columns=feature_names)
        out.to_csv(RESULTS_DIR / f"shap_values_{key}.csv", index=False)

    # Importance summary
    rows = []
    for key, sv in shap_out.items():
        imp = np.abs(sv).mean(axis=0)
        for fn, iv in zip(feature_names, imp):
            rows.append({"model": key, "feature": fn, "mean_abs_shap": iv})
    summary = pd.DataFrame(rows)
    summary.to_csv(RESULTS_DIR / "shap_importance_summary_models.csv", index=False)

    shap_out["feature_names"] = feature_names
    shap_out["X_scaled"]      = X_sc
    return shap_out


# ── plots ─────────────────────────────────────────────────────────────────────

def _beeswarm(shap_vals, X_sc, feature_names, title, out_path, top_n=TOP_N):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = list(np.argsort(mean_abs)[::-1][:top_n][::-1])
    labels   = [_feat_label(feature_names[i]) for i in order]
    sv_sub   = shap_vals[:, order]
    xv_sub   = X_sc[:, order]

    rng    = np.random.default_rng(RANDOM_STATE)
    jitter = rng.uniform(-0.3, 0.3, sv_sub.shape)
    cmap   = _mpl.colormaps["coolwarm"]

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.38)))
    for fi in range(top_n):
        xr = xv_sub[:, fi].max() - xv_sub[:, fi].min() + 1e-9
        colour = cmap((xv_sub[:, fi] - xv_sub[:, fi].min()) / xr)
        ax.scatter(sv_sub[:, fi], fi + jitter[:, fi],
                   c=colour, s=5, alpha=0.45, linewidths=0)

    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("SHAP value (impact on log risk)")
    ax.set_title(title)

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(0, 1))
    sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.01)
    cb.set_label("Feature value", fontsize=8)
    cb.set_ticks([0, 1]); cb.set_ticklabels(["Low", "High"])
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def _bar(shap_vals, feature_names, title, out_path, color, top_n=TOP_N):
    mean_abs = np.abs(shap_vals).mean(axis=0)
    order    = list(np.argsort(mean_abs)[::-1][:top_n][::-1])
    labels   = [_feat_label(feature_names[i]) for i in order]
    vals     = mean_abs[order]

    fig, ax = plt.subplots(figsize=(8, max(5, top_n * 0.35)))
    ax.barh(range(top_n), vals, color=color, alpha=0.85)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: {out_path.name}")


def plot_cv_comparison(cv_results: dict, out_path: Path):
    models = {"FT-Transformer": "ftt", "XGBoost": "xgb", "LightGBM": "lgb"}
    colors = ["#4c72b0", "#dd8452", "#55a868"]
    means  = [np.mean(cv_results[k]) for k in models.values()]
    stds   = [np.std(cv_results[k])  for k in models.values()]

    fig, ax = plt.subplots(figsize=(7, 4))
    x = np.arange(len(models))
    bars = ax.bar(x, means, yerr=stds, capsize=6, width=0.5,
                  color=colors, alpha=0.85, edgecolor="white")

    # Per-fold scatter
    for i, key in enumerate(models.values()):
        folds = cv_results[key]
        ax.scatter([i] * len(folds), folds, color="black", s=20, zorder=5, alpha=0.7)

    ax.set_xticks(x)
    ax.set_xticklabels(list(models.keys()), fontsize=11)
    ax.set_ylabel("C-index (5-fold CV)")
    ax.set_title("Survival Model Comparison — MOFA+ Feature Set")
    ax.set_ylim(max(0, min(means) - 0.1), min(1, max(means) + 0.12))
    ax.axhline(0.5, color="grey", lw=0.8, ls="--", alpha=0.5)

    for bar, mean, std in zip(bars, means, stds):
        ax.text(bar.get_x() + bar.get_width() / 2,
                mean + std + 0.005, f"{mean:.3f}",
                ha="center", va="bottom", fontsize=9, fontweight="bold")

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_three_model_shap(shap_dict: dict, out_path: Path, top_n: int = TOP_N):
    """Side-by-side mean |SHAP| for all three models, top features by union."""
    feature_names = shap_dict["feature_names"]
    models = {"FT-Transformer": "ftt", "XGBoost": "xgb", "LightGBM": "lgb"}
    colors = {"ftt": "#4c72b0", "xgb": "#dd8452", "lgb": "#55a868"}

    imps = {k: np.abs(shap_dict[k]).mean(axis=0) for k in models.values()}

    # Union of top features across all models
    union = set()
    for imp in imps.values():
        union |= set(np.argsort(imp)[::-1][:top_n])
    union = sorted(union, key=lambda i: -sum(imps[k][i] for k in models.values()))[:top_n]
    union = union[::-1]  # ascending for barh

    labels = [_feat_label(feature_names[i]) for i in union]
    x      = np.arange(len(union))
    width  = 0.26

    fig, ax = plt.subplots(figsize=(10, max(6, len(union) * 0.42)))
    offsets = [-width, 0, width]
    for (name, key), offset in zip(models.items(), offsets):
        ax.barh(x + offset, imps[key][union], width,
                label=name, color=colors[key], alpha=0.85)

    ax.set_yticks(x)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Feature Importance — FT-Transformer vs. XGBoost vs. LightGBM")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


def plot_risk_km(X: pd.DataFrame, y: pd.DataFrame, cv_results: dict,
                 device: torch.device, out_path: Path):
    """
    KM curves for low / mid / high risk tertiles using the best model
    (highest mean C-index) refitted on full data.
    """
    model_names = {"ftt": "FT-Transformer", "xgb": "XGBoost", "lgb": "LightGBM"}
    best_key    = max(["ftt", "xgb", "lgb"], key=lambda k: np.mean(cv_results[k]))
    print(f"  Risk-KM plot using best model: {model_names[best_key]} "
          f"(C={np.mean(cv_results[best_key]):.3f})")

    X_np = X.values.astype(np.float32)
    sc   = StandardScaler()
    X_sc = sc.fit_transform(X_np)
    times   = y["OS_time"].values
    events  = y["OS_event"].values

    rng  = np.random.default_rng(RANDOM_STATE)
    vi   = rng.choice(len(X_sc), max(1, int(0.1 * len(X_sc))), replace=False)
    vm   = np.zeros(len(X_sc), dtype=bool); vm[vi] = True

    if best_key == "ftt":
        torch.manual_seed(RANDOM_STATE)
        model = train_fttransformer(X_sc[~vm], y.iloc[~vm], X_sc[vm], y.iloc[vm], device)
        model.eval()
        with torch.no_grad():
            risk = model(torch.FloatTensor(X_sc).to(device)).cpu().numpy()

    elif best_key == "xgb":
        lbl = _surv_label(y)
        bst = xgb.train(
            {"objective": "survival:cox", "max_depth": 5, "learning_rate": 0.05,
             "subsample": 0.8, "colsample_bytree": 0.8, "verbosity": 0},
            xgb.DMatrix(X_sc, label=lbl),
            num_boost_round=600,
            evals=[(xgb.DMatrix(X_sc[vm], label=lbl[vm]), "val")],
            early_stopping_rounds=30, verbose_eval=False,
        )
        risk = bst.predict(xgb.DMatrix(X_sc))

    else:  # lgb
        lbl = _surv_label(y)
        bst = lgb.train(
            {"objective": _lgb_cox_objective, "num_leaves": 31,
             "learning_rate": 0.05, "subsample": 0.8, "verbose": -1},
            lgb.Dataset(X_sc, label=lbl),
            num_boost_round=600,
            feval=_lgb_cox_metric,
            valid_sets=[lgb.Dataset(X_sc[vm], label=lbl[vm])],
            callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(period=-1)],
        )
        risk = bst.predict(X_sc, raw_score=True)

    # Split into tertiles
    q33, q67 = np.percentile(risk, [33.3, 66.7])
    group = np.where(risk <= q33, "Low risk", np.where(risk <= q67, "Mid risk", "High risk"))

    colors = {"Low risk": "#2ecc71", "Mid risk": "#f39c12", "High risk": "#e74c3c"}
    fig, ax = plt.subplots(figsize=(8, 5))
    for grp in ["Low risk", "Mid risk", "High risk"]:
        m = group == grp
        kmf = KaplanMeierFitter()
        kmf.fit(times[m], events[m], label=f"{grp} (n={m.sum()})")
        kmf.plot_survival_function(ax=ax, color=colors[grp], ci_show=True, ci_alpha=0.1)

    # Log-rank test: low vs. high
    lr = logrank_test(
        times[group == "Low risk"],  times[group == "High risk"],
        events[group == "Low risk"], events[group == "High risk"],
    )
    ax.set_title(f"Kaplan–Meier: Risk Tertiles ({model_names[best_key]})\n"
                 f"Log-rank p (low vs high) = {lr.p_value:.2e}")
    ax.set_xlabel("Time (days)")
    ax.set_ylabel("Survival probability")
    ax.set_ylim(0, 1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {out_path.name}")


# ── main pipeline ─────────────────────────────────────────────────────────────

def run_model_comparison(dataset=None) -> dict:
    if dataset is None:
        from src.preprocess import load_multiomics
        dataset = load_multiomics()

    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    mofa_path = RESULTS_DIR / "mofa_factors.csv"
    if not mofa_path.exists():
        raise FileNotFoundError(
            f"MOFA+ factors not found at {mofa_path}. "
            "Run src.unsupervised.mofa_analysis first."
        )

    # ── Feature Set A: MOFA+ + clinical ──────────────────────────────────────
    print("\n=== Building Feature Set A (MOFA+ factors + cancer type + age) ===")
    X_mofa, y_surv = build_mofa_feature_matrix(dataset, mofa_path)

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n=== 5-Fold Cross-Validation (Feature Set A) ===")
    cv_results = cross_validate_models(X_mofa, y_surv, DEVICE, label="MOFA+")

    # Save CV scores
    rows = []
    for fold_i in range(N_FOLDS):
        rows.append({
            "fold":          fold_i + 1,
            "ftt_cindex":    cv_results["ftt"][fold_i],
            "xgb_cindex":    cv_results["xgb"][fold_i],
            "lgb_cindex":    cv_results["lgb"][fold_i],
        })
    cv_df = pd.DataFrame(rows)
    cv_df.to_csv(RESULTS_DIR / "model_cv_scores.csv", index=False)
    print(f"\n  CV scores saved: model_cv_scores.csv")

    # ── Comparison plot ───────────────────────────────────────────────────────
    print("\n=== Generating comparison plot ===")
    plot_cv_comparison(cv_results, FIGURES_DIR / "model_comparison_cindex.png")

    # ── Risk-group KM plot ────────────────────────────────────────────────────
    print("\n=== Risk-group Kaplan–Meier plot ===")
    plot_risk_km(X_mofa, y_surv, cv_results, DEVICE,
                 FIGURES_DIR / "risk_km_best_model.png")

    # ── SHAP analysis ─────────────────────────────────────────────────────────
    print("\n=== SHAP Analysis ===")
    feature_names = list(X_mofa.columns)
    shap_results  = compute_shap_values(cv_results, X_mofa, y_surv, DEVICE)

    model_meta = {
        "ftt": ("FT-Transformer SHAP", "#4c72b0"),
        "xgb": ("XGBoost SHAP",        "#dd8452"),
        "lgb": ("LightGBM SHAP",       "#55a868"),
    }
    print("\n  Generating SHAP plots ...")
    sc_full = StandardScaler()
    X_sc    = sc_full.fit_transform(X_mofa.values.astype(np.float32))

    for key, (title, color) in model_meta.items():
        sv = shap_results[key]
        _beeswarm(sv, X_sc, feature_names,
                  f"{title} — top features (beeswarm)",
                  FIGURES_DIR / f"shap_{key}_beeswarm.png")
        _bar(sv, feature_names,
             f"{title} — Mean |SHAP|",
             FIGURES_DIR / f"shap_{key}_bar.png",
             color=color)

    plot_three_model_shap(shap_results, FIGURES_DIR / "shap_three_model_comparison.png")

    # ── Feature Set B baseline (optional) ────────────────────────────────────
    print("\n=== Feature Set B (Raw PCA baseline) ===")
    try:
        X_raw, y_raw = build_raw_feature_matrix(dataset)
        print("  Running XGBoost on raw features for baseline C-index ...")
        skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        c_raw = []
        for tr, te in skf.split(X_raw.values, y_raw["OS_event"].values):
            sc_b  = StandardScaler()
            Xb_tr = sc_b.fit_transform(X_raw.values[tr])
            Xb_te = sc_b.transform(X_raw.values[te])
            y_tr_b, y_te_b = y_raw.iloc[tr], y_raw.iloc[te]
            lbl_tr = _surv_label(y_tr_b)
            bst    = xgb.train(
                {"objective": "survival:cox", "max_depth": 5, "verbosity": 0},
                xgb.DMatrix(Xb_tr, label=lbl_tr),
                num_boost_round=300, verbose_eval=False,
            )
            risk_b = bst.predict(xgb.DMatrix(Xb_te))
            c_raw.append(concordance_index(y_te_b["OS_time"].values,
                                           -risk_b, y_te_b["OS_event"].values))
        print(f"  Raw feature XGBoost  C-index = {np.mean(c_raw):.3f} ± {np.std(c_raw):.3f}")
        cv_df["xgb_raw_cindex"] = c_raw
        cv_df.to_csv(RESULTS_DIR / "model_cv_scores.csv", index=False)
    except Exception as exc:
        print(f"  [Raw baseline skipped: {exc}]")

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("FINAL C-INDEX SUMMARY")
    print("=" * 60)
    header = f"{'Model':<20} {'Mean':>8} {'Std':>8}  Folds"
    print(header)
    print("-" * 60)
    for name, key in [("FT-Transformer", "ftt"), ("XGBoost", "xgb"), ("LightGBM", "lgb")]:
        arr   = cv_results[key]
        folds = "  ".join(f"{v:.3f}" for v in arr)
        print(f"{name:<20} {np.mean(arr):>8.3f} {np.std(arr):>8.3f}  {folds}")
    print("=" * 60)

    return {
        "cv_results":   cv_results,
        "shap_results": shap_results,
        "X_mofa":       X_mofa,
        "y_surv":       y_surv,
    }


if __name__ == "__main__":
    results = run_model_comparison()
    print("\nModel comparison complete.")
