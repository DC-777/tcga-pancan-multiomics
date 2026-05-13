"""
marker_analysis.py
------------------
Comprehensive multiomics marker analysis linking MOFA factors -> SHAP scores -> gene-level
therapeutic targets. Outputs:
  - marker_shap_scores.csv          (gene × model SHAP contribution, ranked)
  - factor_gene_table.csv           (top genes per factor × modality)
  - therapeutic_targets.csv         (ranked targets with mechanism + evidence)
  - factor_biology_summary.txt      (human-readable summary)
  - figures/marker_*.png            (multiple publication-quality figures)
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE      = Path(r"C:\Users\debad\OneDrive\Documents\DC Academic Research\Cancer Research")
RESULTS   = BASE / "results"
FIGURES   = RESULTS / "figures"
FIGURES.mkdir(parents=True, exist_ok=True)
OUT       = RESULTS

# ── 1. Load Data ───────────────────────────────────────────────────────────────
print("Loading MOFA weights and SHAP data …")

shap_df   = pd.read_csv(RESULTS / "shap_importance_summary_models.csv")
var_df    = pd.read_csv(RESULTS / "mofa_variance.csv", index_col=0)
w_expr    = pd.read_csv(RESULTS / "mofa_weights_expression.csv", index_col=0)
w_mut     = pd.read_csv(RESULTS / "mofa_weights_mutations.csv",  index_col=0)
w_cnv     = pd.read_csv(RESULTS / "mofa_weights_cnv.csv",         index_col=0)

print(f"  Expression weights : {w_expr.shape}")
print(f"  Mutation weights   : {w_mut.shape}")
print(f"  CNV weights        : {w_cnv.shape}")

# ── 2. Factor-level SHAP (avg |SHAP| across models) ───────────────────────────
factor_shap = (
    shap_df[shap_df["feature"].str.startswith("MOFA_")]
    .copy()
)
factor_shap["factor"] = factor_shap["feature"].str.replace("MOFA_", "")
factor_pivot = factor_shap.pivot(index="factor", columns="model", values="mean_abs_shap")
factor_pivot["avg_shap"]  = factor_pivot.mean(axis=1)
factor_pivot["shap_cv"]   = factor_pivot[["ftt","xgb","lgb"]].std(axis=1) / factor_pivot["avg_shap"]
factor_pivot = factor_pivot.sort_values("avg_shap", ascending=False)

# Add variance explained
var_mean = var_df.mean(axis=1)
var_mean.index = [f"Factor{i+1}" for i in range(len(var_mean))]
factor_pivot["var_explained"] = var_mean.reindex(factor_pivot.index)

print("\nTop factors by avg |SHAP|:")
print(factor_pivot[["ftt","xgb","lgb","avg_shap","var_explained"]].round(4).to_string())

# ── 3. Top factors to decode (avg_shap > 0.10) ────────────────────────────────
TOP_FACTORS = factor_pivot[factor_pivot["avg_shap"] > 0.10].index.tolist()
N_TOP_GENES = 25   # genes per factor per modality

print(f"\nFactors selected for deep-dive: {TOP_FACTORS}")

# ── 4. Extract top genes per factor × modality ────────────────────────────────
def top_genes(weight_df, factor, n=N_TOP_GENES, prefix=""):
    col = factor
    if col not in weight_df.columns:
        return pd.DataFrame()
    s = weight_df[col].abs().sort_values(ascending=False).head(n)
    df = pd.DataFrame({
        "gene"     : s.index.str.replace(f"^{prefix}", "", regex=True),
        "abs_weight": s.values,
        "raw_weight": weight_df.loc[s.index, col].values,
        "direction": np.where(weight_df.loc[s.index, col].values > 0, "up", "down"),
        "modality" : "expression" if prefix=="" else ("mutation" if prefix=="M_" else "cnv"),
        "factor"   : factor,
    })
    return df

records = []
for f in TOP_FACTORS:
    records.append(top_genes(w_expr, f, prefix=""))
    records.append(top_genes(w_mut,  f, prefix=""))
    records.append(top_genes(w_cnv,  f, prefix=""))

factor_gene_df = pd.concat(records, ignore_index=True)

# Add factor-level SHAP
factor_gene_df["factor_avg_shap"] = factor_gene_df["factor"].map(factor_pivot["avg_shap"])
# Gene-level contribution = |weight| × factor_avg_shap  (proxy for direct gene influence)
factor_gene_df["gene_shap_contrib"] = factor_gene_df["abs_weight"] * factor_gene_df["factor_avg_shap"]

factor_gene_df.to_csv(OUT / "factor_gene_table.csv", index=False)
print(f"\nFactor-gene table: {factor_gene_df.shape[0]} rows -> factor_gene_table.csv")

# ── 5. Cross-modal gene ranking ────────────────────────────────────────────────
# Aggregate across all factors and modalities -> ranked marker list
gene_agg = (
    factor_gene_df
    .groupby(["gene","modality"])
    .agg(
        total_contrib   = ("gene_shap_contrib", "sum"),
        max_contrib     = ("gene_shap_contrib", "max"),
        n_factors       = ("factor",            "count"),
        directions      = ("direction",          lambda x: ",".join(x.unique())),
    )
    .reset_index()
    .sort_values("total_contrib", ascending=False)
)

# ── 6. Build therapeutic target table ─────────────────────────────────────────
# Curated known therapeutic relevance (oncogenes/TSGs/pathways)
KNOWN_ONCOGENES = {
    "TP53":"TSG/gain-of-function",  "KRAS":"Oncogene","BRAF":"Oncogene",
    "PIK3CA":"Oncogene","IDH1":"Oncogene","ATRX":"TSG","APC":"TSG",
    "EGFR":"Oncogene","MYC":"Oncogene","ERBB2":"Oncogene","CDK4":"Oncogene",
    "CDKN2A":"TSG","PTEN":"TSG","RB1":"TSG","VHL":"TSG","BRCA1":"TSG",
    "BRCA2":"TSG","NOTCH1":"Oncogene","AR":"Oncogene","ESR1":"Oncogene",
    "KMT2D":"TSG","KMT2C":"TSG","ARID1A":"TSG","KLK3":"Oncogene",
    "KLK2":"Oncogene",
}

PATHWAY_MAP = {
    # Proliferation / cell cycle
    "KRT5":"Epithelial differentiation","KRT6A":"Epithelial differentiation",
    "KRT14":"Epithelial differentiation","KRT16":"Epithelial differentiation",
    "KRT17":"Epithelial differentiation","KRT13":"Epithelial differentiation",
    "KRT15":"Epithelial differentiation","KRT19":"Epithelial differentiation",
    "TP63":"Squamous differentiation",
    # Immune / immunoglobulin
    "IGHG1":"B-cell/Immunoglobulin","IGHG2":"B-cell/Immunoglobulin",
    "IGHG3":"B-cell/Immunoglobulin","IGHG4":"B-cell/Immunoglobulin",
    "IGHA1":"B-cell/Immunoglobulin","IGLC1":"B-cell/Immunoglobulin",
    "IGLC2":"B-cell/Immunoglobulin","IGLC3":"B-cell/Immunoglobulin",
    "IGKC":"B-cell/Immunoglobulin",
    # Lung surfactant
    "SFTPB":"Lung surfactant","SFTPA2":"Lung surfactant",
    # GI/hepatic
    "ALDOB":"Hepatocyte/GI","HNF4A":"Hepatocyte TF","CDHR5":"Intestinal epithelium",
    "FOXA1":"Luminal/hepatic TF","HNF1B":"Renal/GI TF",
    "TFF3":"Mucin/GI","SPDEF":"Luminal epithelial TF",
    "AGR2":"ER stress / mucin","MUC13":"Mucin/GI",
    # Neural / glioma
    "GFAP":"Astrocyte/Glioma","BCAN":"Glioma ECM","PTPRZ1":"Glioma/neural",
    "MIR9-1HG":"Neural lncRNA",
    # Neuroendocrine
    "CHGA":"Neuroendocrine","CHGB":"Neuroendocrine",
    # Prostate
    "KLK3":"Prostate (PSA)","KLK2":"Prostate kallikrein",
    "AR":"Androgen receptor",
    # Breast / luminal
    "FOXA1":"Luminal breast TF","PPP1R1B":"DARPP-32/breast",
    "CEACAM5":"GI/lung tumor antigen","CEACAM6":"CEA family",
    # DNA damage / repair
    "TP53":"DNA damage response","ATRX":"Chromatin remodeling",
    "KMT2D":"Histone methyltransferase","KMT2C":"Histone methyltransferase",
    "ARID1A":"SWI/SNF chromatin","BRCA1":"DNA repair","BRCA2":"DNA repair",
    # Signaling
    "KRAS":"RAS-MAPK","BRAF":"RAS-MAPK","PIK3CA":"PI3K-AKT",
    "PTEN":"PI3K-AKT","IDH1":"TCA cycle/epigenetics",
    "APC":"WNT pathway","EGFR":"EGFR-RAS",
    # MEK
    "GPX2":"ROS/oxidative stress",
    # Steroid
    "ESR1":"Estrogen receptor",
    # Liver
    "ALB":"Hepatocyte","FGG":"Coagulation",
    "HP":"Acute phase","HMGCS2":"Ketogenesis",
    # Squamous / skin
    "SFN":"Stratifin/14-3-3","SERPINB5":"Maspin TSG",
    "S100P":"Calcium binding/metastasis","S100A7":"Calcium/squamous",
    "S100A14":"Calcium/epithelial",
    # Other
    "MSLN":"Mesothelioma antigen","LCN2":"Inflammatory",
    "MMP1":"ECM remodeling","PIGR":"Mucosal immunity",
}

THERAPEUTIC_MECHANISM = {
    # Knockdown targets (high expression = bad)
    "KRAS":"Small molecule (MRTX849/sotorasib for G12C); pan-RAS degraders in dev.",
    "BRAF":"Vemurafenib/dabrafenib (V600E); type II inhibitors for other mutations.",
    "PIK3CA":"Alpelisib (PIK3CA-mut breast); pan-PI3K inhibitors (idelalisib).",
    "EGFR":"Erlotinib/gefitinib/osimertinib; EGFR-targeted ADC (amivantamab).",
    "IDH1":"Ivosidenib (IDH1-mut AML/glioma/cholangiocarcinoma).",
    "ERBB2":"Trastuzumab/pertuzumab/T-DM1/T-DXd; HER2 bispecific antibodies.",
    "AR":"Enzalutamide/apalutamide/darolutamide; AR degraders (ARV-110).",
    "ESR1":"Fulvestrant/elacestrant; ESR1-mut endocrine-resistant breast.",
    "MYC":"BET bromodomain inhibitors (JQ1/OTX015); Aurora B; indirect via CDK7.",
    "CDK4":"Palbociclib/ribociclib/abemaciclib; CDK4/6 inhibitors.",
    "NOTCH1":"GSI (γ-secretase inhibitors); anti-DLL3/DLL4 antibodies.",
    "KLK3":"Lutetium PSMA radioligand (177Lu-PSMA-617); PSMA CAR-T.",
    "FOXA1":"Pioneer TF; no direct inhibitor; indirect via AR/ESR1 axis.",
    "GFAP":"Glioma marker; no direct therapeutic; guides TMZ/RT; glioCAR.",
    "MSLN":"Anti-mesothelin ADC (anetumab); CAR-T (CART-meso); SS1P immunotoxin.",
    "CEACAM5":"Tusamitamab ravtansine (CEA-ADC); CEA-CD3 bispecific (cibisatamab).",
    "CHGA":"Neuroendocrine target; somatostatin analog (octreotide); PRRT.",
    # Boosting targets (low expression = bad)
    "TP53":"Gene therapy / p53 reactivation (APR-246/eprenetapopt); MDM2 inhibitors.",
    "ARID1A":"SWI/SNF loss -> synthetic lethality with HDAC/EZH2 inhibitors (tazemetostat).",
    "KMT2D":"MLL4 loss -> synthetic lethality with PRMT5 inhibitors (GSK3326595).",
    "PTEN":"Loss -> PI3K dependence; alpelisib in PIK3CA co-mut; AKT inhibitors.",
    "RB1":"Loss -> CDK4/6 independence; E2F inhibitors; Aurora A inhibitors.",
    "APC":"WNT loss -> porcupine inhibitors (WNT974); beta-catenin PROTAC.",
    "ATRX":"ALT mechanism -> PARP inhibitors; ATM/ATR inhibitors in ATRX-null glioma.",
    "BRCA1":"PARP inhibitors (olaparib/niraparib/rucaparib); platinum agents.",
    "BRCA2":"PARP inhibitors; RAD51 inhibitors; POLQ inhibitors.",
    "IDH1":"Ivosidenib; IDH1-mut tumors also sensitive to IDH-mut-PARP synthetic lethality.",
    "VHL":"HIF-2α inhibitors (belzutifan) in VHL-deficient RCC.",
    # Expression targets
    "MMP1":"Marimastat (MMPI); ECM remodeling inhibition; indirect via TME targeting.",
    "S100P":"Pentamidine (S100P inhibitor); RAGE pathway blockade.",
    "LCN2":"Anti-LCN2 antibodies; combination with VEGF inhibition in gastric/CRC.",
    "GPX2":"No direct inhibitor; ROS amplification strategies (auranofin/arsenic trioxide).",
    "PIGR":"pIgR-mediated transcytosis; targeted delivery platform for mucosal tumors.",
    "SFN":"14-3-3σ upregulation in luminal tumors; indirect via cell cycle checkpoint.",
    "SERPINB5":"Maspin is a TSG; viral delivery / methylation reversal strategies.",
    "SFTPB":"Lung lineage marker; no direct therapeutic yet; ADC target potential.",
    "TFF3":"Anti-TFF3 antibodies in GI/breast; TFF3-targeted radiotherapy explored.",
    "PPP1R1B":"DARPP-32 in breast/gastric; CDK5 pathway; no approved inhibitor.",
    "HNF4A":"Nuclear receptor; no direct inhibitor; guides transcriptional reprogramming.",
    "ALDOB":"Metabolic target in HCC; glycolysis inhibition (2-DG); fructose pathway.",
    "BCAN":"Brevican in glioma ECM; anti-brevican immunotherapy; CSPG4 pathway.",
    "PTPRZ1":"Receptor tyrosine phosphatase in glioma; pleiotrophin antagonism; glioCAR.",
    "AGR2":"ER-stress / mucin folding; anti-AGR2 antibodies (P3X-AGR2) in lung/breast.",
    "SPDEF":"Luminal TF; indirect via ETS factor inhibition.",
    "IGHG1":"Tumor-infiltrating B-cells; combination with checkpoint inhibitors.",
    "KRT5":"Squamous lineage; indirect via p63/EGFR axis.",
    "TTN":"Passenger mutator (large gene); hypermutation biomarker; not a direct target.",
}

# Annotate gene_agg
gene_agg["role"]           = gene_agg["gene"].map(KNOWN_ONCOGENES).fillna("Unknown")
gene_agg["pathway"]        = gene_agg["gene"].map(PATHWAY_MAP).fillna("Unknown")
gene_agg["tx_mechanism"]   = gene_agg["gene"].map(THERAPEUTIC_MECHANISM).fillna("Under investigation")
gene_agg["actionable"]     = gene_agg["role"].apply(lambda r: "Yes" if r != "Unknown" else "Partial")

# Save
gene_agg.to_csv(OUT / "marker_shap_scores.csv", index=False)
print(f"\nMarker SHAP scores: {gene_agg.shape[0]} entries -> marker_shap_scores.csv")

# ── 7. Curated therapeutic target table ──────────────────────────────────────
# Take top 40 across all modalities
tgt = gene_agg.sort_values("total_contrib", ascending=False).head(60)
tgt.to_csv(OUT / "therapeutic_targets.csv", index=False)
print(f"Therapeutic target table: {len(tgt)} targets -> therapeutic_targets.csv")

# ── 8. Summary text ───────────────────────────────────────────────────────────
def summarize_factor(f, n=15):
    fp = factor_pivot.loc[f]
    lines = [
        f"\n{'='*70}",
        f"FACTOR {f}  |  avg |SHAP| = {fp['avg_shap']:.4f}  |  var explained = {fp['var_explained']:.3f}",
        f"  FTT={fp['ftt']:.4f}  XGB={fp['xgb']:.4f}  LGB={fp['lgb']:.4f}  (consistency CV={fp['shap_cv']:.3f})",
        f"{'='*70}",
    ]
    for mod, wdf in [("Expression", w_expr), ("Mutation", w_mut), ("CNV", w_cnv)]:
        col = f
        if col not in wdf.columns: continue
        s = wdf[col].abs().sort_values(ascending=False).head(n)
        top_genes_str = ", ".join(
            [f"{g}({'(up)' if wdf.loc[g,col]>0 else '(down)'})"
             for g in s.index[:n]]
        )
        lines.append(f"  [{mod}] top genes: {top_genes_str}")
        known = [g for g in s.index if g in KNOWN_ONCOGENES]
        if known:
            lines.append(f"    -> Druggable/known: {', '.join(known)}")
    return "\n".join(lines)

summary_path = OUT / "factor_biology_summary.txt"
with open(summary_path, "w") as fh:
    fh.write("TCGA PAN-CANCER MULTIOMICS MARKER ANALYSIS\n")
    fh.write("Factor-level biology decoded from MOFA+ weights × SHAP scores\n")
    fh.write(f"Top factors: {TOP_FACTORS}\n")
    for f in TOP_FACTORS:
        fh.write(summarize_factor(f))
    fh.write("\n\n")
    fh.write("="*70 + "\n")
    fh.write("TOP 30 GENE-LEVEL THERAPEUTIC TARGETS\n")
    fh.write("="*70 + "\n")
    fh.write(
        gene_agg[["gene","modality","total_contrib","n_factors","role","pathway","tx_mechanism"]]
        .head(30).to_string(index=False)
    )

print(f"\nSummary written -> {summary_path}")

# ── 9. FIGURE 1: Factor SHAP × variance bubble chart ─────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
for f in factor_pivot.index:
    shap = factor_pivot.loc[f, "avg_shap"]
    var  = factor_pivot.loc[f, "var_explained"] * 100
    cv   = factor_pivot.loc[f, "shap_cv"]
    color = "#e63946" if f in TOP_FACTORS else "#a8dadc"
    size  = 200 * shap / factor_pivot["avg_shap"].max() + 40
    ax.scatter(var, shap, s=size, c=color,
               alpha=0.85, edgecolors="white", linewidths=0.8, zorder=3)
    ax.annotate(f.replace("Factor","F"), (var, shap),
                textcoords="offset points", xytext=(5, 3), fontsize=8.5,
                fontweight="bold" if f in TOP_FACTORS else "normal")

ax.axhline(0.10, ls="--", c="#999", lw=1, label="SHAP threshold (0.10)")
ax.set_xlabel("Variance Explained (%)", fontsize=12)
ax.set_ylabel("Avg |SHAP| across 3 models", fontsize=12)
ax.set_title("MOFA+ Factor Importance: SHAP Score vs. Variance Explained\n"
             "(bubble size ∝ SHAP; red = selected for deep-dive)", fontsize=12)
ax.legend(fontsize=9)
ax.set_facecolor("#f8f9fa")
fig.tight_layout()
fig.savefig(FIGURES / "marker_factor_importance.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  Saved: marker_factor_importance.png")

# ── 10. FIGURE 2: Horizontal bar — top genes by gene_shap_contrib ─────────────
MOD_COLORS = {"expression":"#2196F3", "mutation":"#FF5722", "cnv":"#4CAF50"}

top40 = gene_agg.sort_values("total_contrib", ascending=True).tail(40)
fig, ax = plt.subplots(figsize=(10, 14))
bars = ax.barh(
    range(len(top40)),
    top40["total_contrib"],
    color=[MOD_COLORS.get(m,"#888") for m in top40["modality"]],
    alpha=0.85, edgecolor="white", height=0.72
)
ax.set_yticks(range(len(top40)))
ax.set_yticklabels(
    [f"{row.gene}  ({row.modality[:3].upper()})"
     for _, row in top40.iterrows()],
    fontsize=9
)
ax.set_xlabel("Gene-level SHAP contribution (|weight| × factor SHAP)", fontsize=11)
ax.set_title("Top 40 Multiomics Markers by SHAP Contribution\n"
             "Color: modality   Sorted by aggregate contribution across factors", fontsize=11)
# Legend
patches = [mpatches.Patch(color=c, label=m.capitalize())
           for m, c in MOD_COLORS.items()]
ax.legend(handles=patches, loc="lower right", fontsize=9)
ax.set_facecolor("#f8f9fa")
fig.tight_layout()
fig.savefig(FIGURES / "marker_top40_genes.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  Saved: marker_top40_genes.png")

# ── 11. FIGURE 3: Per-factor heatmap (top genes × factors) ───────────────────
# Expression only — top 20 genes per top factor
all_expr_genes = {}
for f in TOP_FACTORS:
    s = w_expr[f].abs().sort_values(ascending=False).head(20)
    all_expr_genes[f] = s.index.tolist()

union_genes = []
seen = set()
for genes in all_expr_genes.values():
    for g in genes:
        if g not in seen:
            union_genes.append(g)
            seen.add(g)

heat_df = w_expr.loc[[g for g in union_genes if g in w_expr.index], TOP_FACTORS]
heat_df = heat_df.loc[heat_df.abs().max(axis=1) > 0.02]   # trim noise

fig, ax = plt.subplots(figsize=(max(8, len(TOP_FACTORS)*1.4), max(12, len(heat_df)*0.35)))
vmax = heat_df.abs().max().max()
im = ax.imshow(heat_df.values, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
ax.set_xticks(range(len(TOP_FACTORS)))
ax.set_xticklabels(TOP_FACTORS, rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(heat_df)))
ax.set_yticklabels(heat_df.index, fontsize=8)
fig.colorbar(im, ax=ax, label="MOFA+ weight (expression)", fraction=0.02, pad=0.02)
ax.set_title("Expression Gene Weights across Top Survival Factors\n"
             "(Red = high expression loads factor up; Blue = down)", fontsize=11)
fig.tight_layout()
fig.savefig(FIGURES / "marker_gene_heatmap.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  Saved: marker_gene_heatmap.png")

# ── 12. FIGURE 4: Mutation marker landscape ──────────────────────────────────
mut_records = []
for f in TOP_FACTORS:
    s = w_mut[f].abs().sort_values(ascending=False).head(15)
    for g in s.index:
        mut_records.append({
            "gene": g,
            "factor": f,
            "weight": w_mut.loc[g, f],
            "abs_weight": abs(w_mut.loc[g, f]),
            "factor_shap": factor_pivot.loc[f, "avg_shap"],
            "gene_contrib": abs(w_mut.loc[g, f]) * factor_pivot.loc[f, "avg_shap"],
        })
mut_df = pd.DataFrame(mut_records)
mut_top = (mut_df.groupby("gene")["gene_contrib"].sum()
           .sort_values(ascending=False).head(25))

fig, axes = plt.subplots(1, 2, figsize=(14, 8))

# Left: ranked bar
axes[0].barh(range(len(mut_top)), mut_top.values[::-1],
             color=["#e63946" if g in KNOWN_ONCOGENES else "#457b9d"
                    for g in mut_top.index[::-1]],
             alpha=0.85, edgecolor="white")
axes[0].set_yticks(range(len(mut_top)))
axes[0].set_yticklabels(mut_top.index[::-1], fontsize=9)
axes[0].set_xlabel("Gene SHAP contribution (mutation view)", fontsize=10)
axes[0].set_title("Top Mutation Markers\n(red = known driver gene)", fontsize=10)
axes[0].set_facecolor("#f8f9fa")

# Right: factor × mutation gene heatmap
mut_genes_top = mut_top.index.tolist()
mut_heat = w_mut.loc[[g for g in mut_genes_top if g in w_mut.index], TOP_FACTORS]
vmax2 = mut_heat.abs().max().max()
im2 = axes[1].imshow(mut_heat.values, cmap="PuOr", vmin=-vmax2, vmax=vmax2, aspect="auto")
axes[1].set_xticks(range(len(TOP_FACTORS)))
axes[1].set_xticklabels(TOP_FACTORS, rotation=45, ha="right", fontsize=8)
axes[1].set_yticks(range(len(mut_heat)))
axes[1].set_yticklabels(mut_heat.index, fontsize=8)
fig.colorbar(im2, ax=axes[1], label="MOFA+ weight (mutation)", fraction=0.04, pad=0.03)
axes[1].set_title("Mutation Weights: Top Genes × Top Factors", fontsize=10)
fig.suptitle("Somatic Mutation Landscape in Survival-Associated MOFA Factors",
             fontsize=12, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "marker_mutation_landscape.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  Saved: marker_mutation_landscape.png")

# ── 13. FIGURE 5: Pathway bubble chart ───────────────────────────────────────
pathway_agg = (
    gene_agg[gene_agg["pathway"] != "Unknown"]
    .groupby("pathway")
    .agg(
        total_contrib = ("total_contrib", "sum"),
        n_genes       = ("gene",          "count"),
    )
    .sort_values("total_contrib", ascending=False)
    .head(18)
)

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(
    range(len(pathway_agg)),
    pathway_agg["total_contrib"],
    s=pathway_agg["n_genes"] * 100,
    c=pathway_agg["total_contrib"],
    cmap="YlOrRd", alpha=0.85,
    edgecolors="white", linewidths=0.8, zorder=3
)
ax.set_xticks(range(len(pathway_agg)))
ax.set_xticklabels(pathway_agg.index, rotation=40, ha="right", fontsize=9)
ax.set_ylabel("Aggregate SHAP contribution", fontsize=11)
ax.set_title("Pathway-level SHAP Contribution\n(bubble size ∝ number of genes in pathway)",
             fontsize=11)
ax.set_facecolor("#f8f9fa")
fig.colorbar(scatter, ax=ax, label="Total SHAP contribution", fraction=0.02)
fig.tight_layout()
fig.savefig(FIGURES / "marker_pathway_bubble.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  Saved: marker_pathway_bubble.png")

# ── 14. FIGURE 6: SHAP consistency across models (scatter matrix) ─────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 5), sharey=False)
pairs = [("ftt","xgb","FTT vs XGB"),("ftt","lgb","FTT vs LGB"),("xgb","lgb","XGB vs LGB")]

fp2 = factor_pivot.reset_index()
for ax, (m1, m2, title) in zip(axes, pairs):
    ax.scatter(fp2[m1], fp2[m2],
               s=fp2["avg_shap"]*800+30,
               c=fp2["avg_shap"], cmap="Reds", alpha=0.85,
               edgecolors="white", linewidths=0.7)
    for _, row in fp2.iterrows():
        ax.annotate(row["factor"].replace("Factor","F"),
                    (row[m1], row[m2]), textcoords="offset points",
                    xytext=(4, 2), fontsize=7.5)
    mn = min(fp2[[m1,m2]].min())
    mx = max(fp2[[m1,m2]].max())
    ax.plot([mn,mx],[mn,mx], "k--", lw=0.8, alpha=0.5)
    ax.set_xlabel(f"{m1.upper()} |SHAP|", fontsize=10)
    ax.set_ylabel(f"{m2.upper()} |SHAP|", fontsize=10)
    ax.set_title(title, fontsize=10)
    ax.set_facecolor("#f8f9fa")

fig.suptitle("Factor SHAP Score Consistency Across Models\n"
             "(points on diagonal = fully consistent signal)", fontsize=11, fontweight="bold")
fig.tight_layout()
fig.savefig(FIGURES / "marker_model_consistency.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  Saved: marker_model_consistency.png")

# ── 15. FIGURE 7: Actionable target summary ───────────────────────────────────
actionable = gene_agg[
    (gene_agg["role"] != "Unknown") |
    (gene_agg["gene"].isin(THERAPEUTIC_MECHANISM.keys()))
].sort_values("total_contrib", ascending=False).head(30)

fig, ax = plt.subplots(figsize=(11, 9))
role_colors = {
    "Oncogene":        "#e63946",
    "TSG/gain-of-function": "#ff6b35",
    "TSG":             "#2196F3",
    "Unknown":         "#adb5bd",
}
bar_colors = [role_colors.get(r, "#adb5bd") for r in actionable["role"]]

ax.barh(range(len(actionable)), actionable["total_contrib"].values,
        color=bar_colors, alpha=0.88, edgecolor="white", height=0.7)
ax.set_yticks(range(len(actionable)))
labels = []
for _, row in actionable.iterrows():
    mec = row["tx_mechanism"]
    short = mec[:55] + "…" if len(mec) > 55 else mec
    labels.append(f"{row.gene}  |  {row.modality[:3].upper()}  |  {short}")
ax.set_yticklabels(labels, fontsize=7.8)
ax.set_xlabel("Gene-level SHAP contribution", fontsize=11)
ax.set_title("Actionable Therapeutic Targets from Multiomics SHAP Analysis\n"
             "Color: role (red=oncogene, blue=TSG, orange=GoF-TSG)", fontsize=11)
patches = [mpatches.Patch(color=c, label=r) for r, c in role_colors.items() if r != "Unknown"]
ax.legend(handles=patches, loc="lower right", fontsize=9)
ax.set_facecolor("#f8f9fa")
fig.tight_layout()
fig.savefig(FIGURES / "marker_actionable_targets.png", dpi=180, bbox_inches="tight")
plt.close(fig)
print("  Saved: marker_actionable_targets.png")

print("\n" + "="*70)
print("MARKER ANALYSIS COMPLETE")
print("="*70)
print(f"Outputs in {RESULTS}:")
print("  marker_shap_scores.csv")
print("  factor_gene_table.csv")
print("  therapeutic_targets.csv")
print("  factor_biology_summary.txt")
print("  figures/marker_factor_importance.png")
print("  figures/marker_top40_genes.png")
print("  figures/marker_gene_heatmap.png")
print("  figures/marker_mutation_landscape.png")
print("  figures/marker_pathway_bubble.png")
print("  figures/marker_model_consistency.png")
print("  figures/marker_actionable_targets.png")
print("="*70)
