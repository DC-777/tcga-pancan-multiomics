# TCGA Pan-Cancer Multiomics Survival Analysis

> **Identifying therapeutic targets from pan-cancer multiomics data using MOFA+ factor analysis, gradient-boosting survival models, SHAP attribution, modality ablation, and diff-diff causal inference**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project builds a full end-to-end pipeline that:

1. **Downloads** the TCGA Pan-Cancer Atlas (33 cancer types, ~11 000 patients) from UCSC Xena
2. **Decomposes** three omics modalities — gene expression, somatic mutations, copy-number variation — into 18 interpretable factors using **MOFA+**
3. **Predicts overall survival** with three models: Feature Tabular Transformer (**FTT**), **XGBoost**, and **LightGBM** (best C-index: **0.7622** — expression-only MOFA+)
4. **Attributes** model predictions to MOFA+ factors using **SHAP**, then traces factors back to individual genes
5. **Ablates** the modality contribution: expression-only vs. mutation-only vs. CNV-only vs. joint 3-modal factors
6. **Validates causality** with `diff-diff` staggered DiD, Triple Difference synthetic lethality, HonestDiD, and TROP
7. **Nominates therapeutic targets** with actionability annotations and drug-mechanism mapping
8. **Delivers** results as a self-contained interactive HTML dashboard (8 panels), two Word manuscripts, and publication-ready figures

### Key findings

| Factor | Biology | avg \|SHAP\| | Var. explained | Key drugs |
|--------|---------|-------------|---------------|-----------|
| **Factor8** | Pan-cancer invasion + multi-driver mutations (TP53/KRAS/IDH1/APC) | **0.2831** | 3.2 % | APR-246, sotorasib, ivosidenib |
| **Factor6** | 9p24.1 immune-checkpoint amplicon (CD274/PD-L1, JAK2) | 0.2194 | 0.8 % | Pembrolizumab, ruxolitinib |
| **Factor1** | Squamous lineage / TP63 / 3q26.3 amplicon | 0.1624 | 6.1 % | Enfortumab vedotin |
| **Factor15** | Luminal de-differentiation (GATA3↓, PTEN/ARID1A mut) | 0.1197 | 2.5 % | Alpelisib, tazemetostat |
| **Factor12** | Thyroid (BRAF V600E) vs. IDH-mutant glioma axis | 0.1196 | 5.3 % | Dabrafenib+trametinib, ivosidenib |

> **Key insight 1 — SHAP vs variance:** Prognostic relevance (SHAP) and variance explained by a MOFA+ factor are orthogonal. Factor8 carries the highest survival signal despite explaining the least variance.

### Modality ablation

| Condition | FTT C-index | XGBoost | LightGBM |
|-----------|-------------|---------|----------|
| **Joint 3-modal** | 0.7512 ± 0.006 | 0.7465 ± 0.008 | 0.7451 ± 0.008 |
| **Expression only** ★ | **0.7622 ± 0.007** | **0.7595 ± 0.009** | **0.7569 ± 0.010** |
| Mutation only | 0.7520 ± 0.006 | 0.7402 ± 0.004 | 0.7317 ± 0.007 |
| CNV only | 0.7493 ± 0.004 | 0.7424 ± 0.004 | 0.7400 ± 0.007 |

> **Key insight 2 — Modality ablation:** Expression-only MOFA+ factors outperform the joint 3-modal model by **+0.011–0.013 C-index** across all architectures. When all factor capacity is dedicated to continuous RNA-seq data, MOFA+ extracts purer prognostic programmes without dilution from sparse binary mutations or noisy CNV. The joint model remains essential for biological interpretability and causal target discovery (DDD co-dependencies).

### Causal validation (diff-diff)

| Factor | ATT | p-value | Status |
|--------|-----|---------|--------|
| **Factor6** (9p24.1 immune checkpoint) | −0.486 | < 10⁻¹² | ✓ Causally validated |
| **Factor15** (luminal de-differentiation) | −0.285 | < 10⁻¹⁷ | ✓ Causally validated |
| Factor1 (squamous lineage) | −0.129 | 0.028 | ✓ Validated |
| Factor8 (invasion/drivers) | +0.298 | 0.134 | Confounded |
| **KRAS × Factor8 DDD** | −0.472 | 0.022 | ✓ Synergistic co-dependency |

> **Key insight 3 — Causal layer:** Factor8's top SHAP rank reflects cancer-type composition confounding. Factor6 and Factor15 are the primary causally validated targets. The KRAS × Factor8 Triple Difference (p = 0.022) nominates KRAS inhibitor + invasion suppression as a combination therapy strategy invisible to SHAP alone.

---

## Repository structure

```
tcga-pancan-multiomics/
├── download_tcga.py          # Step 1 – Download TCGA data from UCSC Xena
├── convert_to_csv.py         # Step 2 – Decompress .gz → .csv
├── load_tcga.py              # Convenience loader for downstream scripts
├── build_report.py           # Generate HTML summary report
│
├── src/
│   ├── preprocess.py                     # Alignment, filtering, imputation
│   ├── unsupervised/
│   │   ├── mofa_analysis.py              # MOFA+ factor decomposition (modalities param)
│   │   ├── clustering.py                 # K-means + Leiden clustering
│   │   └── dim_reduction.py             # PCA / UMAP
│   ├── supervised/
│   │   ├── cox_models.py                 # Cox PH + ElasticNet
│   │   ├── deepsurv.py                   # DeepSurv neural baseline
│   │   ├── model_comparison.py           # FTT / XGB / LGB + 5-fold CV
│   │   └── shap_analysis.py              # SHAP computation & plots
│   └── analysis/
│       ├── marker_analysis.py            # Gene-level SHAP attribution
│       ├── modality_ablation.py          # Expression vs. Mut vs. CNV vs. Joint ablation
│       ├── did_causal_layer.py           # diff-diff causal inference (CS, DDD, HonestDiD, TROP)
│       └── build_manuscript.py          # Word document generation
│
├── results/
│   ├── dashboard.html                    # Interactive browser dashboard (8 panels)
│   ├── graphrag.html                     # Graphical RAG — knowledge graph + Q&A
│   ├── concept_map.html                  # Pipeline concept map
│   ├── manuscript.docx                   # Main manuscript (Word)
│   ├── supplementary.docx               # Supplementary material (Word)
│   ├── figures/                          # All publication PNG figures
│   ├── mofa_factors.csv                  # Joint 3-modal factor scores (7,902 × 18)
│   ├── mofa_factors_expr_only.csv        # Expression-only factor scores
│   ├── mofa_factors_mut_only.csv         # Mutation-only factor scores
│   ├── mofa_factors_cnv_only.csv         # CNV-only factor scores
│   ├── ablation_cindex_summary.csv       # 4 conditions × 3 models C-index table
│   ├── mofa_weights_*.csv               # Per-modality MOFA factor weights
│   ├── shap_importance_summary_models.csv
│   ├── therapeutic_targets.csv
│   ├── did_att_estimates.csv             # Callaway-Sant'Anna ATT per factor
│   ├── did_ddd_synergy.csv              # Triple Difference gene-factor co-dependencies
│   ├── did_sensitivity.csv              # HonestDiD bounds for Factor8
│   └── ...                              # Other small result tables
│
├── data/                                 # NOT tracked – download separately
│   └── xena_pancan/
│       ├── expression/
│       ├── mutations/
│       ├── methylation/
│       ├── copy_number/
│       ├── mirna/
│       ├── protein/
│       └── clinical/
│
├── requirements.txt
├── environment.yml
└── pyproject.toml
```

---

## Quick start

### 1. Clone and install

```bash
git clone https://github.com/debaditya-chakraborty/tcga-pancan-multiomics.git
cd tcga-pancan-multiomics
```

**Option A — pip (venv)**

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Option B — conda**

```bash
conda env create -f environment.yml
conda activate tcga-multiomics
```

> **GPU acceleration:** Replace `torch==2.12.0` in `requirements.txt` with `torch==2.12.0+cu121` (CUDA 12.1) and install from `https://download.pytorch.org/whl/cu121`.

---

### 2. Download the TCGA data

```bash
# Full download (~15 GB, dominated by methylation)
python download_tcga.py

# Fast smoke test (~2 GB – expression + mutations + clinical only)
python download_tcga.py --modalities expression,mutations,clinical

# Custom data directory
python download_tcga.py --data-dir /path/to/data
```

Files are saved to `data/xena_pancan/` with one subfolder per modality. Already-downloaded files are skipped automatically.

### 3. Convert compressed files to CSV

```bash
python convert_to_csv.py
```

Outputs go to `data/csv/`. Large files (methylation ~38 GB) are skipped by default; pass `--include-methylation` to convert them.

### 4. Run the full analysis pipeline

Each script is self-contained and reads from `data/csv/` or `results/`:

```bash
# Preprocessing & alignment
python -m src.preprocess

# Unsupervised: MOFA+ + UMAP + clustering
python -m src.unsupervised.mofa_analysis
python -m src.unsupervised.dim_reduction
python -m src.unsupervised.clustering

# Supervised: survival models + SHAP
python -m src.supervised.model_comparison
python -m src.supervised.shap_analysis

# Marker & therapeutic target analysis
python -m src.analysis.marker_analysis

# Build manuscripts (requires results/ from above)
python -m src.analysis.build_manuscript
```

### 5. Run the modality ablation

```bash
python -m src.analysis.modality_ablation
```

Trains single-modality MOFA+ (expression-only, mutation-only, CNV-only) and runs 5-fold CV on each.
Reuses the existing joint `results/mofa_factors.csv` — no re-training needed for the joint condition.

Outputs: `results/ablation_cindex_summary.csv`, `results/figures/ablation_cindex_comparison.png`,
`results/figures/ablation_variance_heatmap.png`, and per-modality factor files.

> **Key result:** Expression-only MOFA+ factors achieve C-index **0.7622** (FTT), **+0.011** above the joint model — the single richest modality for survival prediction. Joint model remains essential for biological interpretability and causal co-dependency mapping.

### 7. Run the causal inference layer (diff-diff)

```bash
python -m src.analysis.did_causal_layer
```

This runs five sequential analyses using the `diff-diff` library (v3.3):
- **CallawaySantAnna** staggered event study (per top-5 MOFA factor)
- **BaconDecomposition** to diagnose TWFE bias
- **TripleDifference** synthetic lethality for 4 gene × factor pairs
- **HonestDiD** Rambachan-Roth sensitivity bounds for Factor8
- **TROP** nuclear-norm factor-adjusted robustness check

Outputs: `results/did_att_estimates.csv`, `results/did_ddd_synergy.csv`,
`results/did_sensitivity.csv`, `results/did_summary.txt`, and 8 new figures.

### 8. View results

| Output | How to open |
|--------|-------------|
| `results/dashboard.html` | Open in any browser — **8 panels** (Overview, Models, MOFA+, SHAP, Targets, Findings, Causal DiD, Modality Ablation) |
| `results/graphrag.html` | Open in any browser — Graphical RAG knowledge graph + Q&A |
| `results/concept_map.html` | Open in any browser — Pipeline concept map |
| `results/manuscript.docx` | Microsoft Word / LibreOffice |
| `results/supplementary.docx` | Microsoft Word / LibreOffice |
| `results/figures/` | PNG files, directly usable in publications |
| `results/ablation_cindex_summary.csv` | 4-condition modality ablation summary table |

---

## Data sources

All data are sourced from the **UCSC Xena Pan-Cancer Atlas** hub (open access, no GDC token required):

| Modality | File | Size |
|----------|------|------|
| Gene expression (TPM log2) | `tcga_RSEM_gene_tpm.gz` | ~1.8 GB |
| Gene expression (Hugo norm) | `tcga_RSEM_Hugo_norm_count.gz` | ~1.8 GB |
| Somatic mutations (MC3 MAF) | `mc3.v0.2.8.PUBLIC.xena.gz` | ~80 MB |
| DNA methylation HM450 | `...HumanMethylation450...xena.gz` | ~10 GB |
| Copy number (SNP6 segments) | `broad.mit.edu...seg.gz` | ~200 MB |
| Copy number (GISTIC2) | `...all_thresholded.by_genes.gz` | ~60 MB |
| miRNA expression | `pancanMiRs...xena.gz` | ~35 MB |
| Protein expression (RPPA) | `TCGA-RPPA-pancan-clean.xena.gz` | ~5 MB |
| Clinical phenotype | `TCGA_phenotype_denseDataOnlyDownload.tsv.gz` | ~2 MB |
| Survival | `Survival_SupplementalTable_S1_20171025_xena_sp.gz` | ~1 MB |

**Citation for data:** Goldman MJ et al. *Visualizing and interpreting cancer genomics data via the Xena platform.* Nature Biotechnology (2020). https://doi.org/10.1038/s41587-020-0546-8

---

## Methods summary

### Multi-omics integration — MOFA+

MOFA+ (Multi-Omics Factor Analysis v2) decomposes gene expression, somatic mutation, and copy-number matrices jointly into 18 latent factors. Factor weights identify which genes drive each factor in each modality; factor scores per patient are used as survival model features.

### Survival models

Three models are trained in 5-fold cross-validation on MOFA+ factor scores + clinical covariates:

| Model | Architecture | Mean C-index |
|-------|-------------|-------------|
| **FTT** | Feature Tabular Transformer | **0.7512** |
| XGBoost | Gradient-boosted trees | 0.7465 |
| LightGBM | Gradient-boosted trees | 0.7451 |

### SHAP attribution

SHAP values are computed for each model separately. Factor importance is the average `|SHAP|` across the three models, weighted by inter-model consistency (inverse CV). Gene-level importance is estimated as `|MOFA_weight| × factor_avg_SHAP`, aggregated across the top-5 factors.

### Modality ablation

MOFA+ is retrained three times with a single-view subset (expression-only, mutation-only, CNV-only), using the same feature selection, likelihood functions, and `N_FACTORS=20` as the joint run. The `modalities` parameter to `run_mofa_pipeline()` controls which views are passed. Survival models are re-trained on each single-modality factor set under identical 5-fold CV splits (`random_state=42`), enabling paired comparison. No SHAP is computed during ablation runs.

### Causal inference (diff-diff)

Callaway-Sant'Anna (2021) staggered DiD on a cancer-type × AJCC-stage pseudo-panel. Treatment = high MOFA factor score (above cancer-type-specific median). Outcome = z-scored OS time. Supplemented by Bacon Decomposition (TWFE bias), Triple Difference (gene-factor synthetic lethality), HonestDiD (Rambachan-Roth 2023 sensitivity bounds), and TROP (nuclear-norm panel estimator).

### Therapeutic target nomination

Genes are annotated with:
- **Role**: Oncogene / TSG / Immune / Lineage factor
- **Pathway**: RAS-MAPK, PI3K-AKT, DNA-damage response, etc.
- **Drug / mechanism**: FDA-approved agents or clinical-stage candidates
- **Actionability**: Yes (approved/phase III) / Partial (phase I-II or biomarker) / No

---

## Requirements

- Python 3.10+
- 32 GB RAM recommended (methylation matrix is large)
- ~20 GB disk for raw data; ~50 GB for all converted CSVs
- CUDA GPU optional (speeds up FTT training; CPU fallback works)

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

## Citation

If you use this pipeline or results, please cite:

```bibtex
@software{chakraborty2026tcga,
  author  = {Chakraborty, Debaditya},
  title   = {TCGA Pan-Cancer Multiomics Survival Analysis},
  year    = {2026},
  url     = {https://github.com/debaditya-chakraborty/tcga-pancan-multiomics},
}
```
