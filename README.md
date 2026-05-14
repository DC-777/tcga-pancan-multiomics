# TCGA Pan-Cancer Multiomics Survival Analysis

> **Identifying therapeutic targets from pan-cancer multiomics data using MOFA+ factor analysis, gradient-boosting survival models, and SHAP attribution**

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This project builds a full end-to-end pipeline that:

1. **Downloads** the TCGA Pan-Cancer Atlas (33 cancer types, ~11 000 patients) from UCSC Xena
2. **Decomposes** three omics modalities — gene expression, somatic mutations, copy-number variation — into 18 interpretable factors using **MOFA+**
3. **Predicts overall survival** with three models: Feature Tabular Transformer (**FTT**), **XGBoost**, and **LightGBM** (best C-index: **0.7512**)
4. **Attributes** model predictions to MOFA+ factors using **SHAP**, then traces factors back to individual genes
5. **Nominates therapeutic targets** with actionability annotations and drug-mechanism mapping
6. **Delivers** results as a self-contained interactive HTML dashboard, two Word manuscripts, and publication-ready figures

### Key findings

| Factor | Biology | avg \|SHAP\| | Var. explained | Key drugs |
|--------|---------|-------------|---------------|-----------|
| **Factor8** | Pan-cancer invasion + multi-driver mutations (TP53/KRAS/IDH1/APC) | **0.2831** | 3.2 % | APR-246, sotorasib, ivosidenib |
| **Factor6** | 9p24.1 immune-checkpoint amplicon (CD274/PD-L1, JAK2) | 0.2194 | 0.8 % | Pembrolizumab, ruxolitinib |
| **Factor1** | Squamous lineage / TP63 / 3q26.3 amplicon | 0.1624 | 6.1 % | Enfortumab vedotin |
| **Factor15** | Luminal de-differentiation (GATA3↓, PTEN/ARID1A mut) | 0.1197 | 2.5 % | Alpelisib, tazemetostat |
| **Factor12** | Thyroid (BRAF V600E) vs. IDH-mutant glioma axis | 0.1196 | 5.3 % | Dabrafenib+trametinib, ivosidenib |

> **Key insight:** Prognostic relevance (SHAP) and variance explained by a MOFA+ factor are orthogonal. Factor8 carries the highest survival signal despite explaining the least variance — underscoring the need for SHAP-guided factor selection rather than variance-based cutoffs.

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
│   │   ├── mofa_analysis.py              # MOFA+ factor decomposition
│   │   ├── clustering.py                 # K-means + Leiden clustering
│   │   └── dim_reduction.py              # PCA / UMAP
│   ├── supervised/
│   │   ├── cox_models.py                 # Cox PH + ElasticNet
│   │   ├── deepsurv.py                   # DeepSurv neural baseline
│   │   ├── model_comparison.py           # FTT / XGB / LGB + 5-fold CV
│   │   └── shap_analysis.py              # SHAP computation & plots
│   └── analysis/
│       ├── marker_analysis.py            # Gene-level SHAP attribution
│       └── build_manuscript.py           # Word document generation
│
├── results/
│   ├── dashboard.html                    # Interactive browser dashboard
│   ├── manuscript.docx                   # Main manuscript (Word)
│   ├── supplementary.docx               # Supplementary material (Word)
│   ├── figures/                          # All publication PNG figures
│   ├── mofa_weights_*.csv               # Per-modality MOFA factor weights
│   ├── shap_importance_summary_models.csv
│   ├── therapeutic_targets.csv
│   ├── factor_biology_summary.txt
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

### 5. Run the causal inference layer (diff-diff)

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

### 6. View results

| Output | How to open |
|--------|-------------|
| `results/dashboard.html` | Open in any browser — 7 panels including Causal DiD |
| `results/manuscript.docx` | Microsoft Word / LibreOffice |
| `results/supplementary.docx` | Microsoft Word / LibreOffice |
| `results/figures/` | PNG files, directly usable in publications |

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
