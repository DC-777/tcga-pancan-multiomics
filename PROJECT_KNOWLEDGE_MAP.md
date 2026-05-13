# TCGA Pan-Cancer Multi-Omics Project Knowledge Map

Purpose: this file is an editable onboarding and training map for a future AI agent that needs to understand, rerun, debug, extend, or explain the entire cancer multi-omics modeling project.

Last updated: 2026-05-13

Project root:

```text
C:\Users\debad\OneDrive\Documents\DC Academic Research\Cancer Research
```

Primary deliverable folders:

```text
data/
results/
results/figures/
src/
src/analysis/
src/supervised/
src/unsupervised/
```

## 1. Project Summary

This project performs an end-to-end TCGA pan-cancer multi-omics analysis. It downloads open-access UCSC Xena / TCGA resources, converts them to local CSV assets, aligns primary tumor samples across omics modalities, discovers molecular structure with PCA/UMAP and clustering, trains survival models, interprets models with SHAP, decodes MOFA-style latent factors back to genes, and ranks marker / therapeutic target hypotheses.

The current final aligned cohort contains:

| Quantity | Current value |
|---|---:|
| Primary tumor samples | 7,902 |
| Cancer types | 31 |
| Expression features | 5,000 |
| Mutation features | 1,490 |
| CNV features | 24,776 |
| Samples with OS time | 7,787 |
| Death events | 2,170 |

Important: older generated report text in the repository may mention a smaller survival subset. Trust the current CSV outputs in `results/`, especially `aligned_clinical.csv`, over stale prose.

## 2. Mental Model For A New Agent

Think of the project as five linked layers:

1. Data layer: download TCGA/Xena files and convert them to CSV.
2. Alignment layer: normalize TCGA barcodes, filter primary tumors, intersect modalities.
3. Representation layer: PCA/UMAP, clustering, and MOFA-style latent factors.
4. Prediction layer: survival modeling with Cox, DeepSurv, FT-Transformer, XGBoost, and LightGBM.
5. Interpretation layer: SHAP, factor-to-gene decoding, marker ranking, therapeutic annotation, reports.

Most downstream analysis assumes that `results/aligned_expression.csv`, `results/aligned_mutations.csv`, `results/aligned_cnv.csv`, and `results/aligned_clinical.csv` already exist.

## 3. Key Data Sources

Data are downloaded by `download_tcga.py`.

| Modality | Source / file family | Used in final model? | Notes |
|---|---|---:|---|
| Expression | UCSC TOIL Xena RSEM gene TPM / Hugo normalized count | Yes | Main expression matrix. Final aligned matrix uses top 5,000 variable genes. |
| Mutations | PanCanAtlas MC3 somatic mutation calls | Yes | Nonsynonymous calls pivoted into binary sample-by-gene matrix. |
| Copy number | TCGA Xena GISTIC2 gene-level thresholded values | Yes | Values are -2, -1, 0, 1, 2. |
| Clinical phenotype | PanCanAtlas Xena phenotype dense table | Yes | Provides cancer type / primary tumor labels. |
| Survival | TCGA-CDR survival supplemental table | Yes | Provides OS time, OS event, age at diagnosis. |
| Methylation | HM450 beta values | Not in final aligned model | Converter supports chunking; file is very large. |
| miRNA | Pan-cancer miRNA expression | Not in final aligned model | Loaders exist. |
| Protein | RPPA pan-cancer protein matrix | Not in final aligned model | Loaders exist. |

## 4. File And Module Map

### Root scripts

| File | Role |
|---|---|
| `download_tcga.py` | Downloads open-access TCGA pan-cancer files from UCSC Xena hubs. |
| `convert_to_csv.py` | Converts downloaded TSV/GZ files to CSV. Methylation supports chunked conversion. |
| `load_tcga.py` | Convenience loaders for raw downloaded Xena files. |
| `build_report.py` | Generates an HTML report from existing result CSVs and PNGs. Some embedded prose may be stale. |
| `PROJECT_KNOWLEDGE_MAP.md` | This agent-facing editable knowledge map. |

### Preprocessing

| File | Role |
|---|---|
| `src/preprocess.py` | Loads expression, mutation, CNV, phenotype, and survival; aligns all modalities; writes aligned matrices to `results/`. |

### Unsupervised analysis

| File | Role |
|---|---|
| `src/unsupervised/dim_reduction.py` | StandardScaler + PCA + UMAP on expression; writes UMAP coordinates and figures. |
| `src/unsupervised/clustering.py` | MiniBatch K-means scan and Leiden graph clustering in PCA space; writes cluster labels and plots. |
| `src/unsupervised/mofa_analysis.py` | Builds multi-view matrices, trains MOFA+ if available or NMF fallback, writes factor scores, weights, variance, and figures. |

### Supervised survival modeling

| File | Role |
|---|---|
| `src/supervised/cox_models.py` | Cox Ridge baseline survival model with ElasticNet feature preselection. |
| `src/supervised/deepsurv.py` | DeepSurv neural Cox model using the same raw feature matrix as Cox. |
| `src/supervised/shap_analysis.py` | SHAP-style interpretation for Cox and DeepSurv raw feature models. |
| `src/supervised/model_comparison.py` | Final FT-Transformer, XGBoost, and LightGBM survival comparison using MOFA factors, cancer type, and age. |

### Marker and manuscript analysis

| File | Role |
|---|---|
| `src/analysis/marker_analysis.py` | Links MOFA factor SHAP importance to gene weights; creates marker, pathway, and therapeutic target outputs. |
| `src/analysis/build_manuscript.py` | Builds manuscript / supplementary artifacts. Inspect before rerunning. |

## 5. Core Data Contracts

Future agents must preserve these contracts unless intentionally refactoring the pipeline.

### TCGA barcode contract

Sample IDs are standardized to 15-character TCGA sample barcodes such as:

```text
TCGA-19-1787-01
```

Primary tumors are filtered by checking that the barcode ends in `01`.

### Aligned matrix contract

All aligned matrices in `results/` must share the same row index and row order.

| File | Shape concept |
|---|---|
| `aligned_expression.csv` | samples x expression genes |
| `aligned_mutations.csv` | samples x binary mutation genes |
| `aligned_cnv.csv` | samples x CNV genes |
| `aligned_clinical.csv` | samples x clinical fields |

Clinical columns currently include:

```text
cancer_type
OS_time
OS_event
age_at_dx
cancer_type_cdr
```

### Survival contract

Survival modeling expects:

```text
OS_time > 0
OS_event in {0, 1}
```

Censored observations have `OS_event = 0`; events/deaths have `OS_event = 1`.

### Feature matrix contract for Cox / DeepSurv

`src/supervised/cox_models.py::build_feature_matrix()` creates:

| Feature block | Description |
|---|---|
| `EXPR_PC1..EXPR_PC50` | PCA components from scaled expression fitted on all aligned samples |
| `MUT_<gene>` | Top 100 frequent mutation genes |
| `CNV_PC1..CNV_PC30` | PCA components from scaled CNV fitted on all aligned samples |

ElasticNet preselection reduces this to roughly 80 features before Cox / DeepSurv.

### Feature matrix contract for final model comparison

`src/supervised/model_comparison.py::build_mofa_feature_matrix()` creates:

| Feature block | Description |
|---|---|
| `MOFA_Factor1..MOFA_FactorK` | Factor scores from `results/mofa_factors.csv` |
| `CT_<cancer type>` | One-hot encoded cancer type |
| `age_at_dx` | Median-imputed age at diagnosis |

## 6. Pipeline Execution Order

Use this order when rebuilding from raw data:

```text
1. python download_tcga.py
2. python convert_to_csv.py
3. python -m src.preprocess
4. python -m src.unsupervised.dim_reduction
5. python -m src.unsupervised.clustering
6. python -m src.unsupervised.mofa_analysis
7. python -m src.supervised.cox_models
8. python -m src.supervised.deepsurv
9. python -m src.supervised.shap_analysis
10. python -m src.supervised.model_comparison
11. python src/analysis/marker_analysis.py
12. python build_report.py
```

Notes:

- The environment used to generate existing outputs may not have `python` on PATH in every shell. Check the user's active Python environment before rerunning.
- Some steps are compute-heavy, especially CNV PCA, DeepSurv, model comparison, SHAP, and MOFA.
- `mofa_analysis.py` tries `mofapy2`; if unavailable it falls back to an NMF-like proxy. That distinction matters for scientific reporting.

## 7. Current Major Results

### Unsupervised structure

K-means currently selected a coarse 2-cluster solution:

| Cluster | Samples |
|---|---:|
| K0 | 809 |
| K1 | 7,093 |

Leiden currently finds 22 clusters, with cluster sizes from 64 to 1,338 samples.

Interpretation:

- Expression-derived global structure is strongly lineage / cancer-type driven.
- Leiden gives a more granular community structure than K-means.
- Use cancer type composition plots before making biological claims about clusters.

### Final survival model comparison

Cross-validated C-index from `results/model_cv_scores.csv`:

| Model | Mean C-index | SD |
|---|---:|---:|
| FT-Transformer | 0.7512 | 0.0063 |
| XGBoost | 0.7465 | 0.0082 |
| LightGBM | 0.7451 | 0.0075 |
| XGBoost raw PCA baseline | 0.7237 | 0.0532 |

Interpretation:

- FT-Transformer is the strongest current model by mean C-index.
- XGBoost and LightGBM are close, suggesting the signal is robust across learner families.
- The raw PCA baseline is weaker and more variable, supporting MOFA-style factor features as a useful representation.

### Top cross-model SHAP features

Top features by average mean absolute SHAP across FT-Transformer, XGBoost, and LightGBM:

| Rank | Feature | Interpretation |
|---:|---|---|
| 1 | `age_at_dx` | Age at diagnosis is the strongest overall prognostic covariate. |
| 2 | `MOFA_Factor8` | Strong latent multi-omics prognostic factor. |
| 3 | `MOFA_Factor6` | Strong latent multi-omics prognostic factor. |
| 4 | `CT_prostate adenocarcinoma` | Cancer-type baseline prognosis signal. |
| 5 | `MOFA_Factor1` | High-variance latent factor with prognostic value. |
| 6 | `CT_breast invasive carcinoma` | Cancer-type baseline prognosis signal. |
| 7 | `MOFA_Factor15` | Prognostic latent factor. |
| 8 | `MOFA_Factor12` | Prognostic latent factor. |

### MOFA variance

Top factors by total variance across the three views:

| Factor | Total R2 across views |
|---|---:|
| Factor1 | 0.1827 |
| Factor4 | 0.1735 |
| Factor12 | 0.1575 |
| Factor9 | 0.1545 |
| Factor7 | 0.1398 |
| Factor11 | 0.1291 |
| Factor16 | 0.1091 |
| Factor8 | 0.0964 |

Interpretation:

- High variance explained is not the same as high survival importance.
- Factor8 is especially important because it has strong survival SHAP despite not being the top variance factor.

### Raw Cox / DeepSurv feature interpretation

From `results/shap_importance_summary.csv`, top average raw-model SHAP features include:

```text
EXPR_PC14
EXPR_PC9
EXPR_PC4
CNV_PC15
EXPR_PC50
EXPR_PC42
MUT_USH2A
EXPR_PC37
EXPR_PC46
MUT_CSMD1
MUT_LAMA1
```

Interpretation:

- Raw-feature models identify expression PCs as dominant.
- CNV_PC15 appears in both Cox and DeepSurv interpretation.
- Mutation features such as `USH2A`, `CSMD1`, and `LAMA1` appear as recurrent raw-branch signals, but these should be interpreted carefully because mutation burden and cancer-type confounding can influence pan-cancer mutation features.

### Marker / therapeutic target synthesis

Top rows in `results/therapeutic_targets.csv` include:

```text
TP53
TTN
RLN1
RLN2
INSL4
PDCD1LG2
CD274
RCL1
PLGRKT
JAK2
KIAA1432
KRT16
```

Interpretation:

- `TP53` is the strongest aggregated marker in the current table.
- Immune/checkpoint-linked genes such as `CD274` and `PDCD1LG2` appear in the ranked marker output.
- Many candidates are marked `Partial` or `Unknown`, meaning they are hypothesis-generating and require validation.
- `TTN` should be treated as a likely hypermutation / passenger-burden biomarker rather than a direct therapeutic target.

## 8. Major Findings To Preserve In Reports

1. Pan-cancer expression structure is strongly organized by cancer type and tissue lineage.
2. Leiden clustering provides more granular molecular communities than K-means.
3. Final survival modeling performs best with MOFA factors plus age and cancer-type context.
4. FT-Transformer currently has the best mean cross-validated C-index, but tree models are close.
5. Age at diagnosis is the single strongest cross-model SHAP feature.
6. MOFA_Factor8, Factor6, Factor1, Factor15, and Factor12 are important prognostic latent factors.
7. Marker synthesis nominates TP53 and several immune / epithelial / signaling candidates, but most target claims remain exploratory.
8. Current survival coverage is much larger than older report text claimed; use current CSVs.

## 9. Known Pitfalls And Agent Warnings

### Do not trust stale generated prose blindly

`build_report.py` and prior HTML reports contain useful structure, but some narrative values may be stale. Always recompute key counts from CSVs.

### Check whether MOFA+ was real MOFA or fallback NMF

`src/unsupervised/mofa_analysis.py` falls back to NMF if `mofapy2` is not installed. A future agent must verify the run log or environment before describing the factors as true MOFA+.

### Avoid leaking cancer type into biological overclaims

Cancer-type one-hot variables are strong predictors. A gene/factor may be prognostic because it captures lineage or tumor type rather than a pan-cancer causal survival mechanism.

### Beware mutation burden confounding

Genes such as `TTN` may reflect tumor mutation burden or gene length rather than a direct driver or therapeutic target.

### Be careful with survival labels

XGBoost / LightGBM survival labels use a signed convention in `model_comparison.py`:

```text
positive time = event
negative time = censored
```

Do not mix this with lifelines format, which keeps time and event as separate columns.

### Preserve sample ordering

Many downstream CSVs rely on aligned row order. When joining data, always align by sample barcode and check shapes before model fitting.

### Large files

Methylation data are very large. Do not casually load full methylation CSV into memory.

## 10. How To Validate A Rerun

After rerunning preprocessing:

```text
Check aligned matrix row counts match.
Check aligned matrix sample indexes are identical.
Check primary tumor suffix is -01.
Check OS_event counts are nonzero and plausible.
Check cancer type distribution resembles current results.
```

After rerunning UMAP / clustering:

```text
results/umap_coords.csv exists.
results/cluster_labels.csv exists.
UMAP figures render and are not blank.
Leiden labels contain a reasonable number of clusters.
```

After rerunning MOFA:

```text
results/mofa_factors.csv exists.
results/mofa_variance.csv exists.
results/mofa_weights_expression.csv exists.
results/mofa_weights_mutations.csv exists.
results/mofa_weights_cnv.csv exists.
Confirm whether mofapy2 or fallback NMF was used.
```

After rerunning survival models:

```text
results/model_cv_scores.csv exists.
Mean C-index should be comfortably above 0.5.
Fold-to-fold variance should not be extreme unless cohort/features changed.
Risk KM figure should show plausible separation.
```

After rerunning marker analysis:

```text
results/marker_shap_scores.csv exists.
results/factor_gene_table.csv exists.
results/therapeutic_targets.csv exists.
results/factor_biology_summary.txt exists.
Top targets should be reviewed for biological plausibility and confounding.
```

## 11. How To Extend The Project

### Add methylation

1. Convert methylation in chunks with `convert_to_csv.py`.
2. Add a methylation loader that avoids full in-memory loading unless resources allow.
3. Add a methylation view to MOFA.
4. Reevaluate model comparison and marker synthesis.

### Add miRNA or RPPA

1. Confirm CSV conversion exists.
2. Align sample barcodes to the same 15-character primary tumor IDs.
3. Add view-specific variance filtering.
4. Update `mofa_analysis.py` likelihoods and feature naming.
5. Update marker analysis if interpreting non-gene features.

### Improve survival modeling

Potential improvements:

- Cancer-type-stratified survival models.
- Leave-one-cancer-type-out validation.
- Time-dependent C-index or integrated Brier score.
- Competing endpoints such as PFI, DFI, or DSS from TCGA-CDR.
- Nested cross-validation for hyperparameter tuning.
- Explicit mutation burden covariate.

### Improve biological validity

Potential improvements:

- External validation on ICGC, CPTAC, or individual TCGA cancer cohorts.
- Pathway enrichment of top factor genes.
- Cancer-type-specific SHAP and factor decoding.
- Adjustment for tumor purity, stage, treatment, and batch effects where available.
- Compare marker candidates against CIViC, OncoKB, DGIdb, MSigDB, and DepMap.

## 12. Suggested Prompt For A New AI Agent

Use this prompt to initialize a future agent:

```text
You are working on the TCGA Pan-Cancer Multi-Omics project at:
C:\Users\debad\OneDrive\Documents\DC Academic Research\Cancer Research

Read PROJECT_KNOWLEDGE_MAP.md first. Then inspect current CSV outputs in results/
before trusting older reports. Your job is to preserve the data contracts:
aligned matrices are samples x features, indexed by 15-character TCGA primary
tumor sample barcodes. The main pipeline is download -> convert -> preprocess ->
UMAP/clustering -> MOFA -> Cox/DeepSurv -> model comparison -> SHAP ->
marker analysis -> report.

When making claims, cite current result tables and figures. Be cautious about
cancer-type confounding, mutation burden confounding, stale generated prose, and
whether mofa_analysis.py used real mofapy2 or its fallback NMF proxy.
```

## 13. Editable TODOs For Future Agents

- [ ] Verify the exact Python environment and dependency versions used for the final run.
- [ ] Confirm whether final `mofa_factors.csv` came from `mofapy2` or fallback NMF.
- [ ] Add a small `run_pipeline.ps1` or `Makefile` with reproducible commands.
- [ ] Add automated validation checks for aligned matrix shapes and sample indexes.
- [ ] Add a `results_manifest.md` listing every output and which script generated it.
- [ ] Rebuild `build_report.py` so it no longer contains stale hard-coded counts.
- [ ] Add cancer-type-stratified model evaluation.
- [ ] Add pathway enrichment for top factor genes.
- [ ] Add external validation plan.
- [ ] Add clear manuscript claims ranked by confidence level.

## 14. Current Important Outputs

| Output | Meaning |
|---|---|
| `results/aligned_expression.csv` | Final expression matrix. |
| `results/aligned_mutations.csv` | Final binary mutation matrix. |
| `results/aligned_cnv.csv` | Final CNV matrix. |
| `results/aligned_clinical.csv` | Final clinical / survival matrix. |
| `results/umap_coords.csv` | UMAP coordinates and annotations. |
| `results/cluster_labels.csv` | K-means and Leiden labels per sample. |
| `results/mofa_factors.csv` | Latent factor scores per sample. |
| `results/mofa_variance.csv` | Factor variance explained per omics view. |
| `results/mofa_weights_*.csv` | Feature weights per factor and modality. |
| `results/model_cv_scores.csv` | FT-Transformer / XGBoost / LightGBM CV scores. |
| `results/shap_importance_summary_models.csv` | Cross-model SHAP summary for final models. |
| `results/shap_importance_summary.csv` | Cox / DeepSurv raw feature SHAP summary. |
| `results/marker_shap_scores.csv` | Aggregated gene-level marker contributions. |
| `results/factor_gene_table.csv` | Factor-by-modality top genes. |
| `results/therapeutic_targets.csv` | Ranked marker / therapeutic target table. |
| `results/factor_biology_summary.txt` | Text summary of top factor biology. |
| `results/methodology_results_findings_2026.html` | Consolidated HTML report created from current outputs. |

## 15. Confidence Levels For Claims

Use these labels in future reports:

| Confidence | Appropriate claim type |
|---|---|
| High | Cohort counts, matrix dimensions, file outputs, model CV scores from CSVs. |
| Medium | Relative model ranking, SHAP feature importance, cluster/cancer-type associations. |
| Low to medium | Biological interpretation of MOFA factors without external validation. |
| Low | Therapeutic target claims, unless supported by external curated databases and cancer-specific validation. |

## 16. Quick Agent Checklist Before Answering User Questions

Before answering a scientific or modeling question:

- [ ] Did I check the relevant current CSV rather than relying on memory?
- [ ] Did I distinguish current outputs from stale report prose?
- [ ] Did I account for cancer-type confounding?
- [ ] Did I explain whether a result is predictive, associative, or mechanistic?
- [ ] Did I avoid overclaiming therapeutic actionability?
- [ ] Did I include exact file paths when pointing the user to artifacts?

