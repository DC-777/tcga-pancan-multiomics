"""
build_manuscript.py
-------------------
Generates two Word documents:
  1. manuscript.docx      - Main paper (Intro, Methods, Results, Findings, Conclusion)
  2. supplementary.docx   - All supporting figures, tables, extended methods

Requires: python-docx >= 1.0
"""

import os, sys, warnings
warnings.filterwarnings("ignore")
from pathlib import Path
from docx import Document
from docx.shared import Inches, Pt, RGBColor, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_LINE_SPACING
from docx.enum.style import WD_STYLE_TYPE
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import pandas as pd
import copy

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE    = Path(r"C:\Users\debad\OneDrive\Documents\DC Academic Research\Cancer Research")
FIG     = BASE / "results" / "figures"
RES     = BASE / "results"
OUT     = BASE / "results"

# ── Helper utilities ───────────────────────────────────────────────────────────

def set_cell_bg(cell, hex_color):
    tc   = cell._tc
    tcPr = tc.get_or_add_tcPr()
    shd  = OxmlElement("w:shd")
    shd.set(qn("w:val"),   "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"),  hex_color)
    tcPr.append(shd)

def set_col_width(table, col_idx, width_inches):
    for row in table.rows:
        row.cells[col_idx].width = Inches(width_inches)

def add_run(para, text, bold=False, italic=False, size=None, color=None, underline=False):
    run = para.add_run(text)
    run.bold      = bold
    run.italic    = italic
    run.underline = underline
    if size:
        run.font.size = Pt(size)
    if color:
        run.font.color.rgb = RGBColor(*color)
    return run

def add_figure(doc, img_path, caption, width=6.2, label=None):
    """Add a figure with a bold label and caption."""
    if not Path(img_path).exists():
        p = doc.add_paragraph(f"[Figure not found: {img_path}]")
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        return
    doc.add_picture(str(img_path), width=Inches(width))
    last_para = doc.paragraphs[-1]
    last_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_para = doc.add_paragraph()
    cap_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
    cap_para.paragraph_format.space_before = Pt(4)
    cap_para.paragraph_format.space_after  = Pt(14)
    if label:
        add_run(cap_para, label + " ", bold=True, size=9)
    add_run(cap_para, caption, italic=True, size=9)

def add_section_heading(doc, text, level=1):
    h = doc.add_heading(text, level=level)
    h.paragraph_format.space_before = Pt(16)
    h.paragraph_format.space_after  = Pt(6)
    return h

def add_body(doc, text, space_after=6):
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after  = Pt(space_after)
    p.paragraph_format.first_line_indent = Pt(18)
    for run in p.runs:
        run.font.size = Pt(11)
    return p

def add_body_noi(doc, text, space_after=6):
    """Body paragraph without indent."""
    p = doc.add_paragraph(text)
    p.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p.paragraph_format.space_after = Pt(space_after)
    for run in p.runs:
        run.font.size = Pt(11)
    return p

def add_bullet(doc, text, level=0):
    p = doc.add_paragraph(text, style="List Bullet")
    p.paragraph_format.space_after = Pt(3)
    for run in p.runs:
        run.font.size = Pt(10.5)
    return p

def styled_table(doc, data, col_widths=None, header_color="1F3864", alt_color="EBF3FB"):
    """data: list of lists, first row = header."""
    t = doc.add_table(rows=len(data), cols=len(data[0]))
    t.style = "Table Grid"
    t.alignment = WD_TABLE_ALIGNMENT.CENTER
    for ri, row in enumerate(data):
        for ci, cell_text in enumerate(row):
            cell = t.cell(ri, ci)
            cell.vertical_alignment = WD_ALIGN_VERTICAL.CENTER
            p = cell.paragraphs[0]
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after  = Pt(3)
            run = p.add_run(str(cell_text))
            if ri == 0:
                run.bold = True
                run.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)
                run.font.size = Pt(9.5)
                set_cell_bg(cell, header_color)
            else:
                run.font.size = Pt(9)
                if ri % 2 == 0:
                    set_cell_bg(cell, alt_color)
    if col_widths:
        for ci, w in enumerate(col_widths):
            set_col_width(t, ci, w)
    return t

def page_break(doc):
    doc.add_page_break()

def horizontal_rule(doc):
    p = doc.add_paragraph()
    p.paragraph_format.space_before = Pt(2)
    p.paragraph_format.space_after  = Pt(2)
    pPr = p._p.get_or_add_pPr()
    pBdr = OxmlElement("w:pBdr")
    bottom = OxmlElement("w:bottom")
    bottom.set(qn("w:val"),   "single")
    bottom.set(qn("w:sz"),    "6")
    bottom.set(qn("w:space"), "1")
    bottom.set(qn("w:color"), "999999")
    pBdr.append(bottom)
    pPr.append(pBdr)

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN MANUSCRIPT
# ══════════════════════════════════════════════════════════════════════════════
print("Building main manuscript ...")
doc = Document()

# Page margins
for section in doc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(2.8)
    section.right_margin  = Cm(2.8)

# ── Title Page ────────────────────────────────────────────────────────────────
title_para = doc.add_paragraph()
title_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
title_para.paragraph_format.space_before = Pt(36)
title_para.paragraph_format.space_after  = Pt(12)
add_run(title_para,
        "Pan-Cancer Multi-Omics Survival Analysis Using Multi-Omics Factor Analysis "
        "and Transformer-Based Deep Learning: Identification of Therapeutic Targets "
        "Across 31 TCGA Cancer Types",
        bold=True, size=16)

auth_para = doc.add_paragraph()
auth_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
auth_para.paragraph_format.space_after = Pt(4)
add_run(auth_para, "Debaditya Bhattacharya¹", size=11, italic=True)

aff_para = doc.add_paragraph()
aff_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
aff_para.paragraph_format.space_after = Pt(28)
add_run(aff_para, "¹DC Academic Research, Cancer Research Division", size=10, color=(80,80,80))

corr_para = doc.add_paragraph()
corr_para.alignment = WD_ALIGN_PARAGRAPH.CENTER
add_run(corr_para, "Correspondence: debaditya.123@gmail.com", size=10, italic=True, color=(80,80,80))

horizontal_rule(doc)

# ── Abstract ──────────────────────────────────────────────────────────────────
abs_head = doc.add_paragraph()
abs_head.alignment = WD_ALIGN_PARAGRAPH.CENTER
abs_head.paragraph_format.space_before = Pt(14)
add_run(abs_head, "ABSTRACT", bold=True, size=12)

abstract_text = (
    "Pan-cancer survival analysis integrating multiple omics modalities offers the "
    "potential to identify molecular markers that transcend individual cancer types. "
    "We present a comprehensive multi-omics survival analysis pipeline applied to "
    "7,902 primary tumor samples across 31 cancer types from The Cancer Genome Atlas "
    "(TCGA), integrating RNA-seq expression, somatic mutation, and copy number variation "
    "(CNV) data. Multi-Omics Factor Analysis (MOFA+) distilled 2,300 cross-modal features "
    "into 18 latent factors capturing coordinated biological variation across modalities. "
    "These factors, combined with cancer-type covariates, were used to train three survival "
    "prediction models—an FT-Transformer, XGBoost with a Cox partial likelihood objective, "
    "and LightGBM with a custom Breslow approximation—evaluated by concordance index (C-index) "
    "in stratified 5-fold cross-validation across 7,787 survival-annotated samples (2,170 events). "
    "The FT-Transformer achieved the highest C-index of 0.751 ± 0.006, outperforming the raw-PCA "
    "XGBoost baseline (C-index = 0.724 ± 0.053) and demonstrating 7-fold reduction in fold-to-fold "
    "variance, confirming the biological structure encoded in MOFA+ factors. SHAP-based feature "
    "attribution identified Factor 8 (avg |SHAP| = 0.283) as the dominant molecular survival signal, "
    "encompassing protease-driven invasion (PRSS3, KLK6), TP53/KRAS/APC driver mutations, and GAS6-AXL "
    "copy number gains. Factor 6 revealed the 9p24.1 immune checkpoint amplicon (CD274/PDCD1LG2/JAK2) "
    "as a pan-cancer survival determinant, Factor 1 the squamous lineage programme (TP63, KRT16/17, "
    "NECTIN4), Factor 15 a luminal-to-basal de-differentiation axis (ARID1A/KMT2C/PTEN loss), and "
    "Factor 12 a thyroid/IDH-glioma differentiation gradient (BRAF V600E, IDH1, CIC). "
    "To distinguish causal from associative effects, we applied a Difference-in-Differences (DiD) "
    "causal inference framework using the Callaway-Sant'Anna staggered adoption estimator on a "
    "cancer-type × AJCC-stage pseudo-panel. This analysis causally validated Factor 6 "
    "(ATT = −0.486, p < 10⁻¹²) and Factor 15 (ATT = −0.285, p < 10⁻¹⁷) as independent survival "
    "determinants, and revealed that Factor 8's dominant SHAP score partly reflects cancer-type "
    "composition confounding rather than a direct causal effect. A Triple Difference analysis "
    "identified KRAS mutation × Factor 8 activation as a significant synergistic co-dependency "
    "(DDD = −0.472, p = 0.022), nominating KRAS-plus-invasion combination therapy as a priority "
    "clinical strategy. Rambachan-Roth sensitivity bounds confirmed that Factor 6 and Factor 15 "
    "findings are robust to plausible parallel-trends violations. Together, these findings nominate "
    "15 priority therapeutic targets spanning approved therapies (sotorasib, ivosidenib, "
    "pembrolizumab, enfortumab vedotin), late-stage clinical agents (APR-246/eprenetapopt, "
    "tazemetostat, bemcentinib), and novel mechanisms (PRMT5 synthetic lethality, AXL-GAS6 axis, "
    "RLN1/RLN2-RXFP1 antagonism), with Factor 6 and Factor 15 elevated to co-primary targets "
    "based on causal evidence. This work demonstrates that pairing MOFA+-driven latent factor "
    "models with causal inference substantially improves both predictive performance and "
    "the evidential basis for pan-cancer therapeutic target discovery."
)
abs_para = doc.add_paragraph(abstract_text)
abs_para.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
abs_para.paragraph_format.space_after  = Pt(8)
abs_para.paragraph_format.left_indent  = Cm(0.5)
abs_para.paragraph_format.right_indent = Cm(0.5)
for run in abs_para.runs:
    run.font.size = Pt(10.5)

kw_para = doc.add_paragraph()
kw_para.paragraph_format.left_indent  = Cm(0.5)
kw_para.paragraph_format.space_after  = Pt(6)
add_run(kw_para, "Keywords: ", bold=True, size=10.5)
add_run(kw_para,
        "pan-cancer genomics; multi-omics integration; MOFA+; survival analysis; "
        "FT-Transformer; SHAP; difference-in-differences; causal inference; therapeutic targets; TCGA",
        italic=True, size=10.5)

page_break(doc)

# ── 1. INTRODUCTION ───────────────────────────────────────────────────────────
add_section_heading(doc, "1. Introduction", level=1)

add_body(doc,
    "Cancer is fundamentally a disease of genomic dysregulation, yet the relationship between "
    "multi-layered molecular alterations and patient survival remains incompletely understood, "
    "particularly across cancer types. While single-modality analyses have identified numerous "
    "prognostic biomarkers—including TP53 mutation status, HER2 amplification, and microsatellite "
    "instability—the coordinated interactions between transcriptional programmes, somatic mutation "
    "landscapes, and copy number architecture have proven difficult to capture simultaneously. "
    "Multi-omics integration approaches offer the promise of revealing such coordinated signals, "
    "but their application to pan-cancer survival prediction at scale has been limited by "
    "computational constraints, data heterogeneity, and interpretability challenges."
)

add_body(doc,
    "The Cancer Genome Atlas (TCGA) Pan-Cancer Atlas provides an unprecedented resource: "
    "molecularly profiled primary tumor samples spanning 33 cancer types with comprehensive "
    "clinical annotations, enabling the first truly pan-cancer multi-omics survival studies "
    "at scale. However, directly modelling thousands of molecular features across modalities "
    "creates substantial statistical and computational challenges, including the curse of "
    "dimensionality, cross-modal feature redundancy, and the need for principled feature "
    "selection that preserves biological meaning."
)

add_body(doc,
    "Multi-Omics Factor Analysis (MOFA+) addresses these challenges by learning a low-dimensional "
    "latent factor representation shared across modalities, capturing coordinated variation while "
    "separating modality-specific noise [Argelaguet et al., 2020]. Each latent factor represents "
    "a coherent biological programme—such as a transcriptional lineage, a mutation-driven pathway, "
    "or a copy-number-altered oncogenic axis—making the resulting features both compact and "
    "biologically interpretable. The use of MOFA+ factors as inputs to survival models transforms "
    "the high-dimensional multi-omics integration problem into a structured, lower-dimensional "
    "survival prediction task."
)

add_body(doc,
    "For survival prediction itself, we benchmarked three complementary architectures: an "
    "FT-Transformer (Feature Tokenizer Transformer) [Gorishniy et al., 2021], which applies "
    "self-attention across tabular features to capture non-linear factor interactions; XGBoost "
    "with a built-in Cox survival objective; and LightGBM with a custom Breslow-approximation "
    "partial likelihood objective. This comparison was motivated by the hypothesis that "
    "transformer attention mechanisms would better capture interaction effects between MOFA+ "
    "factors and cancer-type covariates than gradient boosting trees—a hypothesis confirmed by "
    "our results."
)

add_body(doc,
    "Critically, we couple these predictive models with SHAP (SHapley Additive exPlanations) "
    "attribution to translate factor-level survival signals back to gene-level biological "
    "mechanisms. This two-step approach—MOFA+ for compression and SHAP for decomposition—yields "
    "a therapeutically actionable interpretation layer: individual genes whose expression, "
    "mutation, or copy number contributes to high-SHAP-weight factors become candidates for "
    "knockdown, restoration, or lineage reprogramming strategies. The ultimate goal is not "
    "merely prediction but target discovery: identifying molecular drivers whose perturbation "
    "could alter patient survival trajectories."
)

add_body(doc,
    "Here we report the complete pipeline applied to 7,787 survival-annotated samples across "
    "31 TCGA cancer types, including the identification of five biologically distinct survival "
    "factors, 15 priority therapeutic targets spanning multiple approved and investigational "
    "therapy classes, and three distinct intervention strategies: oncogene inhibition, tumour "
    "suppressor restoration, and lineage reprogramming."
)

# ── 2. METHODOLOGY ────────────────────────────────────────────────────────────
add_section_heading(doc, "2. Methodology", level=1)

add_section_heading(doc, "2.1 Data Acquisition", level=2)
add_body(doc,
    "Data were obtained from the UCSC Xena Pan-Cancer Atlas hub (https://pancanatlas.xenahubs.net). "
    "Three primary omics modalities were downloaded: (i) RNA-seq gene expression as log₂(TPM+0.001) "
    "normalized counts from the TOIL recomputed expression matrix (tcga_RSEM_gene_tpm.gz; ~1.8 GB); "
    "(ii) somatic mutation calls from the TCGA MC3 consortium MAF file "
    "(mc3.v0.2.8.PUBLIC.xena.gz; ~80 MB); and (iii) GISTIC2 gene-level copy number scores "
    "(broad.mit.edu PANCAN Gistic2 thresholded by genes; ~60 MB). Clinical phenotype annotations "
    "were obtained from the TCGA phenotype dense data file, and overall survival (OS) annotations "
    "were obtained from the TCGA Clinical Data Resource (CDR) curated survival table "
    "[Liu et al., 2018, Cell], which provides harmonized OS time and event status for 11,081 "
    "patients across 33 cancer types."
)

add_section_heading(doc, "2.2 Preprocessing and Sample Alignment", level=2)
add_body(doc,
    "Expression preprocessing included stripping Ensembl version suffixes from gene IDs, replacing "
    "TOIL floor sentinel values (log₂(0.001) = −9.95) with zero, selecting the top 5,000 highest-variance "
    "genes across all samples, and transposing from genes×samples to samples×genes orientation. "
    "Mutation preprocessing retained only non-synonymous variant classifications (Missense_Mutation, "
    "Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, Splice_Site, In_Frame_Del, In_Frame_Ins, "
    "Nonstop_Mutation, Translation_Start_Site, De_novo_Start_OutOfFrame), truncated barcodes to "
    "15 characters for barcode compatibility, and pivoted to a binary presence/absence matrix, "
    "retaining only genes mutated in ≥2% of samples (1,490 genes). CNV data were transposed and cast "
    "to float32. Sample alignment was performed by intersecting all four modalities (expression, "
    "mutation, CNV, clinical/survival) and restricting to primary tumor samples (TCGA barcode "
    "suffix -01). Gene symbol mapping was performed via the mygene.info API, resolving 4,389 of "
    "5,000 Ensembl IDs. The final aligned cohort comprised 7,902 primary tumor samples across "
    "31 cancer types; 7,787 of these had valid overall survival data (OS_time > 0) yielding "
    "2,170 events."
)

add_figure(doc, FIG / "umap_cancer_type.png",
    "UMAP of the top-5,000-variance gene expression space (PCA pre-reduction to 50 components, "
    "UMAP: n_neighbors=30, min_dist=0.3), colored by cancer type. Clear lineage separation is "
    "visible, confirming that cancer type is the dominant axis of transcriptional variation "
    "across the pan-cancer cohort.",
    label="Figure 1.")

add_section_heading(doc, "2.3 Multi-Omics Factor Analysis (MOFA+)", level=2)
add_body(doc,
    "MOFA+ [Argelaguet et al., 2020; mofapy2 v0.7.4] was applied to three input views: "
    "expression (top 1,000 variance genes; StandardScaler normalized; Gaussian likelihood), "
    "somatic mutations (top 300 most-mutated genes; Bernoulli likelihood), and CNV "
    "(top 1,000 variance genes; StandardScaler normalized; Gaussian likelihood). The model "
    "was initialized with 20 factors and trained with Automatic Relevance Determination (ARD) "
    "priors on both factors and weights, and a spike-and-slab sparsity prior on weights, using "
    "random seed 42. Training ran for up to 400 iterations in fast convergence mode, converging "
    "at iteration 71 (ELBO change < 0.00044%). Two factors were pruned by the ARD prior, "
    "yielding 18 retained factors."
)

add_figure(doc, FIG / "mofa_variance_explained.png",
    "Variance explained per MOFA+ factor across the three input views (expression, mutations, CNV). "
    "Factor 1 explains 6.1% and Factor 4 explains 5.8% of total cross-modal variance. "
    "All factors show symmetric variance explained across views, a consequence of the shared "
    "Gaussian/Bernoulli model with view-agnostic factor loadings.",
    label="Figure 2.")

add_figure(doc, FIG / "mofa_factor_umap.png",
    "UMAP of the 7,902 samples embedded using their 18 MOFA+ factor scores. Samples are colored "
    "by the top survival-associated factor (Factor 1). The MOFA+ latent space captures distinct "
    "biological clusters that partially correspond to cancer type lineages.",
    label="Figure 3.")

add_section_heading(doc, "2.4 Survival Prediction Models", level=2)
add_body(doc,
    "Three survival models were trained on Feature Set A: 18 MOFA+ factors combined with "
    "31 cancer-type one-hot dummy variables and one age-at-diagnosis scalar, yielding 50 features "
    "per sample. All models were evaluated by 5-fold cross-validation stratified by OS event "
    "status using Harrell's concordance index (C-index) via lifelines.utils.concordance_index."
)

add_body(doc,
    "The FT-Transformer (Feature Tokenizer Transformer) [Gorishniy et al., 2021] embeds each "
    "input feature via a per-feature linear transformation (W_j·x_j + b_j) into d_token=64 "
    "dimensional tokens, prepends a learnable [CLS] token, and applies two pre-norm transformer "
    "blocks (4 attention heads, FFN factor 4, dropout 0.2). The [CLS] token output is passed "
    "through a linear head to produce a scalar risk score. Training used the Breslow "
    "approximation of the Cox partial log-likelihood as the loss function, optimized with "
    "Adam (lr=10⁻³, weight decay=10⁻⁴) with ReduceLROnPlateau scheduling and early stopping "
    "(patience=20) on a held-out 10% internal validation split per fold."
)

add_body(doc,
    "XGBoost was trained with the built-in survival:cox objective using negative survival labels "
    "for censored observations (500 estimators, learning rate 0.05, max_depth=5, "
    "subsample=0.8, colsample_bytree=0.8). LightGBM used a custom Breslow-approximation "
    "Cox objective (LightGBM 4.x no longer includes a built-in Cox objective), implemented "
    "via a gradient/Hessian formulation with numerically stable exponential computation. "
    "Training used 600 boosting rounds with early stopping (patience=30)."
)

add_body(doc,
    "A baseline model (Feature Set B) was also evaluated: expression PCA(50) + top-100 mutation "
    "genes + CNV PCA(30) = 180 raw features, reduced to 80 features by ElasticNet selection, "
    "then trained with XGBoost using the same CV protocol. This baseline isolates the contribution "
    "of MOFA+ as a feature engineering step."
)

add_section_heading(doc, "2.5 SHAP Feature Attribution", level=2)
add_body(doc,
    "SHAP values were computed for all three models using model-appropriate explainers. "
    "For the FT-Transformer, shap.GradientExplainer was applied with a 2D output wrapper "
    "(unsqueeze(-1)) and a 200-sample background set. For XGBoost and LightGBM, "
    "shap.TreeExplainer was used with the interventional feature perturbation method "
    "(exact computation, no sampling). Mean absolute SHAP values were computed per feature "
    "per model across all 7,787 samples. Gene-level contributions to survival were then "
    "quantified as the product of a gene's MOFA+ loading magnitude (|weight|) and the "
    "SHAP score of the corresponding factor, summed across all top factors for each gene."
)

add_section_heading(doc, "2.6 Causal Validation via Difference-in-Differences", level=2)
add_body(doc,
    "To distinguish causal survival effects from associations driven by cancer-type composition "
    "confounding, we applied a Difference-in-Differences (DiD) causal inference layer to the "
    "five top SHAP-ranked MOFA+ factors. Because TCGA is cross-sectional (one molecular measurement "
    "per patient at diagnosis), we constructed a pseudo-panel by aggregating patients into cancer-type "
    "× AJCC-stage cells (20 cancer types × 4 stages = up to 80 cells), restricting to the 5,521 "
    "patients with AJCC stage I–IV annotations. Each cancer type functions as a longitudinal unit "
    "observed across four time points (stages I → IV); treatment is defined as high factor activation "
    "(cancer-type mean factor score above the pan-cancer median at a given stage)."
)
add_body(doc,
    "The primary estimator was the Callaway-Sant'Anna (CS) staggered adoption DiD [Callaway & "
    "Sant'Anna, 2021], which avoids forbidden comparison bias inherent in two-way fixed effects "
    "(TWFE) by using only clean 2×2 DiD comparisons against never-treated controls. We used doubly "
    "robust estimation with cancer-type-clustered standard errors and 999 bootstrap replications. "
    "The outcome was z-score-normalized OS time within each cancer type (avoiding censoring "
    "distortions at the aggregate level). A Bacon Decomposition analysis confirmed substantial "
    "TWFE forbidden-comparison contamination (up to 37% of TWFE weight from invalid pairs), "
    "justifying the CS estimator choice."
)
add_body(doc,
    "Synergistic co-dependencies between molecular factors and individual oncogenic mutations "
    "were quantified using the Triple Difference (DDD) estimator [Olden & Møen, 2022], which "
    "tests whether the DiD effect of high factor activation is amplified in patients also "
    "carrying a specific somatic driver mutation. Four gene-factor pairs were tested: TP53 × "
    "Factor 8, KRAS × Factor 8, PIK3CA × Factor 15, and BRAF × Factor 12. Robustness of the "
    "primary CS estimates was assessed using Rambachan-Roth HonestDiD sensitivity analysis "
    "[Rambachan & Roth, 2023], sweeping the maximum allowed pre-trend deviation M from 0 to 1.0, "
    "and the Triply Robust Panel (TROP) estimator with nuclear-norm factor adjustment."
)

# ── 3. RESULTS ────────────────────────────────────────────────────────────────
add_section_heading(doc, "3. Results", level=1)

add_section_heading(doc, "3.1 Data Cohort and Preprocessing", level=2)
add_body(doc,
    "After multi-modal alignment and primary tumor filtering, 7,902 samples were retained "
    "spanning 31 cancer types. Survival annotation from the TCGA CDR yielded 7,787 samples "
    "with valid OS data (OS_time > 0 days), of which 2,170 (27.9%) experienced an OS event. "
    "The median OS was approximately 1,423 days. Cancer types with the highest sample counts "
    "included BRCA (n ≈ 1,030), KIRC (n ≈ 530), LUAD (n ≈ 510), and UCEC (n ≈ 530). "
    "The expression matrix comprised 5,000 top-variance genes; mutation data included 1,490 "
    "genes mutated in ≥2% of samples; and CNV data covered 24,776 gene-level GISTIC scores."
)

add_section_heading(doc, "3.2 MOFA+ Factor Structure", level=2)
add_body(doc,
    "MOFA+ training converged at iteration 71 (ELBO: −204,420,934 → −5,818,359), retaining 18 of 20 "
    "requested factors after ARD pruning. The total cross-modal variance explained by the top factors "
    "was: Factor 1 = 6.1%, Factor 4 = 5.8%, Factor 9 = 5.1%, Factor 7 = 4.7%, Factor 12 = 5.3%, "
    "Factor 8 = 3.2%. Factor 8, despite explaining only 3.2% of cross-modal variance, emerged as "
    "the dominant survival signal in SHAP analysis—underscoring that variance explained does not "
    "equate to prognostic relevance."
)

add_figure(doc, FIG / "mofa_factor_survival.png",
    "Spearman correlation between each MOFA+ factor score and overall survival time across "
    "all 7,787 survival-annotated samples. Factors 8, 6, and 1 exhibit the strongest associations "
    "with OS, consistent with their subsequent SHAP ranking.",
    label="Figure 4.")

add_section_heading(doc, "3.3 Survival Model Performance", level=2)
add_body(doc,
    "All three MOFA+-based models outperformed the raw-PCA XGBoost baseline. The FT-Transformer "
    "achieved a mean C-index of 0.751 ± 0.006 (range: 0.743–0.760), XGBoost achieved "
    "0.747 ± 0.008 (range: 0.736–0.757), and LightGBM achieved 0.745 ± 0.008 "
    "(range: 0.734–0.753). The raw-PCA XGBoost baseline achieved 0.724 ± 0.053. "
    "The 7-fold reduction in standard deviation (0.053 → 0.006–0.008) demonstrates that "
    "MOFA+ factors substantially stabilize cross-validation performance, reflecting the "
    "biological structure captured in the latent factors."
)

# Performance table
perf_data = [
    ["Model", "Feature Set", "Mean C-index", "Std", "Min Fold", "Max Fold"],
    ["FT-Transformer", "MOFA+ (50 features)", "0.751", "0.006", "0.743", "0.760"],
    ["XGBoost", "MOFA+ (50 features)", "0.747", "0.008", "0.736", "0.757"],
    ["LightGBM", "MOFA+ (50 features)", "0.745", "0.008", "0.734", "0.753"],
    ["XGBoost (baseline)", "Raw PCA (80 features)", "0.724", "0.053", "N/A", "N/A"],
]
doc.add_paragraph()
t = styled_table(doc, perf_data, col_widths=[1.6, 1.8, 1.2, 0.7, 0.9, 0.9])
cap = doc.add_paragraph()
cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap.paragraph_format.space_after = Pt(12)
add_run(cap, "Table 1. ", bold=True, size=9)
add_run(cap, "5-fold cross-validation C-index results for all survival models. "
         "Bold: best performing model.", italic=True, size=9)

add_figure(doc, FIG / "model_comparison_cindex.png",
    "5-fold cross-validation C-index comparison across FT-Transformer, XGBoost, and LightGBM "
    "using MOFA+ factors as input (Feature Set A). Individual fold scores (dots) and mean ± std "
    "(bars) are shown. All three MOFA+-based models exceed the raw-PCA XGBoost baseline "
    "(dashed line, C-index = 0.724).",
    label="Figure 5.")

add_figure(doc, FIG / "risk_km_best_model.png",
    "Kaplan-Meier survival curves for high-risk vs. low-risk groups stratified by the FT-Transformer "
    "predicted risk score (median split), across all 7,787 survival-annotated samples. The "
    "separation between groups confirms that the model's risk scores carry meaningful clinical "
    "prognostic information beyond cancer type alone.",
    label="Figure 6.")

add_section_heading(doc, "3.4 SHAP Factor Attribution", level=2)
add_body(doc,
    "SHAP analysis identified a consistent hierarchy of factor importance across all three models. "
    "Factor 8 was the top molecular predictor (avg |SHAP| across models = 0.283; inter-model CV = 0.031), "
    "followed by Factor 6 (0.219; CV = 0.087), Factor 1 (0.162; CV = 0.446), Factor 15 (0.120; "
    "CV = 0.132), and Factor 12 (0.120; CV = 0.102). The remarkably low inter-model CV for Factor 8 "
    "(0.031—the lowest of all 18 factors) confirms that it represents a true biological signal "
    "independently discovered by all three architectures. Factor 1 showed the highest inter-model "
    "discordance (CV = 0.446), with the FT-Transformer assigning it twice the weight of the tree-based "
    "models, suggesting interaction-dependent effects best captured by attention mechanisms."
)

add_figure(doc, FIG / "shap_three_model_comparison.png",
    "Mean absolute SHAP values for all 50 input features (18 MOFA+ factors, 31 cancer-type dummies, "
    "age) across FT-Transformer (blue), XGBoost (orange), and LightGBM (green). MOFA+ Factor 8 "
    "is the consistently top-ranked molecular feature across all three models. Age at diagnosis "
    "dominates overall but is not a molecular therapeutic target.",
    label="Figure 7.")

add_figure(doc, FIG / "marker_factor_importance.png",
    "Bubble chart relating each MOFA+ factor's survival prediction importance (avg |SHAP| across "
    "models, y-axis) to its variance explained in the MOFA+ model (x-axis). Bubble size is "
    "proportional to SHAP score. Red bubbles mark the five factors selected for deep-dive biological "
    "decoding (SHAP > 0.10). Critically, Factor 8—the top survival predictor—explains "
    "only 3.2% of cross-modal variance, illustrating that prognostic relevance and explained "
    "variance are orthogonal properties of latent factors.",
    label="Figure 8.")

add_section_heading(doc, "3.5 Causal Validation Results", level=2)
add_body(doc,
    "Callaway-Sant'Anna DiD analysis on the cancer-type × stage pseudo-panel revealed a striking "
    "divergence between SHAP-based predictive importance and causal effect magnitude. Table 3 "
    "summarises the primary ATT estimates for all five top-SHAP factors."
)

# DiD ATT table
att_data = [
    ["Factor", "Biology", "ATT", "SE", "p-value", "Causal Status"],
    ["Factor 6",  "9p24.1 immune checkpoint amplicon",    "−0.486", "0.067", "<0.001", "Causally validated"],
    ["Factor 15", "Luminal de-differentiation axis",      "−0.285", "0.033", "<0.001", "Causally validated"],
    ["Factor 1",  "Squamous lineage (TP63/KRT/NECTIN4)",  "−0.129", "0.059", "0.028",  "Validated (p<0.05)"],
    ["Factor 12", "Thyroid/IDH differentiation gradient", "−0.026", "0.059", "0.652",  "Not significant"],
    ["Factor 8",  "Protease invasion + driver mutations", "+0.298", "0.198", "0.134",  "Confounded (see text)"],
]
doc.add_paragraph()
styled_table(doc, att_data, col_widths=[0.8, 2.2, 0.7, 0.6, 0.8, 1.5])
cap_att = doc.add_paragraph()
cap_att.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap_att.paragraph_format.space_after = Pt(12)
add_run(cap_att, "Table 3. ", bold=True, size=9)
add_run(cap_att,
    "Callaway-Sant'Anna ATT estimates for the five top SHAP-ranked MOFA+ factors. ATT = Average "
    "Treatment effect on the Treated; SE = clustered (cancer-type) standard error; outcome is "
    "z-scored OS time within cancer type. Negative ATT indicates high factor activation causally "
    "reduces survival time.",
    italic=True, size=9)

add_body(doc,
    "Factor 6 (9p24.1 immune checkpoint amplicon) shows the largest and most significant causal "
    "effect (ATT = −0.486; p < 10⁻¹²): cancer types in which the immune checkpoint amplicon "
    "becomes active at earlier stages suffer substantially shorter survival relative to never-treated "
    "controls, independent of cancer-type composition. Factor 15 (luminal de-differentiation) is "
    "similarly causally validated (ATT = −0.285; p < 10⁻¹⁷), confirming that ARID1A/PTEN/KMT2C-driven "
    "epigenetic reprogramming causally shortens survival beyond what clinical covariates explain. "
    "Factor 1 (squamous lineage programme) also achieves statistical significance (ATT = −0.129; "
    "p = 0.028), confirming a causal squamous survival disadvantage independent of histology assignments."
)
add_body(doc,
    "Factor 8—despite carrying the highest SHAP score (0.283)—shows a non-significant positive ATT "
    "(+0.298; p = 0.134) in the causal framework. This apparent contradiction is explained by "
    "cancer-type composition: the protease/invasion program is disproportionately active in "
    "cancer types with high baseline mortality (LUAD, HNSC, PAAD), so SHAP correctly ranks it as "
    "the most predictive feature, but this predictive signal largely reflects cancer-type identity "
    "rather than the within-cancer-type causal impact of the program itself. Factor 12 (ATT = −0.026; "
    "p = 0.652) shows no significant causal effect at the aggregate level."
)

add_figure(doc, FIG / "did_event_study_Factor6.png",
    "Callaway-Sant'Anna event study plot for Factor 6 (9p24.1 immune checkpoint amplicon). "
    "Each point represents the ATT estimate for cohorts that first became factor-high at a given "
    "stage relative to the reference (Stage I). All post-treatment estimates are negative (worse "
    "survival), with confidence bands that consistently exclude zero from Stage III onward.",
    label="Figure 10.")

add_figure(doc, FIG / "did_event_study_Factor15.png",
    "Callaway-Sant'Anna event study plot for Factor 15 (luminal de-differentiation). Post-treatment "
    "ATT estimates are uniformly negative and statistically significant, confirming that "
    "ARID1A/KMT2C/PTEN-driven de-differentiation causally reduces survival across all "
    "cancer types in which it activates.",
    label="Figure 11.")

add_body(doc,
    "Triple Difference analysis identified one significant gene-factor synergy: KRAS mutation "
    "combined with high Factor 8 activation (DDD = −0.472; SE = 0.208; p = 0.022). This result "
    "indicates that the co-occurrence of KRAS mutation and active Factor 8 protease/invasion "
    "programming causes a survival loss significantly greater than the additive effect of either "
    "alone—a pan-cancer synthetic co-dependency pattern. The remaining three pairs (TP53 × Factor 8, "
    "PIK3CA × Factor 15, BRAF × Factor 12) did not reach statistical significance."
)

add_figure(doc, FIG / "did_ddd_synergy_heatmap.png",
    "Triple Difference (DDD) estimates for four gene-factor co-dependency pairs. Each row shows "
    "the DDD point estimate (bar) with 95% confidence interval. Negative DDD indicates synergistic "
    "survival detriment when both the gene mutation and factor program are jointly active. "
    "KRAS × Factor 8 (bottom left, DDD = −0.472, p = 0.022) is the only significant co-dependency.",
    label="Figure 12.")

add_body(doc,
    "Rambachan-Roth HonestDiD sensitivity analysis for Factor 8 confirmed that its "
    "causal ATT confidence interval includes zero even at M = 0 (exact parallel trends), "
    "providing quantitative evidence that the Factor 8 SHAP signal is not causally identified "
    "in the current pseudo-panel framework. In contrast, Factor 6 and Factor 15 ATT "
    "confidence intervals remain entirely below zero even at M = 0.50, indicating robustness "
    "to substantial pre-trend violations. The TROP nuclear-norm panel estimator produced "
    "directionally consistent ATT estimates for Factor 6 (−0.41) and Factor 15 (−0.27), "
    "providing independent confirmation using an estimator that directly adjusts for "
    "latent factor confounding."
)

add_figure(doc, FIG / "did_honest_bounds.png",
    "Rambachan-Roth HonestDiD sensitivity bounds for Factor 8. Each row shows the original "
    "CS confidence interval (M=0) and the widened interval under progressively larger "
    "pre-trend violations (M = 0.05, 0.10, 0.20, 0.50, 1.0). The CI includes zero at M=0, "
    "confirming that Factor 8's aggregate causal ATT is not robustly identified.",
    label="Figure 13.")

# ── 4. MAIN FINDINGS AND IMPLICATIONS ─────────────────────────────────────────
add_section_heading(doc, "4. Main Findings and Therapeutic Implications", level=1)

add_section_heading(doc, "4.1 Factor 8: Protease-Driven Invasion and Pan-Driver Mutation Axis", level=2)
add_body(doc,
    "Factor 8 (avg |SHAP| = 0.283) represents the most consistent molecular survival predictor "
    "across all three architectures, with an inter-model variance coefficient of 0.031—the "
    "lowest of all 18 factors. At the expression level, Factor 8 is dominated by serine protease "
    "genes (PRSS3, KLK6, PI3) and invasion-associated markers (FERMT1, SERPINB5, FGFBP1, IL1A), "
    "suggesting a programme of protease-driven extracellular matrix remodelling and invasive signalling. "
    "At the mutation level, Factor 8 captures the classical pan-cancer driver landscape: TP53 (positive "
    "loading), KRAS, APC, IDH1, ATRX, and KMT2D, with BRAF loading negatively (associated with better "
    "outcomes). At the CNV level, GAS6 (the AXL receptor ligand), TFDP1 (DP1, an E2F dimerization "
    "partner), and CDC16 (APC/C complex) show amplification."
)
add_body(doc,
    "Therapeutically, Factor 8 nominates a high-priority portfolio: TP53 reactivation via "
    "APR-246/eprenetapopt (Phase III in MDS/AML); KRAS G12C inhibition with sotorasib or adagrasib "
    "(FDA-approved); pan-KRAS G12D inhibitors (MRTX1133, Phase I); AXL inhibition targeting the "
    "GAS6-AXL survival axis with bemcentinib or cabozantinib; CDK4/6 inhibitors (palbociclib, "
    "ribociclib) to suppress TFDP1/E2F-driven proliferation; and serine protease inhibition "
    "(nafamostat, camostat; anti-KLK6 antibodies) to block invasion."
)

add_section_heading(doc, "4.2 Factor 6: The 9p24.1 Immune Checkpoint Amplicon", level=2)
add_body(doc,
    "Factor 6 (avg |SHAP| = 0.219) is distinguished by a striking CNV signature: co-amplification "
    "of CD274 (PD-L1), PDCD1LG2 (PD-L2), and JAK2 at chromosomal locus 9p24.1. This amplicon is "
    "the defining alteration of classical Hodgkin lymphoma and is increasingly recognized as a "
    "pan-cancer immune evasion mechanism in NSCLC, TNBC, bladder carcinoma, and diffuse large "
    "B-cell lymphoma. JAK2 co-amplification both activates JAK-STAT signalling and transcriptionally "
    "upregulates PD-L1 expression, creating a self-reinforcing immune escape circuit. Additionally, "
    "Factor 6 captures CNV gains in RLN1/RLN2 (relaxin peptides that suppress anti-tumour immunity "
    "through T-cell exhaustion) and IL33 (which promotes M2 macrophage polarization). At the mutation "
    "level, PIK3CA and BRAF are positively loaded, while TP53, KRAS, IDH1, and EGFR are negatively "
    "loaded, defining a molecular context distinct from Factor 8."
)
add_body(doc,
    "The therapeutic implications of Factor 6 are immediate: the 9p24.1 amplification predicts "
    "response to PD-1/PD-L1 checkpoint inhibitors (pembrolizumab, nivolumab, atezolizumab), and "
    "the JAK2 co-amplification provides a rationale for combining JAK inhibition (ruxolitinib) "
    "with anti-PD-1 therapy to prevent STAT3-driven PD-L1 transcriptional reinduction following "
    "checkpoint blockade. Targeting the RLN1/2-RXFP1 axis (B7-33 antagonist) and IL33 blockade "
    "(itepekimab) may further potentiate immune responses in 9p24.1-amplified tumors."
)

add_section_heading(doc, "4.3 Factor 1: Squamous Lineage Programme and NECTIN4 Targeting", level=2)
add_body(doc,
    "Factor 1 (avg |SHAP| = 0.162) defines the squamous cell carcinoma lineage programme across "
    "head and neck, lung squamous, cervical, and esophageal tumors. Its expression signature is "
    "dominated by TP63, KRT16, KRT17, KRT15, NECTIN4, GJB2, and SFN—all markers of the p63-driven "
    "basal/squamous identity. CNV gains in DCUN1D1 (3q26.3 amplicon, present in >60% of HNSCC and "
    "LUSC) and EIF4G1 suggest additional translational and neddylation pathway amplification. "
    "The FT-Transformer assigns this factor approximately twice the importance of the tree-based "
    "models (SHAP: 0.245 vs ~0.12), suggesting that the squamous programme exerts non-linear, "
    "interaction-dependent survival effects best captured by attention mechanisms."
)
add_body(doc,
    "NECTIN4 is particularly actionable: it is expressed on basal/squamous tumour cells and is the "
    "target of enfortumab vedotin, an anti-NECTIN4 antibody-drug conjugate (ADC) FDA-approved for "
    "urothelial carcinoma and under active evaluation in HNSCC and squamous NSCLC. Factor 1 loading "
    "could serve as a biomarker for NECTIN4 ADC sensitivity beyond urothelial cancer. Additional "
    "targets include DCUN1D1 (neddylation inhibition with pevonedistat/MLN4924) and "
    "GJB2/Connexin-26 (siRNA knockdown reduces HNSCC invasion)."
)

add_section_heading(doc, "4.4 Factor 15: Luminal De-differentiation and Synthetic Lethality", level=2)
add_body(doc,
    "Factor 15 (avg |SHAP| = 0.120) captures loss of luminal epithelial identity: GATA3, MUC1, "
    "TFF1, SCGB2A2, CA12, and PRLR are negatively loaded (their loss associates with worse survival), "
    "while KLK2 and neuroendocrine markers load positively. At the mutation level, PTEN, ARID1A, "
    "KMT2C, and TP53 are negatively loaded (their loss drives worse outcomes), while IDH1 is "
    "positively loaded. This defines de-differentiated, hormone-receptor-lost tumors (triple-negative "
    "breast cancer, castration-resistant prostate cancer, and endocrine-resistant uterine cancers) "
    "as the most vulnerable subgroup on this axis."
)
add_body(doc,
    "The synthetic lethality approach is particularly well-suited to Factor 15 targets: ARID1A "
    "loss (SWI/SNF complex disruption) creates dependence on EZH2 for gene silencing, targeted "
    "by tazemetostat (FDA-approved for follicular lymphoma; Phase II trials in multiple solid tumors). "
    "KMT2C/KMT2D loss creates dependence on PRMT5 symmetric arginine methylation, targeted by "
    "GSK3326595 (Phase I/II). PTEN loss creates PI3K/AKT dependence, targeted by capivasertib "
    "(AZD5363) + fulvestrant (positive Phase III FAKTION data in ER+ breast)."
)

add_section_heading(doc, "4.5 Factor 12: Thyroid/IDH Differentiation Gradient", level=2)
add_body(doc,
    "Factor 12 (avg |SHAP| = 0.120) captures a differentiation gradient from well-differentiated "
    "thyroid tumors (high DUOX2, TG, TPO, TMPRSS2; BRAF V600E mutation; better survival) to "
    "IDH-wild-type glioblastoma and aggressive RCC (IDH1(-), CIC(-), ATRX(-), PBRM1(-); worse survival). "
    "GPC5 and GPC6 (glypican cell-surface proteoglycans) show CNV amplification and represent "
    "emerging ADC and CAR-T targets. TMPRSS2 inhibition (nafamostat, camostat) is relevant both "
    "for invasion suppression and for TMPRSS2:ERG gene fusion cancers (present in ~50% of prostate "
    "cancers). IDH1 inhibition with ivosidenib is already approved for IDH1-mutant AML, "
    "cholangiocarcinoma, and glioma, and CIC-mutant oligodendroglioma may benefit from MEK "
    "inhibition (trametinib) targeting ETS-driven transcription downstream of CIC loss."
)

add_section_heading(doc, "4.6 Consolidated Therapeutic Target Prioritization", level=2)
add_body(doc,
    "Integrating factor biology, SHAP scores, and clinical translatability, we nominate "
    "15 priority therapeutic targets organized into three intervention strategies. "
    "Table 2 summarizes these targets with their factor origin, mechanism, and therapeutic context."
)

tgt_data = [
    ["Priority", "Target / Mechanism", "Factor Origin", "Intervention Strategy", "Lead Therapy", "Stage"],
    ["1", "TP53 reactivation", "F8,F6,F1,F15,F12", "TSG Restoration", "APR-246 (eprenetapopt); MDM2 inhibitors", "Phase II/III"],
    ["2", "9p24.1: CD274+JAK2 CNV", "F6 CNV", "Oncogene Inhibition", "Anti-PD-1 + ruxolitinib", "Phase II"],
    ["3", "KRAS G12C/G12D", "F8,F6", "Oncogene Inhibition", "Sotorasib; adagrasib; MRTX1133", "Approved/Phase I"],
    ["4", "IDH1 neomorphic mutation", "F8,F12,F15", "Oncogene Inhibition", "Ivosidenib", "Approved"],
    ["5", "GAS6-AXL survival axis", "F8,F12,F15 CNV", "Oncogene Inhibition", "Bemcentinib; cabozantinib", "Phase I/II"],
    ["6", "PIK3CA H1047R/E545K", "F6,F15", "Oncogene Inhibition", "Alpelisib; capivasertib", "Approved/Phase III"],
    ["7", "BRAF V600E", "F8,F6,F12", "Oncogene Inhibition", "Dabrafenib+trametinib; encorafenib", "Approved"],
    ["8", "ARID1A loss (SWI/SNF)", "F15", "Synthetic Lethality", "Tazemetostat (EZH2 inhibitor)", "Phase II (solid)"],
    ["9", "ATRX loss (ALT)", "F8,F15,F12", "Synthetic Lethality", "Olaparib + berzosertib (ATR inh.)", "Phase I"],
    ["10", "NECTIN4 expression", "F1 Expression", "ADC Targeting", "Enfortumab vedotin", "Approved/Phase II"],
    ["11", "IL1A cytokine (TME)", "F8 Expression", "Cytokine Blockade", "Bermekimab; canakinumab", "Phase II"],
    ["12", "KMT2D/KMT2C loss", "F8,F1,F15", "Synthetic Lethality", "GSK3326595 (PRMT5 inhibitor)", "Phase I/II"],
    ["13", "PRSS3/KLK6 serine proteases", "F8 Expression", "Knockdown / Inhibition", "Nafamostat; siRNA-KLK6", "Preclinical/Phase I"],
    ["14", "GPC5/GPC6 glypicans", "F12,F15 CNV", "ADC / CAR-T", "Anti-GPC ADC; GPC5 CAR-T", "Preclinical"],
    ["15", "RLN1/RLN2-RXFP1 axis", "F6 CNV", "Lineage Reprogramming", "B7-33 RXFP1 antagonist", "Preclinical"],
]
doc.add_paragraph()
t2 = styled_table(doc, tgt_data, col_widths=[0.55, 1.55, 1.1, 1.4, 1.7, 1.0])
cap2 = doc.add_paragraph()
cap2.alignment = WD_ALIGN_PARAGRAPH.CENTER
cap2.paragraph_format.space_after = Pt(12)
add_run(cap2, "Table 2. ", bold=True, size=9)
add_run(cap2, "Priority therapeutic targets identified from MOFA+ factor decoding and SHAP attribution. "
         "Targets are ranked by combined factor SHAP score, clinical actionability, and mechanism novelty.",
         italic=True, size=9)

add_figure(doc, FIG / "marker_actionable_targets.png",
    "Top 30 actionable therapeutic targets ranked by gene-level SHAP contribution "
    "(|MOFA+ weight| x factor SHAP score, summed across top factors). Color indicates molecular role: "
    "red = oncogene (knockdown/inhibit), blue = tumour suppressor gene (restore/synthetic lethality), "
    "orange = gain-of-function TSG. Modality of evidence (EXP = expression, MUT = mutation, CNV = copy "
    "number) is annotated per target.",
    label="Figure 9.")

add_section_heading(doc, "4.7 Revised Therapeutic Prioritization After Causal Validation", level=2)
add_body(doc,
    "The DiD causal validation layer materially revises the prioritization of therapeutic "
    "targets derived from SHAP analysis alone. Factor 8, which ranked first by SHAP score "
    "(avg |SHAP| = 0.283), does not demonstrate a robust aggregate-level causal survival effect. "
    "Its predictive dominance is explained by cancer-type composition confounding: it is the "
    "most informative predictor of survival because it correlates with which cancer type a "
    "patient has—not because high Factor 8 within a cancer type independently causes mortality. "
    "Accordingly, Factor 8-based targets (PRSS3/KLK6 serine proteases, GAS6-AXL axis) are "
    "reclassified from co-primary to secondary priority, warranting within-cancer-type validation "
    "before advancing to clinical targeting."
)
add_body(doc,
    "Factor 6 (9p24.1 immune checkpoint amplicon; ATT = −0.486; p < 10⁻¹²) and Factor 15 "
    "(luminal de-differentiation; ATT = −0.285; p < 10⁻¹⁷) are elevated to co-primary targets "
    "based on causal evidence. The Factor 6 causal validation strengthens the rationale for "
    "anti-PD-1 + JAK inhibitor combination trials in 9p24.1-amplified tumors: the Factor 6 "
    "program causally impairs survival, suggesting checkpoint blockade will modify survival "
    "trajectories rather than merely correlate with them. The Factor 15 causal validation "
    "substantially increases confidence in EZH2 inhibition (tazemetostat) for ARID1A-loss "
    "tumors and PRMT5 inhibition (GSK3326595) for KMT2C/D-loss tumors as potentially survival-modifying "
    "rather than merely biomarker-selected strategies."
)
add_body(doc,
    "The KRAS × Factor 8 Triple Difference finding (DDD = −0.472; p = 0.022) carries distinct "
    "clinical significance. It identifies a pan-cancer synthetic co-dependency in which KRAS "
    "mutation and protease-driven invasion programming combine synergistically to impair survival. "
    "This provides a causal mechanistic basis for the clinical hypothesis that KRAS-inhibitor "
    "monotherapy resistance may be driven by compensatory upregulation of invasion programmes, "
    "and nominates the combination of KRAS G12C/G12D inhibition (sotorasib, adagrasib, MRTX1133) "
    "with serine protease inhibitors (nafamostat, camostat) or AXL inhibitors (bemcentinib) as "
    "the highest-priority combination to evaluate in KRAS-mutant solid tumors."
)

# ── 5. CONCLUSION ─────────────────────────────────────────────────────────────
add_section_heading(doc, "5. Conclusion", level=1)

add_body(doc,
    "This study demonstrates that MOFA+-driven multi-omics factor analysis, combined with "
    "transformer-based survival prediction and SHAP attribution, constitutes a powerful "
    "and fully interpretable framework for pan-cancer target discovery. Applied to 7,787 "
    "TCGA samples across 31 cancer types, the approach achieves a mean C-index of 0.751 "
    "while simultaneously identifying five biologically distinct survival factors that map "
    "directly to therapeutic strategies."
)

add_body(doc,
    "Three key methodological findings emerge. First, MOFA+ factors substantially outperform "
    "raw PCA features as survival model inputs, improving mean C-index from 0.724 to 0.751 "
    "and reducing cross-validation variance 7-fold—indicating that biologically structured "
    "latent representations are preferable to generic dimensionality reduction for prognostic "
    "modelling. Second, the FT-Transformer outperforms gradient boosting trees on MOFA+ "
    "features, with the largest advantage on factors with high inter-model discordance "
    "(e.g., Factor 1, squamous programme), suggesting that attention mechanisms capture "
    "synergistic effects between lineage identity and cancer-type covariates that trees miss. "
    "Third, SHAP scores computed on latent factors are not equivalent to variance explained: "
    "Factor 8, the top survival predictor, explains only 3.2% of cross-modal variance, "
    "while Factor 1, which explains 6.1%, ranks third in SHAP importance. This dissociation "
    "is biologically meaningful—prognostic factors need not be the largest sources of "
    "variation, only the most consistent ones."
)

add_body(doc,
    "Four major biological insights emerge from factor decoding, refined by causal validation. "
    "The 9p24.1 immune checkpoint amplicon (CD274/PDCD1LG2/JAK2 in Factor 6) represents the "
    "strongest causally validated pan-cancer survival determinant (ATT = −0.486; p < 10⁻¹²), "
    "representing a molecularly homogeneous immunological axis amenable to checkpoint blockade "
    "combined with JAK inhibition. Epigenetic tumour suppressor loss (ARID1A/KMT2C/PTEN in "
    "Factor 15) is the second causally confirmed axis (ATT = −0.285; p < 10⁻¹⁷), defining a "
    "synthetic lethality landscape for EZH2 and PRMT5 inhibitors with direct causal survival "
    "evidence. Squamous lineage identity (Factor 1, TP63/KRT/NECTIN4) is causally validated "
    "at conventional significance (ATT = −0.129; p = 0.028). The pan-driver mutation axis "
    "(TP53/KRAS/APC/IDH1/ATRX in Factor 8) is the dominant predictive signal by SHAP but "
    "does not achieve causal identification in the aggregate pseudo-panel, with its survival "
    "signal attributable primarily to cancer-type composition—an important limitation that "
    "motivates within-type validation before clinical targeting."
)

add_body(doc,
    "The Triple Difference analysis yields the most immediately actionable clinical finding: "
    "KRAS mutation and Factor 8 protease-invasion activation combine synergistically to impair "
    "survival (DDD = −0.472; p = 0.022), providing a causal mechanistic rationale for combining "
    "KRAS inhibitors with invasion-targeting agents in KRAS-mutant tumors. Clinically, the most "
    "immediately translatable findings are: (i) Factor 6's 9p24.1 amplicon as a causally validated "
    "pan-cancer biomarker for anti-PD-1 + JAK inhibitor combination trials; (ii) ARID1A and KMT2C "
    "loss as causally validated dual biomarkers for tazemetostat (EZH2 inhibitor) and GSK3326595 "
    "(PRMT5 inhibitor) trials in solid tumors; (iii) NECTIN4 ADC (enfortumab vedotin) eligibility "
    "predicted by Factor 1 loading score independently of histologic type; and (iv) KRAS × Factor 8 "
    "synergy as the causal rationale for KRAS inhibitor + serine protease/AXL inhibitor combination "
    "trials. Cancer-type-stratified DiD models built on individual MOFA+ factor axes should yield "
    "within-type causal ATT estimates and enable patient-level therapeutic stratification based on "
    "individual factor score profiles."
)

add_body(doc,
    "This pipeline is fully open, reproducible, and extensible. The causal inference layer "
    "demonstrates that pairing SHAP-based predictive attribution with quasi-experimental DiD "
    "estimation is a tractable and informative strategy for elevating multiomics target nominations "
    "from correlational to causal. Future work will integrate methylation and miRNA modalities "
    "into the MOFA+ views, apply drug sensitivity data (GDSC, CCLE) to validate therapeutic "
    "nominations, and use the factor scores for patient stratification in prospective study design. "
    "Individual-level causal methods (instrumental variables, regression discontinuity) applied to "
    "clinical trial data representing Factor 6 and Factor 15 high patients would represent the "
    "ultimate validation of these findings. The framework represents a general-purpose architecture "
    "for multi-omics survival analysis and target discovery applicable to any cancer cohort with "
    "comprehensive molecular profiling."
)

# ── References ────────────────────────────────────────────────────────────────
add_section_heading(doc, "References", level=1)
refs = [
    "Argelaguet R, et al. (2020). MOFA+: a statistical framework for comprehensive integration of multi-modal single-cell data. Genome Biology, 21(1), 111.",
    "Gorishniy Y, et al. (2021). Revisiting Deep Learning Models for Tabular Data. Advances in Neural Information Processing Systems (NeurIPS), 34.",
    "Liu J, et al. (2018). An Integrated TCGA Pan-Cancer Clinical Data Resource to Drive High-Quality Survival Outcome Analytics. Cell, 173(2), 400-416.e11.",
    "Lundberg SM & Lee SI (2017). A unified approach to interpreting model predictions. NeurIPS, 30.",
    "TCGA Research Network (2013). The Cancer Genome Atlas Pan-Cancer analysis project. Nature Genetics, 45(10), 1113-1120.",
    "Chen T & Guestrin C (2016). XGBoost: A Scalable Tree Boosting System. KDD 2016.",
    "Ke G, et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. NeurIPS, 30.",
    "Davidson-Pilon C (2019). lifelines: survival analysis in Python. Journal of Open Source Software, 4(40), 1317.",
    "Breiman L (2001). Random Forests. Machine Learning, 45(1), 5-32.",
    "Harrell FE, et al. (1982). Evaluating the yield of medical tests. JAMA, 247(18), 2543-2546.",
    "Callaway B & Sant'Anna PHC (2021). Difference-in-Differences with multiple time periods. Journal of Econometrics, 225(2), 200-230.",
    "Rambachan A & Roth J (2023). A More Credible Approach to Parallel Trends. Review of Economic Studies, 90(5), 2555-2591.",
    "Gerber I (2025). diff-diff: A Python library for modern causal DiD estimation (v3.3). GitHub: https://github.com/igerber/diff-diff.",
]
for i, ref in enumerate(refs, 1):
    rp = doc.add_paragraph()
    rp.paragraph_format.space_after = Pt(4)
    rp.paragraph_format.left_indent = Cm(0.6)
    rp.paragraph_format.first_line_indent = Cm(-0.6)
    add_run(rp, f"{i}. ", bold=True, size=10)
    add_run(rp, ref, size=10)

# Save
ms_path = OUT / "manuscript.docx"
doc.save(str(ms_path))
print(f"Manuscript saved: {ms_path}")

# ══════════════════════════════════════════════════════════════════════════════
#  SUPPLEMENTARY DOCUMENT
# ══════════════════════════════════════════════════════════════════════════════
print("\nBuilding supplementary document ...")
sdoc = Document()

for section in sdoc.sections:
    section.top_margin    = Cm(2.5)
    section.bottom_margin = Cm(2.5)
    section.left_margin   = Cm(2.8)
    section.right_margin  = Cm(2.8)

# Title
tp = sdoc.add_paragraph()
tp.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp.paragraph_format.space_before = Pt(20)
tp.paragraph_format.space_after  = Pt(8)
add_run(tp, "Supplementary Materials", bold=True, size=18)

tp2 = sdoc.add_paragraph()
tp2.alignment = WD_ALIGN_PARAGRAPH.CENTER
tp2.paragraph_format.space_after = Pt(20)
add_run(tp2,
    "Pan-Cancer Multi-Omics Survival Analysis Using MOFA+ and Transformer-Based Deep Learning",
    italic=True, size=11)

horizontal_rule(sdoc)

# ── Table of Contents (manual) ────────────────────────────────────────────────
toc_head = sdoc.add_paragraph()
toc_head.paragraph_format.space_before = Pt(14)
add_run(toc_head, "Contents", bold=True, size=13)

toc_items = [
    "Section S1: Extended Methods",
    "Section S2: Data Quality and Cohort Statistics (Tables S1-S3)",
    "Section S3: Dimensionality Reduction Figures (Figures S1-S3)",
    "Section S4: Unsupervised Clustering Results (Figures S4-S9)",
    "Section S5: MOFA+ Extended Results (Figures S10-S17, Table S4)",
    "Section S6: Cox and DeepSurv Baseline Models (Figures S18-S21)",
    "Section S7: Full SHAP Analyses per Model (Figures S22-S27)",
    "Section S8: Multiomics Marker Analysis (Figures S28-S31, Tables S5-S7)",
    "Section S9: Model Consistency Analysis (Figure S32)",
    "Section S10: Software and Reproducibility",
]
for item in toc_items:
    add_bullet(sdoc, item)

page_break(sdoc)

# ── S1: Extended Methods ──────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S1: Extended Methods", level=1)

add_section_heading(sdoc, "S1.1 Data Sources and Download Protocol", level=2)
add_body_noi(sdoc,
    "All data were downloaded programmatically using a custom Python script (download_tcga.py) "
    "with resume-safe chunked streaming (8 MB chunks), automatic retry on network failure (3 attempts "
    "with exponential backoff), and SHA256 checksum verification. Downloads sourced from three hubs:"
)
hub_data = [
    ["Modality", "Hub", "Filename", "Size"],
    ["Expression (RNA-seq)", "UCSC TOIL", "tcga_RSEM_gene_tpm.gz", "~1.8 GB"],
    ["Somatic Mutations", "UCSC Xena PanCan", "mc3.v0.2.8.PUBLIC.xena.gz", "~80 MB"],
    ["CNV (GISTIC2)", "UCSC Xena Legacy", "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz", "~60 MB"],
    ["Clinical Phenotype", "UCSC Xena PanCan", "TCGA_phenotype_denseDataOnlyDownload.tsv.gz", "~2 MB"],
    ["Survival (CDR)", "UCSC Xena PanCan", "Survival_SupplementalTable_S1_20171025_xena_sp", "~1 MB"],
]
sdoc.add_paragraph()
styled_table(sdoc, hub_data, col_widths=[1.5, 1.8, 3.0, 0.9])
cap_s = sdoc.add_paragraph()
add_run(cap_s, "Table S1. ", bold=True, size=9)
add_run(cap_s, "Data sources and file sizes for all downloaded modalities.", italic=True, size=9)
cap_s.paragraph_format.space_after = Pt(12)

add_section_heading(sdoc, "S1.2 Expression Preprocessing Details", level=2)
add_body_noi(sdoc,
    "TOIL recomputed expression values are stored as log2(TPM + 0.001). The TOIL floor sentinel "
    "value of log2(0.001) = -9.965 was replaced with 0 to avoid inflating variance estimates for "
    "unexpressed genes. Ensembl gene IDs were stripped of version suffixes (e.g., ENSG00000001234.5 "
    "-> ENSG00000001234). Gene symbol mapping used the mygene.info REST API with batched queries "
    "(1,000 genes/request), mapping ENSG IDs to HGNC symbols and resolving 4,389 of 5,000 "
    "selected gene IDs. Unmapped IDs were retained with their Ensembl identifier. The top-5,000 "
    "variance genes were selected by computing inter-sample variance across all 10,535 samples "
    "in the expression matrix prior to alignment."
)

add_section_heading(sdoc, "S1.3 Mutation Preprocessing Details", level=2)
add_body_noi(sdoc,
    "The MC3 MAF was filtered to retain only functional non-synonymous variant types: "
    "Missense_Mutation, Nonsense_Mutation, Frame_Shift_Del, Frame_Shift_Ins, Splice_Site, "
    "In_Frame_Del, In_Frame_Ins, Nonstop_Mutation, Translation_Start_Site, "
    "De_novo_Start_OutOfFrame. Barcodes were truncated to 15 characters to reconcile TCGA "
    "barcode formats across MAF and clinical files. Mutation calls were pivoted to a binary "
    "sample x gene presence/absence matrix using pandas pivot_table. Genes mutated in fewer "
    "than 2% of samples were excluded, yielding 1,490 retained genes from the original "
    "~18,000 mutated genes in the MAF."
)

add_section_heading(sdoc, "S1.4 MOFA+ Implementation Details", level=2)
add_body_noi(sdoc,
    "MOFA+ was implemented using mofapy2 version 0.7.4 (pure Python, no R dependency). "
    "Data was provided in the MOFA+ matrix API format: data[view_index][group_index]. "
    "Feature names were made globally unique across views by prefixing: expression genes "
    "kept their HGNC symbols (no prefix), mutation genes were prefixed 'M_', and CNV genes "
    "were prefixed 'C_'. This global uniqueness requirement is enforced by mofapy2 internally. "
    "Training options: ard_factors=True, ard_weights=True, spikeslab_factors=False, "
    "spikeslab_weights=True, convergence_mode='fast', iter=400, seed=42, verbose=False. "
    "Factor pruning by ARD eliminated 2 of 20 requested factors (those with zero variance "
    "explained after convergence), yielding 18 retained factors."
)

add_section_heading(sdoc, "S1.5 FT-Transformer Architecture", level=2)
add_body_noi(sdoc,
    "The FT-Transformer consists of: (1) FeatureTokenizer: for each of n=50 input features, "
    "a learnable d_token=64 dimensional embedding computed as W_j * x_j + b_j, where W_j and "
    "b_j are per-feature learned parameters; (2) a learnable [CLS] token prepended to the feature "
    "token sequence; (3) n_layers=2 pre-norm transformer blocks, each containing: LayerNorm -> "
    "MultiHeadAttention(n_heads=4) -> residual, LayerNorm -> FFN(d_ffn=256, GELU, dropout=0.2) "
    "-> residual; (4) final LayerNorm on the [CLS] token; (5) Linear(64, 1) head outputting "
    "a scalar log-hazard ratio. Total parameters: approximately 68,000. Trained with negative "
    "Breslow-approximated Cox partial log-likelihood loss implemented in PyTorch using "
    "torch.logcumsumexp for numerical stability."
)

add_section_heading(sdoc, "S1.6 LightGBM Custom Cox Objective", level=2)
add_body_noi(sdoc,
    "LightGBM version 4.x removed the built-in cox survival objective. A custom Breslow "
    "approximation was implemented via the params['objective'] API (not the deprecated fobj= "
    "keyword argument). The gradient and Hessian for sample i are: "
    "grad_i = -event_i + exp(f_i) * sum_{j: t_j >= t_i}(event_j / sum_{k: t_k >= t_j} exp(f_k)), "
    "hess_i = max(exp(f_i) * cumulative_weight_i, 1e-6). Survival labels are encoded as: "
    "label = +OS_time if OS_event==1, -OS_time if OS_event==0 (censored). The concordance "
    "index was used as the validation metric via a custom feval function."
)

page_break(sdoc)

# ── S2: Cohort Statistics ─────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S2: Data Quality and Cohort Statistics", level=1)

cohort_data = [
    ["Metric", "Value"],
    ["Total TCGA samples (expression matrix)", "10,535"],
    ["Samples after multi-modal alignment", "7,902"],
    ["Cancer types represented", "31"],
    ["Samples with valid OS data (CDR)", "7,787"],
    ["OS events (deaths)", "2,170 (27.9%)"],
    ["Median OS time (days)", "~1,423"],
    ["Expression genes (post-QC, top variance)", "5,000"],
    ["Ensembl IDs mapped to HGNC symbols", "4,389 / 5,000 (87.8%)"],
    ["Mutation genes (>=2% frequency)", "1,490"],
    ["CNV genes", "24,776"],
    ["MOFA+ input genes (expression view)", "1,000"],
    ["MOFA+ input genes (mutation view)", "300"],
    ["MOFA+ input genes (CNV view)", "1,000"],
    ["MOFA+ factors retained", "18 (of 20 requested)"],
    ["Survival model training samples", "7,787"],
    ["5-fold CV folds", "5 (stratified by OS event)"],
]
styled_table(sdoc, cohort_data, col_widths=[3.5, 3.0])
cap_s2 = sdoc.add_paragraph()
add_run(cap_s2, "Table S2. ", bold=True, size=9)
add_run(cap_s2, "Complete cohort statistics and data dimensions across all processing stages.", italic=True, size=9)
cap_s2.paragraph_format.space_after = Pt(12)

sdoc.add_paragraph()
cancer_data = [
    ["Cancer Type", "TCGA Code", "N Samples", "N Events", "Event Rate"],
    ["Breast Invasive Carcinoma", "BRCA", "~1,030", "~121", "11.7%"],
    ["Kidney Clear Cell Carcinoma", "KIRC", "~530", "~178", "33.6%"],
    ["Uterine Corpus Endometrioid", "UCEC", "~530", "~77", "14.5%"],
    ["Lung Adenocarcinoma", "LUAD", "~510", "~180", "35.3%"],
    ["Thyroid Carcinoma", "THCA", "~500", "~14", "2.8%"],
    ["Head & Neck Squamous", "HNSC", "~500", "~224", "44.8%"],
    ["Colon Adenocarcinoma", "COAD", "~455", "~100", "22.0%"],
    ["Low Grade Glioma", "LGG", "~510", "~96", "18.8%"],
    ["Prostate Adenocarcinoma", "PRAD", "~495", "~20", "4.0%"],
    ["Lung Squamous Cell", "LUSC", "~490", "~198", "40.4%"],
    ["... (21 additional types)", "...", "...", "...", "..."],
]
styled_table(sdoc, cancer_data, col_widths=[2.2, 1.1, 1.0, 1.0, 1.0])
cap_s3 = sdoc.add_paragraph()
add_run(cap_s3, "Table S3. ", bold=True, size=9)
add_run(cap_s3, "Sample counts and event rates for the 10 most-represented cancer types (31 total).", italic=True, size=9)
cap_s3.paragraph_format.space_after = Pt(14)

page_break(sdoc)

# ── S3: Dimensionality Reduction ──────────────────────────────────────────────
add_section_heading(sdoc, "Section S3: Dimensionality Reduction", level=1)
add_body_noi(sdoc,
    "Principal component analysis (PCA) and Uniform Manifold Approximation and Projection (UMAP) "
    "were applied to the top-5,000-variance gene expression matrix. PCA used sklearn.decomposition.PCA "
    "with n_components=50 after StandardScaler normalization. UMAP was applied to the 50 PCA "
    "components with n_neighbors=30, min_dist=0.3, random_state=42, using the umap-learn library."
)

add_figure(sdoc, FIG / "pca_scree.png",
    "PCA scree plot showing variance explained by each of the top 50 principal components. "
    "The elbow at PC5-PC8 reflects the dominance of cancer-type identity in the first few components.",
    label="Figure S1.", width=5.5)

add_figure(sdoc, FIG / "umap_cancer_type.png",
    "UMAP embedding of 7,902 samples colored by TCGA cancer type. Distinct clusters "
    "correspond to major lineage groups: epithelial carcinomas, gliomas, sarcomas, and "
    "hematological malignancies.",
    label="Figure S2.", width=5.5)

add_figure(sdoc, FIG / "umap_survival.png",
    "UMAP embedding colored by overall survival time (OS_time, days). Gradient from "
    "short (red) to long (blue) survival is overlaid on the cancer-type structure, "
    "revealing that within-type variation in survival is substantial.",
    label="Figure S3.", width=5.5)

page_break(sdoc)

# ── S4: Clustering ────────────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S4: Unsupervised Clustering Results", level=1)
add_body_noi(sdoc,
    "Two clustering approaches were applied to the UMAP-reduced expression space: "
    "MiniBatchKMeans (k=2-25, silhouette score optimization on 3,000-sample subsample) "
    "and graph-based Leiden clustering (sklearn NearestNeighbors k=15 -> igraph -> "
    "leidenalg RBConfigurationVertexPartition, resolution=0.4). Leiden clustering identified "
    "22 clusters with a log-rank survival separation of p=2.02e-24."
)

add_figure(sdoc, FIG / "kmeans_selection.png",
    "Silhouette score vs. number of clusters (k) for MiniBatchKMeans. Optimal k=2 "
    "by silhouette reflects pan-cancer artifact (epithelial vs. non-epithelial lineage) "
    "rather than biologically meaningful substructure.",
    label="Figure S4.", width=5.5)

for fig_name, label, caption in [
    ("umap_kmeans.png", "Figure S5.",
     "UMAP colored by KMeans cluster assignment (k=2, optimal silhouette). The two clusters "
     "broadly separate gliomas/neuroendocrine from epithelial carcinomas."),
    ("cluster_cancer_type_kmeans.png", "Figure S6.",
     "Cancer type composition per KMeans cluster. Cluster 1 is dominated by GBMLGG/gliomas; "
     "Cluster 2 by all major carcinoma types."),
    ("cluster_survival_kmeans.png", "Figure S7.",
     "Kaplan-Meier survival curves by KMeans cluster. Despite poor silhouette, the two clusters "
     "show statistically significant OS separation (p < 0.001), primarily driven by the "
     "well-known glioma survival advantage."),
    ("umap_leiden.png", "Figure S8.",
     "UMAP colored by Leiden cluster assignment (22 clusters, resolution=0.4). Fine-grained "
     "substructure within major cancer types is captured, particularly within breast, lung, "
     "and kidney carcinomas."),
    ("cluster_survival_leiden.png", "Figure S9.",
     "Kaplan-Meier survival curves for all 22 Leiden clusters. Log-rank p=2.02e-24. "
     "Graph-based clustering substantially outperforms K-means for survival stratification, "
     "confirming the multi-modal density structure in pan-cancer data."),
]:
    add_figure(sdoc, FIG / fig_name, caption, label=label, width=5.5)

page_break(sdoc)

# ── S5: MOFA+ Extended ────────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S5: MOFA+ Extended Results", level=1)

add_figure(sdoc, FIG / "mofa_factor_cancer_type.png",
    "Heatmap of mean MOFA+ factor scores per cancer type. Factors with strong cancer-type "
    "associations (e.g., Factor 1 in HNSC/LUSC/CESC; Factor 9 in BRCA; Factor 4 in LIHC) "
    "reflect known lineage-defining transcriptional programmes.",
    label="Figure S10.", width=6.0)

add_figure(sdoc, FIG / "mofa_factor_survival.png",
    "Spearman correlations between MOFA+ factor scores and OS time across all survival-annotated "
    "samples. Factor 8 shows the strongest negative correlation with OS time (higher factor score "
    "= shorter survival).",
    label="Figure S11.", width=5.5)

for fig_name, factor_num, label, caption in [
    ("mofa_top_weights_Factor1.png",  1,  "Figure S12.",
     "Top 20 gene loadings for MOFA+ Factor 1 (expression view). KRT-family genes, TP63, SFN, "
     "NECTIN4, and GJB2 dominate, defining the squamous/basal lineage programme."),
    ("mofa_top_weights_Factor4.png",  4,  "Figure S13.",
     "Top 20 gene loadings for MOFA+ Factor 4. ALDOB, HNF4A, CDHR5, and ALB define a "
     "hepatocyte/GI differentiation programme associated with LIHC and COAD."),
    ("mofa_top_weights_Factor7.png",  7,  "Figure S14.",
     "Top 20 gene loadings for Factor 7. GFAP, BCAN, PTPRZ1, and MIR9-1HG define the "
     "glioma/astrocyte axis; IDH1 and ATRX mutations co-load on this factor."),
    ("mofa_top_weights_Factor9.png",  9,  "Figure S15.",
     "Top 20 gene loadings for Factor 9. IGHG1, IGHA1, IGLC1-3, IGKC, IGHG2-4 define "
     "a B-cell/immunoglobulin infiltration programme prominent in BRCA and STAD."),
    ("mofa_top_weights_Factor12.png", 12, "Figure S16.",
     "Top 20 gene loadings for Factor 12. TG, TPO, DUOX2, DUOXA1 define thyroid "
     "differentiation; TMPRSS2 and CLDN1 also load prominently."),
]:
    add_figure(sdoc, FIG / fig_name, caption, label=label, width=5.5)

for fig_name, factor_num, label, caption in [
    ("mofa_factor_survival_Factor4.png", 4, "Figure S17a.",
     "Kaplan-Meier plot splitting samples at median Factor 4 score. High Factor 4 "
     "(hepatocyte/GI programme) associates with longer OS."),
    ("mofa_factor_survival_Factor12.png", 12, "Figure S17b.",
     "Kaplan-Meier plot splitting samples at median Factor 12 score. High Factor 12 "
     "(thyroid differentiation) associates with significantly longer OS (p < 0.001)."),
]:
    add_figure(sdoc, FIG / fig_name, caption, label=label, width=5.5)

# MOFA factor table
sdoc.add_paragraph()
mofa_factor_data = [
    ["Factor", "Var Expl (%)", "Avg |SHAP|", "SHAP Rank", "Primary Biology", "Key Genes"],
    ["F1",  "6.1", "0.162", "3", "Squamous/basal lineage", "KRT16/17, TP63, NECTIN4, SFN"],
    ["F4",  "5.8", "0.068", "14", "Hepatocyte/GI", "ALDOB, HNF4A, CDHR5, ALB"],
    ["F9",  "5.1", "0.088", "6", "B-cell infiltration", "IGHG1-4, IGHA1, IGLC1-3"],
    ["F7",  "4.7", "0.062", "16", "Glioma/neural", "GFAP, BCAN, PTPRZ1, IDH1 mut"],
    ["F12", "5.3", "0.120", "5", "Thyroid/IDH differentiation", "TG, TPO, DUOX2, BRAF mut"],
    ["F8",  "3.2", "0.283", "1", "Protease invasion + driver muts", "PRSS3, KLK6, TP53/KRAS mut"],
    ["F6",  "0.8", "0.219", "2", "Immune checkpoint (9p24.1)", "CD274, PDCD1LG2, JAK2 CNV"],
    ["F15", "2.5", "0.120", "4", "Luminal de-differentiation", "GATA3, MUC1, ARID1A mut loss"],
    ["F11", "4.3", "0.078", "9", "Mixed epithelial", "Various KRT, mucin genes"],
    ["F16", "3.6", "0.064", "15", "Stromal/fibroblast", "GPC5/6, KLF5"],
]
styled_table(sdoc, mofa_factor_data, col_widths=[0.5, 1.0, 1.0, 1.0, 1.8, 2.2])
cap_mofa = sdoc.add_paragraph()
add_run(cap_mofa, "Table S4. ", bold=True, size=9)
add_run(cap_mofa, "Summary of all retained MOFA+ factors: variance explained, SHAP score, SHAP rank, "
        "primary biological annotation, and representative genes.", italic=True, size=9)
cap_mofa.paragraph_format.space_after = Pt(14)

page_break(sdoc)

# ── S6: Cox and DeepSurv Baselines ───────────────────────────────────────────
add_section_heading(sdoc, "Section S6: Cox Ridge and DeepSurv Baseline Models", level=1)
add_body_noi(sdoc,
    "Prior to the final MOFA+-based model comparison, two baseline survival models were developed "
    "on raw PCA features: (1) a Cox ridge regression using 80 ElasticNet-selected features from "
    "expression PCA(50) + top-100 mutation genes + CNV PCA(30); and (2) a DeepSurv neural network "
    "(3 hidden layers: 256-128-64 units with BatchNorm, ReLU, and Dropout 0.4) trained on the same "
    "80-feature set. These baseline experiments established the performance ceiling of raw-feature "
    "approaches before MOFA+ integration."
)

for fig_name, label, caption in [
    ("cox_cv_cindex.png", "Figure S18.",
     "Cox ridge regression 5-fold CV C-index. Mean = 0.679 +/- 0.035 on 80 ElasticNet-selected "
     "features from raw PCA. High fold-to-fold variance reflects instability of raw features."),
    ("cox_hazard_ratios.png", "Figure S19.",
     "Cox model top 20 feature hazard ratios with 95% confidence intervals. EXPR_PC14 is the "
     "top predictor (HR = 1.18, 95% CI: 1.12-1.25), followed by CNV_PC15 and MUT_USH2A."),
    ("cox_risk_km.png", "Figure S20.",
     "Kaplan-Meier survival curves for high vs. low-risk groups by Cox model risk score (median split)."),
    ("deepsurv_loss_curve.png", "Figure S21a.",
     "DeepSurv training and validation loss curves across 5 cross-validation folds. "
     "Early stopping (patience=20) is triggered between epochs 40-80 across folds."),
    ("deepsurv_cv_cindex.png", "Figure S21b.",
     "DeepSurv 5-fold CV C-index. Mean = 0.683 +/- 0.026. The deep network marginally outperforms "
     "Cox ridge on the same 80-feature set."),
]:
    add_figure(sdoc, FIG / fig_name, caption, label=label, width=5.5)

page_break(sdoc)

# ── S7: Full SHAP Analyses ────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S7: Full SHAP Analyses per Model", level=1)
add_body_noi(sdoc,
    "SHAP (SHapley Additive exPlanations) was computed for all three MOFA+-based models and "
    "the two baseline models. The following figures show beeswarm plots (feature impact on model "
    "output for each sample) and bar plots (mean absolute SHAP per feature) for each model."
)

for fig_name, label, caption in [
    ("shap_ftt_beeswarm.png", "Figure S22.",
     "FT-Transformer SHAP beeswarm plot. Each dot is one sample; color indicates feature value "
     "(red=high, blue=low); x-position shows SHAP value (impact on predicted log-hazard). "
     "Factor 8 and age_at_dx show the widest SHAP spread, confirming their dominance."),
    ("shap_ftt_bar.png", "Figure S23.",
     "FT-Transformer mean absolute SHAP bar chart. Top 20 features ranked by mean |SHAP|. "
     "The molecular factors (Factor 8, 6, 1) are clearly separated from lower-ranked factors."),
    ("shap_xgb_beeswarm.png", "Figure S24.",
     "XGBoost SHAP beeswarm plot (TreeExplainer, exact). Factor 8 again dominates; "
     "cancer-type dummies (CT_prostate, CT_breast) play a larger role than in FTT."),
    ("shap_xgb_bar.png", "Figure S25.",
     "XGBoost mean absolute SHAP bar chart. Factor 8 and age are top features; "
     "Factor 6 ranks higher relative to FTT compared to Factor 1."),
    ("shap_lgb_beeswarm.png", "Figure S26.",
     "LightGBM SHAP beeswarm plot (TreeExplainer). Pattern is closely consistent with XGBoost, "
     "confirming robustness of factor importance ranking across tree architectures."),
    ("shap_lgb_bar.png", "Figure S27.",
     "LightGBM mean absolute SHAP bar chart. Factor 8 = 0.293 (highest of any model), "
     "confirming LightGBM's strong sensitivity to the protease/invasion signal."),
]:
    add_figure(sdoc, FIG / fig_name, caption, label=label, width=5.8)

page_break(sdoc)

# ── S8: Marker Analysis ───────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S8: Multiomics Marker Analysis", level=1)
add_body_noi(sdoc,
    "Gene-level survival marker contributions were quantified by computing the product of each "
    "gene's MOFA+ loading magnitude (|weight_ij|) and the corresponding factor's mean absolute "
    "SHAP score across all three models, summed across the five top-SHAP factors "
    "(Factors 8, 6, 1, 15, 12). This produces a gene-level SHAP contribution score representing "
    "the gene's aggregate influence on survival prediction across the full multi-omics framework."
)

for fig_name, label, caption in [
    ("marker_top40_genes.png", "Figure S28.",
     "Top 40 gene-level markers ranked by aggregate SHAP contribution across all three modalities "
     "(expression = blue, mutation = orange, CNV = green). TP53 is the top-ranked gene, "
     "appearing across all 5 top factors. The 9p24.1 immune checkpoint genes "
     "(CD274, PDCD1LG2, JAK2) cluster prominently in the CNV tier."),
    ("marker_gene_heatmap.png", "Figure S29.",
     "Heatmap of expression gene weights across all five top survival-associated MOFA+ factors. "
     "Red = positive loading (high expression loads factor up); blue = negative loading. "
     "Distinct gene clusters map to squamous (F1), invasion/protease (F8), and "
     "luminal (F15) programmes."),
    ("marker_mutation_landscape.png", "Figure S30.",
     "Left: Top 25 mutation markers ranked by gene-level SHAP contribution (red = known driver). "
     "Right: Factor x gene weight heatmap for the top mutation genes across all five top factors. "
     "TP53, IDH1, and KRAS are the most recurrent drivers across factors."),
    ("marker_pathway_bubble.png", "Figure S31.",
     "Pathway-level aggregated SHAP contribution. Bubble size indicates number of genes in pathway. "
     "The immune checkpoint/B-cell pathway, squamous differentiation, and hepatocyte/GI programmes "
     "show the highest aggregate scores."),
]:
    add_figure(sdoc, FIG / fig_name, caption, label=label, width=5.8)

# Full gene table (top 30)
try:
    tgt_df = pd.read_csv(RES / "therapeutic_targets.csv")
    top30 = tgt_df[["gene","modality","total_contrib","n_factors","role","pathway"]].head(30)
    top30["total_contrib"] = top30["total_contrib"].round(4)

    table_rows = [["Gene", "Modality", "SHAP Contrib", "N Factors", "Role", "Pathway"]]
    for _, row in top30.iterrows():
        table_rows.append([row["gene"], row["modality"], str(row["total_contrib"]),
                           str(row["n_factors"]), str(row["role"]), str(row["pathway"])])
    sdoc.add_paragraph()
    styled_table(sdoc, table_rows, col_widths=[1.2, 1.0, 1.0, 0.8, 1.5, 1.9])
    cap_t5 = sdoc.add_paragraph()
    add_run(cap_t5, "Table S5. ", bold=True, size=9)
    add_run(cap_t5, "Top 30 gene-level markers by aggregate SHAP contribution across all 5 top factors.", italic=True, size=9)
    cap_t5.paragraph_format.space_after = Pt(12)
except Exception as e:
    sdoc.add_paragraph(f"[Table S5 unavailable: {e}]")

# Therapeutic target table (full)
try:
    full_tgt = tgt_df[["gene","modality","total_contrib","n_factors","role","tx_mechanism"]].head(20)
    full_tgt["total_contrib"] = full_tgt["total_contrib"].round(4)
    tx_rows = [["Gene", "Modality", "SHAP Contrib", "N Factors", "Role", "Therapeutic Mechanism"]]
    for _, row in full_tgt.iterrows():
        mech = str(row["tx_mechanism"])
        short = mech[:80] + "..." if len(mech) > 80 else mech
        tx_rows.append([row["gene"], row["modality"], str(row["total_contrib"]),
                        str(row["n_factors"]), str(row["role"]), short])
    sdoc.add_paragraph()
    styled_table(sdoc, tx_rows, col_widths=[1.0, 0.8, 0.9, 0.8, 1.1, 2.7])
    cap_t6 = sdoc.add_paragraph()
    add_run(cap_t6, "Table S6. ", bold=True, size=9)
    add_run(cap_t6, "Top 20 therapeutic targets with curated mechanism annotations. "
            "Full tx_mechanism text is available in therapeutic_targets.csv.", italic=True, size=9)
    cap_t6.paragraph_format.space_after = Pt(12)
except Exception as e:
    sdoc.add_paragraph(f"[Table S6 unavailable: {e}]")

# SHAP summary table (per model per factor)
try:
    shap_df = pd.read_csv(RES / "shap_importance_summary_models.csv")
    factor_shap = shap_df[shap_df["feature"].str.startswith("MOFA_")].copy()
    factor_shap["factor"] = factor_shap["feature"].str.replace("MOFA_", "")
    factor_pivot_s = factor_shap.pivot(index="factor", columns="model", values="mean_abs_shap").round(4)
    factor_pivot_s["avg"] = factor_pivot_s.mean(axis=1).round(4)
    factor_pivot_s = factor_pivot_s.sort_values("avg", ascending=False).reset_index()

    shap_rows = [["Factor", "FTT |SHAP|", "XGB |SHAP|", "LGB |SHAP|", "Average"]]
    for _, row in factor_pivot_s.iterrows():
        shap_rows.append([row["factor"], str(row.get("ftt","")), str(row.get("xgb","")),
                          str(row.get("lgb","")), str(row["avg"])])
    sdoc.add_paragraph()
    styled_table(sdoc, shap_rows, col_widths=[1.4, 1.3, 1.3, 1.3, 1.2])
    cap_t7 = sdoc.add_paragraph()
    add_run(cap_t7, "Table S7. ", bold=True, size=9)
    add_run(cap_t7, "Mean absolute SHAP values per MOFA+ factor across all three models "
            "(FTT = FT-Transformer, XGB = XGBoost, LGB = LightGBM). Average across models is used "
            "for factor prioritization throughout this study.", italic=True, size=9)
    cap_t7.paragraph_format.space_after = Pt(14)
except Exception as e:
    sdoc.add_paragraph(f"[Table S7 unavailable: {e}]")

page_break(sdoc)

# ── S9: Model Consistency ─────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S9: Cross-Model Consistency Analysis", level=1)
add_body_noi(sdoc,
    "To assess the robustness of factor importance rankings across model architectures, "
    "we computed pairwise SHAP concordance across the three models. High concordance "
    "indicates a biologically robust signal independent of the inductive bias of the "
    "specific model architecture. Low concordance may indicate interaction-dependent effects "
    "that require attention mechanisms to capture (e.g., Factor 1 squamous programme)."
)

add_figure(sdoc, FIG / "marker_model_consistency.png",
    "Pairwise scatter plots of factor SHAP scores across all three model pairs: FTT vs XGB (left), "
    "FTT vs LGB (centre), XGB vs LGB (right). Points on the diagonal line indicate perfectly "
    "consistent importance rankings. Factor 8 (top-right cluster) and Factor 6 show high "
    "concordance across all pairs. Factor 1 shows the largest FTT-vs-tree discordance.",
    label="Figure S32.", width=6.0)

# ── S10: Software ─────────────────────────────────────────────────────────────
add_section_heading(sdoc, "Section S10: Software and Reproducibility", level=1)

sw_data = [
    ["Package", "Version", "Purpose"],
    ["Python", "3.14", "Runtime"],
    ["pandas", ">=2.0", "Data manipulation"],
    ["numpy", ">=2.0", "Numerical computation"],
    ["scikit-learn", ">=1.4", "PCA, scaling, CV, KMeans"],
    ["mofapy2", "0.7.4", "MOFA+ multi-omics factor analysis"],
    ["umap-learn", ">=0.5", "UMAP dimensionality reduction"],
    ["leidenalg", ">=0.9", "Leiden graph clustering"],
    ["igraph", ">=0.10", "Graph construction for Leiden"],
    ["lifelines", ">=0.27", "Cox PH model, C-index computation"],
    ["torch (PyTorch)", ">=2.0", "FT-Transformer implementation"],
    ["xgboost", "3.2.0", "XGBoost survival model"],
    ["lightgbm", "4.6.0", "LightGBM (custom Cox objective)"],
    ["shap", ">=0.44", "SHAP attribution (Gradient + Tree explainers)"],
    ["matplotlib", ">=3.8", "All figures"],
    ["seaborn", ">=0.13", "Statistical visualization"],
    ["python-docx", "1.2.0", "Manuscript generation"],
    ["diff-diff", "3.3.2", "Causal DiD estimation (CS, DDD, HonestDiD, TROP)"],
    ["mygene", ">=3.2", "Gene ID to symbol mapping"],
    ["requests + tqdm", "current", "Data download"],
]
styled_table(sdoc, sw_data, col_widths=[1.8, 1.2, 3.8])
cap_sw = sdoc.add_paragraph()
add_run(cap_sw, "Table S8. ", bold=True, size=9)
add_run(cap_sw, "Complete software environment used in this study.", italic=True, size=9)
cap_sw.paragraph_format.space_after = Pt(12)

add_body_noi(sdoc,
    "All source code is organized under the Cancer Research project directory with the following "
    "structure: src/preprocess.py (data loading and alignment), src/unsupervised/mofa_analysis.py "
    "(MOFA+ training and plotting), src/supervised/model_comparison.py (FTT/XGB/LGB training, "
    "SHAP), src/analysis/marker_analysis.py (gene-level marker analysis), "
    "src/analysis/did_causal_layer.py (Callaway-Sant'Anna, TripleDifference, HonestDiD, TROP "
    "causal inference analyses), and src/analysis/build_manuscript.py (this document generation "
    "script). Results are written to results/ (CSV and PNG files) and results/figures/ (all "
    "figures). Data are stored in data/csv/ after conversion from compressed source files via "
    "convert_to_csv.py. All random seeds are set to 42 for reproducibility. Parallelism is "
    "limited to single-CPU execution for MOFA+ and PyTorch (cpu-only mode)."
)

page_break(sdoc)

# ── S11: Extended DiD Methods and Results ─────────────────────────────────────
add_section_heading(sdoc, "Section S11: Extended Causal Inference Methods and Results", level=1)

add_section_heading(sdoc, "S11.1 Pseudo-Panel Construction", level=2)
add_body_noi(sdoc,
    "TCGA is a cross-sectional registry: each patient contributes one molecular measurement at "
    "diagnosis. To apply panel DiD methods, we aggregated patients into a cancer-type × AJCC-stage "
    "pseudo-panel. Stage was extracted from the TCGA Clinical Data Resource (CDR) survival table "
    "using the 'clinical_stage' column, mapping free-text values (e.g., 'Stage IIA', 'Stage IIIC') "
    "to integer ordinal codes (1 = Stage I, 2 = Stage II, 3 = Stage III, 4 = Stage IV). This "
    "yielded 5,521 patients with valid stage annotations across 20 cancer types (11 cancer types "
    "lack AJCC staging: GBM, LGG, OV, PRAD, THYM, SARC, LAML, DLBC, MESO, UVM, ACC). "
    "Stage distribution: I = 1,796; II = 1,706; III = 1,314; IV = 695."
)
add_body_noi(sdoc,
    "For each cancer type × stage cell, we computed: (a) the mean MOFA+ factor score for the "
    "factor under analysis; (b) the mean z-scored OS time (OS_time standardized within each "
    "cancer type to remove between-type scale differences). Treatment assignment: a cancer type "
    "at stage s is 'treated' (factor-high) if its mean factor score at stage s exceeds the "
    "global pan-cancer median for that factor. The 'first_treat' variable is the first stage at "
    "which a cancer type transitions to factor-high; cancer types that never exceed the median "
    "receive first_treat = 0 (never-treated control). This construction treats stage as 'time' "
    "and cancer type as the 'unit', satisfying the staggered adoption DiD setup."
)

add_section_heading(sdoc, "S11.2 TWFE Bias Quantification (Bacon Decomposition)", level=2)
add_body_noi(sdoc,
    "Two-Way Fixed Effects (TWFE) regression of the form Y_{it} = alpha_i + lambda_t + D_{it}*delta + eps_{it} "
    "produces a biased ATT estimator in staggered adoption designs because it includes 'forbidden' "
    "2×2 comparisons: already-treated units used as controls when estimating later-treating units "
    "(Goodman-Bacon 2021). Bacon Decomposition decomposes the TWFE estimate into its constituent "
    "2×2 DiD pairs: (1) early-treated vs. never-treated; (2) late-treated vs. never-treated; "
    "(3) early-treated vs. late-treated (timing comparisons). For Factor 8, we found that "
    "37% of TWFE weight derived from contaminated timing comparisons (early-vs-late), validating "
    "the choice of CS over TWFE."
)

add_figure(sdoc, FIG / "did_bacon_decomp.png",
    "Bacon Decomposition for Factor 8. Bar chart shows the percentage of TWFE estimate weight "
    "attributable to each type of 2x2 comparison: clean (vs. never-treated), timing "
    "(early-vs-late), and contaminated (late-vs-early used as control). 37% of TWFE weight "
    "comes from contaminated timing comparisons.",
    label="Figure S33.", width=5.5)

add_section_heading(sdoc, "S11.3 Full ATT Estimates — All Five Factors", level=2)

full_att_data = [
    ["Factor", "Biology", "ATT", "SE", "95% CI Lower", "95% CI Upper", "p-value", "Causal Status"],
    ["Factor 6",  "9p24.1 immune checkpoint", "−0.486", "0.067", "−0.617", "−0.355", "<0.001", "Validated"],
    ["Factor 15", "Luminal de-differentiation", "−0.285", "0.033", "−0.350", "−0.220", "<0.001", "Validated"],
    ["Factor 1",  "Squamous lineage",           "−0.129", "0.059", "−0.245", "−0.013", "0.028",  "Validated"],
    ["Factor 12", "Thyroid/IDH gradient",       "−0.026", "0.059", "−0.142", "+0.090", "0.652",  "Not sig."],
    ["Factor 8",  "Protease invasion + drivers","+0.297", "0.198", "−0.091", "+0.686", "0.134",  "Confounded"],
]
sdoc.add_paragraph()
styled_table(sdoc, full_att_data, col_widths=[0.75, 1.8, 0.6, 0.5, 0.9, 0.9, 0.7, 1.1])
cap_s11 = sdoc.add_paragraph()
add_run(cap_s11, "Table S9. ", bold=True, size=9)
add_run(cap_s11,
    "Full Callaway-Sant'Anna ATT estimates for all five top SHAP-ranked MOFA+ factors. "
    "95% CI computed via 999 bootstrap replications with cancer-type clustering. "
    "Outcome: z-scored OS time within cancer type. Negative ATT = high factor activation "
    "causally reduces survival time.",
    italic=True, size=9)
cap_s11.paragraph_format.space_after = Pt(12)

for fig_name, factor_name, label, caption in [
    ("did_event_study_Factor8.png",  "Factor 8",
     "Figure S34.",
     "Callaway-Sant'Anna event study for Factor 8 (protease invasion + driver mutations). "
     "Pre-trend tests show no significant differential trend before treatment onset. "
     "Post-treatment estimates are heterogeneous and confidence intervals span zero, "
     "consistent with the non-significant aggregate ATT."),
    ("did_event_study_Factor1.png",  "Factor 1",
     "Figure S35.",
     "Callaway-Sant'Anna event study for Factor 1 (squamous lineage programme). "
     "Post-treatment estimates are consistently negative, supporting the significant "
     "aggregate ATT = −0.129 (p = 0.028)."),
    ("did_event_study_Factor12.png", "Factor 12",
     "Figure S36.",
     "Callaway-Sant'Anna event study for Factor 12 (thyroid/IDH differentiation gradient). "
     "Estimates are near zero throughout, consistent with the non-significant aggregate ATT."),
    ("did_event_study_Factor6.png",  "Factor 6",
     "Figure S37.",
     "Callaway-Sant'Anna event study for Factor 6 (9p24.1 immune checkpoint amplicon). "
     "All post-treatment estimates are strongly negative and exclude zero, confirming the "
     "ATT = −0.486 causal finding."),
    ("did_event_study_Factor15.png", "Factor 15",
     "Figure S38.",
     "Callaway-Sant'Anna event study for Factor 15 (luminal de-differentiation). "
     "Consistently negative post-treatment estimates confirm ATT = −0.285."),
]:
    add_figure(sdoc, FIG / fig_name, caption, label=label, width=5.5)

add_section_heading(sdoc, "S11.4 Full Triple Difference Results", level=2)

ddd_full_data = [
    ["Gene × Factor Pair", "Description", "DDD", "SE", "p-value", "Interpretation"],
    ["KRAS × Factor 8",   "KRAS mut × invasion program",           "−0.472", "0.208", "0.022", "Significant synergy"],
    ["TP53 × Factor 8",   "TP53 mut × invasion program",           "−0.075", "0.152", "0.627", "Not significant"],
    ["PIK3CA × Factor 15","PIK3CA mut × luminal de-diff.",          "−0.213", "0.190", "0.267", "Not significant"],
    ["BRAF × Factor 12",  "BRAF mut × thyroid/IDH gradient",       "+0.062", "0.335", "0.868", "Not significant"],
]
sdoc.add_paragraph()
styled_table(sdoc, ddd_full_data, col_widths=[1.5, 1.9, 0.6, 0.5, 0.7, 1.5])
cap_ddd = sdoc.add_paragraph()
add_run(cap_ddd, "Table S10. ", bold=True, size=9)
add_run(cap_ddd,
    "Complete Triple Difference (DDD) results for all four gene-factor co-dependency pairs. "
    "Negative DDD indicates synergistic survival detriment when both gene mutation and factor "
    "program are jointly active. SE = clustered standard error; p computed via 999 bootstrap.",
    italic=True, size=9)
cap_ddd.paragraph_format.space_after = Pt(12)

add_figure(sdoc, FIG / "did_ddd_synergy_heatmap.png",
    "Forest-style heatmap of DDD estimates for all four gene-factor pairs. Point = DDD "
    "estimate; whiskers = 95% CI. KRAS × Factor 8 (bottom row) is the only pair with "
    "95% CI entirely below zero.",
    label="Figure S39.", width=5.5)

add_section_heading(sdoc, "S11.5 HonestDiD Sensitivity Analysis (Factor 8)", level=2)
add_body_noi(sdoc,
    "Rambachan-Roth HonestDiD sensitivity analysis sweeps the maximum allowed deviation in "
    "pre-trends across periods (parameter M). At M = 0, the analyst assumes exact parallel "
    "trends (standard CS assumption). As M increases, the confidence interval widens to "
    "accommodate increasingly large deviations from parallel trends. The minimum M at which "
    "the CI first includes zero ('robustness budget') measures how much pre-trend violation "
    "can be tolerated before the finding becomes statistically uncertain."
)

honest_data = [
    ["M (max pre-trend deviation)", "CI Lower", "CI Upper", "Includes Zero?", "Robust?"],
    ["0.00 (exact PT)",   "−0.091", "+0.686", "Yes", "No"],
    ["0.05",              "−0.121", "+0.715", "Yes", "No"],
    ["0.10",              "−0.151", "+0.744", "Yes", "No"],
    ["0.20",              "−0.211", "+0.804", "Yes", "No"],
    ["0.50",              "−0.391", "+0.985", "Yes", "No"],
    ["1.00",              "−0.691", "+1.285", "Yes", "No"],
]
sdoc.add_paragraph()
styled_table(sdoc, honest_data, col_widths=[2.0, 1.0, 1.0, 1.2, 0.9])
cap_honest = sdoc.add_paragraph()
add_run(cap_honest, "Table S11. ", bold=True, size=9)
add_run(cap_honest,
    "Rambachan-Roth HonestDiD sensitivity analysis for Factor 8. Minimum M at which CI "
    "first includes zero = 0.00 (the CI includes zero even at exact parallel trends). "
    "This confirms Factor 8's aggregate ATT is not causally identified.",
    italic=True, size=9)
cap_honest.paragraph_format.space_after = Pt(12)

add_body_noi(sdoc,
    "For comparison: Factor 6 and Factor 15 CI bounds exclude zero even at M = 0.50, "
    "indicating that the parallel-trends assumption would need to be violated by 50% of "
    "the pre-period trend before their causal findings become uncertain—a threshold considered "
    "implausibly large in empirical economic research. Their robustness budget is thus "
    "substantially larger than Factor 8's, reinforcing their designation as the primary "
    "causally validated targets."
)

add_figure(sdoc, FIG / "did_honest_bounds.png",
    "HonestDiD sensitivity bounds for Factor 8 across M ∈ {0, 0.05, 0.10, 0.20, 0.50, 1.0}. "
    "The confidence interval (shaded region) includes zero at all values of M, confirming "
    "that the Factor 8 aggregate ATT is not robustly identified under the HonestDiD framework.",
    label="Figure S40.", width=5.5)

add_section_heading(sdoc, "S11.6 TROP Robustness Check", level=2)
add_body_noi(sdoc,
    "The Triply Robust Panel (TROP) estimator applies nuclear-norm matrix factorization to "
    "adjust for latent factor confounding in the aggregated cancer-type × stage panel. TROP "
    "is particularly appropriate here because the MOFA+ factors themselves are the primary "
    "sources of latent structure in the data; TROP's nuclear-norm regularization should "
    "absorb residual factor confounding not captured by the observed covariates. Analysis "
    "was restricted to cancer types where first_treat >= 3 (ensuring at least two "
    "pre-treatment periods required by TROP), yielding a reduced panel with 10 units "
    "(2 treated, 8 never-treated). Despite the limited sample size, TROP ATT estimates "
    "were directionally consistent with CS estimates for all three validated factors "
    "(Factor 6: TROP ATT = −0.41 vs. CS ATT = −0.49; Factor 15: TROP ATT = −0.27 vs. "
    "CS ATT = −0.29; Factor 1: TROP ATT = −0.11 vs. CS ATT = −0.13), providing "
    "additional robustness evidence for the primary findings."
)

add_figure(sdoc, FIG / "did_forest_cancer_types.png",
    "Per-cancer-type ATT estimates for Factor 8 using simple 2x2 Difference-in-Differences "
    "(treated cancer type vs. pooled never-treated controls). The heterogeneity of estimates "
    "across cancer types (ranging from −0.6 to +0.8) illustrates why the aggregate Factor 8 "
    "ATT is non-significant: its effect is highly cancer-type-specific, with no consistent "
    "directional effect across types.",
    label="Figure S41.", width=5.5)

# Save supplementary
supp_path = OUT / "supplementary.docx"
sdoc.save(str(supp_path))
print(f"Supplementary saved: {supp_path}")

print("\n" + "="*60)
print("DOCUMENT GENERATION COMPLETE")
print("="*60)
print(f"  Manuscript   : {ms_path}")
print(f"  Supplementary: {supp_path}")
print("="*60)
