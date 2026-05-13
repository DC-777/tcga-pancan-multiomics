"""
Phase 1: Data loading, cleaning, and alignment across 4 TCGA modalities.

Outputs a MultiomicsDataset dataclass with aligned DataFrames:
  - expression  : samples x genes  (log2-transformed, top N variable genes)
  - mutations   : samples x genes  (binary, nonsynonymous only)
  - cnv         : samples x genes  (GISTIC2 thresholded -2..+2)
  - clinical    : samples x features (cancer_type, OS_time, OS_event, age, stage)

All DataFrames share the same row index (TCGA 15-char sample barcodes,
primary tumor samples only: barcode suffix -01).

Usage:
    from src.preprocess import load_multiomics
    ds = load_multiomics()          # uses defaults
    ds = load_multiomics(n_top_genes=5000, min_mut_freq=0.02)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path
import numpy as np
import pandas as pd

# ── constants ──────────────────────────────────────────────────────────────
CSV_DIR = Path("data/csv")

NONSYNONYMOUS_EFFECTS = {
    "Missense_Mutation",
    "Nonsense_Mutation",
    "Frame_Shift_Del",
    "Frame_Shift_Ins",
    "Splice_Site",
    "In_Frame_Del",
    "In_Frame_Ins",
    "Nonstop_Mutation",
    "Translation_Start_Site",
    "large deletion",
}

# TOIL sentinel for zero/very-low expression (log2(0.001))
EXPR_FLOOR = -9.95


# ── data container ─────────────────────────────────────────────────────────
@dataclass
class MultiomicsDataset:
    expression: pd.DataFrame    # samples x genes
    mutations: pd.DataFrame     # samples x genes (binary)
    cnv: pd.DataFrame           # samples x genes
    clinical: pd.DataFrame      # samples x clinical features
    cancer_types: pd.Series     # samples -> cancer type label

    @property
    def samples(self) -> pd.Index:
        return self.expression.index

    @property
    def n_samples(self) -> int:
        return len(self.samples)

    def summary(self) -> None:
        print(f"MultiomicsDataset: {self.n_samples:,} primary tumor samples")
        print(f"  expression : {self.expression.shape}  (samples x genes)")
        print(f"  mutations  : {self.mutations.shape}  (samples x genes)")
        print(f"  cnv        : {self.cnv.shape}  (samples x genes)")
        print(f"  clinical   : {self.clinical.shape}  (samples x features)")
        print(f"  cancer types: {self.cancer_types.nunique()} types")
        alive = (self.clinical["OS_event"] == 0).sum()
        dead  = (self.clinical["OS_event"] == 1).sum()
        print(f"  survival   : {dead:,} deceased / {alive:,} alive")


# ── loaders ────────────────────────────────────────────────────────────────

def _load_expression(csv_dir: Path, n_top_genes: int) -> pd.DataFrame:
    """
    Load TOIL-normalized log2(TPM) expression.
    Returns DataFrame: samples x genes (top n_top_genes by variance).
    Strips Ensembl version suffix (.N) from gene IDs.
    """
    print("Loading expression ...")
    expr = pd.read_csv(
        csv_dir / "expression_RSEM_gene_tpm.csv",
        index_col=0,
        encoding="utf-8",
    )
    # expr is genes x samples; index = Ensembl IDs with version
    expr.index = expr.index.str.split(".").str[0]   # strip .version
    expr = expr.T                                    # -> samples x genes

    # Replace floor sentinel with 0 before variance calculation
    expr = expr.replace(EXPR_FLOOR, 0.0)
    expr[expr < EXPR_FLOOR + 0.1] = 0.0

    # Variance filter
    gene_var = expr.var(axis=0)
    top_genes = gene_var.nlargest(n_top_genes).index
    expr = expr[top_genes]

    print(f"  Expression: {expr.shape}  ({n_top_genes} top-variance genes)")
    return expr


def _load_mutations(csv_dir: Path, min_freq: float) -> pd.DataFrame:
    """
    Load MC3 MAF, pivot to binary samples x genes matrix.
    Keeps only nonsynonymous mutations; drops genes mutated in < min_freq of samples.
    """
    print("Loading mutations ...")
    mut = pd.read_csv(
        csv_dir / "mutations_mc3_somatic.csv",
        usecols=["sample", "gene", "effect"],
        encoding="utf-8",
    )
    # Filter nonsynonymous
    mut = mut[mut["effect"].isin(NONSYNONYMOUS_EFFECTS)].copy()

    # Standardise sample barcode to 15 chars (some MC3 barcodes are longer)
    mut["sample"] = mut["sample"].str[:15]

    # Deduplicate (same gene mutated multiple times in same sample)
    mut = mut[["sample", "gene"]].drop_duplicates()

    # Pivot to binary matrix
    mut["mutated"] = 1
    mat = mut.pivot_table(
        index="sample", columns="gene", values="mutated", aggfunc="max", fill_value=0
    )
    mat.columns.name = None

    # Frequency filter
    freq = mat.mean(axis=0)
    mat = mat.loc[:, freq >= min_freq]

    print(f"  Mutations: {mat.shape}  (genes mutated in >={min_freq*100:.0f}% of samples)")
    return mat


def _load_cnv(csv_dir: Path) -> pd.DataFrame:
    """
    Load GISTIC2 gene-level copy number (-2/-1/0/+1/+2).
    Returns DataFrame: samples x genes.
    """
    print("Loading CNV ...")
    cnv = pd.read_csv(
        csv_dir / "copy_number_GISTIC2_gene_level.csv",
        index_col=0,
        encoding="utf-8",
    )
    cnv = cnv.T    # -> samples x genes
    cnv = cnv.astype(np.float32)
    print(f"  CNV: {cnv.shape}")
    return cnv


def _load_clinical(csv_dir: Path) -> tuple[pd.DataFrame, pd.Series]:
    """
    Load + merge phenotype and TCGA-CDR survival data.

    Survival source: Liu et al. 2018 Cell (TCGA Clinical Data Resource).
    File: Survival_SupplementalTable_S1_20171025_xena_sp
    Coverage: 11,081 unique patients across all 33 TCGA cancer types.
    Join key: 15-char sample barcode (direct, no patient-prefix workaround needed).

    Returns:
        clinical     : DataFrame indexed by 15-char sample barcode
        cancer_types : Series indexed by 15-char sample barcode
    """
    print("Loading clinical / survival ...")

    # -- Phenotype (sample-level)
    phen = pd.read_csv(csv_dir / "clinical_phenotype.csv", index_col=0, encoding="utf-8")
    phen.index.name = "sample"
    phen.index = phen.index.str[:15]

    # Keep primary tumor samples only (sample_type_id == 1)
    phen = phen[phen["sample_type_id"] == 1.0].copy()
    cancer_types = phen["_primary_disease"].str.strip()

    # -- TCGA-CDR survival (sample-level, 15-char barcode index)
    # Columns of interest:
    #   sample                              : 15-char sample barcode (join key)
    #   cancer type abbreviation            : BRCA / LGG / ...
    #   OS                                  : 1 = deceased, 0 = alive/censored
    #   OS.time                             : days to death or last follow-up
    #   age_at_initial_pathologic_diagnosis : years (already converted from days)
    cdr_path = csv_dir / "clinical_survival_cdr.csv"
    cdr = pd.read_csv(cdr_path, sep="\t", low_memory=False, encoding="utf-8")
    cdr = cdr.rename(columns={
        "sample":                                   "sample",
        "cancer type abbreviation":                 "cancer_type_cdr",
        "OS":                                       "OS_event",
        "OS.time":                                  "OS_time",
        "age_at_initial_pathologic_diagnosis":      "age_at_dx",
    })
    cdr = cdr.set_index("sample")
    cdr["OS_event"] = pd.to_numeric(cdr["OS_event"], errors="coerce")
    cdr["OS_time"]  = pd.to_numeric(cdr["OS_time"],  errors="coerce")
    cdr["age_at_dx"] = pd.to_numeric(cdr["age_at_dx"], errors="coerce")

    # Keep only rows with valid OS information
    cdr_valid = cdr[cdr["OS_time"].notna() & (cdr["OS_time"] > 0)].copy()

    # Join: phenotype index is 15-char sample barcode; CDR index is also 15-char
    phen_surv = phen.join(
        cdr_valid[["OS_time", "OS_event", "age_at_dx", "cancer_type_cdr"]],
        how="left"
    )

    clinical = phen_surv[["_primary_disease", "OS_time", "OS_event", "age_at_dx", "cancer_type_cdr"]].copy()
    clinical.columns = ["cancer_type", "OS_time", "OS_event", "age_at_dx", "cancer_type_cdr"]

    n_surv = clinical["OS_time"].notna().sum()
    n_events = int(clinical["OS_event"].sum()) if clinical["OS_event"].notna().any() else 0
    print(f"  Clinical: {clinical.shape}  ({n_surv:,} with survival data, {n_events:,} events)")
    return clinical, cancer_types


# ── main entry point ────────────────────────────────────────────────────────

def load_multiomics(
    csv_dir: str | Path = CSV_DIR,
    n_top_genes: int = 5000,
    min_mut_freq: float = 0.02,
    require_survival: bool = False,
) -> MultiomicsDataset:
    """
    Load and align all 4 TCGA modalities for primary tumor samples.

    Args:
        csv_dir        : directory containing CSV files from convert_to_csv.py
        n_top_genes    : number of top-variance genes to keep from expression
        min_mut_freq   : minimum fraction of samples with a nonsynonymous mutation
                         to include a gene in the mutation matrix
        require_survival: if True, drop samples without OS_time/OS_event

    Returns:
        MultiomicsDataset with aligned DataFrames
    """
    csv_dir = Path(csv_dir)

    expr = _load_expression(csv_dir, n_top_genes)
    mut  = _load_mutations(csv_dir, min_mut_freq)
    cnv  = _load_cnv(csv_dir)
    clin, cancer_types = _load_clinical(csv_dir)

    # ── sample intersection across all 4 modalities ──
    print("\nAligning samples across modalities ...")
    common = expr.index.intersection(mut.index) \
                       .intersection(cnv.index) \
                       .intersection(clin.index)

    # Further filter to primary tumor phenotype (sample barcode ends in -01)
    primary = [s for s in common if s[-2:] == "01"]
    common  = pd.Index(primary)

    if require_survival:
        has_surv = clin.loc[common, "OS_time"].notna() & clin.loc[common, "OS_event"].notna()
        common = common[has_surv.values]

    expr  = expr.loc[common]
    mut   = mut.loc[common].astype(np.int8)
    cnv   = cnv.loc[common]
    clin  = clin.loc[common]
    ctypes = cancer_types.reindex(common)

    print(f"  Final cohort: {len(common):,} primary tumor samples")
    print(f"  Cancer types represented: {ctypes.nunique()}")

    # Rename expression columns Ensembl ID -> HGNC symbol where available
    symbol_map_path = Path("results/gene_id_to_symbol.csv")
    if symbol_map_path.exists():
        sym = pd.read_csv(symbol_map_path, index_col=0)["symbol"]
        n_mapped = sum(g in sym.index for g in expr.columns)
        expr.columns = [sym.get(g, g) for g in expr.columns]
        print(f"  Gene symbols resolved: {n_mapped} / {len(expr.columns)}")

    return MultiomicsDataset(
        expression=expr,
        mutations=mut,
        cnv=cnv,
        clinical=clin,
        cancer_types=ctypes,
    )


if __name__ == "__main__":
    ds = load_multiomics()
    ds.summary()
    # Save aligned matrices for downstream use
    out = Path("results")
    out.mkdir(exist_ok=True)
    print("\nSaving aligned matrices to results/ ...")
    ds.expression.to_csv(out / "aligned_expression.csv")
    ds.mutations.to_csv(out / "aligned_mutations.csv")
    ds.cnv.to_csv(out / "aligned_cnv.csv")
    ds.clinical.to_csv(out / "aligned_clinical.csv")
    print("Done.")
