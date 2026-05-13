"""
Convenience loaders for TCGA pan-cancer data downloaded by download_tcga.py.

All functions return pandas DataFrames. For methylation (large file),
pass chunksize to get a TextFileReader iterator instead of reading into memory.

Example:
    from load_tcga import load_expression, load_clinical, load_all

    expr = load_expression("data/xena_pancan")   # genes x samples
    clin = load_clinical("data/xena_pancan")      # samples x features

    data = load_all("data/xena_pancan", modalities=["expression", "clinical"])
"""

import pandas as pd
from pathlib import Path


def _gz(data_dir: str | Path, subfolder: str, filename: str) -> Path:
    return Path(data_dir) / subfolder / filename


def load_expression(
    data_dir: str | Path = "data/xena_pancan",
    normalized: bool = True,
) -> pd.DataFrame:
    """
    Load pan-cancer gene expression matrix.

    normalized=True  → RSEM Hugo normalized counts (log2)
    normalized=False → RSEM TPM (log2)

    Returns DataFrame indexed by gene symbol, columns are TCGA sample barcodes.
    """
    filename = (
        "tcga_RSEM_Hugo_norm_count.gz" if normalized else "tcga_RSEM_gene_tpm.gz"
    )
    path = _gz(data_dir, "expression", filename)
    return pd.read_csv(path, sep="\t", index_col=0, compression="gzip")


def load_mutations(data_dir: str | Path = "data/xena_pancan") -> pd.DataFrame:
    """
    Load MC3 somatic mutation calls (MAF-style).
    Returns a flat DataFrame; one row per mutation event.
    """
    path = _gz(data_dir, "mutations", "mc3.v0.2.8.PUBLIC.xena.gz")
    return pd.read_csv(path, sep="\t", compression="gzip")


def load_methylation(
    data_dir: str | Path = "data/xena_pancan",
    chunksize: int | None = None,
) -> pd.DataFrame:
    """
    Load HM450 DNA methylation beta values (~10 GB uncompressed).

    Pass chunksize (e.g. 1000) to get a TextFileReader iterator to
    process in chunks without loading the full matrix into memory.

    Returns DataFrame (or iterator) indexed by CpG probe ID,
    columns are TCGA sample barcodes.
    """
    filename = (
        "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv"
        ".synapse_download_5096262.xena.gz"
    )
    path = _gz(data_dir, "methylation", filename)
    return pd.read_csv(
        path, sep="\t", index_col=0, compression="gzip", chunksize=chunksize
    )


def load_copy_number(
    data_dir: str | Path = "data/xena_pancan",
    gene_level: bool = True,
) -> pd.DataFrame:
    """
    Load copy number data.

    gene_level=True  → GISTIC2 gene-level thresholded values (-2/-1/0/1/2)
    gene_level=False → SNP6 segment-level copy number (log2 ratio)
    """
    if gene_level:
        filename = "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz"
        return pd.read_csv(
            _gz(data_dir, "copy_number", filename),
            sep="\t", index_col=0, compression="gzip",
        )
    else:
        filename = "SNP6_genomicSegment.gz"
        return pd.read_csv(
            _gz(data_dir, "copy_number", filename),
            sep="\t", compression="gzip",
        )


def load_mirna(data_dir: str | Path = "data/xena_pancan") -> pd.DataFrame:
    """
    Load pan-cancer miRNA expression (RPM, log2 transformed).
    Returns DataFrame indexed by miRNA ID, columns are sample barcodes.
    """
    filename = (
        "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz"
    )
    path = _gz(data_dir, "mirna", filename)
    return pd.read_csv(path, sep="\t", index_col=0, compression="gzip")


def load_protein(data_dir: str | Path = "data/xena_pancan") -> pd.DataFrame:
    """
    Load RPPA protein expression (~200 antibodies).
    Returns DataFrame indexed by antibody/protein, columns are sample barcodes.
    """
    path = _gz(data_dir, "protein", "TCGA-RPPA-pancan-clean.xena.gz")
    return pd.read_csv(path, sep="\t", index_col=0, compression="gzip")


def load_clinical(data_dir: str | Path = "data/xena_pancan") -> pd.DataFrame:
    """
    Load clinical phenotype annotations.
    Returns DataFrame indexed by TCGA sample barcode.
    """
    path = _gz(data_dir, "clinical", "TCGA_phenotype_denseDataOnlyDownload.tsv.gz")
    return pd.read_csv(path, sep="\t", index_col=0, compression="gzip")


def load_survival(data_dir: str | Path = "data/xena_pancan") -> pd.DataFrame:
    """
    Load Pan-Cancer survival/clinical data from GDC API download.
    Returns DataFrame indexed by TCGA case submitter_id.
    """
    path = Path(data_dir) / "clinical" / "gdc_clinical_survival.tsv"
    return pd.read_csv(path, sep="\t", index_col=0)


def load_sample_types(data_dir: str | Path = "data/xena_pancan") -> pd.DataFrame:
    """
    Load sample type annotations (tumor vs. normal, cancer type, primary disease).
    Sourced from TCGA_phenotype_denseDataOnlyDownload.tsv.gz which includes
    sample_type, sample_type_id, and _primary_disease columns.
    """
    return load_clinical(data_dir)


_LOADERS = {
    "expression": load_expression,
    "mutations": load_mutations,
    "copy_number": load_copy_number,
    "mirna": load_mirna,
    "protein": load_protein,
    "clinical": load_clinical,
    "survival": load_survival,
    "sample_types": load_sample_types,
}


def load_all(
    data_dir: str | Path = "data/xena_pancan",
    modalities: list[str] | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Load multiple modalities into a dict keyed by modality name.
    Methylation is excluded by default (too large); add explicitly if needed.

    modalities: list of keys from:
        expression, mutations, copy_number, mirna, protein,
        clinical, survival, sample_types, methylation
    """
    if modalities is None:
        modalities = [k for k in _LOADERS if k != "methylation"]

    result: dict[str, pd.DataFrame] = {}
    for name in modalities:
        if name == "methylation":
            print("Warning: methylation is large — call load_methylation() directly with chunksize.")
            continue
        if name not in _LOADERS:
            raise ValueError(f"Unknown modality '{name}'. Available: {list(_LOADERS)}")
        print(f"Loading {name}...")
        result[name] = _LOADERS[name](data_dir)
        print(f"  {result[name].shape}")
    return result
