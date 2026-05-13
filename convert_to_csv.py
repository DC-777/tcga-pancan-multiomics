"""
Convert downloaded TCGA .gz (TSV) files to CSV format.
Run after download_tcga.py has completed.

Usage:
    python convert_to_csv.py
    python convert_to_csv.py --data-dir data/xena_pancan --out-dir data/csv
    python convert_to_csv.py --skip-methylation   # skip the ~80 GB methylation CSV
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from tqdm import tqdm


# Maps each source .gz file to its output CSV name.
# Methylation is flagged separately because it requires chunked I/O.
CONVERSIONS = [
    # (subfolder, gz_filename, csv_filename, is_large)
    (
        "expression",
        "tcga_RSEM_Hugo_norm_count.gz",
        "expression_RSEM_Hugo_norm_count.csv",
        False,
    ),
    (
        "expression",
        "tcga_RSEM_gene_tpm.gz",
        "expression_RSEM_gene_tpm.csv",
        False,
    ),
    (
        "mutations",
        "mc3.v0.2.8.PUBLIC.xena.gz",
        "mutations_mc3_somatic.csv",
        False,
    ),
    (
        "methylation",
        (
            "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv"
            ".synapse_download_5096262.xena.gz"
        ),
        "methylation_HM450_beta.csv",
        True,  # large — processed in chunks
    ),
    (
        "copy_number",
        "SNP6_genomicSegment.gz",
        "copy_number_SNP6_segments.csv",
        False,
    ),
    (
        "copy_number",
        "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
        "copy_number_GISTIC2_gene_level.csv",
        False,
    ),
    (
        "mirna",
        "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz",
        "mirna_expression.csv",
        False,
    ),
    (
        "protein",
        "TCGA-RPPA-pancan-clean.xena.gz",
        "protein_RPPA.csv",
        False,
    ),
    (
        "clinical",
        "TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
        "clinical_phenotype.csv",
        False,
    ),
    # TCGA-CDR survival (Liu et al. 2018 Cell) — plain TSV, no compression
    (
        "clinical",
        "Survival_SupplementalTable_S1_20171025_xena_sp.tsv",
        "clinical_survival_cdr.csv",
        False,
    ),
]

METHYLATION_CHUNKSIZE = 500  # rows per chunk (~500 CpG probes at a time)


def convert_standard(src: Path, dest: Path) -> None:
    print(f"  Reading {src.name} ...")
    compression = "gzip" if src.suffix == ".gz" else None
    df = pd.read_csv(
        src, sep="\t", index_col=0, compression=compression,
        on_bad_lines="skip", encoding="utf-8", encoding_errors="replace",
    )
    print(f"  Shape: {df.shape}  ->  writing {dest.name} ...")
    df.to_csv(dest, encoding="utf-8")
    size_mb = dest.stat().st_size / 1e6
    print(f"  Saved {dest.name} ({size_mb:.1f} MB)")


def convert_chunked(src: Path, dest: Path, chunksize: int = METHYLATION_CHUNKSIZE) -> None:
    """Write methylation CSV in chunks to avoid loading ~10 GB into RAM at once."""
    print(f"  Reading {src.name} in chunks of {chunksize} rows ...")
    reader = pd.read_csv(src, sep="\t", index_col=0, compression="gzip", chunksize=chunksize)

    header_written = False
    total_rows = 0
    with tqdm(unit="chunk", desc="methylation") as bar:
        for chunk in reader:
            chunk.to_csv(dest, mode="a", header=not header_written)
            header_written = True
            total_rows += len(chunk)
            bar.update(1)
            bar.set_postfix(rows=f"{total_rows:,}")

    size_gb = dest.stat().st_size / 1e9
    print(f"  Saved {dest.name} ({size_gb:.2f} GB, {total_rows:,} CpG probes)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert TCGA Xena .gz files to CSV."
    )
    parser.add_argument(
        "--data-dir",
        default="data/xena_pancan",
        help="Root directory containing downloaded .gz files (default: data/xena_pancan)",
    )
    parser.add_argument(
        "--out-dir",
        default="data/csv",
        help="Output directory for CSV files (default: data/csv)",
    )
    parser.add_argument(
        "--skip-methylation",
        action="store_true",
        help="Skip methylation (saves ~80 GB of disk space)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Reconvert even if the CSV already exists",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ok = failed = skipped = 0

    for subfolder, gz_name, csv_name, is_large in CONVERSIONS:
        if is_large and args.skip_methylation:
            print(f"[skip] {gz_name} (--skip-methylation)")
            skipped += 1
            continue

        src = data_dir / subfolder / gz_name
        dest = out_dir / csv_name

        if not src.exists():
            print(f"[missing] {src} — run download_tcga.py first")
            failed += 1
            continue

        if dest.exists() and dest.stat().st_size > 0 and not args.no_skip:
            size_mb = dest.stat().st_size / 1e6
            print(f"[skip] {csv_name} already exists ({size_mb:.1f} MB)")
            skipped += 1
            continue

        print(f"\n--- {csv_name} ---")
        try:
            if is_large:
                convert_chunked(src, dest)
            else:
                convert_standard(src, dest)
            ok += 1
        except Exception as exc:
            print(f"  [ERROR] {exc}")
            if dest.exists():
                dest.unlink()
            failed += 1

    print(f"\n=== Done: {ok} converted | {skipped} skipped | {failed} failed ===")
    print(f"CSV files saved to: {out_dir.resolve()}")


if __name__ == "__main__":
    main()
