"""
Download TCGA pan-cancer multiomics data from UCSC Xena hubs.
All files are open-access (no GDC token required).

Sources:
  - pancanatlas.xenahubs.net  →  mutations, methylation, miRNA, RPPA, clinical phenotype,
                                  TCGA-CDR survival (Liu et al. 2018 Cell — all 33 types)
  - toil.xenahubs.net         →  gene expression (TOIL-recomputed, cross-study normalised)
  - tcga.xenahubs.net         →  copy number (GISTIC2 + SNP6 segments)

Usage:
    python download_tcga.py
    python download_tcga.py --modalities expression,mutations,clinical
    python download_tcga.py --data-dir /path/to/storage --modalities all
"""

import argparse
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

XENA_PANCAN = "https://pancanatlas.xenahubs.net/download"
XENA_TOIL   = "https://toil.xenahubs.net/download"
XENA_TCGA   = "https://tcga.xenahubs.net/download"
GDC_API     = "https://api.gdc.cancer.gov"

# Registry: modality_key -> list of (url, local_filename, subfolder)
DATASETS: dict[str, list[tuple[str, str, str]]] = {
    "expression": [
        (
            f"{XENA_TOIL}/tcga_RSEM_gene_tpm.gz",
            "tcga_RSEM_gene_tpm.gz",
            "expression",
        ),
        (
            f"{XENA_TOIL}/tcga_RSEM_Hugo_norm_count.gz",
            "tcga_RSEM_Hugo_norm_count.gz",
            "expression",
        ),
    ],
    "mutations": [
        (
            f"{XENA_PANCAN}/mc3.v0.2.8.PUBLIC.xena.gz",
            "mc3.v0.2.8.PUBLIC.xena.gz",
            "mutations",
        ),
    ],
    "methylation": [
        (
            f"{XENA_PANCAN}/jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv"
            ".synapse_download_5096262.xena.gz",
            "jhu-usc.edu_PANCAN_HumanMethylation450.betaValue_whitelisted.tsv"
            ".synapse_download_5096262.xena.gz",
            "methylation",
        ),
    ],
    "copy_number": [
        (
            f"{XENA_TCGA}/TCGA.PANCAN.sampleMap/SNP6_genomicSegment.gz",
            "SNP6_genomicSegment.gz",
            "copy_number",
        ),
        (
            f"{XENA_TCGA}/TCGA.PANCAN.sampleMap/Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
            "Gistic2_CopyNumber_Gistic2_all_thresholded.by_genes.gz",
            "copy_number",
        ),
    ],
    "mirna": [
        (
            f"{XENA_PANCAN}/pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz",
            "pancanMiRs_EBadjOnProtocolPlatformWithoutRepsWithUnCorrectMiRs_08_04_16.xena.gz",
            "mirna",
        ),
    ],
    "protein": [
        (
            f"{XENA_PANCAN}/TCGA-RPPA-pancan-clean.xena.gz",
            "TCGA-RPPA-pancan-clean.xena.gz",
            "protein",
        ),
    ],
    "clinical": [
        (
            f"{XENA_PANCAN}/TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
            "TCGA_phenotype_denseDataOnlyDownload.tsv.gz",
            "clinical",
        ),
        # TCGA Clinical Data Resource (Liu et al. 2018 Cell) — curated OS/PFI/DFI/DSS
        # for all 11,160 TCGA patients across all 33 cancer types.
        (
            f"{XENA_PANCAN}/Survival_SupplementalTable_S1_20171025_xena_sp",
            "Survival_SupplementalTable_S1_20171025_xena_sp.tsv",
            "clinical",
        ),
    ],
}

ALL_MODALITIES = list(DATASETS.keys())
CHUNK_SIZE  = 8 * 1024 * 1024  # 8 MB
MAX_RETRIES = 3
RETRY_DELAY = 10  # seconds


def download_file(url: str, dest: Path, skip_existing: bool = True) -> bool:
    if skip_existing and dest.exists() and dest.stat().st_size > 0:
        print(f"  [skip] {dest.name} already exists ({dest.stat().st_size / 1e6:.1f} MB)")
        return True

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                total = int(r.headers.get("content-length", 0))
                with (
                    open(dest, "wb") as f,
                    tqdm(
                        total=total,
                        unit="B",
                        unit_scale=True,
                        unit_divisor=1024,
                        desc=dest.name[:50],
                        leave=True,
                    ) as bar,
                ):
                    for chunk in r.iter_content(chunk_size=CHUNK_SIZE):
                        f.write(chunk)
                        bar.update(len(chunk))
            return True
        except (requests.RequestException, OSError) as exc:
            print(f"  [attempt {attempt}/{MAX_RETRIES}] Error: {exc}")
            if dest.exists():
                dest.unlink()
            if attempt < MAX_RETRIES:
                print(f"  Retrying in {RETRY_DELAY}s...")
                time.sleep(RETRY_DELAY)

    print(f"  [FAILED] {dest.name} after {MAX_RETRIES} attempts.")
    return False


def verify_downloads(data_dir: Path, modalities: list[str]) -> None:
    print("\n--- Verification ---")
    ok = failed = 0
    for modality in modalities:
        for _url, filename, subfolder in DATASETS[modality]:
            dest = data_dir / subfolder / filename
            if dest.exists() and dest.stat().st_size > 0:
                print(f"  OK  {filename[:60]:<60}  {dest.stat().st_size / 1e6:>8.1f} MB")
                ok += 1
            else:
                print(f"  {'EMPTY' if dest.exists() else 'MISSING'}  {filename}")
                failed += 1

    print(f"\n  {ok} OK  |  {failed} failed/missing")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download TCGA pan-cancer data.")
    parser.add_argument(
        "--modalities",
        default="all",
        help=f"Comma-separated modalities (default: all). Available: {', '.join(ALL_MODALITIES)}",
    )
    parser.add_argument(
        "--data-dir",
        default="data/xena_pancan",
        help="Root directory for downloaded data (default: data/xena_pancan)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Re-download files even if they already exist",
    )
    args = parser.parse_args()

    if args.modalities.strip().lower() == "all":
        modalities = ALL_MODALITIES
    else:
        modalities = [m.strip() for m in args.modalities.split(",")]
        invalid = [m for m in modalities if m not in DATASETS]
        if invalid:
            print(f"Unknown modalities: {invalid}. Available: {ALL_MODALITIES}")
            sys.exit(1)

    data_dir = Path(args.data_dir)
    skip_existing = not args.no_skip

    subfolders = {sf for mod in modalities for _, _, sf in DATASETS[mod]}
    subfolders.add("clinical")
    for sf in subfolders:
        (data_dir / sf).mkdir(parents=True, exist_ok=True)

    print(f"Downloading {len(modalities)} modality group(s): {', '.join(modalities)}")
    print(f"Destination: {data_dir.resolve()}\n")

    success_count = 0
    total_count = sum(len(DATASETS[m]) for m in modalities)

    for modality in modalities:
        print(f"\n=== {modality.upper()} ===")
        for url, filename, subfolder in DATASETS[modality]:
            dest = data_dir / subfolder / filename
            if download_file(url, dest, skip_existing=skip_existing):
                success_count += 1

    verify_downloads(data_dir, modalities)
    print(f"\nDone: {success_count}/{total_count} files downloaded successfully.")


if __name__ == "__main__":
    main()
