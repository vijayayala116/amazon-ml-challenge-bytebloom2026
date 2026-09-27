"""
eda.py

Exploratory Data Analysis for the Business Entity Resolution dataset.
Run this before/alongside feature engineering to understand the data:
  - country balance across the 3 sources
  - missing-value patterns (business_name, business_address, country)
  - legal-suffix / abbreviation noise in business names
  - duplicate-name frequency
  - ground-truth match distribution (singletons, avg matches, S2 vs S3 split)

Usage:
    python eda.py \
        --data-dir dataset/train \
        --ground-truth dataset/train/train_ground_truth.tsv \
        --output-dir eda_report
"""

import argparse
from collections import Counter
from pathlib import Path

import pandas as pd


LEGAL_WORDS = [
    "pvt", "private", "ltd", "limited", "inc", "incorporated",
    "corp", "corporation", "llc", "llp", "co", "company",
]


def country_distribution(df, source_name, report_lines):
    counts = df["country"].value_counts()
    report_lines.append(f"\n[{source_name}] country distribution:")
    report_lines.append(counts.to_string())


def missing_value_report(df, source_name, report_lines):
    report_lines.append(f"\n[{source_name}] missing / empty values:")
    for col in ["business_name", "business_address", "country"]:
        n_missing = (df[col].isna() | (df[col].astype(str).str.strip() == "")).sum()
        pct = n_missing / len(df) * 100
        report_lines.append(f"  {col}: {n_missing} ({pct:.2f}%)")


def legal_suffix_frequency(df, source_name, report_lines):
    text = " ".join(df["business_name"].dropna().astype(str).str.lower())
    tokens = text.split()
    counts = Counter(tokens)
    report_lines.append(f"\n[{source_name}] legal-suffix word frequency:")
    for word in LEGAL_WORDS:
        report_lines.append(f"  {word}: {counts.get(word, 0)}")


def duplicate_name_count(df, source_name, report_lines):
    dupes = df["business_name"].astype(str).str.lower().str.strip().duplicated().sum()
    report_lines.append(
        f"\n[{source_name}] rows sharing an exact (case-insensitive) name with another row: {dupes}"
    )


def ground_truth_report(gt_path, report_lines):
    gt = pd.read_csv(gt_path, sep="\t", dtype=str, keep_default_na=False)

    total = len(gt)
    singletons = (gt["matched_entity_ids"] == "").sum()

    match_counts, s2_count, s3_count = [], 0, 0
    for ids in gt["matched_entity_ids"]:
        if not ids:
            match_counts.append(0)
            continue
        parts = ids.split(",")
        match_counts.append(len(parts))
        s2_count += sum(1 for p in parts if p.startswith("S2-"))
        s3_count += sum(1 for p in parts if p.startswith("S3-"))

    non_singleton = [m for m in match_counts if m > 0]

    report_lines.append("\n[Ground truth] summary:")
    report_lines.append(f"  Total Source1 entities: {total}")
    report_lines.append(f"  Singletons (no match): {singletons} ({singletons / total * 100:.2f}%)")
    report_lines.append(
        f"  Avg matches per non-singleton entity: {sum(non_singleton) / len(non_singleton):.2f}"
    )
    report_lines.append(f"  Max matches for a single entity: {max(match_counts)}")
    report_lines.append(f"  Total match links -> Source2: {s2_count}, Source3: {s3_count}")


def run_eda(data_dir, ground_truth_path, output_dir):
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    report_lines = []

    for i in (1, 2, 3):
        path = next(data_dir.glob(f"*_source{i}.tsv"))
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        source_name = f"Source{i}"

        country_distribution(df, source_name, report_lines)
        missing_value_report(df, source_name, report_lines)
        legal_suffix_frequency(df, source_name, report_lines)
        duplicate_name_count(df, source_name, report_lines)

    if ground_truth_path:
        ground_truth_report(ground_truth_path, report_lines)

    report_text = "\n".join(report_lines)
    print(report_text)

    report_path = output_dir / "eda_summary.txt"
    report_path.write_text(report_text, encoding="utf-8")
    print(f"\nSaved report to {report_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EDA for the entity resolution dataset.")
    parser.add_argument("--data-dir", required=True, help="Directory with *_source1/2/3.tsv files.")
    parser.add_argument("--ground-truth", default=None, help="Path to train_ground_truth.tsv (optional).")
    parser.add_argument("--output-dir", default="eda_report", help="Where to save the report file.")
    args = parser.parse_args()

    run_eda(args.data_dir, args.ground_truth, args.output_dir)