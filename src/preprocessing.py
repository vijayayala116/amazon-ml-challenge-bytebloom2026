import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd


# ---------------------------------------------------------
# 1. Normalize text
# ---------------------------------------------------------
def normalize_text(value):
    """
    Normalize business names and addresses while preserving
    letters, numbers, and Unicode characters from different languages.
    """

    if pd.isna(value):
        return ""

    text = str(value)

    # Normalize Unicode representation
    text = unicodedata.normalize("NFKC", text)

    # Convert to lowercase
    text = text.casefold()

    # Keep Unicode letters, marks, numbers, and whitespace.
    # Remove punctuation and symbols.
    cleaned = []

    for char in text:
        category = unicodedata.category(char)

        if (
            category.startswith("L") or   # Letter
            category.startswith("M") or   # Combining mark
            category.startswith("N") or   # Number
            category.startswith("Z")       # Unicode separator/space
        ):
            cleaned.append(char)
        else:
            cleaned.append(" ")

    text = "".join(cleaned)

    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# 2. Preprocess one chunk
# ---------------------------------------------------------
def preprocess_chunk(df):
    """
    Apply preprocessing to one chunk of the source dataset.
    Original columns are preserved.
    """

    required_columns = [
        "entity_id",
        "business_name",
        "business_address",
        "country",
    ]

    # Check that all required columns exist
    missing_columns = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns: {missing_columns}"
        )

    # Create normalized business name
    df["business_name_normalized"] = (
        df["business_name"]
        .apply(normalize_text)
    )

    # Create normalized business address
    df["business_address_normalized"] = (
        df["business_address"]
        .apply(normalize_text)
    )

    # Create normalized country
    df["country_normalized"] = (
        df["country"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.casefold()
    )

    return df


# ---------------------------------------------------------
# 3. Inspect data quality
# ---------------------------------------------------------
def inspect_file(input_file, chunk_size=100_000):
    """
    Inspect a large TSV file chunk-by-chunk.

    Checks:
    - total rows
    - missing values
    - duplicate entity IDs
    """

    input_file = Path(input_file)

    total_rows = 0

    missing_counts = {
        "entity_id": 0,
        "business_name": 0,
        "business_address": 0,
        "country": 0,
    }

    entity_ids = set()
    duplicate_entity_ids = 0

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
    ):

        total_rows += len(chunk)

        # Count missing/empty values
        for column in missing_counts:
            missing_counts[column] += (
                chunk[column].isna()
                | chunk[column].astype(str).str.strip().eq("")
            ).sum()

        # Check duplicate entity IDs
        for entity_id in chunk["entity_id"]:
            if entity_id in entity_ids:
                duplicate_entity_ids += 1
            else:
                entity_ids.add(entity_id)

    print("\n========== DATA QUALITY REPORT ==========")
    print(f"File: {input_file}")
    print(f"Total rows: {total_rows}")

    print("\nMissing / empty values:")
    for column, count in missing_counts.items():
        print(f"  {column}: {count}")

    print(f"\nDuplicate entity IDs: {duplicate_entity_ids}")
    print("========================================\n")


# ---------------------------------------------------------
# 4. Process a large TSV file in chunks
# ---------------------------------------------------------
def preprocess_file(input_file, output_file, chunk_size=100_000):
    """
    Process a large TSV file chunk-by-chunk so that
    the entire dataset does not need to fit into memory.
    """

    input_file = Path(input_file)
    output_file = Path(output_file)

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    first_chunk = True

    for chunk in pd.read_csv(
        input_file,
        sep="\t",
        chunksize=chunk_size,
        dtype=str,
        keep_default_na=False,
    ):

        # Apply preprocessing
        chunk = preprocess_chunk(chunk)

        # Write processed chunk
        chunk.to_csv(
            output_file,
            sep="\t",
            index=False,
            mode="w" if first_chunk else "a",
            header=first_chunk,
        )

        first_chunk = False

    print(f"Finished: {input_file}")
    print(f"Output:   {output_file}")


# ---------------------------------------------------------
# 5. Main program
# ---------------------------------------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Preprocess business entity resolution TSV files."
    )

    parser.add_argument(
        "--input-dir",
        required=True,
        help="Directory containing input TSV files."
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory where processed TSV files will be saved."
    )

    parser.add_argument(
        "--chunk-size",
        type=int,
        default=100_000,
        help="Number of rows processed at a time."
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # Check input directory
    if not input_dir.exists():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    # Create output directory
    output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    # Process Source 1, Source 2, and Source 3
    input_files = sorted(
        input_dir.glob("*_source[123].tsv")
    )

    if not input_files:
        raise FileNotFoundError(
            f"No source TSV files found in: {input_dir}"
        )

    for input_file in input_files:

        output_file = output_dir / input_file.name

        preprocess_file(
            input_file,
            output_file,
            chunk_size=args.chunk_size
        )