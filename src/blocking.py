"""
blocking.py  (v3 - chunked streaming, memory-safe, frequent progress)

Candidate generation (blocking) for the Business Entity Resolution challenge.

Same blocking strategy as before (country + first-two-tokens-of-core-name
key, vectorized pandas merge, rapidfuzz fine filter, capped generic keys)
but processes Source1 in CHUNKS and streams output to disk, instead of
doing one giant merge across the whole dataset at once.

Why: a single merge across the full Source1 x Source2 x Source3 data can
use a large amount of memory in one shot, and prints nothing until it is
completely finished - if it is slow, or the machine is tight on RAM, it
looks "stuck" with no way to tell what's happening.

v3 instead:
  1. Loads Source2 and Source3 ONCE, with any overly generic blocking key
     capped at --max-group-size rows (so no single key can blow up a
     merge).
  2. Reads Source1 in chunks of --chunk-size rows, and for each chunk:
     merges against Source2 and Source3, scores with rapidfuzz, keeps the
     top candidates, and appends the result straight to the output file.
  3. Prints a progress line after every chunk, so you always know it's
     moving and roughly how much is left.

Peak memory is bounded by one chunk's worth of candidate pairs (plus the
capped Source2/Source3 tables), not the whole dataset.

Usage:
    python blocking.py \
        --preprocessed-dir dataset/preprocessed \
        --output output/candidate_pairs.tsv \
        --max-candidates 15 \
        --similarity-threshold 0.55 \
        --max-group-size 200 \
        --chunk-size 50000

Requires: pandas, rapidfuzz, tqdm
"""

import argparse
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from feature_engineering import canonicalize_legal_terms, core_name


# ---------------------------------------------------------
# 1. Helpers shared by Source2/Source3 (loaded once) and
#    Source1 chunks (loaded repeatedly)
# ---------------------------------------------------------

def add_blocking_columns(df):
    df["core_name"] = df["business_name_normalized"].apply(core_name)
    df["canonical_name"] = df["business_name_normalized"].apply(canonicalize_legal_terms)
    df["blocking_key"] = df["country_normalized"] + "||" + df["core_name"].apply(
        lambda s: " ".join(s.split()[:2])  # first TWO tokens - more selective than one
    )
    return df


def load_and_key(preprocessed_dir, index):
    path = next(Path(preprocessed_dir).glob(f"*_source{index}.tsv"))
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = add_blocking_columns(df)
    return df[["entity_id", "blocking_key", "canonical_name"]]


def cap_group_size(df, max_size):
    """
    If a blocking key matches more than max_size rows, keep only the first
    max_size of them. Bounds the size of any single merge regardless of
    how generic a key turns out to be.
    """
    rank = df.groupby("blocking_key").cumcount()
    return df[rank < max_size].reset_index(drop=True)


def score_and_filter(pairs, max_candidates, similarity_threshold):
    if len(pairs) == 0:
        return pairs
    pairs = pairs.copy()
    pairs["score"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(pairs["canonical_name_s1"], pairs["canonical_name_cand"])
    ]
    pairs = pairs[pairs["score"] >= similarity_threshold]
    pairs = pairs.sort_values("score", ascending=False)
    pairs = pairs.groupby("entity_id_s1", sort=False).head(max_candidates)
    return pairs


# ---------------------------------------------------------
# 2. Main pipeline - chunked over Source1
# ---------------------------------------------------------

def run_blocking(preprocessed_dir, output_path, max_candidates, similarity_threshold,
                  max_group_size, chunk_size):
    t0 = time.time()

    print("Loading Source2 and Source3 (once) ...")
    s2 = cap_group_size(load_and_key(preprocessed_dir, 2), max_group_size)
    s3 = cap_group_size(load_and_key(preprocessed_dir, 3), max_group_size)
    print(f"  Source2 (capped): {len(s2):,}   Source3 (capped): {len(s3):,}   "
          f"({time.time() - t0:.1f}s elapsed)")

    s1_path = next(Path(preprocessed_dir).glob("*_source1.tsv"))
    with open(s1_path, encoding="utf-8") as f:
        total_s1 = sum(1 for _ in f) - 1  # minus header

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True

    progress = tqdm(
        total=total_s1,
        unit="rows",
        unit_scale=True,
        desc="Blocking Source1",
        ncols=100,
    )

    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
        chunk = add_blocking_columns(chunk)
        s1_chunk = chunk[["entity_id", "blocking_key", "canonical_name"]]

        pairs_s2 = score_and_filter(
            s1_chunk.merge(s2, on="blocking_key", suffixes=("_s1", "_cand")),
            max_candidates, similarity_threshold,
        )
        pairs_s3 = score_and_filter(
            s1_chunk.merge(s3, on="blocking_key", suffixes=("_s1", "_cand")),
            max_candidates, similarity_threshold,
        )
        all_pairs = pd.concat([pairs_s2, pairs_s3], ignore_index=True)

        if len(all_pairs) > 0:
            grouped = (
                all_pairs.groupby("entity_id_s1")["entity_id_cand"]
                .apply(lambda ids: ",".join(ids))
            )
        else:
            grouped = pd.Series(dtype=str)

        result_chunk = s1_chunk[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"})
        result_chunk = result_chunk.merge(
            grouped.rename("candidate_entity_ids"),
            left_on="source1_entity_id", right_index=True, how="left",
        )
        result_chunk["candidate_entity_ids"] = result_chunk["candidate_entity_ids"].fillna("")

        result_chunk.to_csv(
            output_path, sep="\t", index=False,
            mode="w" if first_chunk else "a", header=first_chunk,
        )
        first_chunk = False

        progress.update(len(chunk))

    progress.close()
    print(f"\nDone. Candidates written to {output_path}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print("Run validate_blocking_recall.py next to check recall against the ground truth.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blocking / candidate generation (chunked, memory-safe).")
    parser.add_argument("--preprocessed-dir", required=True,
                         help="Directory with preprocessing.py output.")
    parser.add_argument("--output", required=True,
                         help="Path to write candidate_pairs.tsv")
    parser.add_argument("--max-candidates", type=int, default=15,
                         help="Max candidates to keep per source (S2, S3) per Source1 entity.")
    parser.add_argument("--similarity-threshold", type=float, default=0.55,
                         help="Minimum name similarity (0-1) to keep a candidate.")
    parser.add_argument("--max-group-size", type=int, default=200,
                         help="Cap on how many rows a single blocking key can match in S2/S3.")
    parser.add_argument("--chunk-size", type=int, default=50_000,
                         help="How many Source1 rows to process per chunk.")
    args = parser.parse_args()

    run_blocking(args.preprocessed_dir, args.output, args.max_candidates,
                 args.similarity_threshold, args.max_group_size, args.chunk_size)