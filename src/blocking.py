"""
blocking.py  (v5 - name key + address-number key, combined)

Candidate generation (blocking) for the Business Entity Resolution challenge.

WHAT WE LEARNED (measured directly against the training ground truth):
  - A key from 1-2 name tokens (any variant: first two, longest two,
    sorted, etc.) has a hard recall ceiling around 55-58%.
  - Exploding into ALL name tokens raises the theoretical ceiling to
    ~89%, but is NOT computationally usable: generic business words like
    "center", "services", "group", "india" appear in hundreds of
    thousands of records, so a shared-token key on them creates buckets
    too large to score (some over 400,000 rows). Capping those buckets
    down to something computable throws away the very rows you needed -
    tested, this only recovers ~25-34% of true matches in practice, WORSE
    than the simple 2-token key.
  - A completely different, much cheaper signal: the longest number
    (4+ digits - PIN/ZIP/street-number-like) found in the address.
    On its own it matches ~28% of true pairs, and its buckets are
    naturally small (median 4 rows, 99th percentile ~638, vs hundreds
    of thousands for common name words) - so it barely needs capping at
    all, unlike name-token blocking.
  - Combining the address-number key with the name key recovers pairs
    that name-blocking alone misses, at very low extra computational
    cost (small buckets = fast).

This version generates candidates via BOTH the name key AND the address-
number key, and takes the union - keeping the runtime close to the
original ~28-minute full-dataset time (the address pass is cheap because
its buckets are small), while adding recall that name-blocking alone
cannot reach.

Usage:
    python blocking.py \
        --preprocessed-dir dataset/preprocessed \
        --output output/candidate_pairs.tsv \
        --max-candidates 15 \
        --similarity-threshold 0.5 \
        --max-group-size 200 \
        --address-max-group-size 500 \
        --chunk-size 50000

    # Quick smoke test on the first 50,000 Source1 rows only:
    python blocking.py --preprocessed-dir dataset/preprocessed \
        --output output/candidate_pairs_TEST.tsv --limit 50000

Requires: pandas, rapidfuzz, tqdm
"""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz
from tqdm import tqdm

from feature_engineering import canonicalize_legal_terms, core_name

NUM_RE = re.compile(r"\d+")


# ---------------------------------------------------------
# 1. Shared helpers
# ---------------------------------------------------------

def add_core_columns(df):
    df["core_name"] = df["business_name_normalized"].apply(core_name)
    df["canonical_name"] = df["business_name_normalized"].apply(canonicalize_legal_terms)
    return df


def name_key(country, core_name_value):
    """Original, proven key: country + first two tokens of the core name."""
    first_two = " ".join(core_name_value.split()[:2])
    return "N||" + country + "||" + first_two


def longest_addr_number(address_normalized):
    """The longest 4+ digit number in the address (PIN/ZIP/street-number-like)."""
    nums = [n for n in NUM_RE.findall(address_normalized) if len(n) >= 4]
    return max(nums, key=len) if nums else ""


def addr_key(country, address_normalized):
    num = longest_addr_number(address_normalized)
    if not num:
        return None
    return "A||" + country + "||" + num


def add_keys(df):
    df["name_key"] = df["country_normalized"] + "||" + df["core_name"].apply(
        lambda s: " ".join(s.split()[:2])
    )
    df["addr_key"] = [
        addr_key(country, addr) for country, addr in zip(df["country_normalized"], df["business_address_normalized"])
    ]
    return df


def cap_group_size(df, key_col, max_size):
    rank = df.groupby(key_col).cumcount()
    return df[rank < max_size]


def load_candidate_source(preprocessed_dir, index, name_cap, addr_cap):
    path = next(Path(preprocessed_dir).glob(f"*_source{index}.tsv"))
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    df = add_core_columns(df)
    df = add_keys(df)

    name_side = cap_group_size(df[["entity_id", "name_key", "canonical_name"]], "name_key", name_cap)
    addr_side = df[["entity_id", "addr_key", "canonical_name"]].dropna(subset=["addr_key"])
    addr_side = cap_group_size(addr_side, "addr_key", addr_cap)

    return name_side.reset_index(drop=True), addr_side.reset_index(drop=True)


def score_and_filter(pairs, key_col_s1, key_col_cand, max_candidates, similarity_threshold):
    if len(pairs) == 0:
        return pd.DataFrame(columns=["entity_id_s1", "entity_id_cand"])
    pairs = pairs.copy()
    pairs["score"] = [
        fuzz.token_set_ratio(a, b) / 100.0
        for a, b in zip(pairs["canonical_name_s1"], pairs["canonical_name_cand"])
    ]
    pairs = pairs[pairs["score"] >= similarity_threshold]
    pairs = pairs.sort_values("score", ascending=False)
    pairs = pairs.groupby("entity_id_s1", sort=False).head(max_candidates)
    return pairs[["entity_id_s1", "entity_id_cand"]]


# ---------------------------------------------------------
# 2. Main pipeline - chunked over Source1
# ---------------------------------------------------------

def run_blocking(preprocessed_dir, output_path, max_candidates, similarity_threshold,
                  max_group_size, address_max_group_size, chunk_size, limit):
    t0 = time.time()

    print("Loading and keying Source2 and Source3 (once) ...")
    s2_name, s2_addr = load_candidate_source(preprocessed_dir, 2, max_group_size, address_max_group_size)
    s3_name, s3_addr = load_candidate_source(preprocessed_dir, 3, max_group_size, address_max_group_size)
    print(f"  S2: {len(s2_name):,} name-keyed / {len(s2_addr):,} addr-keyed   "
          f"S3: {len(s3_name):,} name-keyed / {len(s3_addr):,} addr-keyed   "
          f"({time.time() - t0:.1f}s elapsed)")

    s1_path = next(Path(preprocessed_dir).glob("*_source1.tsv"))
    with open(s1_path, encoding="utf-8") as f:
        total_s1 = sum(1 for _ in f) - 1
    if limit:
        total_s1 = min(total_s1, limit)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    first_chunk = True
    rows_done = 0

    progress = tqdm(
        total=total_s1, unit="rows", unit_scale=True, desc="Blocking Source1",
        dynamic_ncols=True, mininterval=1.0, ascii=True,
    )

    for chunk in pd.read_csv(s1_path, sep="\t", dtype=str, keep_default_na=False, chunksize=chunk_size):
        if limit and rows_done >= limit:
            break
        if limit and rows_done + len(chunk) > limit:
            chunk = chunk.iloc[: limit - rows_done]

        chunk = add_core_columns(chunk)
        chunk = add_keys(chunk)

        s1_name = chunk[["entity_id", "name_key", "canonical_name"]]
        s1_addr = chunk[["entity_id", "addr_key", "canonical_name"]].dropna(subset=["addr_key"])

        all_pairs = []
        for cand_name, cand_addr, label in [(s2_name, s2_addr, "S2"), (s3_name, s3_addr, "S3")]:
            p_name = s1_name.merge(cand_name, on="name_key", suffixes=("_s1", "_cand"))
            p_name = score_and_filter(p_name, "name_key", "name_key", max_candidates, similarity_threshold)

            p_addr = s1_addr.merge(cand_addr, on="addr_key", suffixes=("_s1", "_cand"))
            p_addr = score_and_filter(p_addr, "addr_key", "addr_key", max_candidates, similarity_threshold)

            all_pairs.append(p_name)
            all_pairs.append(p_addr)

        combined = pd.concat(all_pairs, ignore_index=True)
        combined = combined.drop_duplicates(subset=["entity_id_s1", "entity_id_cand"])

        if len(combined) > 0:
            grouped = (
                combined.groupby("entity_id_s1")["entity_id_cand"]
                .apply(lambda ids: ",".join(sorted(set(ids))[:max_candidates * 2]))
            )
        else:
            grouped = pd.Series(dtype=str)

        result_chunk = chunk[["entity_id"]].rename(columns={"entity_id": "source1_entity_id"})
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

        rows_done += len(chunk)
        progress.update(len(chunk))

    progress.close()
    print(f"\nDone. Candidates written to {output_path}")
    print(f"Total time: {time.time() - t0:.1f}s")
    print("Run validate_blocking_recall.py next to check recall against the ground truth.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Blocking / candidate generation (name key + address-number key).")
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--max-candidates", type=int, default=15,
                         help="Max candidates to keep per source (S2, S3) per Source1 entity, per key type.")
    parser.add_argument("--similarity-threshold", type=float, default=0.5)
    parser.add_argument("--max-group-size", type=int, default=200,
                         help="Cap on rows sharing a name key.")
    parser.add_argument("--address-max-group-size", type=int, default=500,
                         help="Cap on rows sharing an address-number key (naturally smaller buckets, can afford a higher cap).")
    parser.add_argument("--chunk-size", type=int, default=50_000)
    parser.add_argument("--limit", type=int, default=None,
                         help="Only process the first N Source1 rows - use to smoke-test before a full run.")
    args = parser.parse_args()

    run_blocking(args.preprocessed_dir, args.output, args.max_candidates,
                 args.similarity_threshold, args.max_group_size,
                 args.address_max_group_size, args.chunk_size, args.limit)