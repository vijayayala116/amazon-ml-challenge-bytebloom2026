"""
feature_engineering.py  (fast, vectorized)

Computes the SAME feature set as before (name + address similarity,
country match) for every (source1_entity_id, candidate_entity_id) pair -
but using vectorized pandas merges instead of a per-row Python loop with
.loc lookups. That row-by-row approach does not scale once you have tens
of millions of candidate pairs (18+ avg candidates x 2.2M Source1 entities
here) - this version does.

Usage:
    python feature_engineering.py \
        --preprocessed-dir dataset/preprocessed \
        --candidates output/candidate_pairs.tsv \
        --output output/features.tsv

Requires: pandas, rapidfuzz
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz

LEGAL_SUFFIX_MAP = {
    "pvt": "private", "ltd": "limited", "co": "company", "corp": "corporation",
    "inc": "incorporated", "llc": "limited liability company",
    "llp": "limited liability partnership", "assn": "association",
    "intl": "international", "mfg": "manufacturing",
}
LEGAL_WORDS = set(LEGAL_SUFFIX_MAP.keys()) | set(LEGAL_SUFFIX_MAP.values())
NUM_RE = re.compile(r"\d+")


def canonicalize_legal_terms(name):
    if not name:
        return ""
    return " ".join(LEGAL_SUFFIX_MAP.get(t, t) for t in name.split())


def core_name(name):
    if not name:
        return ""
    toks = [t for t in name.split() if t not in LEGAL_WORDS]
    return " ".join(toks) if toks else name


def first_word(s):
    toks = s.split()
    return toks[0] if toks else ""


def last_word(s):
    toks = s.split()
    return toks[-1] if toks else ""


def jaccard_and_common(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        return 1.0, 0
    if not sa or not sb:
        return 0.0, 0
    inter = sa & sb
    return len(inter) / len(sa | sb), len(inter)


def addr_jaccard_and_nums(a, b):
    sa, sb = set(a.split()), set(b.split())
    if not sa and not sb:
        jac = 1.0
    elif not sa or not sb:
        jac = 0.0
    else:
        jac = len(sa & sb) / len(sa | sb)
    na, nb = set(NUM_RE.findall(a)), set(NUM_RE.findall(b))
    common_n = na & nb
    union_n = na | nb
    num_jac = len(common_n) / len(union_n) if union_n else 0.0
    return jac, num_jac, int(bool(common_n)), len(common_n)


def load_source(preprocessed_dir, i):
    path = next(Path(preprocessed_dir).glob(f"*_source{i}.tsv"))
    df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    return df.set_index("entity_id")[
        ["business_name_normalized", "business_address_normalized", "country_normalized"]
    ]


def build_feature_table(candidates_path, preprocessed_dir, output_path):
    print("Loading sources ...")
    s1 = load_source(preprocessed_dir, 1)
    s2 = load_source(preprocessed_dir, 2)
    s3 = load_source(preprocessed_dir, 3)
    all_cand = pd.concat([s2, s3])
    print(f"  Source1: {len(s1):,}  Source2+3 (candidates): {len(all_cand):,}")

    print("Loading candidate pairs and exploding to one row per pair ...")
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)
    cands = cands[cands["candidate_entity_ids"] != ""].copy()
    cands["candidate_entity_id"] = cands["candidate_entity_ids"].str.split(",")
    pairs = cands.explode("candidate_entity_id")[["source1_entity_id", "candidate_entity_id"]]
    pairs = pairs.drop_duplicates().reset_index(drop=True)
    print(f"  Total pairs: {len(pairs):,}")

    print("Joining pairs with Source1/Source2/Source3 data (vectorized merge) ...")
    pairs = pairs.merge(s1, left_on="source1_entity_id", right_index=True, how="left")
    pairs = pairs.rename(columns={
        "business_name_normalized": "name_s1",
        "business_address_normalized": "addr_s1",
        "country_normalized": "country_s1",
    })
    pairs = pairs.merge(all_cand, left_on="candidate_entity_id", right_index=True, how="left")
    pairs = pairs.rename(columns={
        "business_name_normalized": "name_c",
        "business_address_normalized": "addr_c",
        "country_normalized": "country_c",
    })
    before = len(pairs)
    pairs = pairs.dropna(subset=["name_s1", "name_c"]).reset_index(drop=True)
    if len(pairs) < before:
        print(f"  Dropped {before - len(pairs)} pairs referencing unknown entity IDs.")

    print("Computing canonical/core names ...")
    pairs["canon_s1"] = pairs["name_s1"].apply(canonicalize_legal_terms)
    pairs["canon_c"] = pairs["name_c"].apply(canonicalize_legal_terms)
    pairs["core_s1"] = pairs["name_s1"].apply(core_name)
    pairs["core_c"] = pairs["name_c"].apply(core_name)

    print("Scoring name similarity (rapidfuzz, vectorized loop - may take a few minutes) ...")
    pairs["name_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(pairs["canon_s1"], pairs["canon_c"])]
    pairs["name_token_sort_ratio"] = [fuzz.token_sort_ratio(a, b) / 100.0 for a, b in zip(pairs["canon_s1"], pairs["canon_c"])]
    pairs["name_token_set_ratio"] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(pairs["canon_s1"], pairs["canon_c"])]
    pairs["name_partial_ratio"] = [fuzz.partial_ratio(a, b) / 100.0 for a, b in zip(pairs["canon_s1"], pairs["canon_c"])]

    pairs["name_exact_match"] = (pairs["name_s1"] == pairs["name_c"]).astype(int)
    pairs["name_canonical_match"] = (pairs["canon_s1"] == pairs["canon_c"]).astype(int)
    pairs["name_core_match"] = ((pairs["core_s1"] == pairs["core_c"]) & (pairs["core_s1"] != "")).astype(int)
    pairs["name_length_diff"] = (pairs["canon_s1"].str.len() - pairs["canon_c"].str.len()).abs()

    jc = [jaccard_and_common(a, b) for a, b in zip(pairs["canon_s1"], pairs["canon_c"])]
    pairs["name_jaccard"] = [x[0] for x in jc]
    pairs["name_common_token_count"] = [x[1] for x in jc]

    first_s1 = pairs["canon_s1"].apply(first_word)
    first_c = pairs["canon_c"].apply(first_word)
    pairs["name_first_word_match"] = ((first_s1 != "") & (first_s1 == first_c)).astype(int)

    last_s1 = pairs["canon_s1"].apply(last_word)
    last_c = pairs["canon_c"].apply(last_word)
    pairs["name_last_word_match"] = ((last_s1 != "") & (last_s1 == last_c)).astype(int)

    pairs["name_token_count_diff"] = (
        pairs["canon_s1"].str.split().apply(len) - pairs["canon_c"].str.split().apply(len)
    ).abs()

    print("Scoring address similarity ...")
    pairs["address_ratio"] = [fuzz.ratio(a, b) / 100.0 for a, b in zip(pairs["addr_s1"], pairs["addr_c"])]
    pairs["address_token_set_ratio"] = [fuzz.token_set_ratio(a, b) / 100.0 for a, b in zip(pairs["addr_s1"], pairs["addr_c"])]
    pairs["address_missing_either"] = ((pairs["addr_s1"] == "") | (pairs["addr_c"] == "")).astype(int)

    aj = [addr_jaccard_and_nums(a, b) for a, b in zip(pairs["addr_s1"], pairs["addr_c"])]
    pairs["address_jaccard"] = [x[0] for x in aj]
    pairs["address_number_overlap"] = [x[1] for x in aj]
    pairs["address_has_common_number"] = [x[2] for x in aj]
    pairs["address_common_number_count"] = [x[3] for x in aj]
    pairs["address_length_diff"] = (pairs["addr_s1"].str.len() - pairs["addr_c"].str.len()).abs()

    pairs["country_match"] = (pairs["country_s1"] == pairs["country_c"]).astype(int)

    drop_cols = ["name_s1", "name_c", "addr_s1", "addr_c", "country_s1", "country_c",
                 "canon_s1", "canon_c", "core_s1", "core_c"]
    feature_df = pairs.drop(columns=drop_cols)

    id_cols = ["source1_entity_id", "candidate_entity_id"]
    feature_cols = [c for c in feature_df.columns if c not in id_cols]
    feature_df = feature_df[id_cols + feature_cols]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, sep="\t", index=False)
    print(f"\nWrote {len(feature_df):,} feature rows to {output_path}")
    return feature_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compute similarity features for Source1-candidate pairs (fast, vectorized).")
    parser.add_argument("--preprocessed-dir", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    build_feature_table(args.candidates, args.preprocessed_dir, args.output)