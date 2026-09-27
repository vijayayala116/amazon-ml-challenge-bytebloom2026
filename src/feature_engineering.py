"""
feature_engineering.py

Computes pairwise similarity features between a Source 1 business record and
each of its candidate Source 2 / Source 3 records (produced by the blocking
/ candidate-generation step).

Assumes the input files are the OUTPUT of preprocessing.py, i.e. they already
have these extra columns:
    business_name_normalized, business_address_normalized, country_normalized

Input:
  - preprocessed source files (entity_id, business_name, business_address,
    country, business_name_normalized, business_address_normalized,
    country_normalized)
  - candidate_pairs.tsv  (source1_entity_id, candidate_entity_ids)

Output:
  - a feature table: one row per (source1_entity_id, candidate_entity_id)
    pair, ready to feed into the matching model.

Usage:
    python feature_engineering.py \
        --preprocessed-dir dataset/preprocessed \
        --candidates output/candidate_pairs.tsv \
        --output output/features.tsv

Requires: pandas, rapidfuzz  (pip install rapidfuzz)
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


# ---------------------------------------------------------
# 1. Legal-suffix / abbreviation canonicalisation
# ---------------------------------------------------------

# Common legal-entity abbreviations -> canonical long form.
# Applied on already-normalized (lowercase, punctuation-stripped) text, so
# "xyz pvt ltd" and "xyz private limited" end up identical.
LEGAL_SUFFIX_MAP = {
    "pvt": "private",
    "ltd": "limited",
    "co": "company",
    "corp": "corporation",
    "inc": "incorporated",
    "llc": "limited liability company",
    "llp": "limited liability partnership",
    "assn": "association",
    "intl": "international",
    "mfg": "manufacturing",
}


def canonicalize_legal_terms(normalized_name):
    """Expand common legal-entity abbreviations to a canonical long form."""
    if not normalized_name:
        return ""
    tokens = normalized_name.split()
    tokens = [LEGAL_SUFFIX_MAP.get(tok, tok) for tok in tokens]
    return " ".join(tokens)


def core_name(normalized_name):
    """
    Strip legal-entity words entirely, leaving just the 'core' brand name.
    Useful as a high-precision blocking key and as a standalone feature.
    """
    if not normalized_name:
        return ""
    legal_words = set(LEGAL_SUFFIX_MAP.values()) | set(LEGAL_SUFFIX_MAP.keys())
    tokens = [t for t in normalized_name.split() if t not in legal_words]
    return " ".join(tokens) if tokens else normalized_name


# ---------------------------------------------------------
# 2. Pairwise similarity features
# ---------------------------------------------------------

def _token_jaccard(a, b):
    set_a, set_b = set(a.split()), set(b.split())
    if not set_a and not set_b:
        return 1.0
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def _extract_numbers(text):
    """Pull out digit sequences (street numbers, PIN/ZIP codes) from text."""
    return set(re.findall(r"\d+", text or ""))


def name_features(name1_norm, name2_norm):
    canon1 = canonicalize_legal_terms(name1_norm)
    canon2 = canonicalize_legal_terms(name2_norm)
    core1 = core_name(name1_norm)
    core2 = core_name(name2_norm)

    return {
        "name_exact_match": int(name1_norm == name2_norm),
        "name_canonical_match": int(canon1 == canon2),
        "name_core_match": int(core1 == core2 and bool(core1)),
        "name_ratio": fuzz.ratio(canon1, canon2) / 100.0,
        "name_token_sort_ratio": fuzz.token_sort_ratio(canon1, canon2) / 100.0,
        "name_token_set_ratio": fuzz.token_set_ratio(canon1, canon2) / 100.0,
        "name_partial_ratio": fuzz.partial_ratio(canon1, canon2) / 100.0,
        "name_jaccard": _token_jaccard(canon1, canon2),
        "name_first_word_match": int(
            bool(canon1) and canon1.split()[0] == (canon2.split()[0] if canon2 else None)
        ),
        "name_length_diff": abs(len(canon1) - len(canon2)),
    }


def address_features(addr1_norm, addr2_norm):
    nums1, nums2 = _extract_numbers(addr1_norm), _extract_numbers(addr2_norm)
    both_have_numbers = bool(nums1) and bool(nums2)

    return {
        "address_missing_either": int(not addr1_norm or not addr2_norm),
        "address_ratio": fuzz.ratio(addr1_norm, addr2_norm) / 100.0,
        "address_token_set_ratio": fuzz.token_set_ratio(addr1_norm, addr2_norm) / 100.0,
        "address_jaccard": _token_jaccard(addr1_norm, addr2_norm),
        "address_number_overlap": (
            len(nums1 & nums2) / len(nums1 | nums2) if both_have_numbers else 0.0
        ),
        "address_has_common_number": int(bool(nums1 & nums2)) if both_have_numbers else 0,
    }


def build_pair_features(row1, row2):
    """
    row1: Series/dict for a Source 1 record (normalized columns present)
    row2: Series/dict for a candidate Source 2 / Source 3 record
    """
    features = {}
    features.update(name_features(row1["business_name_normalized"], row2["business_name_normalized"]))
    features.update(address_features(row1["business_address_normalized"], row2["business_address_normalized"]))
    features["country_match"] = int(row1["country_normalized"] == row2["country_normalized"])
    return features


# ---------------------------------------------------------
# 3. Build the feature table from candidate_pairs.tsv
# ---------------------------------------------------------

def load_preprocessed_sources(preprocessed_dir):
    """
    Loads the three preprocessed source files (output of preprocessing.py),
    indexed by entity_id for fast lookup.

    NOTE on memory: this keeps all three sources in RAM as DataFrames
    indexed by entity_id. For this dataset's size that should fit on a
    normal laptop (~1-2 GB). If it does not fit on your machine, swap this
    for a dict built via a single streaming read, or a sqlite lookup table.
    """
    preprocessed_dir = Path(preprocessed_dir)
    frames = {}
    for i in (1, 2, 3):
        path = next(preprocessed_dir.glob(f"*_source{i}.tsv"))
        df = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
        frames[f"source{i}"] = df.set_index("entity_id")
    return frames


def _lookup(frames, entity_id):
    """entity_id prefix (S1-/S2-/S3-) tells us which frame to look in."""
    source_key = {"S1": "source1", "S2": "source2", "S3": "source3"}[entity_id[:2]]
    return frames[source_key].loc[entity_id]


def build_feature_table(candidates_path, preprocessed_dir, output_path):
    frames = load_preprocessed_sources(preprocessed_dir)
    candidates = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)

    rows = []
    for _, row in candidates.iterrows():
        s1_id = row["source1_entity_id"]
        candidate_ids = row["candidate_entity_ids"]

        if not candidate_ids:
            continue  # no candidates for this Source1 entity (will become a predicted singleton)

        s1_record = _lookup(frames, s1_id)

        for cand_id in candidate_ids.split(","):
            cand_record = _lookup(frames, cand_id)
            feats = build_pair_features(s1_record, cand_record)
            feats["source1_entity_id"] = s1_id
            feats["candidate_entity_id"] = cand_id
            rows.append(feats)

    feature_df = pd.DataFrame(rows)

    # Put ID columns first, features after
    id_cols = ["source1_entity_id", "candidate_entity_id"]
    feature_cols = [c for c in feature_df.columns if c not in id_cols]
    feature_df = feature_df[id_cols + feature_cols]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feature_df.to_csv(output_path, sep="\t", index=False)

    print(f"Wrote {len(feature_df)} feature rows to {output_path}")
    return feature_df


# ---------------------------------------------------------
# 4. CLI
# ---------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compute similarity features for Source1-candidate pairs."
    )
    parser.add_argument(
        "--preprocessed-dir", required=True,
        help="Directory containing preprocessing.py output (normalized columns present)."
    )
    parser.add_argument(
        "--candidates", required=True,
        help="Path to candidate_pairs.tsv (output of the blocking step)."
    )
    parser.add_argument(
        "--output", required=True,
        help="Path to write the feature table (TSV)."
    )
    args = parser.parse_args()

    build_feature_table(args.candidates, args.preprocessed_dir, args.output)