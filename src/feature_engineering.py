"""
feature_engineering.py

Computes pairwise similarity features between a Source 1 business record and
each of its candidate Source 2 / Source 3 records.

Input:
  - preprocessed source files
  - candidate_pairs.tsv

Output:
  - one feature row per (source1_entity_id, candidate_entity_id) pair

Usage:
    python feature_engineering.py \
        --preprocessed-dir dataset/preprocessed \
        --candidates output/candidate_pairs.tsv \
        --output output/features.tsv

Requires:
    pandas
    rapidfuzz
"""

import argparse
import re
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz


# =========================================================
# 1. LEGAL TERM CANONICALISATION
# =========================================================

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
    """
    Expand common legal-entity abbreviations.

    Example:
        xyz pvt ltd
        ->
        xyz private limited
    """

    if not normalized_name:
        return ""

    tokens = normalized_name.split()

    tokens = [
        LEGAL_SUFFIX_MAP.get(token, token)
        for token in tokens
    ]

    return " ".join(tokens)


def core_name(normalized_name):
    """
    Remove legal-entity words and keep the core business name.
    """

    if not normalized_name:
        return ""

    legal_words = (
        set(LEGAL_SUFFIX_MAP.values())
        | set(LEGAL_SUFFIX_MAP.keys())
    )

    tokens = [
        token
        for token in normalized_name.split()
        if token not in legal_words
    ]

    return " ".join(tokens) if tokens else normalized_name


# =========================================================
# 2. BASIC TEXT HELPERS
# =========================================================

def _safe_text(value):
    """
    Convert a value safely to string.
    """

    if value is None:
        return ""

    if pd.isna(value):
        return ""

    return str(value).strip()


def _tokenize(text):
    """
    Convert text into whitespace-separated tokens.
    """

    text = _safe_text(text)

    if not text:
        return []

    return text.split()


def _token_jaccard(a, b):
    """
    Token-level Jaccard similarity.
    """

    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))

    if not tokens_a and not tokens_b:
        return 1.0

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a & tokens_b) / len(tokens_a | tokens_b)


def _common_token_count(a, b):
    """
    Number of common tokens between two strings.
    """

    tokens_a = set(_tokenize(a))
    tokens_b = set(_tokenize(b))

    if not tokens_a or not tokens_b:
        return 0

    return len(tokens_a & tokens_b)


def _extract_numbers(text):
    """
    Extract digit sequences.

    Useful for:
        street numbers
        building numbers
        PIN/ZIP codes
        unit numbers
    """

    text = _safe_text(text)

    return set(
        re.findall(r"\d+", text)
    )


def _length_difference(a, b):
    """
    Absolute character-length difference.
    """

    return abs(
        len(_safe_text(a))
        - len(_safe_text(b))
    )


def _first_word(text):
    tokens = _tokenize(text)

    if not tokens:
        return ""

    return tokens[0]


def _last_word(text):
    tokens = _tokenize(text)

    if not tokens:
        return ""

    return tokens[-1]


def _first_two_words(text):
    tokens = _tokenize(text)

    if len(tokens) < 2:
        return ""

    return " ".join(tokens[:2])


# =========================================================
# 3. NAME FEATURES
# =========================================================

def name_features(name1_norm, name2_norm):

    name1_norm = _safe_text(name1_norm)
    name2_norm = _safe_text(name2_norm)

    # Canonical names
    canon1 = canonicalize_legal_terms(
        name1_norm
    )

    canon2 = canonicalize_legal_terms(
        name2_norm
    )

    # Core names
    core1 = core_name(
        name1_norm
    )

    core2 = core_name(
        name2_norm
    )

    # Token sets
    tokens1 = set(
        _tokenize(canon1)
    )

    tokens2 = set(
        _tokenize(canon2)
    )

    common_tokens = tokens1.intersection(
        tokens2
    )

    # First/last words
    first1 = _first_word(canon1)
    first2 = _first_word(canon2)

    last1 = _last_word(canon1)
    last2 = _last_word(canon2)

    first_two1 = _first_two_words(canon1)
    first_two2 = _first_two_words(canon2)

    return {

        # -------------------------------------------------
        # Exact / canonical matching
        # -------------------------------------------------

        "name_exact_match": int(
            name1_norm == name2_norm
            and bool(name1_norm)
        ),

        "name_canonical_match": int(
            canon1 == canon2
            and bool(canon1)
        ),

        "name_core_match": int(
            core1 == core2
            and bool(core1)
        ),

        # -------------------------------------------------
        # Fuzzy similarity
        # -------------------------------------------------

        "name_ratio": (
            fuzz.ratio(
                canon1,
                canon2
            ) / 100.0
        ),

        "name_token_sort_ratio": (
            fuzz.token_sort_ratio(
                canon1,
                canon2
            ) / 100.0
        ),

        "name_token_set_ratio": (
            fuzz.token_set_ratio(
                canon1,
                canon2
            ) / 100.0
        ),

        "name_partial_ratio": (
            fuzz.partial_ratio(
                canon1,
                canon2
            ) / 100.0
        ),

        # -------------------------------------------------
        # Token similarity
        # -------------------------------------------------

        "name_jaccard": _token_jaccard(
            canon1,
            canon2
        ),

        "name_common_token_count": len(
            common_tokens
        ),

        # -------------------------------------------------
        # Word-position features
        # -------------------------------------------------

        "name_first_word_match": int(
            bool(first1)
            and bool(first2)
            and first1 == first2
        ),

        "name_last_word_match": int(
            bool(last1)
            and bool(last2)
            and last1 == last2
        ),

        "name_first_two_words_match": int(
            bool(first_two1)
            and bool(first_two2)
            and first_two1 == first_two2
        ),

        # -------------------------------------------------
        # Length features
        # -------------------------------------------------

        "name_length_diff": _length_difference(
            canon1,
            canon2
        ),

        "name_token_count_diff": abs(
            len(_tokenize(canon1))
            - len(_tokenize(canon2))
        ),
    }


# =========================================================
# 4. ADDRESS FEATURES
# =========================================================

def address_features(addr1_norm, addr2_norm):

    addr1_norm = _safe_text(addr1_norm)
    addr2_norm = _safe_text(addr2_norm)

    nums1 = _extract_numbers(
        addr1_norm
    )

    nums2 = _extract_numbers(
        addr2_norm
    )

    common_numbers = nums1.intersection(
        nums2
    )

    union_numbers = nums1.union(
        nums2
    )

    # Number Jaccard
    if union_numbers:
        number_jaccard = (
            len(common_numbers)
            / len(union_numbers)
        )
    else:
        number_jaccard = 0.0

    return {

        # -------------------------------------------------
        # Missingness
        # -------------------------------------------------

        "address_missing_either": int(
            not addr1_norm
            or not addr2_norm
        ),

        # -------------------------------------------------
        # Fuzzy similarity
        # -------------------------------------------------

        "address_ratio": (
            fuzz.ratio(
                addr1_norm,
                addr2_norm
            ) / 100.0
        ),

        "address_token_set_ratio": (
            fuzz.token_set_ratio(
                addr1_norm,
                addr2_norm
            ) / 100.0
        ),

        # -------------------------------------------------
        # Token similarity
        # -------------------------------------------------

        "address_jaccard": _token_jaccard(
            addr1_norm,
            addr2_norm
        ),

        "address_common_token_count": (
            _common_token_count(
                addr1_norm,
                addr2_norm
            )
        ),

        # -------------------------------------------------
        # Number-based features
        # -------------------------------------------------

        "address_number_overlap": (
            number_jaccard
        ),

        "address_has_common_number": int(
            bool(common_numbers)
        ),

        "address_common_number_count": (
            len(common_numbers)
        ),

        # -------------------------------------------------
        # Length features
        # -------------------------------------------------

        "address_length_diff": (
            _length_difference(
                addr1_norm,
                addr2_norm
            )
        ),

        "address_token_count_diff": abs(
            len(_tokenize(addr1_norm))
            - len(_tokenize(addr2_norm))
        ),
    }


# =========================================================
# 5. BUILD FEATURES FOR ONE PAIR
# =========================================================

def build_pair_features(row1, row2):
    """
    row1 = Source 1 record
    row2 = Source 2 / Source 3 candidate record
    """

    features = {}

    # -----------------------------------------------------
    # Name features
    # -----------------------------------------------------

    features.update(
        name_features(
            row1["business_name_normalized"],
            row2["business_name_normalized"],
        )
    )

    # -----------------------------------------------------
    # Address features
    # -----------------------------------------------------

    features.update(
        address_features(
            row1["business_address_normalized"],
            row2["business_address_normalized"],
        )
    )

    # -----------------------------------------------------
    # Country
    # -----------------------------------------------------

    country1 = _safe_text(
        row1["country_normalized"]
    )

    country2 = _safe_text(
        row2["country_normalized"]
    )

    features["country_match"] = int(
        bool(country1)
        and bool(country2)
        and country1 == country2
    )

    return features


# =========================================================
# 6. LOAD PREPROCESSED SOURCES
# =========================================================

def load_preprocessed_sources(preprocessed_dir):
    """
    Load Source1, Source2 and Source3.

    Expected filenames:
        *_source1.tsv
        *_source2.tsv
        *_source3.tsv

    The files are indexed by entity_id.
    """

    preprocessed_dir = Path(
        preprocessed_dir
    )

    frames = {}

    for i in (1, 2, 3):

        matching_files = list(
            preprocessed_dir.glob(
                f"*_source{i}.tsv"
            )
        )

        if not matching_files:

            raise FileNotFoundError(
                f"No preprocessed Source{i} file found "
                f"in {preprocessed_dir}"
            )

        path = matching_files[0]

        print(
            f"Loading Source{i}: {path}"
        )

        df = pd.read_csv(
            path,
            sep="\t",
            dtype=str,
            keep_default_na=False,
        )

        required_columns = [
            "entity_id",
            "business_name_normalized",
            "business_address_normalized",
            "country_normalized",
        ]

        missing = [
            col
            for col in required_columns
            if col not in df.columns
        ]

        if missing:

            raise ValueError(
                f"Source{i} is missing columns: "
                f"{missing}"
            )

        frames[f"source{i}"] = (
            df.set_index("entity_id")
        )

    return frames


# =========================================================
# 7. ENTITY LOOKUP
# =========================================================

def _lookup(frames, entity_id):
    """
    Determine Source1/2/3 from entity ID prefix.

    Example:
        S1-xxxxx -> source1
        S2-xxxxx -> source2
        S3-xxxxx -> source3
    """

    entity_id = str(
        entity_id
    )

    prefix = entity_id[:2]

    source_key = {
        "S1": "source1",
        "S2": "source2",
        "S3": "source3",
    }

    if prefix not in source_key:

        raise ValueError(
            f"Unknown entity ID prefix: {entity_id}"
        )

    source_name = source_key[prefix]

    try:

        return frames[
            source_name
        ].loc[entity_id]

    except KeyError:

        raise KeyError(
            f"Entity ID {entity_id} "
            f"not found in {source_name}"
        )


# =========================================================
# 8. BUILD COMPLETE FEATURE TABLE
# =========================================================

def build_feature_table(
    candidates_path,
    preprocessed_dir,
    output_path
):
    """
    Convert candidate pairs into pairwise feature rows.
    """

    print("=" * 70)
    print("FEATURE ENGINEERING")
    print("=" * 70)

    # -----------------------------------------------------
    # Load sources
    # -----------------------------------------------------

    frames = load_preprocessed_sources(
        preprocessed_dir
    )

    print(
        "\nLoading candidate pairs..."
    )

    candidates = pd.read_csv(
        candidates_path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
    )

    required_candidate_columns = [
        "source1_entity_id",
        "candidate_entity_ids",
    ]

    missing = [
        col
        for col in required_candidate_columns
        if col not in candidates.columns
    ]

    if missing:

        raise ValueError(
            "candidate_pairs.tsv is missing columns: "
            f"{missing}"
        )

    print(
        f"Source1 records in candidate file: "
        f"{len(candidates)}"
    )

    # -----------------------------------------------------
    # Feature rows
    # -----------------------------------------------------

    rows = []

    processed_entities = 0
    total_pairs = 0

    for _, row in candidates.iterrows():

        s1_id = row[
            "source1_entity_id"
        ]

        candidate_ids = row[
            "candidate_entity_ids"
        ]

        processed_entities += 1

        # No candidates
        if not candidate_ids:
            continue

        # -------------------------------------------------
        # Source1 lookup
        # -------------------------------------------------

        s1_record = _lookup(
            frames,
            s1_id
        )

        # -------------------------------------------------
        # Candidate IDs
        # -------------------------------------------------

        candidate_list = [
            candidate.strip()
            for candidate
            in candidate_ids.split(",")
            if candidate.strip()
        ]

        # Remove duplicates while preserving order
        candidate_list = list(
            dict.fromkeys(
                candidate_list
            )
        )

        # -------------------------------------------------
        # Generate features
        # -------------------------------------------------

        for cand_id in candidate_list:

            try:

                candidate_record = _lookup(
                    frames,
                    cand_id
                )

            except KeyError:

                # Ignore invalid candidate IDs
                # rather than crashing the entire pipeline.
                continue

            features = build_pair_features(
                s1_record,
                candidate_record
            )

            # IDs
            features[
                "source1_entity_id"
            ] = s1_id

            features[
                "candidate_entity_id"
            ] = cand_id

            rows.append(
                features
            )

            total_pairs += 1

    # -----------------------------------------------------
    # Create DataFrame
    # -----------------------------------------------------

    if not rows:

        print(
            "\nWARNING: No candidate pairs "
            "were converted into features."
        )

        feature_df = pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )

    else:

        feature_df = pd.DataFrame(
            rows
        )

    # -----------------------------------------------------
    # Column order
    # -----------------------------------------------------

    id_cols = [
        "source1_entity_id",
        "candidate_entity_id",
    ]

    feature_cols = [
        col
        for col in feature_df.columns
        if col not in id_cols
    ]

    feature_df = feature_df[
        id_cols + feature_cols
    ]

    # -----------------------------------------------------
    # Save
    # -----------------------------------------------------

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    feature_df.to_csv(
        output_path,
        sep="\t",
        index=False
    )

    # -----------------------------------------------------
    # Summary
    # -----------------------------------------------------

    print("\n" + "=" * 70)
    print("FEATURE ENGINEERING COMPLETE")
    print("=" * 70)

    print(
        f"Source1 entities processed : "
        f"{processed_entities}"
    )

    print(
        f"Candidate pairs processed  : "
        f"{total_pairs}"
    )

    print(
        f"Feature columns             : "
        f"{len(feature_cols)}"
    )

    print(
        f"Output file                 : "
        f"{output_path}"
    )

    print(
        f"Output rows                 : "
        f"{len(feature_df)}"
    )

    return feature_df


# =========================================================
# 9. CLI
# =========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "Compute similarity features "
            "for Source1-candidate pairs."
        )
    )

    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        help=(
            "Directory containing "
            "preprocessing.py output."
        ),
    )

    parser.add_argument(
        "--candidates",
        required=True,
        help=(
            "Path to candidate_pairs.tsv "
            "generated by blocking.py."
        ),
    )

    parser.add_argument(
        "--output",
        required=True,
        help=(
            "Path to write the feature "
            "table TSV."
        ),
    )

    args = parser.parse_args()

    build_feature_table(
        candidates_path=args.candidates,
        preprocessed_dir=args.preprocessed_dir,
        output_path=args.output,
    )
