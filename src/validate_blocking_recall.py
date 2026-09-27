"""
validate_blocking_recall.py

Checks blocking quality against the training ground truth:
  - Recall: of all TRUE matches, what fraction did blocking's candidate
    set actually contain? (This is the ceiling your matching model can
    ever reach - a missed candidate can never be recovered later.)
  - Average candidate-set size per Source1 entity (smaller is better,
    per the challenge's updated scoring rule - as long as recall holds up).

Usage:
    python validate_blocking_recall.py \
        --candidates output/candidate_pairs.tsv \
        --ground-truth dataset/train/train_ground_truth.tsv
"""

import argparse
import pandas as pd


def validate(candidates_path, ground_truth_path):
    cands = pd.read_csv(candidates_path, sep="\t", dtype=str, keep_default_na=False)
    gt = pd.read_csv(ground_truth_path, sep="\t", dtype=str, keep_default_na=False)

    cand_map = dict(zip(cands["source1_entity_id"], cands["candidate_entity_ids"]))

    total_true_matches = 0
    recalled = 0
    perfect_entities = 0  # entities where ALL true matches were recalled
    non_singleton_entities = 0

    total_candidate_count = 0

    for _, row in gt.iterrows():
        s1_id = row["source1_entity_id"]
        true_ids = set(row["matched_entity_ids"].split(",")) if row["matched_entity_ids"] else set()

        cand_str = cand_map.get(s1_id, "")
        cand_ids = set(cand_str.split(",")) if cand_str else set()
        total_candidate_count += len(cand_ids)

        if true_ids:
            non_singleton_entities += 1
            found = true_ids & cand_ids
            total_true_matches += len(true_ids)
            recalled += len(found)
            if found == true_ids:
                perfect_entities += 1

    print("=" * 50)
    print("BLOCKING QUALITY REPORT")
    print("=" * 50)
    print(f"Total Source1 entities in ground truth: {len(gt)}")
    print(f"Non-singleton entities (have >=1 true match): {non_singleton_entities}")
    print(f"Total true match links: {total_true_matches}")
    print(f"Recalled by blocking: {recalled}")
    print(f"RECALL: {recalled / total_true_matches * 100:.2f}%")
    print(f"Entities with ALL true matches recalled: {perfect_entities} "
          f"({perfect_entities / non_singleton_entities * 100:.2f}%)")
    print(f"Average candidate-set size per entity: {total_candidate_count / len(gt):.2f}")
    print("=" * 50)
    print("Rule of thumb: aim for recall > 90% before worrying about")
    print("shrinking the candidate set further - a missed true match here")
    print("can never be recovered by the matching model downstream.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate blocking recall against ground truth.")
    parser.add_argument("--candidates", required=True, help="Path to candidate_pairs.tsv")
    parser.add_argument("--ground-truth", required=True, help="Path to train_ground_truth.tsv")
    args = parser.parse_args()

    validate(args.candidates, args.ground_truth)