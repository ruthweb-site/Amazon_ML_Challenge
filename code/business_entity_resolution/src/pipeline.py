import os
import sys
import argparse
import pandas as pd
from typing import Dict, List, Set

# Ensure src directory is in sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from preprocessor import normalize_business_name, normalize_address
from blocking import InvertedIndexBlocker
from matching import compute_similarity
from evaluate import evaluate_predictions

def run_pipeline(
    s1_path: str,
    s2_path: str,
    s3_path: str,
    gt_path: str = None,
    output_dir: str = "output",
    threshold: float = 0.75,
    max_candidates: int = 25,
    sample_limit: int = None
):
    os.makedirs(output_dir, exist_ok=True)
    matching_tsv = os.path.join(output_dir, "matching_results.tsv")
    candidate_tsv = os.path.join(output_dir, "candidate_pairs.tsv")

    print(f"Loading Source 1 from {s1_path}...")
    df_s1 = pd.read_csv(s1_path, sep="\t", nrows=sample_limit)
    print(f"Loaded {len(df_s1)} Source 1 records.")

    print(f"Loading Source 2 from {s2_path}...")
    df_s2 = pd.read_csv(s2_path, sep="\t", nrows=sample_limit * 5 if sample_limit else None)
    print(f"Loaded {len(df_s2)} Source 2 records.")

    print(f"Loading Source 3 from {s3_path}...")
    df_s3 = pd.read_csv(s3_path, sep="\t", nrows=sample_limit * 5 if sample_limit else None)
    print(f"Loaded {len(df_s3)} Source 3 records.")

    # Preprocessing
    print("Normalizing names and addresses...")
    df_s1["clean_name"] = df_s1["business_name"].fillna("").apply(normalize_business_name)
    df_s1["clean_address"] = df_s1["business_address"].fillna("").apply(normalize_address)

    df_s2["clean_name"] = df_s2["business_name"].fillna("").apply(normalize_business_name)
    df_s2["clean_address"] = df_s2["business_address"].fillna("").apply(normalize_address)

    df_s3["clean_name"] = df_s3["business_name"].fillna("").apply(normalize_business_name)
    df_s3["clean_address"] = df_s3["business_address"].fillna("").apply(normalize_address)

    # Step 1: Candidate Generation (Blocking)
    print("Building Inverted Index Blocker for Source 2 and 3...")
    blocker = InvertedIndexBlocker(max_candidates_per_entity=max_candidates)
    blocker.index_target_records(df_s2)
    blocker.index_target_records(df_s3)
    print(f"Indexed {len(blocker.target_records)} target entities.")

    # Step 2: Generate candidates and score
    print("Generating candidate pairs and computing similarity scores...")
    candidate_records = []
    matching_records = []
    
    pred_dict: Dict[str, Set[str]] = {}

    for _, row in df_s1.iterrows():
        s1_id = row["entity_id"]
        s1_name = row["clean_name"]
        s1_addr = row["clean_address"]
        country = str(row.get("country", "")).strip()

        # Candidates from blocking
        candidates = blocker.generate_candidates_for_s1(s1_id, s1_name, country)
        candidate_records.append({
            "source1_entity_id": s1_id,
            "candidate_entity_ids": ",".join(candidates)
        })

        # Score candidates
        matched_ids = []
        for cand_id in candidates:
            target = blocker.target_records.get(cand_id)
            if not target:
                continue
            sim = compute_similarity(s1_name, s1_addr, target["clean_name"], target["clean_address"])
            if sim >= threshold:
                matched_ids.append(cand_id)

        matching_records.append({
            "source1_entity_id": s1_id,
            "matched_entity_ids": ",".join(matched_ids)
        })
        pred_dict[s1_id] = set(matched_ids)

    # Write output TSVs
    print(f"Writing candidate pairs to {candidate_tsv}...")
    df_cand = pd.DataFrame(candidate_records)
    df_cand.to_csv(candidate_tsv, sep="\t", index=False)

    print(f"Writing matching results to {matching_tsv}...")
    df_match = pd.DataFrame(matching_records)
    df_match.to_csv(matching_tsv, sep="\t", index=False)

    print(f"Candidate and Matching files written successfully!")

    # Step 3: Local Evaluation if Ground Truth is provided
    if gt_path and os.path.exists(gt_path):
        print("Evaluating predictions against Ground Truth...")
        s1_ids_set = set(df_s1["entity_id"])
        gt_dict: Dict[str, Set[str]] = {}
        for chunk in pd.read_csv(gt_path, sep="\t", chunksize=100000):
            matched_chunk = chunk[chunk["source1_entity_id"].isin(s1_ids_set)]
            for _, r in matched_chunk.iterrows():
                sid = str(r["source1_entity_id"])
                m = str(r.get("matched_entity_ids", "")).strip()
                gt_dict[sid] = set(m.split(",")) if m and m != "nan" else set()
            if len(gt_dict) >= len(s1_ids_set):
                break

        metrics = evaluate_predictions(gt_dict, pred_dict)
        print("=== EVALUATION RESULTS ===")
        print(f"Macro F_0.5 Score: {metrics['macro_f05']:.4f}")
        print(f"Total Evaluated Entities: {metrics['total_entities']}")
        print(f"Singleton Count: {metrics['singleton_count']} (Accuracy: {metrics['singleton_accuracy']:.2%})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Entity Resolution Pipeline")
    parser.add_argument("--s1", required=True, help="Path to Source 1 TSV")
    parser.add_argument("--s2", required=True, help="Path to Source 2 TSV")
    parser.add_argument("--s3", required=True, help="Path to Source 3 TSV")
    parser.add_argument("--gt", default=None, help="Path to Ground Truth TSV (optional)")
    parser.add_argument("--output-dir", default="output", help="Output directory")
    parser.add_argument("--threshold", type=float, default=0.75, help="Matching similarity threshold")
    parser.add_argument("--max-candidates", type=int, default=25, help="Max candidates per entity")
    parser.add_argument("--sample-limit", type=int, default=None, help="Limit number of rows for testing")

    args = parser.parse_args()
    run_pipeline(
        s1_path=args.s1,
        s2_path=args.s2,
        s3_path=args.s3,
        gt_path=args.gt,
        output_dir=args.output_dir,
        threshold=args.threshold,
        max_candidates=args.max_candidates,
        sample_limit=args.sample_limit
    )
