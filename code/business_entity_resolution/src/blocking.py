import os
import pandas as pd
from collections import defaultdict
from typing import Dict, List, Set, Tuple

def extract_blocking_keys(name: str) -> List[str]:
    """
    Extracts high-quality blocking keys:
    1. First significant word (if len >= 3)
    2. Character 3-grams of significant tokens for typo tolerance
    """
    tokens = [t for t in name.split() if len(t) >= 3 and not t.isdigit()]
    keys = []
    
    # Prefix blocking: first 2 tokens
    if tokens:
        keys.append(f"w1:{tokens[0]}")
        if len(tokens) > 1:
            keys.append(f"w2:{tokens[1]}")
            keys.append(f"w12:{tokens[0]}_{tokens[1]}")
            
    # Character 3-grams for typo-resilient candidate generation
    for t in tokens[:3]:
        if len(t) >= 4:
            for i in range(len(t) - 2):
                keys.append(f"ng:{t[i:i+3]}")
                
    return keys

class InvertedIndexBlocker:
    """
    Fast, memory-efficient candidate generator using an inverted index.
    Restricts candidate searches strictly within the same country.
    """
    def __init__(self, max_candidates_per_entity: int = 20):
        self.max_candidates = max_candidates_per_entity
        # index: country -> token -> list of target entity_ids
        self.index: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))
        self.target_records: Dict[str, Dict[str, str]] = {}

    def index_target_records(self, df: pd.DataFrame):
        """Index Source 2 and Source 3 records."""
        for _, row in df.iterrows():
            eid = row["entity_id"]
            country = str(row.get("country", "")).strip().upper()
            norm_name = str(row.get("clean_name", "")).strip()
            
            self.target_records[eid] = {
                "clean_name": norm_name,
                "clean_address": str(row.get("clean_address", "")).strip(),
                "country": country
            }
            
            keys = extract_blocking_keys(norm_name)
            for k in set(keys):
                self.index[country][k].append(eid)

    def generate_candidates_for_s1(self, s1_id: str, norm_name: str, country: str) -> List[str]:
        """Find candidate target IDs sharing blocking keys within the same country."""
        country = country.strip().upper()
        country_index = self.index.get(country)
        if not country_index:
            return []
            
        keys = extract_blocking_keys(norm_name)
        if not keys:
            return []
            
        # Count key overlaps
        candidate_counts: Dict[str, int] = defaultdict(int)
        for k in set(keys):
            if k in country_index:
                # Add weight: full words have higher weight than 3-grams
                weight = 3 if k.startswith("w1:") or k.startswith("w2:") else 1
                for target_id in country_index[k]:
                    candidate_counts[target_id] += weight
                    
        if not candidate_counts:
            return []
            
        # Rank by overlap count and return top-K
        sorted_candidates = sorted(candidate_counts.items(), key=lambda x: x[1], reverse=True)
        return [c[0] for c in sorted_candidates[:self.max_candidates]]
