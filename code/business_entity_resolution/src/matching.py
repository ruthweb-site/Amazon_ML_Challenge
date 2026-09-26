from typing import Dict, List, Set, Tuple

try:
    from rapidfuzz import fuzz
    HAS_RAPIDFUZZ = True
except ImportError:
    HAS_RAPIDFUZZ = False

def jaccard_similarity(set1: Set[str], set2: Set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not set1 or not set2:
        return 0.0
    intersection = len(set1 & set2)
    union = len(set1 | set2)
    return intersection / union if union > 0 else 0.0

def levenshtein_ratio(s1: str, s2: str) -> float:
    """Fallback Levenshtein similarity ratio between 0.0 and 1.0."""
    if not s1 or not s2:
        return 0.0
    if s1 == s2:
        return 1.0
    
    # Simple dynamic programming Levenshtein distance
    m, n = len(s1), len(s2)
    if m < n:
        s1, s2 = s2, s1
        m, n = n, m
        
    prev = list(range(n + 1))
    for i, c1 in enumerate(s1):
        curr = [i + 1] * (n + 1)
        for j, c2 in enumerate(s2):
            insertions = prev[j + 1] + 1
            deletions = curr[j] + 1
            substitutions = prev[j] + (c1 != c2)
            curr[j + 1] = min(insertions, deletions, substitutions)
        prev = curr
        
    dist = prev[n]
    max_len = max(m, n)
    return 1.0 - (dist / max_len)

def compute_similarity(s1_name: str, s1_addr: str, target_name: str, target_addr: str) -> float:
    """
    Computes a combined similarity score between Source 1 record and a Candidate record.
    Returns a score between 0.0 and 1.0.
    """
    # 1. Name Similarity
    if HAS_RAPIDFUZZ:
        token_sort = fuzz.token_sort_ratio(s1_name, target_name) / 100.0
        token_set = fuzz.token_set_ratio(s1_name, target_name) / 100.0
        ratio = fuzz.ratio(s1_name, target_name) / 100.0
        name_score = 0.5 * token_set + 0.3 * token_sort + 0.2 * ratio
    else:
        s1_tokens = set(s1_name.split())
        target_tokens = set(target_name.split())
        token_jaccard = jaccard_similarity(s1_tokens, target_tokens)
        lev = levenshtein_ratio(s1_name, target_name)
        name_score = 0.6 * token_jaccard + 0.4 * lev

    # 2. Address Similarity (if both present)
    if s1_addr and target_addr:
        s1_addr_tokens = set(s1_addr.split())
        target_addr_tokens = set(target_addr.split())
        addr_score = jaccard_similarity(s1_addr_tokens, target_addr_tokens)
        
        # When address is present, require both name and address agreement
        final_score = 0.70 * name_score + 0.30 * addr_score
    else:
        # If address is missing in one of them, rely on high-precision name score
        final_score = name_score * 0.95

    return final_score
