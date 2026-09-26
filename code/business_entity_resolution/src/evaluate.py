from typing import Dict, List, Set

def compute_entity_f05(true_matches: Set[str], pred_matches: Set[str]) -> float:
    """
    Computes F_0.5 score for a single Source 1 entity.
    F_0.5 = (1.25 * Precision * Recall) / (0.25 * Precision + Recall)
    
    Rules from official instructions:
    - If true_matches is empty (singleton):
        - 1.0 if pred_matches is empty
        - 0.0 if pred_matches is not empty (false merge)
    - If true_matches is non-empty:
        - 0.0 if pred_matches is empty
        - Otherwise calculate Precision and Recall:
            Precision = TP / len(pred_matches)
            Recall = TP / len(true_matches)
    """
    is_singleton = len(true_matches) == 0
    if is_singleton:
        return 1.0 if len(pred_matches) == 0 else 0.0
    
    if len(pred_matches) == 0:
        return 0.0
    
    tp = len(true_matches & pred_matches)
    if tp == 0:
        return 0.0
    
    precision = tp / len(pred_matches)
    recall = tp / len(true_matches)
    
    denom = 0.25 * precision + recall
    if denom == 0:
        return 0.0
    
    f05 = (1.25 * precision * recall) / denom
    return f05

def evaluate_predictions(ground_truth: Dict[str, Set[str]], predictions: Dict[str, Set[str]]) -> Dict[str, float]:
    """
    Calculates Macro-averaged F_0.5 across all Source 1 entities in ground_truth.
    """
    total_f05 = 0.0
    total_count = len(ground_truth)
    
    singleton_count = 0
    singleton_correct = 0
    
    for s1_id, true_set in ground_truth.items():
        pred_set = predictions.get(s1_id, set())
        score = compute_entity_f05(true_set, pred_set)
        total_f05 += score
        
        if len(true_set) == 0:
            singleton_count += 1
            if len(pred_set) == 0:
                singleton_correct += 1
                
    macro_f05 = total_f05 / total_count if total_count > 0 else 0.0
    
    return {
        "macro_f05": macro_f05,
        "total_entities": total_count,
        "singleton_count": singleton_count,
        "singleton_accuracy": singleton_correct / singleton_count if singleton_count > 0 else 1.0
    }
