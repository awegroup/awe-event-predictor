from __future__ import annotations
import numpy as np
from typing import Dict, Literal, Optional
from sklearn.metrics import (
    precision_recall_curve,
    f1_score,
    precision_score,
    recall_score,
    average_precision_score
)


def tune_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    method: Literal["max_f1", "max_pr_auc", "balanced_precision_recall"] = "max_f1"
) -> Dict:
    """
    Tune decision threshold on validation set using probabilities.
    
    Parameters
    ----------
    y_true : np.ndarray
        True labels (validation set).
    y_proba : np.ndarray
        Predicted probabilities for positive class (class 1).
        Should be 1D array of probabilities for class 1, or 2D array where we extract column 1.
    method : str
        Method to select optimal threshold:
        - 'max_f1': Maximize F1 score
        - 'max_pr_auc': Maximize PR-AUC (uses precision-recall curve)
        - 'balanced_precision_recall': Find threshold where precision ≈ recall
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'optimal_threshold': Optimal threshold value
        - 'method': Method used
        - 'f1_at_threshold': F1 score at optimal threshold
        - 'precision_at_threshold': Precision at optimal threshold
        - 'recall_at_threshold': Recall at optimal threshold
        - 'pr_auc': PR-AUC score
    """
    # Ensure y_proba is 1D array of probabilities for positive class
    if y_proba.ndim == 2:
        if y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            raise ValueError(f"Expected 2 columns in y_proba, got {y_proba.shape[1]}")
    else:
        y_proba_pos = y_proba
    
    # Compute precision-recall curve
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba_pos)
    
    # Note: precision_recall_curve returns:
    # - precision and recall arrays of length n_thresholds + 1
    # - thresholds array of length n_thresholds
    # The last element of precision/recall corresponds to threshold = 0.0 (all predictions positive)
    
    # Compute F1 at each threshold
    # We iterate over thresholds (which has length len(precision) - 1)
    f1_scores = []
    for i in range(len(thresholds)):
        p = precision[i]
        r = recall[i]
        if p + r > 0:
            f1 = 2 * (p * r) / (p + r)
        else:
            f1 = 0.0
        f1_scores.append(f1)
    
    f1_scores = np.array(f1_scores)
    
    # Also add the last point (threshold = 0.0, all positive)
    if len(precision) > len(thresholds):
        p_last = precision[-1]
        r_last = recall[-1]
        if p_last + r_last > 0:
            f1_last = 2 * (p_last * r_last) / (p_last + r_last)
        else:
            f1_last = 0.0
        # Append threshold 0.0 and its F1 score
        thresholds = np.append(thresholds, [0.0])
        f1_scores = np.append(f1_scores, [f1_last])
    
    # Compute PR-AUC
    pr_auc = average_precision_score(y_true, y_proba_pos)
    
    # Select optimal threshold based on method
    if method == "max_f1":
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        
    elif method == "max_pr_auc":
        # PR-AUC is already computed, but we want to maximize F1 while having good PR-AUC
        # So we'll use max_f1 as a proxy (PR-AUC is already optimized during hyperparameter tuning)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        
    elif method == "balanced_precision_recall":
        # Find threshold where precision and recall are most balanced
        # Minimize |precision - recall|
        # Use all precision/recall values except the last one (which corresponds to threshold=0.0)
        if len(precision) == len(thresholds):
            # Normal case: precision/recall have one more element than thresholds
            balance_scores = np.abs(precision[:-1] - recall[:-1])
        else:
            # Edge case: use all available
            balance_scores = np.abs(precision[:len(thresholds)] - recall[:len(thresholds)])
        optimal_idx = np.argmin(balance_scores)
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f1_scores[optimal_idx]
        optimal_precision = precision[optimal_idx]
        optimal_recall = recall[optimal_idx]
        
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return {
        'optimal_threshold': float(optimal_threshold),
        'method': method,
        'f1_at_threshold': float(optimal_f1),
        'precision_at_threshold': float(optimal_precision),
        'recall_at_threshold': float(optimal_recall),
        'pr_auc': float(pr_auc),
        'all_thresholds': thresholds.tolist(),
        'all_f1_scores': f1_scores.tolist(),
        'all_precision': precision.tolist(),
        'all_recall': recall.tolist()
    }


def apply_threshold(y_proba: np.ndarray, threshold: float) -> np.ndarray:
    """
    Apply threshold to probabilities to get binary predictions.
    
    Parameters
    ----------
    y_proba : np.ndarray
        Predicted probabilities. Can be 1D (positive class) or 2D (both classes).
    threshold : float
        Decision threshold.
    
    Returns
    -------
    np.ndarray
        Binary predictions (0 or 1).
    """
    # Extract positive class probabilities
    if y_proba.ndim == 2:
        if y_proba.shape[1] == 2:
            y_proba_pos = y_proba[:, 1]
        else:
            raise ValueError(f"Expected 2 columns in y_proba, got {y_proba.shape[1]}")
    else:
        y_proba_pos = y_proba
    
    return (y_proba_pos >= threshold).astype(int)

