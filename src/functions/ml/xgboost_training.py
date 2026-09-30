from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Literal, List
try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    print("Warning: xgboost not installed. Install with: pip install xgboost")
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, make_scorer
)
import warnings
warnings.filterwarnings('ignore')

def format_feature_name(feature_name: str) -> str:
    """
    Format feature name for display.
    
    Parameters
    ----------
    feature_name : str
        Feature name in format "{signal_name}_{feature_type}" (e.g., "V_kite_0_mag_mean")
        or legacy format "{signal_name}_feat_{feat_idx}" for backward compatibility
        
    Returns
    -------
    str
        Formatted feature name (returns as-is if already formatted, or formats legacy names)
    """
    # Check if it's already in the new format (contains actual feature type names)
    # Legacy format: "{signal}_feat_{idx}"
    # New format: "{signal}_{feature_type}"
    
    # If it contains "_feat_", it's legacy format - try to parse it
    if "_feat_" in feature_name:
        # Legacy format - just return as-is with a note that it's legacy
        # (This shouldn't happen with new code, but handle for backward compatibility)
        return feature_name.replace("_feat_", "_")
    
    # Already in new format, return as-is
    return feature_name


def tune_xgboost_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring: Literal["macro_f1", "balanced_accuracy", "f1", "f1_class1", "pr_auc"] = "macro_f1",
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict:
    """
    Tune XGBoost hyperparameters using random search.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    scoring : str
        Metric to optimize: 'macro_f1', 'balanced_accuracy', 'f1', 'f1_class1', or 'pr_auc'.
        'f1_class1' optimizes F1 score specifically for the faulty class (label 1).
        'pr_auc' optimizes Precision-Recall Area Under Curve (PR-AUC).
    cv : int
        Number of cross-validation folds.
    n_iter : int
        Number of iterations for random search.
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
        
    Returns
    -------
    dict
        Best parameters and best score.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Install with: pip install xgboost")
    
    print(f"\nTuning XGBoost hyperparameters (optimizing {scoring})...")
    
    # Map scoring to sklearn scorer
    if scoring == 'f1_class1':
        # Custom scorer for F1 on class 1 (faulty class)
        sklearn_scorer = make_scorer(
            f1_score,
            pos_label=1,
            average='binary',
            zero_division=0
        )
    elif scoring == 'pr_auc':
        # Custom scorer for PR-AUC (average precision score)
        from sklearn.metrics import get_scorer
        sklearn_scorer = get_scorer('average_precision')
    else:
        scoring_map = {
            'macro_f1': 'f1_macro',
            'balanced_accuracy': 'balanced_accuracy',
            'f1': 'f1'
        }
        if scoring not in scoring_map:
            raise ValueError(f"Unknown scoring: {scoring}")
        sklearn_scorer = scoring_map[scoring]
    
    # Define parameter distributions for RandomizedSearchCV
    # Focused on most effective XGBoost hyperparameters
    param_distributions = {
        'n_estimators': [50, 100, 200, 300, 500],
        'max_depth': [3, 4, 5, 6, 7, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
        'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0],
        'min_child_weight': [1, 3, 5, 7],
        'gamma': [0, 0.1, 0.2, 0.3, 0.4],
        'reg_alpha': [0, 0.01, 0.1, 1.0],
        'reg_lambda': [1, 1.5, 2, 3],
        'scale_pos_weight': [1, 2, 3, 5, 10]  # For handling class imbalance
    }
    
    # Create XGBoost classifier
    xgb_classifier = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        use_label_encoder=False,
        random_state=random_state,
        n_jobs=1  # XGBoost handles parallelism internally
    )
    
    search = RandomizedSearchCV(
        xgb_classifier,
        param_distributions=param_distributions,
        n_iter=n_iter,
        cv=cv,
        scoring=sklearn_scorer,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1,
        error_score=np.nan  # Skip invalid combinations
    )
    
    search.fit(X_train, y_train)
    
    print(f"\n  Best parameters: {search.best_params_}")
    print(f"  Best CV score ({scoring}): {search.best_score_:.4f}")
    
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_estimator': search.best_estimator_
    }


def train_xgboost(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyperparameters: Optional[Dict] = None,
    feature_names: Optional[List[str]] = None
) -> Dict:
    """
    Train XGBoost model and evaluate on test set.
    
    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    y_train : np.ndarray
        Training labels.
    X_test : np.ndarray
        Test features.
    y_test : np.ndarray
        Test labels.
    hyperparameters : dict, optional
        XGBoost hyperparameters. If None, uses defaults.
    feature_names : list, optional
        List of feature names. If provided, will be used to display meaningful feature names
        in importance output instead of "Feature {idx}".
        
    Returns
    -------
    dict
        Trained model, predictions, probabilities, and metrics.
    """
    if not XGBOOST_AVAILABLE:
        raise ImportError("xgboost is not installed. Install with: pip install xgboost")
    
    print("\nTraining XGBoost model...")
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'n_estimators': 100,
            'max_depth': 6,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'min_child_weight': 1,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 1,
            'scale_pos_weight': 1,
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'use_label_encoder': False,
            'random_state': 42,
            'n_jobs': 1
        }
    else:
        hyperparameters = hyperparameters.copy()
        # Ensure required XGBoost parameters are set
        if 'objective' not in hyperparameters:
            hyperparameters['objective'] = 'binary:logistic'
        if 'eval_metric' not in hyperparameters:
            hyperparameters['eval_metric'] = 'logloss'
        if 'use_label_encoder' not in hyperparameters:
            hyperparameters['use_label_encoder'] = False
        if 'random_state' not in hyperparameters:
            hyperparameters['random_state'] = 42
        if 'n_jobs' not in hyperparameters:
            hyperparameters['n_jobs'] = 1
    
    # Create and train model
    xgb_model = xgb.XGBClassifier(**hyperparameters)
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    
    # Probabilities
    y_proba = xgb_model.predict_proba(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print("\nTest Set Performance:")
    print_metrics(metrics)
    
    # Extract XGBoost built-in feature importance
    feature_importance = None
    try:
        # Get feature importance (gain, weight, cover, or total_gain)
        importance_gain = xgb_model.feature_importances_
        
        # Create a dictionary similar to SVM feature importance structure for compatibility
        feature_importance = {
            'importance_scores': importance_gain,
            'ranked_indices': np.argsort(importance_gain)[::-1],  # Descending order
            'feature_importance': importance_gain,
            'importance_type': 'gain'  # XGBoost uses gain by default
        }
        
        print(f"\n  XGBoost feature importance computed for {len(importance_gain)} features")
        print(f"  Top 10 most important features:")
        top_10 = feature_importance['ranked_indices'][:10]
        for i, idx in enumerate(top_10, 1):
            if feature_names is not None and idx < len(feature_names):
                # Feature names are already in format "signal_name_feature_type"
                # Just use them directly (format_feature_name handles legacy format if needed)
                formatted_name = format_feature_name(feature_names[idx])
                print(f"    {i}. {formatted_name}: importance={importance_gain[idx]:.6f}")
            else:
                print(f"    {i}. Feature {idx}: importance={importance_gain[idx]:.6f}")
    except Exception as e:
        print(f"  Warning: Could not extract XGBoost feature importance: {e}")
        feature_importance = None
    
    results = {
        'model': xgb_model,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics
    }
    
    if feature_importance is not None:
        results['feature_importance'] = feature_importance
    
    return results


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_proba: Optional[np.ndarray] = None) -> Dict:
    """Compute comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
        'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
        'macro_f1': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
    }
    
    if y_proba is not None:
        # ROC-AUC and PR-AUC require probabilities
        try:
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['pr_auc'] = average_precision_score(y_true, y_proba[:, 1])
        except Exception as e:
            print(f"  Warning: Could not compute AUC metrics: {e}")
            metrics['roc_auc'] = None
            metrics['pr_auc'] = None
    
    return metrics


def print_metrics(metrics: Dict):
    """Print classification metrics in a readable format."""
    print(f"  Accuracy:           {metrics['accuracy']:.4f}")
    print(f"  Precision:          {metrics['precision']:.4f}")
    print(f"  Recall:             {metrics['recall']:.4f}")
    print(f"  F1-score:           {metrics['f1']:.4f}")
    print(f"  Macro F1:           {metrics['macro_f1']:.4f}")
    print(f"  Balanced Accuracy:  {metrics['balanced_accuracy']:.4f}")
    
    if metrics.get('roc_auc') is not None:
        print(f"  ROC-AUC:            {metrics['roc_auc']:.4f}")
    if metrics.get('pr_auc') is not None:
        print(f"  PR-AUC:             {metrics['pr_auc']:.4f}")
    
    print(f"\n  Confusion Matrix:")
    cm = metrics['confusion_matrix']
    print(f"    [[{cm[0][0]:4d}, {cm[0][1]:4d}],")
    print(f"     [{cm[1][0]:4d}, {cm[1][1]:4d}]]")
    print(f"    (TN={cm[0][0]}, FP={cm[0][1]}, FN={cm[1][0]}, TP={cm[1][1]})")

