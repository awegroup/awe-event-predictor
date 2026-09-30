from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Literal
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    balanced_accuracy_score, roc_auc_score, average_precision_score,
    confusion_matrix, classification_report, make_scorer
)
import warnings
warnings.filterwarnings('ignore')


def balanced_accuracy_scorer(y_true, y_pred):
    """Custom scorer for balanced accuracy."""
    return balanced_accuracy_score(y_true, y_pred)


def macro_f1_scorer(y_true, y_pred):
    """Custom scorer for macro F1."""
    return f1_score(y_true, y_pred, average='macro')


def f1_class1_scorer(y_true, y_pred):
    """Custom scorer for F1 score on class 1 (faulty class)."""
    return f1_score(y_true, y_pred, pos_label=1, average='binary', zero_division=0)


def validate_and_clean_features(X: np.ndarray) -> np.ndarray:
    """
    Validate and clean feature matrix (handle NaN, inf, etc.).
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
        
    Returns
    -------
    np.ndarray
        Cleaned feature matrix.
    """
    X_clean = X.copy()
    
    # Replace inf with NaN
    X_clean[np.isinf(X_clean)] = np.nan
    
    # Count NaN values
    nan_count = np.isnan(X_clean).sum()
    if nan_count > 0:
        print(f"  Warning: Found {nan_count} NaN values in features")
        # Replace NaN with column median
        for col_idx in range(X_clean.shape[1]):
            col = X_clean[:, col_idx]
            if np.isnan(col).any():
                median_val = np.nanmedian(col)
                if np.isnan(median_val):
                    median_val = 0.0  # Fallback if entire column is NaN
                X_clean[np.isnan(col), col_idx] = median_val
    
    # Check for constant features
    constant_features = []
    for col_idx in range(X_clean.shape[1]):
        col = X_clean[:, col_idx]
        if np.std(col) < 1e-10:
            constant_features.append(col_idx)
    
    if constant_features:
        print(f"  Warning: Found {len(constant_features)} constant features (std < 1e-10)")
        # Optionally remove constant features (for now, we keep them)
    
    return X_clean


def scale_features(
    X_train: np.ndarray,
    X_test: np.ndarray,
    method: Literal["standard", "minmax"] = "standard",
    clip_percentiles: Optional[tuple[float, float]] = (1.0, 99.0)
) -> tuple[np.ndarray, np.ndarray, object]:
    """
    Scale features with optional percentile clipping (train-only).

    Parameters
    ----------
    X_train : np.ndarray
        Training features.
    X_test : np.ndarray
        Test features.
    method : str
        'standard' or 'minmax'
    clip_percentiles : tuple or None
        (low, high) percentiles for clipping. If None, no clipping.

    Returns
    -------
    X_train_scaled, X_test_scaled, scaler
    """
    print(f"\nScaling features using {method} scaling...")

    X_train_proc = X_train.copy()
    X_test_proc  = X_test.copy()

    # --------------------------------------------------
    # 1. CLIP EXTREME VALUES (TRAIN-ONLY)
    # --------------------------------------------------
    if clip_percentiles is not None:
        p_low, p_high = clip_percentiles
        print(f"  Clipping features to {p_low}–{p_high} percentiles (train only)")

        lower_bounds = np.percentile(X_train_proc, p_low, axis=0)
        upper_bounds = np.percentile(X_train_proc, p_high, axis=0)

        # Handle degenerate columns
        mask = lower_bounds >= upper_bounds
        if np.any(mask):
            upper_bounds[mask] = lower_bounds[mask] + 1e-12

        X_train_proc = np.clip(X_train_proc, lower_bounds, upper_bounds)
        X_test_proc  = np.clip(X_test_proc,  lower_bounds, upper_bounds)

    # --------------------------------------------------
    # 2. SCALE
    # --------------------------------------------------
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError(f"Unknown scaling method: {method}")

    X_train_scaled = scaler.fit_transform(X_train_proc)
    X_test_scaled  = scaler.transform(X_test_proc)

    print(
        f"  Train mean={X_train_scaled.mean():.4f}, "
        f"std={X_train_scaled.std():.4f}"
    )
    print(
        f"  Test  mean={X_test_scaled.mean():.4f}, "
        f"std={X_test_scaled.std():.4f}"
    )

    return X_train_scaled, X_test_scaled, scaler


def tune_logistic_regression_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring: Literal["macro_f1", "balanced_accuracy", "f1", "f1_class1", "pr_auc"] = "macro_f1",
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict:
    """
    Tune Logistic Regression hyperparameters using random search.
    
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
    print(f"\nTuning Logistic Regression hyperparameters (optimizing {scoring})...")
    
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
        # Note: Logistic Regression already provides probability estimates
        # Use built-in sklearn scorer for average_precision
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
    # We'll use a simpler approach: separate grids for compatible combinations
    # and let RandomizedSearchCV handle sampling with error handling
    
    # Simplified parameter grid - focusing on most common and compatible combinations
    # L2 works with all solvers, L1 works with liblinear and saga
    param_distributions = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
        'penalty': ['l2', 'l1'],  # Exclude elasticnet for simplicity
        'solver': ['lbfgs', 'liblinear', 'saga'],
        'class_weight': [None, 'balanced'],
        'max_iter': [100, 500, 1000, 2000]
    }
    
    # Use RandomizedSearchCV
    # Note: sklearn will skip invalid parameter combinations automatically
    lr = LogisticRegression(random_state=random_state, max_iter=1000)
    
    search = RandomizedSearchCV(
        lr,
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


def train_logistic_regression(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyperparameters: Optional[Dict] = None
) -> Dict:
    """
    Train Logistic Regression model and evaluate on test set.
    
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
        Logistic Regression hyperparameters. If None, uses defaults.
        
    Returns
    -------
    dict
        Trained model, predictions, probabilities, and metrics.
    """
    print("\nTraining Logistic Regression model...")
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'C': 1.0,
            'penalty': 'l2',
            'solver': 'lbfgs',
            'class_weight': 'balanced',
            'max_iter': 1000,
            'random_state': 42
        }
    else:
        hyperparameters = hyperparameters.copy()
        if 'random_state' not in hyperparameters:
            hyperparameters['random_state'] = 42
        if 'max_iter' not in hyperparameters:
            hyperparameters['max_iter'] = 1000
    
    # Create and train model
    lr = LogisticRegression(**hyperparameters)
    lr.fit(X_train, y_train)
    
    # Predictions
    y_pred = lr.predict(X_test)
    
    # Probabilities (posterior P(y|x))
    y_proba = lr.predict_proba(X_test)
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print("\nTest Set Performance:")
    print_metrics(metrics)
    
    return {
        'model': lr,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'metrics': metrics
    }


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

