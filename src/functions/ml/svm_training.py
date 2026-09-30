from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Literal
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
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



def tune_svm_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    scoring: Literal["macro_f1", "balanced_accuracy", "f1", "f1_class1", "pr_auc"] = "macro_f1",
    cv: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
    n_jobs: int = -1,
    probability: bool = True
) -> Dict:
    """
    Tune SVM hyperparameters using grid search or random search.
    
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
        Number of iterations for random search (if using).
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
    probability : bool
        Whether to enable probability estimates in SVM. If False, uses classic margin optimization.
        
    Returns
    -------
    dict
        Best parameters and best score.
    """
    print(f"\nTuning SVM hyperparameters (optimizing {scoring})...")
    
    # Define optimized parameter grid (aggressively reduced for faster tuning)
    # Focused on most effective values based on SVM best practices:
    # - C: Reduced to 3 most common values (1, 10, 100 cover most cases)
    # - gamma: 'scale' is usually best, plus one numeric value (0.01) as backup
    # - kernel: 'rbf' is most common, 'linear' is fast alternative
    # - class_weight: Keep both options (important for imbalanced data)
    # This reduces search space from 180 to 12 combinations (75% reduction)
    param_grid = {
        'C': [0.1, 1, 10, 100],
        'gamma': ['scale', 0.01],
        'kernel': ['rbf', 'linear'],
        'class_weight': [None, 'balanced']
    }
    
    # Calculate total combinations and cap n_iter to avoid redundant iterations
    total_combinations = 1
    for param_values in param_grid.values():
        total_combinations *= len(param_values)
    effective_n_iter = min(n_iter, total_combinations)
    if effective_n_iter < n_iter:
        print(f"  Note: Reduced n_iter from {n_iter} to {effective_n_iter} (total combinations: {total_combinations})")
    
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
        # Note: This requires probability estimates, which SVC already has enabled
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
    
    # Use RandomizedSearchCV for faster search
    svm = SVC(probability=probability, random_state=random_state)
    
    search = RandomizedSearchCV(
        svm,
        param_distributions=param_grid,
        n_iter=effective_n_iter,
        cv=cv,
        scoring=sklearn_scorer,
        n_jobs=n_jobs,
        random_state=random_state,
        verbose=1
    )
    
    search.fit(X_train, y_train)
    
    print(f"\n  Best parameters: {search.best_params_}")
    print(f"  Best CV score ({scoring}): {search.best_score_:.4f}")
    
    return {
        'best_params': search.best_params_,
        'best_score': search.best_score_,
        'best_estimator': search.best_estimator_
    }


def train_svm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    hyperparameters: Optional[Dict] = None,
    probability: bool = True,
    compute_feature_importance: bool = True,
    feature_importance_n_repeats: int = 10,
    feature_importance_boundary_threshold: float = 0.1,
    feature_importance_adaptive_threshold: bool = True,
    feature_importance_min_boundary_faults: int = 10
) -> Dict:
    """
    Train SVM model and evaluate on test set.
    
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
        SVM hyperparameters. If None, uses defaults.
    probability : bool
        Whether to enable probability estimates.
    compute_feature_importance : bool
        Whether to compute margin-based feature importance.
    feature_importance_n_repeats : int
        Number of permutations per feature for importance computation.
    feature_importance_boundary_threshold : float
        Fixed threshold for identifying boundary samples (absolute margin < threshold).
        Only used if feature_importance_adaptive_threshold is False.
    feature_importance_adaptive_threshold : bool
        If True, uses percentile-based adaptive threshold for boundary region.
    feature_importance_min_boundary_faults : int
        Minimum number of fault-class samples required in boundary region (adaptive mode).
        
    Returns
    -------
    dict
        Trained model, predictions, probabilities, metrics, and feature importance.
    """
    print("\nTraining SVM model...")
    
    # Default hyperparameters
    if hyperparameters is None:
        hyperparameters = {
            'C': 1.0,
            'gamma': 'scale',
            'kernel': 'gamma',
            'class_weight': None,
            'probability': probability,
            'random_state': 42
        }
    else:
        hyperparameters = hyperparameters.copy()
        hyperparameters['probability'] = probability
        if 'random_state' not in hyperparameters:
            hyperparameters['random_state'] = 42
    
    # Create and train model
    svm = SVC(**hyperparameters)
    svm.fit(X_train, y_train)
    
    # Predictions (uses decision function / margin)
    y_pred = svm.predict(X_test)
    
    # Probabilities or decision function
    y_proba = None
    if probability:
        # Probability-based: use predict_proba
        y_proba = svm.predict_proba(X_test)
        print("  Using probability-based predictions (predict_proba)")
    else:
        # Margin-based: use decision_function (distance from hyperplane)
        # Note: decision_function returns signed distance, not probabilities
        # We store it in y_proba for compatibility, but it's not a probability
        decision_scores = svm.decision_function(X_test)
        # Convert to 2D array format for compatibility (though not true probabilities)
        # decision_function returns 1D for binary classification
        if decision_scores.ndim == 1:
            # For binary classification, create 2D array with [negative_class_score, positive_class_score]
            # We approximate: prob_class_0 = sigmoid(-decision), prob_class_1 = sigmoid(decision)
            # But for margin mode, we just store the decision scores
            y_proba = np.column_stack([-decision_scores, decision_scores])
        else:
            y_proba = decision_scores
        print("  Using margin-based predictions (decision_function)")
        print(f"  Decision function range: [{decision_scores.min():.4f}, {decision_scores.max():.4f}]")
    
    # Compute metrics
    metrics = compute_metrics(y_test, y_pred, y_proba)
    
    print("\nTest Set Performance:")
    print_metrics(metrics)
    
    # Compute margin-based feature importance
    feature_importance = None
    if compute_feature_importance:
        try:
            feature_importance = compute_margin_based_feature_importance(
                model=svm,
                X=X_train,
                y=y_train,
                n_repeats=feature_importance_n_repeats,
                boundary_threshold=feature_importance_boundary_threshold,
                adaptive_threshold=feature_importance_adaptive_threshold,
                min_boundary_faults=feature_importance_min_boundary_faults,
                random_state=42
            )
        except Exception as e:
            print(f"\n  Warning: Could not compute feature importance: {e}")
            print("  Continuing without feature importance...")
            feature_importance = None
    
    results = {
        'model': svm,
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
        # Check if y_proba contains probabilities (sum to ~1) or decision scores
        # For margin mode, y_proba contains decision function scores, not probabilities
        try:
            # Extract positive class scores/probabilities
            if y_proba.ndim == 2 and y_proba.shape[1] == 2:
                pos_scores = y_proba[:, 1]
            else:
                pos_scores = y_proba
            
            # Check if these are probabilities (should be in [0, 1]) or decision scores
            # Decision scores can be negative or > 1, probabilities are always in [0, 1]
            is_probability = np.all((pos_scores >= 0) & (pos_scores <= 1))
            if y_proba.ndim == 2:
                # For 2D, check if rows sum to ~1 (probabilities) or not (decision scores)
                row_sums = y_proba.sum(axis=1)
                is_probability = is_probability and np.allclose(row_sums, 1.0, atol=0.1)
            
            if is_probability:
                # True probabilities - can compute AUC metrics
                metrics['roc_auc'] = roc_auc_score(y_true, pos_scores)
                metrics['pr_auc'] = average_precision_score(y_true, pos_scores)
            else:
                # Decision scores - cannot compute AUC metrics
                metrics['roc_auc'] = None
                metrics['pr_auc'] = None
                print(f"  Note: AUC metrics not available for margin-based predictions (decision scores, not probabilities)")
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


def select_features_by_importance(
    feature_importance: Dict,
    top_n_margin: int = 10,
    top_m_boundary: int = 10
) -> np.ndarray:
    """
    Select features based on margin and boundary importance rankings.
    
    Selects top N features by margin importance and top M features by boundary importance,
    then returns the union (concatenation) of these feature indices.
    
    Parameters
    ----------
    feature_importance : dict
        Dictionary returned by compute_margin_based_feature_importance.
    top_n_margin : int
        Number of top features to select based on margin importance.
    top_m_boundary : int
        Number of top features to select based on boundary importance.
        
    Returns
    -------
    np.ndarray
        Array of selected feature indices (sorted, with duplicates removed).
    """
    margin_ranked = feature_importance.get('margin_ranked_indices', [])
    boundary_ranked = feature_importance.get('boundary_ranked_indices', [])
    
    if len(margin_ranked) == 0 or len(boundary_ranked) == 0:
        raise ValueError("Feature importance rankings not found. Run compute_margin_based_feature_importance first.")
    
    # Select top N margin features
    top_margin_features = margin_ranked[:top_n_margin]
    
    # Select top M boundary features
    top_boundary_features = boundary_ranked[:top_m_boundary]
    
    # Combine and remove duplicates (preserve order)
    selected_features = np.unique(np.concatenate([top_margin_features, top_boundary_features]))
    
    print(f"\n  Feature Selection:")
    print(f"    Top {top_n_margin} margin features: {top_margin_features.tolist()}")
    print(f"    Top {top_m_boundary} boundary features: {top_boundary_features.tolist()}")
    print(f"    Total selected features: {len(selected_features)} (union of margin + boundary)")
    print(f"    Selected feature indices: {selected_features.tolist()}")
    
    return selected_features


def compute_margin_based_feature_importance(
    model: SVC,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    boundary_threshold: float = 0.1,
    adaptive_threshold: bool = True,
    min_boundary_faults: int = 100,
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict:
    """
    Compute feature importance based on margin (decision function) changes.
    
    For each feature, randomly permute it and measure:
    1. Change in mean absolute margin (|decision_function|)
    2. Change in boundary sample macro F1 score
       (samples near decision boundary, where |margin| < threshold)
    
    Features that cause larger reductions in margin or F1 when permuted are more important.
    
    Parameters
    ----------
    model : SVC
        Trained SVM model.
    X : np.ndarray
        Feature matrix (test set or validation set).
    y : np.ndarray
        True labels.
    n_repeats : int
        Number of times to permute each feature (for stability).
    boundary_threshold : float
        Fixed threshold for identifying boundary samples (absolute margin < threshold).
        Only used if adaptive_threshold is False.
    adaptive_threshold : bool
        If True, uses percentile-based adaptive threshold to ensure sufficient
        minority-class samples in boundary region. If False, uses fixed boundary_threshold.
    min_boundary_faults : int
        Minimum number of fault-class samples required in boundary region when using
        adaptive threshold. Tries percentiles [5, 10, 15, 20, 25] to find suitable threshold.
    random_state : int
        Random seed for reproducibility.
    n_jobs : int
        Number of parallel jobs (-1 for all cores).
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'importance_scores': Array of importance scores (higher = more important)
        - 'margin_change_mean': Mean change in absolute margin per feature
        - 'margin_change_std': Std of change in absolute margin per feature
        - 'boundary_change_mean': Mean change in boundary sample macro F1 per feature
        - 'boundary_change_std': Std of change in boundary sample macro F1 per feature
        - 'ranked_indices': Indices sorted by importance (descending)
        - 'feature_importance': Combined importance scores
    """
    print("\n" + "="*70)
    print("COMPUTING MARGIN-BASED FEATURE IMPORTANCE")
    print("="*70)
    print(f"  Using {n_repeats} permutations per feature...")
    
    np.random.seed(random_state)
    
    # Get baseline margins (decision function)
    baseline_margins = model.decision_function(X)
    if baseline_margins.ndim > 1:
        # For multi-class, use the margin for the true class
        baseline_margins = baseline_margins[np.arange(len(y)), y]
    
    baseline_abs_margin = np.abs(baseline_margins)
    baseline_mean_abs_margin = np.mean(baseline_abs_margin)
    
    # Determine boundary threshold (adaptive or fixed)
    selected_percentile = None  # Track which percentile was used (for adaptive mode)
    
    if adaptive_threshold:
        print(f"  Using adaptive percentile-based threshold (min_boundary_faults={min_boundary_faults})...")
        
        # Try increasing percentiles until we have enough fault samples
        percentiles_to_try = [5, 10, 15, 20, 25, 30]
        selected_threshold = None
        
        for p in percentiles_to_try:
            threshold = np.percentile(baseline_abs_margin, p)
            boundary_mask_candidate = baseline_abs_margin < threshold
            n_faults = np.sum(y[boundary_mask_candidate] == 1)
            
            print(f"    Trying p={p}%: threshold={threshold:.4f}, n_faults={n_faults}")
            
            if n_faults >= min_boundary_faults:
                selected_threshold = threshold
                selected_percentile = p
                break
        
        if selected_threshold is None:
            # If no percentile gives enough faults, use the largest percentile tried
            selected_percentile = percentiles_to_try[-1]
            selected_threshold = np.percentile(baseline_abs_margin, selected_percentile)
            print(f"    Warning: Could not find threshold with {min_boundary_faults} faults.")
            print(f"    Using p={selected_percentile}% (threshold={selected_threshold:.4f}) as fallback.")
        else:
            print(f"    Selected p={selected_percentile}% (threshold={selected_threshold:.4f})")
        
        boundary_threshold = selected_threshold
    else:
        print(f"  Using fixed boundary threshold: {boundary_threshold}")
    
    # Identify boundary samples (samples near decision boundary)
    boundary_mask = baseline_abs_margin < boundary_threshold
    n_boundary_samples = np.sum(boundary_mask)
    
    # Check if we have sufficient samples and class representation in boundary region
    baseline_boundary_f1 = 0.0
    sufficient_boundary_samples = False
    
    if n_boundary_samples > 0:
        baseline_boundary_predictions = model.predict(X[boundary_mask])
        y_boundary = y[boundary_mask]
        
        # Check minority class presence in boundary region
        n_boundary_class_0 = np.sum(y_boundary == 0)
        n_boundary_class_1 = np.sum(y_boundary == 1)
        
        # Require at least 3 samples of each class for meaningful F1 calculation
        if n_boundary_class_0 >= 3 and n_boundary_class_1 >= 3:
            # Use macro F1 for balanced evaluation across both classes
            baseline_boundary_f1 = f1_score(y_boundary, baseline_boundary_predictions, average='macro', zero_division=0)
            sufficient_boundary_samples = True
        else:
            print(f"  Warning: Insufficient class representation in boundary region")
            print(f"    Class 0: {n_boundary_class_0} samples, Class 1 (fault): {n_boundary_class_1} samples")
            print(f"    Boundary importance will be disabled (need ≥3 samples per class)")
    
    print(f"\n  Baseline mean |margin|: {baseline_mean_abs_margin:.4f}")
    print(f"  Boundary threshold used: {boundary_threshold:.4f}")
    print(f"  Number of boundary samples: {n_boundary_samples}")
    if sufficient_boundary_samples:
        print(f"  Baseline boundary macro F1: {baseline_boundary_f1:.4f}")
    else:
        print(f"  Baseline boundary macro F1: N/A (insufficient samples)")
    
    n_features = X.shape[1]
    n_samples = X.shape[0]
    
    # Storage for results
    margin_changes = np.zeros((n_features, n_repeats))
    boundary_changes = np.zeros((n_features, n_repeats))
    
    print(f"\n  Processing {n_features} features...")
    
    # Process each feature
    for feature_idx in range(n_features):
        if (feature_idx + 1) % max(1, n_features // 10) == 0:
            print(f"    Progress: {feature_idx + 1}/{n_features} features...")
        
        # Repeat permutation multiple times for stability
        for repeat in range(n_repeats):
            # Create permuted copy of X
            X_permuted = X.copy()
            
            # Permute feature j
            perm_indices = np.random.permutation(n_samples)
            X_permuted[:, feature_idx] = X[perm_indices, feature_idx]
            
            # Get margins with permuted feature
            permuted_margins = model.decision_function(X_permuted)
            if permuted_margins.ndim > 1:
                permuted_margins = permuted_margins[np.arange(len(y)), y]
            
            permuted_abs_margin = np.abs(permuted_margins)
            permuted_mean_abs_margin = np.mean(permuted_abs_margin)
            
            # Measure change in mean absolute margin
            # Negative change means margin decreased (feature is important)
            margin_change = baseline_mean_abs_margin - permuted_mean_abs_margin
            margin_changes[feature_idx, repeat] = margin_change
            
            # Measure change in boundary sample classification
            if sufficient_boundary_samples:
                permuted_boundary_predictions = model.predict(X_permuted[boundary_mask])
                y_boundary = y[boundary_mask]
                
                # Calculate macro F1 for permuted predictions
                permuted_boundary_f1 = f1_score(y_boundary, permuted_boundary_predictions, average='macro', zero_division=0)
                
                # Change in F1 (positive change means F1 decreased when feature is permuted → feature is important)
                boundary_change = baseline_boundary_f1 - permuted_boundary_f1
                boundary_changes[feature_idx, repeat] = boundary_change
            else:
                # If insufficient boundary samples, boundary importance is not computed
                boundary_changes[feature_idx, repeat] = 0.0
    
    # Compute statistics across repeats
    margin_change_mean = np.mean(margin_changes, axis=1)
    margin_change_std = np.std(margin_changes, axis=1)
    boundary_change_mean = np.mean(boundary_changes, axis=1)
    boundary_change_std = np.std(boundary_changes, axis=1)
    
    # Normalize margin changes to [0, 1] range for combination
    # Features that reduce margin more are more important
    if np.max(margin_change_mean) > 0:
        margin_importance = (margin_change_mean - np.min(margin_change_mean)) / (np.max(margin_change_mean) - np.min(margin_change_mean) + 1e-10)
    else:
        margin_importance = np.zeros(n_features)
    
    # Normalize boundary changes to [0, 1] range
    # Features that reduce boundary F1 more are more important
    if np.max(boundary_change_mean) > 0:
        boundary_importance = (boundary_change_mean - np.min(boundary_change_mean)) / (np.max(boundary_change_mean) - np.min(boundary_change_mean) + 1e-10)
    else:
        boundary_importance = np.zeros(n_features)
    
    # Combine both measures (equal weight)
    # Higher score = more important (larger reduction in margin/boundary F1)
    #combined_importance = 0.5 * margin_importance + 0.5 * boundary_importance
    #combined_importance = 1 * margin_importance
    combined_importance = 1 * boundary_importance

    # Rank features by importance (descending)
    ranked_indices = np.argsort(combined_importance)[::-1]
    
    # Separate rankings for margin and boundary importance
    margin_ranked_indices = np.argsort(margin_change_mean)[::-1]  # Descending by margin change
    boundary_ranked_indices = np.argsort(boundary_change_mean)[::-1]  # Descending by boundary change
    
    print(f"\n  Feature importance computation complete!")
    print(f"  Top 5 most important features (by margin change):")
    top_5_margin = margin_ranked_indices[:5]
    for i, idx in enumerate(top_5_margin, 1):
        print(f"    {i}. Feature {idx}: margin_change={margin_change_mean[idx]:.6f} ± {margin_change_std[idx]:.6f}")
    
    if sufficient_boundary_samples:
        print(f"  Top 5 most important features (by boundary F1 change):")
        top_5_boundary = boundary_ranked_indices[:5]
        for i, idx in enumerate(top_5_boundary, 1):
            print(f"    {i}. Feature {idx}: boundary_f1_change={boundary_change_mean[idx]:.6f} ± {boundary_change_std[idx]:.6f}")
    else:
        print(f"  Boundary importance: DISABLED (insufficient minority-class samples in boundary region)")
    
    return {
        'importance_scores': combined_importance,
        'margin_change_mean': margin_change_mean,
        'margin_change_std': margin_change_std,
        'boundary_change_mean': boundary_change_mean,
        'boundary_change_std': boundary_change_std,
        'ranked_indices': ranked_indices,
        'margin_ranked_indices': margin_ranked_indices,  # Separate ranking by margin
        'boundary_ranked_indices': boundary_ranked_indices,  # Separate ranking by boundary
        'margin_importance': margin_importance,  # Normalized margin importance scores
        'boundary_importance': boundary_importance,  # Normalized boundary importance scores
        'feature_importance': combined_importance,
        'baseline_mean_abs_margin': baseline_mean_abs_margin,
        'baseline_boundary_f1': baseline_boundary_f1,  # Baseline macro F1 on boundary samples
        'n_boundary_samples': n_boundary_samples,
        'sufficient_boundary_samples': sufficient_boundary_samples,  # Whether boundary importance is valid
        'boundary_threshold': boundary_threshold,  # Threshold used (adaptive or fixed)
        'boundary_threshold_percentile': selected_percentile,  # Percentile used (None if fixed threshold)
        'adaptive_threshold_used': adaptive_threshold  # Whether adaptive threshold was enabled
    }

