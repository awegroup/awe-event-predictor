from __future__ import annotations
import numpy as np
from typing import Dict, Optional, Literal, List
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

from functions.ml.train_test_split import time_based_train_test_split
from functions.ml.class_balancing import balance_classes
from functions.ml.svm_training import (
    validate_and_clean_features,
    scale_features,
    tune_svm_hyperparameters,
    train_svm,
    compute_margin_based_feature_importance,
    select_features_by_importance
)
from functions.ml.logistic_regression_training import (
    tune_logistic_regression_hyperparameters,
    train_logistic_regression
)
from functions.ml.threshold_tuning import tune_threshold as tune_decision_threshold, apply_threshold


def compute_per_dataset_metrics(
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    dataset_indices: np.ndarray,
    dataset_names: List[str]
) -> Dict[str, Dict]:
    """
    Compute metrics for each dataset independently.
    
    Parameters
    ----------
    y_test : np.ndarray
        True labels for all test samples
    y_pred : np.ndarray
        Predicted labels for all test samples
    y_proba : Optional[np.ndarray]
        Predicted probabilities for all test samples (can be None for margin-based SVM)
    dataset_indices : np.ndarray
        Array indicating which dataset each test sample belongs to
    dataset_names : List[str]
        List of dataset names corresponding to indices
    
    Returns
    -------
    Dict[str, Dict]
        Dictionary mapping dataset names to their metrics
    """
    from functions.ml.svm_training import compute_metrics
    
    per_dataset_metrics = {}
    unique_indices = np.unique(dataset_indices)
    
    for dataset_idx in unique_indices:
        # Get mask for current dataset
        mask = dataset_indices == dataset_idx
        
        # Get dataset name
        if dataset_idx < len(dataset_names):
            dataset_name = dataset_names[dataset_idx]
        else:
            dataset_name = f"Dataset_{dataset_idx}"
        
        # Extract data for this dataset
        y_test_dataset = y_test[mask]
        y_pred_dataset = y_pred[mask]
        y_proba_dataset = y_proba[mask] if y_proba is not None else None
        
        # Compute metrics
        metrics = compute_metrics(y_test_dataset, y_pred_dataset, y_proba_dataset)
        metrics['n_samples'] = len(y_test_dataset)
        metrics['n_class_0'] = np.sum(y_test_dataset == 0)
        metrics['n_class_1'] = np.sum(y_test_dataset == 1)
        
        per_dataset_metrics[dataset_name] = metrics
    
    return per_dataset_metrics


def remove_low_variance_features(X_train, X_test, eps=1e-8):
    """
    Remove features with near-zero variance (train-only decision).

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    eps : float
        Threshold for standard deviation below which features are considered low variance

    Returns
    -------
    X_train_clean : np.ndarray
        Training features with low variance features removed
    X_test_clean : np.ndarray
        Test features with same features removed
    keep_mask : np.ndarray
        Boolean mask indicating which features were kept
    """
    stds = X_train.std(axis=0)
    keep_mask = stds > eps
    removed = np.sum(~keep_mask)
    if removed > 0:
        print(f"  Removed {removed} near-constant features (std < {eps})")
    return X_train[:, keep_mask], X_test[:, keep_mask], keep_mask


def remove_correlated_features_fun(X_train, X_test, threshold=0.95):
    """
    Remove highly correlated features (train-only decision).

    Parameters
    ----------
    X_train : np.ndarray
        Training feature matrix
    X_test : np.ndarray
        Test feature matrix
    threshold : float
        Correlation threshold above which features are considered highly correlated

    Returns
    -------
    X_train_clean : np.ndarray
        Training features with correlated features removed
    X_test_clean : np.ndarray
        Test features with same features removed
    keep_idx : list
        List of indices of features that were kept
    """
    corr = np.corrcoef(X_train, rowvar=False)
    upper = np.triu(np.abs(corr), k=1)

    to_remove = set()
    for i in range(upper.shape[0]):
        for j in range(i + 1, upper.shape[1]):
            if upper[i, j] > threshold:
                to_remove.add(j)

    keep_idx = [i for i in range(X_train.shape[1]) if i not in to_remove]

    removed = len(to_remove)
    if removed > 0:
        print(f"  Removed {removed} highly correlated features (correlation > {threshold})")

    return X_train[:, keep_idx], X_test[:, keep_idx], keep_idx


def run_training_pipeline(
    X: np.ndarray,
    y: np.ndarray,
    frame_times: Optional[List[np.ndarray]] = None,
    X_test: Optional[np.ndarray] = None,
    y_test: Optional[np.ndarray] = None,
    X_test_in_domain: Optional[np.ndarray] = None,
    y_test_in_domain: Optional[np.ndarray] = None,
    test_dataset_indices: Optional[np.ndarray] = None,
    test_dataset_names: Optional[List[str]] = None,
    in_domain_dataset_indices: Optional[np.ndarray] = None,
    in_domain_dataset_names: Optional[List[str]] = None,
    test_size: float = 0.2,
    model_type: Literal["svm", "logistic_regression"] = "svm",
    scaling_method: Literal["standard", "minmax"] = "standard",
    svm_mode: Literal["probability", "margin"] = "probability",
    balancing_method: Literal["smote", "downsample", "upsample", "nearmiss", "nearmiss1", "nearmiss2", "nearmiss3", "tomek_links", "tomek_links_then_nearmiss", "none"] = "smote",
    balance_validation: bool = False,
    downsample_ratio: float = 1.0,
    nearmiss_version: int = 1,
    nearmiss_n_neighbors: int = 3,
    tune_hyperparameters: bool = True,
    hyperparameter_scoring: Literal["macro_f1", "balanced_accuracy", "f1", "f1_class1", "pr_auc"] = "macro_f1",
    cv_folds: int = 5,
    n_iter: int = 50,
    random_state: int = 42,
    tune_threshold: bool = True,
    threshold_tuning_method: Literal["max_f1", "max_pr_auc", "balanced_precision_recall"] = "max_f1",
    validation_size: float = 0.2,
    compute_feature_importance: bool = True,
    feature_importance_n_repeats: int = 10,
    feature_importance_boundary_threshold: float = 0.1,
    feature_importance_adaptive_threshold: bool = True,
    feature_importance_min_boundary_faults: int = 100,
    use_feature_selection: bool = False,
    top_n_margin_features: int = 10,
    top_m_boundary_features: int = 10,
    remove_correlated_features: bool = True,
    correlation_threshold: float = 0.95,
    preselected_feature_indices: Optional[np.ndarray] = None
) -> Dict:
    """
    Generic training pipeline that supports multiple model types.
    Correctly handles external train/test splits and ensures both sets are cleaned before scaling.
    
    Parameters
    ----------
    X : np.ndarray
        Training features.
    y : np.ndarray
        Training labels.
    frame_times : Optional[List[np.ndarray]]
        Frame times for time-based splitting (if not using external splits).
    X_test : Optional[np.ndarray]
        External test features (out-of-domain).
    y_test : Optional[np.ndarray]
        External test labels (out-of-domain).
    X_test_in_domain : Optional[np.ndarray]
        In-domain test features (held-out from training data, no balancing applied).
    y_test_in_domain : Optional[np.ndarray]
        In-domain test labels.
    test_dataset_indices : Optional[np.ndarray]
        Array indicating which dataset each out-of-domain test sample belongs to (for per-dataset metrics).
    test_dataset_names : Optional[List[str]]
        List of dataset names corresponding to indices in test_dataset_indices.
    in_domain_dataset_indices : Optional[np.ndarray]
        Array indicating which dataset each in-domain test sample belongs to (for per-dataset metrics).
    in_domain_dataset_names : Optional[List[str]]
        List of dataset names corresponding to indices in in_domain_dataset_indices.
    test_size : float
        Fraction of data to use for test set (only used if X_test is None).
    model_type : str
        Type of model to train: 'svm' or 'logistic_regression'
    svm_mode : str
        SVM training mode: 'probability' (use predict_proba) or 'margin' (use decision_function)
        Only used when model_type='svm'. When 'margin', threshold tuning is disabled.
    balancing_method : str
        Method for class balancing: 'smote', 'downsample', 'upsample', 'nearmiss', etc.
    balance_validation : bool
        If True, applies the same balancing to validation set as training set.
        If False (recommended), validation set keeps natural class distribution.
        Note: Keeping natural distribution is better for threshold tuning and realistic PR-AUC.
    compute_feature_importance : bool
        Whether to compute margin-based feature importance (SVM only).
    feature_importance_n_repeats : int
        Number of permutations per feature for importance computation.
    feature_importance_boundary_threshold : float
        Fixed threshold for identifying boundary samples (only used if adaptive_threshold=False).
    feature_importance_adaptive_threshold : bool
        If True, uses percentile-based adaptive threshold for boundary region.
    feature_importance_min_boundary_faults : int
        Minimum number of fault-class samples required in boundary region (adaptive mode).
    use_feature_selection : bool
        Whether to perform feature selection based on importance (requires compute_feature_importance=True).
        If True, will compute importance, select top features, then retrain with selected features.
    top_n_margin_features : int
        Number of top features to select based on margin importance (only if use_feature_selection=True).
    top_m_boundary_features : int
        Number of top features to select based on boundary importance (only if use_feature_selection=True).
    remove_correlated_features : bool
        Whether to remove highly correlated features (correlation > threshold).
    correlation_threshold : float
        Correlation threshold above which features are considered highly correlated (only if remove_correlated_features=True).
    preselected_feature_indices : Optional[np.ndarray]
        Pre-selected feature indices to use (e.g., from a previously trained model).
        If provided, bypasses feature importance computation and uses these indices directly.
        This allows reusing feature selection from one model type (e.g., SVM) on another (e.g., Logistic Regression).
    """

    print("\n" + "="*70)
    print(f"{model_type.upper().replace('_', ' ')} TRAINING PIPELINE")
    print("="*70)

    results = {}

    # ============================================================
    # Step 1: Validate FULL dataset
    # ============================================================
    print("\n[Step 1/9] Validating and cleaning FULL dataset...")
    X_clean = validate_and_clean_features(X)
    results["X_clean_full"] = X_clean

    # ============================================================
    # Step 2: Train/Test Split
    # ============================================================

    if X_test is None:
        # ----------------------
        # INTERNAL TIME-BASED SPLIT
        # ----------------------
        print("\n[Step 2/9] Performing time-based train-test split...")

        X_train, X_test, y_train, y_test, train_indices, test_indices = time_based_train_test_split(
            X_clean, y, frame_times,
            test_size=test_size,
            stratify=True,
            min_test_samples=5
        )

        results["train_indices"] = train_indices
        results["test_indices"] = test_indices

        # Clean both splits
        print("\n[Step 2a] Cleaning training set...")
        X_train_clean = validate_and_clean_features(X_train)

        print("\n[Step 2b] Cleaning test set...")
        X_test_clean = validate_and_clean_features(X_test)

    else:
        # ----------------------
        # EXTERNAL SPLIT PROVIDED
        # ----------------------
        print("\n[Step 2/9] Using EXTERNAL train-test split...")

        # For external split:
        # X = full training set
        # y = full training labels
        # X_test, y_test = out-of-domain test set provided externally
        # X_test_in_domain, y_test_in_domain = optional in-domain test set

        X_train = X
        y_train = y                 # <-- FIXED
        y_test = y_test             # <-- remains the same

        print("\n[Step 2a] Cleaning training set (external)...")
        X_train_clean = validate_and_clean_features(X_train)

        print("\n[Step 2b] Cleaning out-of-domain test set (external)...")
        X_test_clean = validate_and_clean_features(X_test)
        
        # Process in-domain test set if provided
        if X_test_in_domain is not None and y_test_in_domain is not None:
            print("\n[Step 2c-in-domain] Cleaning in-domain test set...")
            X_test_in_domain_clean = validate_and_clean_features(X_test_in_domain)
        else:
            X_test_in_domain_clean = None

    # ============================================================
    # Step 2d: Remove low variance features
    # ============================================================
    print("\n[Step 2d/9] Removing low variance features...")
    X_train_clean, X_test_clean, variance_mask = remove_low_variance_features(
        X_train_clean, X_test_clean, eps=1e-8
    )
    # Apply same transformation to in-domain test if present
    if X_test_in_domain_clean is not None:
        X_test_in_domain_clean = X_test_in_domain_clean[:, variance_mask]
    results["variance_mask"] = variance_mask

    # ============================================================
    # Step 2e: Remove highly correlated features
    # ============================================================
    if remove_correlated_features:
        print(f"\n[Step 2e/9] Removing highly correlated features (threshold: {correlation_threshold})...")
        X_train_clean, X_test_clean, correlation_keep_idx = remove_correlated_features_fun(
            X_train_clean, X_test_clean, threshold=correlation_threshold
        )
        # Apply same transformation to in-domain test if present
        if X_test_in_domain_clean is not None:
            X_test_in_domain_clean = X_test_in_domain_clean[:, correlation_keep_idx]
        results["correlation_keep_idx"] = correlation_keep_idx
    else:
        print("\n[Step 2e/9] Skipping removal of correlated features...")
        # Keep all features (create identity mapping)
        correlation_keep_idx = list(range(X_train_clean.shape[1]))
        results["correlation_keep_idx"] = correlation_keep_idx

    # ============================================================
    # Step 3: Scaling
    # ============================================================
    print("\n[Step 3/9] Scaling features...")

    X_train_scaled, X_test_scaled, scaler = scale_features(
        X_train_clean,
        X_test_clean,
        method=scaling_method
    )
    results["scaler"] = scaler
    
    # Scale in-domain test set if present (using same scaler)
    if X_test_in_domain_clean is not None:
        print("  Scaling in-domain test set...")
        X_test_in_domain_scaled = scaler.transform(X_test_in_domain_clean)
        print(f"  In-domain test mean={np.nanmean(X_test_in_domain_scaled):.4f}, std={np.nanstd(X_test_in_domain_scaled):.4f}")
    else:
        X_test_in_domain_scaled = None

    # Diagnostic print
    print(f"  Train mean={np.nanmean(X_train_scaled):.4f}, std={np.nanstd(X_train_scaled):.4f}")
    print(f"  Out-of-domain test mean={np.nanmean(X_test_scaled):.4f}, std={np.nanstd(X_test_scaled):.4f}")

    # ============================================================
    # Step 4: Split into Train/Validation (BEFORE balancing)
    # ============================================================
    # Determine if we need validation split
    need_validation_split = use_feature_selection or tune_threshold
    
    X_train_unbalanced = X_train_scaled
    y_train_unbalanced = y_train
    X_val = None
    y_val = None
    
    if need_validation_split:
        print(f"\n[Step 4/9] Splitting training data into train/validation (BEFORE balancing)...")
        print(f"  Validation set will be used for: ", end="")
        purposes = []
        if use_feature_selection:
            purposes.append("feature importance computation")
        if tune_threshold:
            purposes.append("threshold tuning")
        print(" and ".join(purposes))
        
        if balance_validation:
            print(f"  Note: Validation set will be BALANCED (same as training set)")
        else:
            print(f"  Note: Validation set will maintain NATURAL class distribution (no balancing)")
        
        # Split BEFORE balancing
        from sklearn.model_selection import train_test_split
        X_train_unbalanced, X_val, y_train_unbalanced, y_val = train_test_split(
            X_train_scaled,
            y_train,
            test_size=validation_size,
            random_state=random_state,
            stratify=y_train if len(np.unique(y_train)) > 1 else None
        )
        
        print(f"  Training set (before balancing): {len(X_train_unbalanced)} samples")
        print(f"    Class 0: {np.sum(y_train_unbalanced == 0)}, Class 1: {np.sum(y_train_unbalanced == 1)}")
        print(f"  Validation set (before balancing): {len(X_val)} samples")
        print(f"    Class 0: {np.sum(y_val == 0)}, Class 1: {np.sum(y_val == 1)}")
    else:
        print(f"\n[Step 4/9] Skipping validation split (not needed)")

    # ============================================================
    # Step 4a: Class Balancing
    # ============================================================
    if balance_validation and X_val is not None:
        print("\n[Step 4a/9] Balancing classes (training AND validation sets)...")
    else:
        print("\n[Step 4a/9] Balancing classes (training set only)...")

    X_train_balanced, y_train_balanced = balance_classes(
        X_train_unbalanced,
        y_train_unbalanced,
        method=balancing_method,
        random_state=random_state,
        downsample_ratio=downsample_ratio,
        nearmiss_version=nearmiss_version,
        nearmiss_n_neighbors=nearmiss_n_neighbors
    )
    
    print(f"  Balanced training set: {len(X_train_balanced)} samples")
    print(f"    Class 0: {np.sum(y_train_balanced == 0)}, Class 1: {np.sum(y_train_balanced == 1)}")
    
    # Balance validation set if requested
    if balance_validation and X_val is not None:
        print(f"\n  Balancing validation set...")
        X_val, y_val = balance_classes(
            X_val,
            y_val,
            method=balancing_method,
            random_state=random_state + 1,  # Different seed for validation
            downsample_ratio=downsample_ratio,
            nearmiss_version=nearmiss_version,
            nearmiss_n_neighbors=nearmiss_n_neighbors
        )
        print(f"  Balanced validation set: {len(X_val)} samples")
        print(f"    Class 0: {np.sum(y_val == 0)}, Class 1: {np.sum(y_val == 1)}")
    
    # X_train_final will be used for actual training (after potential feature selection)
    X_train_final = X_train_balanced
    y_train_final = y_train_balanced

    # ============================================================
    # Step 4.5: Feature Selection (if enabled) - Part 1: Initial Hyperparameter Tuning
    # ============================================================
    selected_feature_indices = None
    feature_importance_results = None
    initial_tuning_results = None
    
    # Handle feature selection: either use pre-selected features or compute new ones
    if preselected_feature_indices is not None:
        # Use pre-selected features from a previous model
        print("\n[Step 4.5/9] Using PRE-SELECTED features from previous model...")
        selected_feature_indices = np.array(preselected_feature_indices, dtype=int)
        
        # Validate feature indices
        if len(selected_feature_indices) == 0:
            raise ValueError("Pre-selected feature indices array is empty")
        if np.max(selected_feature_indices) >= X_train_final.shape[1]:
            raise ValueError(f"Pre-selected feature indices contain invalid index {np.max(selected_feature_indices)} "
                           f"(max allowed: {X_train_final.shape[1] - 1})")
        
        # Apply feature selection to training, validation, and test sets
        print(f"  Applying feature selection: {X_train_final.shape[1]} -> {len(selected_feature_indices)} features")
        X_train_final = X_train_final[:, selected_feature_indices]
        X_val = X_val[:, selected_feature_indices]
        X_test_scaled = X_test_scaled[:, selected_feature_indices]
        if X_test_in_domain_scaled is not None:
            X_test_in_domain_scaled = X_test_in_domain_scaled[:, selected_feature_indices]
        
        results["selected_feature_indices"] = selected_feature_indices
        results["n_selected_features"] = len(selected_feature_indices)
        results["feature_selection_source"] = "preselected"
        
        print(f"  ✓ Pre-selected feature selection complete: using {len(selected_feature_indices)} features")
        
    elif use_feature_selection:
        if not compute_feature_importance:
            raise ValueError("use_feature_selection requires compute_feature_importance=True")
        if model_type != "svm":
            raise ValueError("Feature selection based on margin importance is only supported for SVM models")
        
        print("\n[Step 4.5a/9] Feature Selection - Tuning hyperparameters on ALL features...")
        
        # First, tune hyperparameters on all features using the training set
        if tune_hyperparameters:
            print(f"  Tuning SVM hyperparameters on all {X_train_final.shape[1]} features...")
            initial_tuning_results = tune_svm_hyperparameters(
                X_train_final,
                y_train_final,
                scoring=hyperparameter_scoring,
                cv=cv_folds,
                n_iter=n_iter,
                random_state=random_state,
                probability=(svm_mode == "probability")
            )
            initial_hyperparams = initial_tuning_results["best_params"]
        else:
            # Use reasonable defaults for feature selection
            print("  Using default hyperparameters for feature importance computation...")
            initial_hyperparams = {
                'C': 1.0,
                'gamma': 'scale',
                'kernel': 'rbf',
                'class_weight': 'balanced',
                'probability': (svm_mode == "probability"),
                'random_state': random_state
            }
        
        print("\n[Step 4.5b/9] Feature Selection - Computing feature importance with tuned model...")
        print(f"  Computing feature importance to select top {top_n_margin_features} margin + top {top_m_boundary_features} boundary features...")
        
        # Train model on training set with tuned hyperparameters
        from sklearn.svm import SVC
        initial_model = SVC(**initial_hyperparams)
        initial_model.fit(X_train_final, y_train_final)
        
        # Compute feature importance on VALIDATION set (unseen data)
        print(f"  Computing feature importance on validation set ({len(X_val)} samples)...")
        X_importance = X_val
        y_importance = y_val
        
        # Compute feature importance
        print("  Computing feature importance...")
        feature_importance_results = compute_margin_based_feature_importance(
            model=initial_model,
            X=X_importance,
            y=y_importance,
            n_repeats=feature_importance_n_repeats,
            boundary_threshold=feature_importance_boundary_threshold,
            adaptive_threshold=feature_importance_adaptive_threshold,
            min_boundary_faults=feature_importance_min_boundary_faults,
            random_state=random_state
        )
        
        # Select features
        selected_feature_indices = select_features_by_importance(
            feature_importance_results,
            top_n_margin=top_n_margin_features,
            top_m_boundary=top_m_boundary_features
        )
        
        # Apply feature selection to training, validation, and test sets
        print(f"\n[Step 4.5c/9] Applying feature selection: {X_train_final.shape[1]} -> {len(selected_feature_indices)} features")
        X_train_final = X_train_final[:, selected_feature_indices]
        X_val = X_val[:, selected_feature_indices]
        X_test_scaled = X_test_scaled[:, selected_feature_indices]
        if X_test_in_domain_scaled is not None:
            X_test_in_domain_scaled = X_test_in_domain_scaled[:, selected_feature_indices]
        
        results["selected_feature_indices"] = selected_feature_indices
        results["feature_importance_selection"] = feature_importance_results
        results["n_selected_features"] = len(selected_feature_indices)
        results["initial_tuning_results_all_features"] = initial_tuning_results
        results["feature_selection_source"] = "computed"
        
        print(f"  ✓ Feature selection complete: using {len(selected_feature_indices)} features")
    
    # ============================================================
    # Step 5: Hyperparameter tuning (or re-tuning after feature selection)
    # ============================================================
    # Validate svm_mode
    if model_type == "svm" and svm_mode not in ["probability", "margin"]:
        raise ValueError(f"Invalid svm_mode: {svm_mode}. Must be 'probability' or 'margin'")
    
    # Auto-disable threshold tuning for margin mode
    if model_type == "svm" and svm_mode == "margin" and tune_threshold:
        print(f"\n⚠️  Threshold tuning disabled for margin-based SVM (no probabilities available)")
        tune_threshold = False
    
    hyperparams = None
    if tune_hyperparameters:
        # If feature selection was used, re-tune on selected features
        if use_feature_selection and selected_feature_indices is not None:
            print(f"\n[Step 5/9] Re-tuning {model_type} hyperparameters on SELECTED features ({len(selected_feature_indices)} features)...")
            print(f"  Note: Initial tuning was done on all features, now re-tuning on selected subset")
        else:
            print(f"\n[Step 5/9] Tuning {model_type} hyperparameters...")

        if model_type == "svm":
            tuning_results = tune_svm_hyperparameters(
                X_train_final,
                y_train_final,
                scoring=hyperparameter_scoring,
                cv=cv_folds,
                n_iter=n_iter,
                random_state=random_state,
                probability=(svm_mode == "probability")
            )
        elif model_type == "logistic_regression":
            tuning_results = tune_logistic_regression_hyperparameters(
                X_train_final,
                y_train_final,
                scoring=hyperparameter_scoring,
                cv=cv_folds,
                n_iter=n_iter,
                random_state=random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        hyperparams = tuning_results["best_params"]
        results["tuning_results"] = tuning_results
        
        # If feature selection was used, store both tuning results
        if use_feature_selection and initial_tuning_results is not None:
            results["tuning_results_selected_features"] = tuning_results
            results["tuning_results_all_features"] = initial_tuning_results
    else:
        print(f"\n[Step 5/9] Skipping hyperparameter tuning (using defaults for {model_type})...")
        if model_type == "svm":
            hyperparams = {
                "C": 1.0,
                "gamma": "scale",
                "kernel": "rbf",
                "class_weight": "balanced"
            }
            if svm_mode == "margin":
                print(f"  Using margin-based SVM (probability=False)")
            else:
                print(f"  Using probability-based SVM (probability=True)")
        elif model_type == "logistic_regression":
            hyperparams = {
                "C": 1.0,
                "penalty": "l2",
                "solver": "lbfgs",
                "class_weight": "balanced",
                "max_iter": 1000
            }
        else:
            raise ValueError(f"Unknown model_type: {model_type}")

    # ============================================================
    # Step 6: Threshold tuning preparation (validation set already created in Step 4a)
    # ============================================================
    optimal_threshold = 0.5  # Default threshold
    threshold_tuning_results = None
    
    if tune_threshold:
        print(f"\n[Step 6/9] Preparing for threshold tuning (using validation set from Step 4a)...")
        print(f"  Validation set: {len(X_val)} samples")
        
        # ============================================================
        # Step 7: Train final model on training set
        # ============================================================
        print(f"\n[Step 7/9] Training final {model_type.upper().replace('_', ' ')} classifier...")
        
        if model_type == "svm":
            use_probability = (svm_mode == "probability")
            # If we already computed feature importance for selection, don't recompute
            compute_final_importance = compute_feature_importance and not use_feature_selection
            final_results = train_svm(
                X_train_final, y_train_final,
                X_test_scaled, y_test,
                hyperparameters=hyperparams,
                probability=use_probability,
                compute_feature_importance=compute_final_importance,
                feature_importance_n_repeats=feature_importance_n_repeats,
                feature_importance_boundary_threshold=feature_importance_boundary_threshold,
                feature_importance_adaptive_threshold=feature_importance_adaptive_threshold,
                feature_importance_min_boundary_faults=feature_importance_min_boundary_faults
            )
            
            # If we used feature selection, add the precomputed importance to results
            if use_feature_selection and feature_importance_results is not None:
                final_results["feature_importance"] = feature_importance_results
        elif model_type == "logistic_regression":
            final_results = train_logistic_regression(
                X_train_final, y_train_final,
                X_test_scaled, y_test,
                hyperparameters=hyperparams
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        # Get probabilities on validation set for threshold tuning
        # Note: This code should only run when tune_threshold=True and svm_mode="probability"
        # (margin mode disables threshold tuning earlier)
        model = final_results["model"]
        if model_type == "svm":
            if svm_mode == "probability":
                y_val_proba = model.predict_proba(X_val)
            else:
                # This should never happen - margin mode disables threshold tuning
                raise ValueError(
                    "Threshold tuning requires probabilities, but svm_mode='margin'. "
                    "This error should not occur - threshold tuning should be disabled for margin mode."
                )
        else:  # logistic_regression
            y_val_proba = model.predict_proba(X_val)
        
        # Tune threshold on validation set
        print(f"\n[Step 8/9] Tuning decision threshold on validation set (method: {threshold_tuning_method})...")
        threshold_tuning_results = tune_decision_threshold(
            y_val,
            y_val_proba,
            method=threshold_tuning_method
        )
        optimal_threshold = threshold_tuning_results["optimal_threshold"]
        
        print(f"  Optimal threshold: {optimal_threshold:.4f}")
        print(f"  F1 at threshold: {threshold_tuning_results['f1_at_threshold']:.4f}")
        print(f"  Precision at threshold: {threshold_tuning_results['precision_at_threshold']:.4f}")
        print(f"  Recall at threshold: {threshold_tuning_results['recall_at_threshold']:.4f}")
        
        # Plot PR-AUC curve on validation set
        num_faulty = np.sum(y_val == 1)
        num_non_faulty = np.sum(y_val == 0)
        pr_auc = threshold_tuning_results['pr_auc']
        precision = threshold_tuning_results['all_precision']
        recall = threshold_tuning_results['all_recall']
        
        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, linewidth=2, label=f'PR-AUC = {pr_auc:.4f}')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title(f'Precision-Recall Curve (Validation Set)\n'
                  f'Faulty samples: {num_faulty}, Non-faulty samples: {num_non_faulty}',
                  fontsize=13)
        plt.grid(True, alpha=0.3)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        
        # Mark the optimal threshold point
        optimal_precision = threshold_tuning_results['precision_at_threshold']
        optimal_recall = threshold_tuning_results['recall_at_threshold']
        plt.plot(optimal_recall, optimal_precision, 'ro', markersize=10, 
                label=f'Optimal threshold = {optimal_threshold:.4f}')
        plt.legend(fontsize=11)
        
        plt.tight_layout()
        plt.show()
        print(f"  PR-AUC: {pr_auc:.4f}")
        print(f"  Validation set: {num_faulty} faulty, {num_non_faulty} non-faulty samples")
        
        # Apply optimal threshold to test set predictions
        print(f"\n[Step 9/9] Applying optimal threshold ({optimal_threshold:.4f}) to test set...")
        y_test_proba = final_results["y_proba"]
        y_pred_optimal = apply_threshold(y_test_proba, optimal_threshold)
        
        # Recompute metrics with optimal threshold
        from functions.ml.svm_training import compute_metrics
        metrics_optimal = compute_metrics(y_test, y_pred_optimal, y_test_proba)
        
        print("\nTest Set Performance (with optimal threshold):")
        from functions.ml.svm_training import print_metrics
        print_metrics(metrics_optimal)
        
        # Update results with optimal threshold predictions and metrics
        final_results["y_pred"] = y_pred_optimal
        final_results["metrics"] = metrics_optimal
        final_results["threshold"] = optimal_threshold
        final_results["threshold_tuning_results"] = threshold_tuning_results
        
        # Compute per-dataset metrics if dataset indices provided
        if test_dataset_indices is not None and test_dataset_names is not None:
            print(f"\n  Computing per-dataset metrics for {len(np.unique(test_dataset_indices))} datasets...")
            per_dataset_metrics = compute_per_dataset_metrics(
                y_test=y_test,
                y_pred=y_pred_optimal,
                y_proba=y_test_proba,
                dataset_indices=test_dataset_indices,
                dataset_names=test_dataset_names
            )
            final_results["per_dataset_metrics"] = per_dataset_metrics
            print(f"  ✓ Per-dataset metrics computed")
        
    else:
        # No threshold tuning - use default 0.5
        print(f"\n[Step 6/9] Skipping threshold tuning (using default threshold=0.5)...")
        print(f"\n[Step 7/9] Training final {model_type.upper().replace('_', ' ')} classifier...")
        
        if model_type == "svm":
            use_probability = (svm_mode == "probability")
            # If we already computed feature importance for selection, don't recompute
            compute_final_importance = compute_feature_importance and not use_feature_selection
            final_results = train_svm(
                X_train_final, y_train_final,
                X_test_scaled, y_test,
                hyperparameters=hyperparams,
                probability=use_probability,
                compute_feature_importance=compute_final_importance,
                feature_importance_n_repeats=feature_importance_n_repeats,
                feature_importance_boundary_threshold=feature_importance_boundary_threshold,
                feature_importance_adaptive_threshold=feature_importance_adaptive_threshold,
                feature_importance_min_boundary_faults=feature_importance_min_boundary_faults
            )
            
            # If we used feature selection, add the precomputed importance to results
            if use_feature_selection and feature_importance_results is not None:
                final_results["feature_importance"] = feature_importance_results
        elif model_type == "logistic_regression":
            final_results = train_logistic_regression(
                X_train_final, y_train_final,
                X_test_scaled, y_test,
                hyperparameters=hyperparams
            )
        else:
            raise ValueError(f"Unknown model_type: {model_type}")
        
        final_results["threshold"] = 0.5  # Default threshold
        
        # Compute per-dataset metrics if dataset indices provided
        if test_dataset_indices is not None and test_dataset_names is not None:
            print(f"\n  Computing per-dataset metrics for {len(np.unique(test_dataset_indices))} datasets...")
            per_dataset_metrics = compute_per_dataset_metrics(
                y_test=y_test,
                y_pred=final_results["y_pred"],
                y_proba=final_results.get("y_proba"),
                dataset_indices=test_dataset_indices,
                dataset_names=test_dataset_names
            )
            final_results["per_dataset_metrics"] = per_dataset_metrics
            print(f"  ✓ Per-dataset metrics computed")

    # ============================================================
    # Step 10: Evaluate on in-domain test set (if provided)
    # ============================================================
    if X_test_in_domain_scaled is not None and y_test_in_domain is not None:
        print("\n" + "="*70)
        print("IN-DOMAIN TEST SET EVALUATION")
        print("="*70)
        print(f"Evaluating on in-domain test set ({len(X_test_in_domain_scaled)} samples)...")
        
        # Get model and make predictions
        model = final_results["model"]
        y_pred_in_domain = model.predict(X_test_in_domain_scaled)
        
        # Get probabilities or decision scores
        y_proba_in_domain = None
        if model_type == "svm":
            if svm_mode == "probability":
                y_proba_in_domain = model.predict_proba(X_test_in_domain_scaled)
            else:
                # Margin mode: use decision function
                decision_scores = model.decision_function(X_test_in_domain_scaled)
                if decision_scores.ndim == 1:
                    y_proba_in_domain = np.column_stack([-decision_scores, decision_scores])
                else:
                    y_proba_in_domain = decision_scores
        elif model_type == "logistic_regression":
            y_proba_in_domain = model.predict_proba(X_test_in_domain_scaled)
        
        # Apply optimal threshold if available (for probability-based models)
        if "threshold" in final_results and y_proba_in_domain is not None:
            optimal_threshold = final_results["threshold"]
            if model_type == "svm" and svm_mode == "probability" or model_type == "logistic_regression":
                y_pred_in_domain = apply_threshold(y_proba_in_domain, optimal_threshold)
                print(f"  Applied optimal threshold: {optimal_threshold:.4f}")
        
        # Compute metrics
        from functions.ml.svm_training import compute_metrics, print_metrics
        metrics_in_domain = compute_metrics(y_test_in_domain, y_pred_in_domain, y_proba_in_domain)
        
        print("\nIn-Domain Test Set Performance:")
        print_metrics(metrics_in_domain)
        
        # Add to results
        final_results["y_pred_in_domain"] = y_pred_in_domain
        final_results["y_proba_in_domain"] = y_proba_in_domain
        final_results["metrics_in_domain"] = metrics_in_domain
        final_results["y_test_in_domain"] = y_test_in_domain
        final_results["X_test_in_domain"] = X_test_in_domain_scaled
        
        # Compute per-dataset metrics for in-domain test set if dataset indices provided
        if in_domain_dataset_indices is not None and in_domain_dataset_names is not None:
            print(f"\n  Computing per-dataset metrics for in-domain test set ({len(np.unique(in_domain_dataset_indices))} datasets)...")
            per_dataset_metrics_in_domain = compute_per_dataset_metrics(
                y_test=y_test_in_domain,
                y_pred=y_pred_in_domain,
                y_proba=y_proba_in_domain,
                dataset_indices=in_domain_dataset_indices,
                dataset_names=in_domain_dataset_names
            )
            final_results["per_dataset_metrics_in_domain"] = per_dataset_metrics_in_domain
            print(f"  ✓ Per-dataset metrics computed for in-domain test set")

    # Add results
    results.update(final_results)
    results["X_train"] = X_train_final
    results["X_test"] = X_test_scaled
    results["y_train"] = y_train_final
    results["y_test"] = y_test
    if X_val is not None:
        results["X_val"] = X_val
        results["y_val"] = y_val
    if threshold_tuning_results is not None:
        results["threshold_tuning_results"] = threshold_tuning_results

    print("\n" + "="*70)
    print("TRAINING PIPELINE COMPLETE")
    print("="*70)

    return results



def identify_critical_points(X: np.ndarray, y: np.ndarray) -> List[str]:
    """
    Identify potential issues and critical points in the data/model.
    
    Returns
    -------
    List[str]
        List of critical points and recommendations.
    """
    critical_points = []
    
    # Check data size
    if len(X) < 100:
        critical_points.append("⚠️  Small dataset: Consider collecting more data or using simpler models")
    
    # Check class imbalance
    unique, counts = np.unique(y, return_counts=True)
    if len(unique) == 2:
        imbalance_ratio = max(counts) / min(counts)
        if imbalance_ratio > 10:
            critical_points.append(f"⚠️  Severe class imbalance ({imbalance_ratio:.1f}:1). Strongly recommend class balancing")
        elif imbalance_ratio > 5:
            critical_points.append(f"⚠️  Moderate class imbalance ({imbalance_ratio:.1f}:1). Consider class balancing")
    
    # Check feature quality
    nan_count = np.isnan(X).sum()
    if nan_count > 0:
        critical_points.append(f"⚠️  Found {nan_count} NaN values in features - ensure proper handling")
    
    inf_count = np.isinf(X).sum()
    if inf_count > 0:
        critical_points.append(f"⚠️  Found {inf_count} infinite values in features - ensure proper handling")
    
    # Check for constant features
    constant_features = np.sum(np.std(X, axis=0) < 1e-10)
    if constant_features > 0:
        critical_points.append(f"⚠️  Found {constant_features} constant features - consider removing them")
    
    # Check feature scale
    feature_std = np.std(X, axis=0)
    if np.max(feature_std) / (np.min(feature_std[feature_std > 0]) + 1e-10) > 1e6:
        critical_points.append("⚠️  Large scale differences between features - feature scaling is critical")
    
    # Check for high dimensionality
    if X.shape[1] > X.shape[0]:
        critical_points.append("⚠️  More features than samples - consider feature selection or dimensionality reduction")
    
    # Recommendations
    critical_points.append("✓  Consider feature selection to reduce dimensionality and improve interpretability")
    critical_points.append("✓  Consider ensemble methods (Random Forest, XGBoost) for comparison")
    critical_points.append("✓  Monitor model performance over time - retrain periodically with new data")
    critical_points.append("✓  Consider model interpretability (SHAP values, feature importance)")
    critical_points.append("✓  Validate on multiple time periods to check for temporal drift")
    critical_points.append("✓  Consider cross-validation strategy that respects temporal order")
    critical_points.append("✓  Save model and scaler for deployment")
    critical_points.append("✓  Document feature engineering steps for reproducibility")
    
    return critical_points

