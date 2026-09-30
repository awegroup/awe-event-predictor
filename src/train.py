"""
Train a model (SVM or logistic regression) on N training flights and evaluate it
on a held-out partition of those flights (in-domain) and on M separate test
flights (out-of-domain).

The training flights are split into train and in-domain test.
The train part is further split into train and validation.
The train data is balanced; validation and test data keep their natural distribution.
Hyperparameters are tuned with K-fold cross-validation on the train data.
The validation data is used for feature-importance selection and threshold tuning.

Usage:
    python src/train.py --dataset data/dataset_1 --model svm --pre-peak-time 0.5
"""

import argparse
import numpy as np
import os

# Our implemented functions
from functions.CSVLoader import CSVLoader
from functions.preprocessing.data_processing import TractionDataProcessor
from functions.preprocessing.build_feature_matrix import build_feature_matrix
from functions.ml.training_pipeline import run_training_pipeline, identify_critical_points
from functions.ml.model_persistence import save_model
from functions.ml.train_test_split import time_based_train_test_split


from functions.cache.feature_cache import (
    load_features_from_cache,
    save_features_to_cache,
    _hash_dict,
    _hash_file,
)
def process_dataset_with_cache(
    df,
    csv_path,
    processor,
    signal_names,
    cache_root,
    cache_params,
    num_sub_windows,
    sub_window_mode
):
    """
    Returns:
        X, y, feature_names
    """
    file_hash = _hash_file(csv_path)
    param_hash = _hash_dict(cache_params)

    cache_filename = f"{os.path.basename(csv_path)}__{file_hash}__{param_hash}.pkl"
    cache_path = os.path.join(cache_root, cache_filename)

    cached = load_features_from_cache(cache_path)
    if cached is not None:
        print(f"Loaded cached features: {os.path.basename(csv_path)}")
        return cached["X"], cached["y"], cached["feature_names"]

    print(f"Computing features: {os.path.basename(csv_path)}")

    results = processor.process(df)

    feature_results = build_feature_matrix(
        data=results["filtered_data"],
        signal_names=signal_names,
        start_indices=results["frame_start_indices"],
        end_indices=results["frame_end_indices"],
        labels=results["frame_labels"],
        num_sub_windows=num_sub_windows,  # Split each 5-second window into N equal sub-windows
        sub_window_mode=sub_window_mode
    )

    payload = {
        "X": feature_results["X_tot"],
        "y": feature_results["Y_tot"],
        "feature_names": feature_results["feature_names"],
        "cache_params": cache_params,
        "file_hash": file_hash,
    }

    save_features_to_cache(cache_path, payload)
    return payload["X"], payload["y"], payload["feature_names"]


# -------------------------
# Parameters definition
# -------------------------

min_height = 20
threshold_on_T = 2500
min_seconds_between_peaks = 30  # seconds
window_length = 5       # seconds (signal history window length)
pre_peak_time = 1.0    # seconds (time before the peak)
post_peak_time = 3.0   # seconds (time after the peak for visualization)
num_sub_windows = 1  # number of sub-windows to split each 5-second window into

# signal names to use for feature extraction
signal_names = [
    "V_kite_0_mag", # m/s
    #'V_kite_1_mag', # m/s
    "kite_0_vx", # m/s
    "kite_0_vy", # m/s
    "kite_0_vz", # m/s
    #"kite_1_vx", # m/s
    #"kite_1_vy", # m/s
    #"kite_1_vz", # m/s
    #"kite_0_latitude",
    #"kite_0_longitude",
    "kite_0_altitude",  # m
    #"kite_0_roll", # deg to be unwrapped (convert to rad first)
    #"kite_0_pitch", # deg to be unwrapped (convert to rad first)
    #"kite_0_yaw", # deg to be unwrapped (convert to rad first)
    #"kite_1_latitude",
    #"kite_1_longitude",
    #"kite_1_altitude", # m 
    #"kite_1_roll", # deg to be unwrapped (convert to rad first)
    #"kite_1_pitch", # deg to be unwrapped (convert to rad first)
    #"kite_1_yaw", # deg to be unwrapped (convert to rad first)

    "ground_tether_reelout_speed",
    "ground_tether_length",
    "ground_tether_force",

    "kite_pos_east", #m
    "kite_pos_north", #m 
    "kite_height", #m

    "kite_elevation", # rad
    "kite_azimuth", # rad
    "kite_distance", # m
    "kite_heading", # seems rad
    "kite_course", # seems rad

    "kite_actual_steering", # % percentage wrt of the range in one direction
    "kite_actual_depower", # %

    "ground_mech_power",

    "kite_turn_rate", 
    "kite_course_rate",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/dataset_1",
                        help="Dataset folder; reads <dataset>/cleaned/{training_data,test_data} (default: data/dataset_1)")
    parser.add_argument("--model", choices=["svm", "logistic_regression"], default="logistic_regression",
                        help="Model type (default: logistic_regression)")
    parser.add_argument("--pre-peak-time", type=float, default=pre_peak_time,
                        help=f"Gap in seconds between the end of a window and the peak (default: {pre_peak_time})")
    parser.add_argument("--feature-selection", action="store_true",
                        help="SVM only: select the top 20 margin + top 20 boundary features, then retrain")
    parser.add_argument("--features-from", default=None, metavar="MODEL_DIR",
                        help="Reuse the selected features of a saved model (e.g. an SVM trained with --feature-selection)")
    parser.add_argument("--output", default=None, metavar="MODEL_DIR",
                        help="Where to save the model and metrics (default: saved_models/<auto name>)")
    parser.add_argument("--n-iter", type=int, default=50, help="Random-search iterations (default: 50)")
    parser.add_argument("--cv-folds", type=int, default=3, help="Cross-validation folds (default: 3)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--cache-dir", default="feature_cache", help="Feature cache folder (default: feature_cache)")
    return parser.parse_args()


if __name__ == "__main__":

    args = parse_args()
    pre_peak_time = args.pre_peak_time
    random_state = args.seed

    train_folder = os.path.join(args.dataset, "cleaned", "training_data")
    test_folder  = os.path.join(args.dataset, "cleaned", "test_data")
    for folder in (train_folder, test_folder):
        if not os.path.isdir(folder):
            raise SystemExit(f"Missing folder: {folder}\nRun `python src/preprocess.py --dataset {args.dataset}` first.")

    sub_window_mode = "concatenate"

    cache_params = {
        "cache_version": "v1",
        "min_height": min_height,
        "threshold_on_T": threshold_on_T,
        "min_seconds_between_peaks": min_seconds_between_peaks,
        "window_length": window_length,
        "pre_peak_time": pre_peak_time,
        "signal_names": signal_names,
        "num_sub_windows": num_sub_windows,  # Include in cache params
        "sub_window_mode": sub_window_mode,  # Include in cache params
    }

    # ----------------------------------------------------------
    # LOAD TRAIN AND TEST DATASETS
    # ----------------------------------------------------------
    train_loader = CSVLoader(train_folder)
    test_loader  = CSVLoader(test_folder)

    train_data = train_loader.load_all()
    test_data  = test_loader.load_all()

    print(f"Loaded {len(train_data)} TRAIN datasets")
    print(f"Loaded {len(test_data)} TEST datasets")

    processor = TractionDataProcessor(
        min_height,
        threshold_on_T,
        min_seconds_between_peaks,
        window_length,
        pre_peak_time,
        plot=False
    )

    # ----------------------------------------------------------
    # STORAGE
    # ----------------------------------------------------------
    X_train_all, Y_train_all = [], []
    X_test_all,  Y_test_all  = [], []
    train_dataset_indices = []  # Track which dataset each training sample belongs to
    train_dataset_names = []  # Store training dataset names

    feature_names = None

    # ----------------------------------------------------------
    # PROCESS TRAIN DATA
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("PROCESSING TRAIN DATASETS")
    print("="*60)

    cache_root_train = os.path.join(args.cache_dir, "train")
    for i, (df, csv_path) in enumerate(zip(train_data, train_loader.files)):
        X_i, y_i, feature_names = process_dataset_with_cache(
            df=df,
            csv_path=csv_path,
            processor=processor,
            signal_names=signal_names,
            cache_root=cache_root_train,
            cache_params=cache_params,
            num_sub_windows=num_sub_windows,
            sub_window_mode=sub_window_mode
        )
        X_train_all.append(X_i)
        Y_train_all.append(y_i)
        
        # Track dataset indices for per-dataset metrics
        train_dataset_indices.extend([i] * len(X_i))
        dataset_name = os.path.basename(csv_path).replace('.csv', '')
        if i == 0 or dataset_name not in train_dataset_names:
            train_dataset_names.append(dataset_name)

    # ----------------------------------------------------------
    # PROCESS TEST DATA
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("PROCESSING TEST DATASETS")
    print("="*60)

    cache_root_test = os.path.join(args.cache_dir, "test")
    test_processing_results = []  # Store processing results for plotting
    test_frame_mapping = []  # Track (dataset_idx, frame_idx) for each frame in concatenated test set
    test_dataset_indices = []  # Track which dataset each sample belongs to
    test_dataset_names = []  # Store dataset names
    
    for i, (df, csv_path) in enumerate(zip(test_data, test_loader.files)):
        # Process to get features (may use cache)
        X_i, y_i, _ = process_dataset_with_cache(
            df=df,
            csv_path=csv_path,
            processor=processor,
            signal_names=signal_names,
            cache_root=cache_root_test,
            cache_params=cache_params,  
            num_sub_windows=num_sub_windows,
            sub_window_mode=sub_window_mode
        )

        X_test_all.append(X_i)
        Y_test_all.append(y_i)
        
        # Track dataset indices for per-dataset metrics
        test_dataset_indices.extend([i] * len(X_i))
        dataset_name = os.path.basename(csv_path).replace('.csv', '')
        if i == 0 or dataset_name not in test_dataset_names:
            test_dataset_names.append(dataset_name)
        
        # Also process to get full results for plotting (needed even if features are cached)
        results = processor.process(df)
        test_processing_results.append(results)
        
        # Track frame mapping: for each frame in this dataset, record (dataset_idx, frame_idx)
        num_frames = len(results["frame_labels"])
        for frame_idx in range(num_frames):
            test_frame_mapping.append((i, frame_idx))


    # ----------------------------------------------------------
    # CONCATENATE
    # ----------------------------------------------------------
    X_train_full = np.vstack(X_train_all)
    Y_train_full = np.hstack(Y_train_all)
    train_dataset_indices = np.array(train_dataset_indices)

    X_test_out_of_domain  = np.vstack(X_test_all)
    Y_test_out_of_domain  = np.hstack(Y_test_all)
    test_dataset_indices = np.array(test_dataset_indices)
    
    print(f"\nTraining data (full): {len(train_dataset_names)} datasets")
    for i, name in enumerate(train_dataset_names):
        n_samples = np.sum(train_dataset_indices == i)
        print(f"  Dataset {i}: {name} ({n_samples} samples)")
    
    print(f"\nOut-of-domain test set: {len(test_dataset_names)} datasets")
    for i, name in enumerate(test_dataset_names):
        n_samples = np.sum(test_dataset_indices == i)
        print(f"  Dataset {i}: {name} ({n_samples} samples)")

    # ----------------------------------------------------------
    # Split training data into train + in-domain test
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("SPLITTING TRAINING DATA")
    print("="*60)
    
    # Reserve a portion of training data as in-domain test set
    # This will NOT undergo balancing and represents the true distribution
    in_domain_test_size = 0.2  # 20% of training data for in-domain test
    
    from sklearn.model_selection import train_test_split
    X_train, X_test_in_domain, Y_train, Y_test_in_domain, train_dataset_indices_split, in_domain_dataset_indices = train_test_split(
        X_train_full,
        Y_train_full,
        train_dataset_indices,
        test_size=in_domain_test_size,
        random_state=random_state,
        stratify=Y_train_full if len(np.unique(Y_train_full)) > 1 else None
    )
    
    print(f"Training data: {X_train.shape[0]} samples")
    print(f"  Class 0: {np.sum(Y_train == 0)}")
    print(f"  Class 1 (faulty): {np.sum(Y_train == 1)}")
    print(f"In-domain test: {X_test_in_domain.shape[0]} samples (no balancing applied)")
    print(f"  Class 0: {np.sum(Y_test_in_domain == 0)}")
    print(f"  Class 1 (faulty): {np.sum(Y_test_in_domain == 1)}")
    
    # Show in-domain test dataset distribution
    print(f"\n  In-domain test set composition:")
    for i, name in enumerate(train_dataset_names):
        n_samples = np.sum(in_domain_dataset_indices == i)
        if n_samples > 0:
            print(f"    Dataset {i}: {name} ({n_samples} samples)")
    
    print(f"\nOut-of-domain test: {X_test_out_of_domain.shape[0]} samples")
    print(f"  Class 0: {np.sum(Y_test_out_of_domain == 0)}")
    print(f"  Class 1 (faulty): {np.sum(Y_test_out_of_domain == 1)}")

    print("\n" + "="*60)
    print("FINAL DATASET SUMMARY")
    print("="*60)
    print(f"TRAIN: {X_train.shape[0]} samples (will undergo balancing)")
    print(f"IN-DOMAIN TEST: {X_test_in_domain.shape[0]} samples (natural distribution)")
    print(f"OUT-OF-DOMAIN TEST: {X_test_out_of_domain.shape[0]} samples (external data)")
    
    print(f"Feature shape: {X_train.shape}")

    # ----------------------------------------------------------
    # Train model (Logistic Regression or SVM)
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("Training Model")
    print("="*60)
    
    # Training parameters
    #test_size = 0.2
    model_type = args.model  # "svm" or "logistic_regression"
    scaling_method = "standard"  # "standard" or "minmax"
    
    # SVM mode switch:
    #   - "probability": Train SVM with probability=True, use predict_proba for predictions
    #                    Enables threshold tuning and probability-based inference
    #   - "margin": Train SVM with probability=False, use classic margin optimization
    #              Uses decision_function for predictions (no probabilities)
    #              Threshold tuning is disabled (SVM uses default decision boundary)
    svm_mode = "probability"  # "probability" or "margin"
    
    # Balancing methods:
    #   - "smote": Synthetic Minority Oversampling (adds synthetic samples)
    #   - "downsample": Random downsampling (may remove important boundary samples)
    #   - "upsample": Random upsampling (duplicates minority samples)
    #   - "nearmiss", "nearmiss1", "nearmiss2", "nearmiss3": Intelligent undersampling
    #     (preserves majority samples closest to minority - good for boundary preservation)
    #   - "tomek_links": Removes noisy/boundary pairs (cleaning, may still be imbalanced)
    #   - "tomek_links_then_nearmiss": Combines cleaning + intelligent undersampling
    #   - "none": No balancing
    balancing_method = "downsample"  # Recommended for highly imbalanced data
    balance_validation = False  # If True, applies balancing to validation set; if False, keeps natural distribution (recommended)
    downsample_ratio = 3.0  # Only used for "downsample" method
    nearmiss_version = 1  # 1, 2, or 3 (only used for nearmiss methods)
    nearmiss_n_neighbors = 3  # Number of neighbors for NearMiss
    tune_hyperparameters = True
    hyperparameter_scoring = "pr_auc"  # "macro_f1", "balanced_accuracy", "f1", "f1_class1", or "pr_auc"
    cv_folds = args.cv_folds
    n_iter = args.n_iter  # Number of iterations for random search
    
    # Threshold tuning (only available when svm_mode="probability")
    # When svm_mode="margin", threshold tuning is automatically disabled
    tune_threshold = True  # Tune decision threshold on validation set after hyperparameter tuning
    threshold_tuning_method = "max_pr_auc"  # "max_f1", "max_pr_auc", or "balanced_precision_recall"
    validation_size = 0.3  # Fraction of training data to use for threshold tuning
    
    # Auto-disable threshold tuning for margin mode
    if svm_mode == "margin":
        tune_threshold = False
        print("\n⚠️  Threshold tuning disabled for margin-based SVM (no probabilities available)")
    
    # Feature importance parameters
    compute_feature_importance = args.feature_selection  # Compute margin-based feature importance
    feature_importance_n_repeats = 10  # Number of permutations per feature
    feature_importance_boundary_threshold = 0.2  # Threshold for boundary samples (|margin| < threshold)
    # Feature selection parameters
    use_feature_selection = args.feature_selection  # Enable feature selection based on importance
    top_n_margin_features = 20  # Top N features by margin importance
    top_m_boundary_features = 20  # Top M features by boundary importance
    # Note: Final model will use union of top N margin + top M boundary features
    
    # Feature reuse from another model (e.g., use SVM-selected features for Logistic Regression)
    # Set to a model directory path (--features-from) to reuse its selected features, or None to compute new features
    load_features_from_model = args.features_from  # e.g., "saved_models/svm_prob_FS20m20b_peak0p5"
    # When load_features_from_model is set, it will:
    # 1. Load selected_feature_indices from the specified model's training_params.json
    # 2. Use those exact features for the current model (bypassing feature importance computation)
    # 3. Useful for training different model types with the same feature subset
    
    # Feature filtering parameters
    remove_correlated_features = False  # Remove highly correlated features (correlation > threshold) keep false
    correlation_threshold = 0.95  # Correlation threshold for removing features
    
    # ----------------------------------------------------------
    # Load pre-selected features from another model if specified
    # ----------------------------------------------------------
    preselected_feature_indices = None
    if load_features_from_model is not None:
        print("\n" + "="*60)
        print("LOADING PRE-SELECTED FEATURES FROM MODEL")
        print("="*60)
        print(f"Source model: {load_features_from_model}")
        
        import json
        training_params_path = os.path.join(load_features_from_model, "training_params.json")
        
        if not os.path.exists(training_params_path):
            raise FileNotFoundError(f"Training parameters not found: {training_params_path}")
        
        with open(training_params_path, 'r') as f:
            source_training_params = json.load(f)
        
        if "selected_feature_indices" not in source_training_params:
            raise ValueError(f"No selected_feature_indices found in {training_params_path}. "
                           "The source model may not have used feature selection.")
        
        preselected_feature_indices = np.array(source_training_params["selected_feature_indices"], dtype=int)
        n_selected = len(preselected_feature_indices)
        
        print(f"  ✓ Loaded {n_selected} pre-selected feature indices")
        
        # Load and display selected feature names if available
        if "selected_feature_names" in source_training_params:
            selected_feature_names_from_source = source_training_params["selected_feature_names"]
            print(f"  ✓ Feature names loaded from source model")
            print(f"\n  Pre-selected features ({n_selected} total):")
            for i, fname in enumerate(selected_feature_names_from_source[:10], 1):  # Show first 10
                print(f"    {i:2d}. {fname}")
            if n_selected > 10:
                print(f"    ... and {n_selected - 10} more features")
        
        # Validate compatibility
        print(f"\n  Validating feature compatibility...")
        print(f"    Source model features: {n_selected}")
        print(f"    Current feature matrix: {X_train.shape[1]} features")
        
        if np.max(preselected_feature_indices) >= X_train.shape[1]:
            raise ValueError(f"Feature index mismatch! Source model used features up to index {np.max(preselected_feature_indices)}, "
                           f"but current data only has {X_train.shape[1]} features. "
                           "Ensure the same signal_names and feature extraction settings are used.")
        
        print(f"  ✓ Feature compatibility validated")
        print(f"\n  Will train {model_type.upper()} model using the same {n_selected} features from source model")
        
        # Override feature selection settings to avoid confusion
        use_feature_selection = False
        compute_feature_importance = False
        print(f"\n  Note: Disabled compute_feature_importance and use_feature_selection (using pre-selected features)")
    
    training_results = run_training_pipeline(
        X=X_train,
        y=Y_train,
        X_test=X_test_out_of_domain,
        y_test=Y_test_out_of_domain,
        X_test_in_domain=X_test_in_domain,
        y_test_in_domain=Y_test_in_domain,
        test_dataset_indices=test_dataset_indices,  # Track which dataset each out-of-domain test sample belongs to
        test_dataset_names=test_dataset_names,  # Names of out-of-domain test datasets
        in_domain_dataset_indices=in_domain_dataset_indices,  # Track which dataset each in-domain test sample belongs to
        in_domain_dataset_names=train_dataset_names,  # Names of in-domain datasets (from training data)
        model_type=model_type,
        scaling_method=scaling_method,
        svm_mode=svm_mode,  # Pass SVM mode: "probability" or "margin"
        balancing_method=balancing_method,
        balance_validation=balance_validation,  # Whether to balance validation set
        downsample_ratio=downsample_ratio,
        nearmiss_version=nearmiss_version,
        nearmiss_n_neighbors=nearmiss_n_neighbors,
        tune_hyperparameters=tune_hyperparameters,
        hyperparameter_scoring=hyperparameter_scoring,
        cv_folds=cv_folds,
        n_iter=n_iter,
        tune_threshold=tune_threshold,
        threshold_tuning_method=threshold_tuning_method,
        validation_size=validation_size,
        random_state=random_state,
        compute_feature_importance=compute_feature_importance,
        feature_importance_n_repeats=feature_importance_n_repeats,
        feature_importance_boundary_threshold=feature_importance_boundary_threshold,
        use_feature_selection=use_feature_selection,
        top_n_margin_features=top_n_margin_features,
        top_m_boundary_features=top_m_boundary_features,
        remove_correlated_features=remove_correlated_features,
        correlation_threshold=correlation_threshold,
        preselected_feature_indices=preselected_feature_indices  # Pass pre-selected features if loaded
    )

    results = {
        "model": training_results["model"],
        "model_type": model_type,
        "svm_mode": svm_mode,  # Store SVM mode
        "scaler": training_results["scaler"],
        "training_results": training_results,
        "feature_names": feature_names,
        "X_train": X_train,
        "X_test_out_of_domain": X_test_out_of_domain,
        "Y_train": Y_train,
        "Y_test_out_of_domain": Y_test_out_of_domain,
        "X_test_in_domain": X_test_in_domain,
        "Y_test_in_domain": Y_test_in_domain,
    }
    
    # Store training results
    results["training_results"] = training_results
    results["model"] = training_results["model"]
    results["model_type"] = model_type
    results["svm_mode"] = svm_mode
    results["scaler"] = training_results["scaler"]
    
    # ----------------------------------------------------------
    # Save model, scaler, and metrics to directory
    # ----------------------------------------------------------
    print("\n" + "="*60)
    print("Saving Model and Metrics")
    print("="*60)
    
    # Include SVM mode and feature selection in model save directory name
    mode_suffix = "prob" if svm_mode == "probability" else "margin"
    if "selected_feature_indices" in training_results:
        n_features = len(training_results["selected_feature_indices"])
        if training_results.get("feature_selection_source") == "preselected":
            fs_suffix = f"_PRESELECTED{n_features}f"
        else:
            fs_suffix = f"_FS{top_n_margin_features}m{top_m_boundary_features}b"
    else:
        fs_suffix = "_ALLFEATURES"
    peak_suffix = f"{pre_peak_time:g}".replace(".", "p")
    model_save_dir = args.output or os.path.join(
        "saved_models", f"{model_type}_{mode_suffix}{fs_suffix}_peak{peak_suffix}"
    )

    # Prepare training parameters for saving
    training_params_to_save = {
        #"test_size": test_size,
        "scaling_method": scaling_method,
        "svm_mode": svm_mode,  # Save SVM mode
        "balancing_method": balancing_method,
        "tune_hyperparameters": tune_hyperparameters,
        "hyperparameter_scoring": hyperparameter_scoring,
        "cv_folds": cv_folds,
        "n_iter": n_iter,
        "tune_threshold": tune_threshold,
        "threshold_tuning_method": threshold_tuning_method,
        "validation_size": validation_size,
        "random_state": random_state,
        "dataset": args.dataset,
        "window_length": window_length,
        "pre_peak_time": pre_peak_time,
        "threshold_on_T": threshold_on_T,
        "use_feature_selection": use_feature_selection,
        "top_n_margin_features": top_n_margin_features,
        "top_m_boundary_features": top_m_boundary_features,
        "remove_correlated_features": remove_correlated_features,
        "correlation_threshold": correlation_threshold
    }
    
    # Add selected feature indices if feature selection was used
    if "selected_feature_indices" in training_results:
        training_params_to_save["selected_feature_indices"] = training_results["selected_feature_indices"].tolist()
        training_params_to_save["n_selected_features"] = int(training_results.get("n_selected_features", len(training_results["selected_feature_indices"])))
        training_params_to_save["feature_selection_source"] = training_results.get("feature_selection_source", "computed")
        
        # Add source model path if features were preselected
        if load_features_from_model is not None:
            training_params_to_save["feature_selection_source_model"] = load_features_from_model
        
        # Map selected indices to feature names
        if feature_names is not None:
            selected_feature_names = [feature_names[i] for i in training_results["selected_feature_indices"]]
            training_params_to_save["selected_feature_names"] = selected_feature_names
            
            if training_results.get("feature_selection_source") == "preselected":
                print(f"\n  Using {len(selected_feature_names)} pre-selected features from source model")
            else:
                print(f"\n  Selected {len(selected_feature_names)} features for final model")
                print(f"  Top {top_n_margin_features} margin features + Top {top_m_boundary_features} boundary features")
    
    # Add optimal threshold if available
    if "threshold" in training_results:
        training_params_to_save["optimal_threshold"] = training_results["threshold"]
    
    # Add best hyperparameters if available
    if "tuning_results" in training_results:
        training_params_to_save["best_hyperparameters"] = training_results["tuning_results"].get("best_params", {})
        training_params_to_save["best_cv_score"] = training_results["tuning_results"].get("best_score", None)
    
    # Prepare metadata
    metadata_to_save = {
        "num_train_samples": len(training_results["y_train"]),
        "num_test_samples_out_of_domain": len(training_results["y_test"]),
        "num_test_samples_in_domain": len(Y_test_in_domain),
        "num_features": X_train.shape[1],
        "num_signals": len(signal_names),
        "train_class_distribution": {
            int(cls): int(count) for cls, count in zip(
                *np.unique(training_results["y_train"], return_counts=True)
            )
        },
        "test_out_of_domain_class_distribution": {
            int(cls): int(count) for cls, count in zip(
                *np.unique(training_results["y_test"], return_counts=True)
            )
        },
        "test_in_domain_class_distribution": {
            int(cls): int(count) for cls, count in zip(
                *np.unique(Y_test_in_domain, return_counts=True)
            )
        },
    }
    
    # Prepare test processing data for false positive plotting
    test_processing_data = {
        "test_processing_results": test_processing_results,
        "test_frame_mapping": test_frame_mapping,
        "test_file_paths": test_loader.files,
        "window_length": window_length,
        "pre_peak_time": pre_peak_time,
        "post_peak_time": post_peak_time,
        "threshold_on_T": threshold_on_T
    }
    
    # Extract feature importance if available
    feature_importance_to_save = None
    if "feature_importance" in training_results:
        feature_importance_to_save = training_results["feature_importance"]
        print(f"\n  Feature importance computed: {len(feature_importance_to_save.get('ranked_indices', []))} features ranked")
    
    # Handle feature names: if feature selection was used, filter to selected features
    feature_names_to_save = results["feature_names"]
    if "selected_feature_indices" in training_results and feature_names_to_save is not None:
        selected_indices = training_results["selected_feature_indices"]
        # Map original indices to selected feature names
        selected_feature_names = [feature_names_to_save[i] for i in selected_indices]
        feature_names_to_save = selected_feature_names
        print(f"  Using selected feature names: {len(selected_feature_names)} features")
    
    # Prepare in-domain test results if available
    in_domain_test_results = None
    if "y_pred_in_domain" in training_results:
        in_domain_test_results = {
            "y_test": training_results.get("y_test_in_domain"),
            "y_pred": training_results.get("y_pred_in_domain"),
            "y_proba": training_results.get("y_proba_in_domain"),
            "metrics": training_results.get("metrics_in_domain")
        }
    
    save_model(
        model_dir=model_save_dir,
        model=training_results["model"],
        scaler=training_results["scaler"],
        y_test=training_results["y_test"],
        y_pred=training_results["y_pred"],
        y_proba=training_results["y_proba"],
        metrics=training_results["metrics"],
        model_type=model_type,
        feature_names=feature_names_to_save,  # Use filtered feature names if selection was used
        training_params=training_params_to_save,
        metadata=metadata_to_save,
        X_test=training_results.get("X_test"),  # Save X_test for feature importance analysis
        test_processing_data=test_processing_data,  # Save for false positive plotting
        threshold=training_results.get("threshold", 0.5),  # Save decision threshold
        feature_importance=feature_importance_to_save,  # Save margin-based feature importance
        in_domain_test_results=in_domain_test_results,  # Save in-domain test results
        per_dataset_metrics=training_results.get("per_dataset_metrics"),  # Save per-dataset metrics (out-of-domain)
        per_dataset_metrics_in_domain=training_results.get("per_dataset_metrics_in_domain")  # Save per-dataset metrics (in-domain)
    )
    
    print("\n" + "="*60)
    print("Pipeline Complete!")
    print("="*60)
    print(f"\nModel type: {model_type}")
    if model_type == "svm":
        print(f"SVM mode: {svm_mode}")
        if svm_mode == "probability":
            print("  - Using probability-based predictions (predict_proba)")
        else:
            print("  - Using margin-based predictions (decision_function)")
    print(f"Model saved in: results['model']")
    print(f"Scaler saved in: results['scaler']")
    print(f"Model files saved to: {model_save_dir}")
    print(f"Test predictions: results['training_results']['y_pred']")
    if svm_mode == "probability":
        print(f"Test probabilities: results['training_results']['y_proba']")
    else:
        print(f"Test decision scores: results['training_results']['y_proba'] (margin-based, not probabilities)")
    print(f"Test metrics: results['training_results']['metrics']")
    print(f"\nTo plot metrics, run:")
    print(f"  python src/plot_model_metrics.py {model_save_dir}")
    
    print(f"\nTo analyze feature importance, run:")
    print(f"  python src/analyze_feature_importance.py {model_save_dir} --top-n 20")
    
    # Display top features from margin-based importance if available
    if feature_importance_to_save is not None and results["feature_names"] is not None:
        print(f"\n" + "="*60)
        print("FEATURE IMPORTANCE SUMMARY")
        print("="*60)
        
        # Show top margin features
        margin_ranked = feature_importance_to_save.get("margin_ranked_indices", [])
        margin_changes = feature_importance_to_save.get("margin_change_mean", [])
        if len(margin_ranked) > 0:
            print(f"\n  Top {min(10, len(margin_ranked))} Features (by Margin Importance):")
            top_n_display = min(10, len(margin_ranked))
            for i, feat_idx in enumerate(margin_ranked[:top_n_display], 1):
                feat_name = results["feature_names"][feat_idx] if feat_idx < len(results["feature_names"]) else f"Feature_{feat_idx}"
                margin_change = margin_changes[feat_idx] if feat_idx < len(margin_changes) else 0.0
                print(f"    {i:2d}. {feat_name:50s} (margin_change: {margin_change:.6f})")
        
        # Show top boundary features
        boundary_ranked = feature_importance_to_save.get("boundary_ranked_indices", [])
        boundary_changes = feature_importance_to_save.get("boundary_change_mean", [])
        if len(boundary_ranked) > 0:
            print(f"\n  Top {min(10, len(boundary_ranked))} Features (by Boundary Importance):")
            top_n_display = min(10, len(boundary_ranked))
            for i, feat_idx in enumerate(boundary_ranked[:top_n_display], 1):
                feat_name = results["feature_names"][feat_idx] if feat_idx < len(results["feature_names"]) else f"Feature_{feat_idx}"
                boundary_change = boundary_changes[feat_idx] if feat_idx < len(boundary_changes) else 0.0
                print(f"    {i:2d}. {feat_name:50s} (boundary_change: {boundary_change:.6f})")
        
        # Show selected features if feature selection was used
        if "selected_feature_indices" in training_results:
            selected_indices = training_results["selected_feature_indices"]
            print(f"\n  Selected Features for Final Model ({len(selected_indices)} features):")
            print(f"    Top {top_n_margin_features} margin + Top {top_m_boundary_features} boundary features")
            for i, feat_idx in enumerate(selected_indices, 1):
                feat_name = results["feature_names"][feat_idx] if feat_idx < len(results["feature_names"]) else f"Feature_{feat_idx}"
                is_margin = feat_idx in margin_ranked[:top_n_margin_features] if len(margin_ranked) > 0 else False
                is_boundary = feat_idx in boundary_ranked[:top_m_boundary_features] if len(boundary_ranked) > 0 else False
                source = []
                if is_margin:
                    source.append("margin")
                if is_boundary:
                    source.append("boundary")
                source_str = "+".join(source) if source else "both"
                print(f"    {i:2d}. {feat_name:50s} (from: {source_str})")
            #if len(selected_indices) > 20:
            #    print(f"    ... and {len(selected_indices) - 20} more features")
    
    # Display in-domain test results if available
    if "metrics_in_domain" in training_results:
        print(f"\n" + "="*60)
        print("IN-DOMAIN TEST SET RESULTS SUMMARY")
        print("="*60)
        metrics_in_domain = training_results["metrics_in_domain"]
        print(f"  Accuracy:           {metrics_in_domain['accuracy']:.4f}")
        print(f"  Precision:          {metrics_in_domain['precision']:.4f}")
        print(f"  Recall:             {metrics_in_domain['recall']:.4f}")
        print(f"  F1-score:           {metrics_in_domain['f1']:.4f}")
        print(f"  Macro F1:           {metrics_in_domain['macro_f1']:.4f}")
        print(f"  Balanced Accuracy:  {metrics_in_domain['balanced_accuracy']:.4f}")
        if metrics_in_domain.get('roc_auc') is not None:
            print(f"  ROC-AUC:            {metrics_in_domain['roc_auc']:.4f}")
        if metrics_in_domain.get('pr_auc') is not None:
            print(f"  PR-AUC:             {metrics_in_domain['pr_auc']:.4f}")
        
        print(f"\n  Comparison:")
        print(f"  {'Metric':<25} {'In-Domain':<15} {'Out-of-Domain':<15}")
        print(f"  {'-'*55}")
        metrics_out = training_results["metrics"]
        for metric_name in ['accuracy', 'precision', 'recall', 'f1', 'macro_f1', 'balanced_accuracy']:
            in_val = metrics_in_domain.get(metric_name, 0.0)
            out_val = metrics_out.get(metric_name, 0.0)
            print(f"  {metric_name:<25} {in_val:<15.4f} {out_val:<15.4f}")
    
    # Display per-dataset metrics for in-domain test set if available
    if "per_dataset_metrics_in_domain" in training_results:
        print(f"\n" + "="*60)
        print("PER-DATASET METRICS (IN-DOMAIN TEST)")
        print("="*60)
        per_dataset_metrics_in_domain = training_results["per_dataset_metrics_in_domain"]
        
        # Print header
        print(f"\n  {'Dataset':<50} {'Samples':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'PR-AUC':<10}")
        print(f"  {'-'*120}")
        
        # Print metrics for each dataset
        for dataset_name, metrics in per_dataset_metrics_in_domain.items():
            n_samples = metrics.get('n_samples', 0)
            acc = metrics.get('accuracy', 0.0)
            prec = metrics.get('precision', 0.0)
            rec = metrics.get('recall', 0.0)
            f1 = metrics.get('f1', 0.0)
            pr_auc = metrics.get('pr_auc', 0.0) if metrics.get('pr_auc') is not None else 0.0
            print(f"  {dataset_name:<50} {n_samples:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {pr_auc:<10.4f}")
        
        # Print overall metrics for comparison
        print(f"  {'-'*120}")
        metrics_overall = training_results["metrics_in_domain"]
        print(f"  {'OVERALL (all datasets)':<50} {len(Y_test_in_domain):<10} "
              f"{metrics_overall['accuracy']:<10.4f} {metrics_overall['precision']:<10.4f} "
              f"{metrics_overall['recall']:<10.4f} {metrics_overall['f1']:<10.4f} "
              f"{metrics_overall.get('pr_auc', 0.0):<10.4f}")
    
    # Display per-dataset metrics if available
    if "per_dataset_metrics" in training_results:
        print(f"\n" + "="*60)
        print("PER-DATASET METRICS (OUT-OF-DOMAIN)")
        print("="*60)
        per_dataset_metrics = training_results["per_dataset_metrics"]
        
        # Print header
        print(f"\n  {'Dataset':<50} {'Samples':<10} {'Accuracy':<10} {'Precision':<10} {'Recall':<10} {'F1':<10} {'PR-AUC':<10}")
        print(f"  {'-'*120}")
        
        # Print metrics for each dataset
        for dataset_name, metrics in per_dataset_metrics.items():
            n_samples = metrics.get('n_samples', 0)
            acc = metrics.get('accuracy', 0.0)
            prec = metrics.get('precision', 0.0)
            rec = metrics.get('recall', 0.0)
            f1 = metrics.get('f1', 0.0)
            pr_auc = metrics.get('pr_auc', 0.0) if metrics.get('pr_auc') is not None else 0.0
            print(f"  {dataset_name:<50} {n_samples:<10} {acc:<10.4f} {prec:<10.4f} {rec:<10.4f} {f1:<10.4f} {pr_auc:<10.4f}")
        
        # Print overall metrics for comparison
        print(f"  {'-'*120}")
        metrics_overall = training_results["metrics"]
        print(f"  {'OVERALL (all datasets)':<50} {len(Y_test_out_of_domain):<10} "
              f"{metrics_overall['accuracy']:<10.4f} {metrics_overall['precision']:<10.4f} "
              f"{metrics_overall['recall']:<10.4f} {metrics_overall['f1']:<10.4f} "
              f"{metrics_overall.get('pr_auc', 0.0):<10.4f}")
    
    print(f"\nNote: False positive plots will be generated when running plot_model_metrics.py")
    