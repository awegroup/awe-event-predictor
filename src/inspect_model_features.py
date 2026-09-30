"""
Utility script to inspect the features used by a saved model.

This is useful when you want to:
1. See which features a model selected
2. Verify feature compatibility before reusing features from one model in another
3. Compare feature selections across different models

Usage:
    python inspect_model_features.py <model_directory>
    python inspect_model_features.py <model_directory> --compare <other_model_directory>

Examples:
    # Inspect SVM features
    python src/inspect_model_features.py saved_models/svm_prob_FS20m20b_...
    
    # Compare features between SVM and Logistic Regression
    python src/inspect_model_features.py saved_models/svm_prob_FS20m20b_... --compare ./saved_models/logistic_regression_...
"""

import os
import json
import sys
import argparse


def load_model_features(model_dir):
    """Load feature information from a saved model directory."""
    training_params_path = os.path.join(model_dir, "training_params.json")
    
    if not os.path.exists(training_params_path):
        raise FileNotFoundError(f"Training parameters not found: {training_params_path}")
    
    with open(training_params_path, 'r') as f:
        training_params = json.load(f)
    
    # Load metadata for additional info
    metadata_path = os.path.join(model_dir, "metadata.json")
    metadata = None
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
    
    return training_params, metadata


def print_feature_info(model_dir, training_params, metadata):
    """Print comprehensive feature information."""
    print(f"\nModel Directory: {model_dir}")
    print("=" * 80)
    
    # Basic model info
    model_type = training_params.get("svm_mode", "unknown")
    if "svm" in model_dir.lower():
        model_type = f"SVM ({model_type})"
    elif "logistic" in model_dir.lower():
        model_type = "Logistic Regression"
    
    print(f"\nModel Type: {model_type}")
    
    # Feature selection info
    if "selected_feature_indices" in training_params:
        selected_indices = training_params["selected_feature_indices"]
        n_selected = len(selected_indices)
        
        print(f"\nFeature Selection: YES")
        print(f"  Number of selected features: {n_selected}")
        
        # Feature selection source
        selection_source = training_params.get("feature_selection_source", "computed")
        if selection_source == "preselected":
            print(f"  Selection method: Pre-selected from another model")
            source_model = training_params.get("feature_selection_source_model", "unknown")
            print(f"  Source model: {source_model}")
        else:
            print(f"  Selection method: Computed (margin + boundary importance)")
            top_n_margin = training_params.get("top_n_margin_features", "unknown")
            top_m_boundary = training_params.get("top_m_boundary_features", "unknown")
            print(f"  Top N margin features: {top_n_margin}")
            print(f"  Top M boundary features: {top_m_boundary}")
        
        # Feature indices
        print(f"\n  Selected feature indices:")
        print(f"    {selected_indices}")
        
        # Feature names if available
        if "selected_feature_names" in training_params:
            feature_names = training_params["selected_feature_names"]
            print(f"\n  Selected features ({len(feature_names)} total):")
            for i, fname in enumerate(feature_names, 1):
                print(f"    {i:3d}. {fname}")
        
    else:
        print(f"\nFeature Selection: NO (using all features)")
        if metadata and "num_features" in metadata:
            print(f"  Total features: {metadata['num_features']}")
    
    # Training parameters summary
    print(f"\nTraining Parameters:")
    print(f"  Scaling method: {training_params.get('scaling_method', 'unknown')}")
    print(f"  Balancing method: {training_params.get('balancing_method', 'unknown')}")
    print(f"  Hyperparameter tuning: {training_params.get('tune_hyperparameters', 'unknown')}")
    if training_params.get("tune_threshold"):
        print(f"  Threshold tuning: {training_params.get('threshold_tuning_method', 'unknown')}")
        print(f"  Optimal threshold: {training_params.get('optimal_threshold', 'unknown')}")
    
    # Dataset info
    if metadata:
        print(f"\nDataset Info:")
        print(f"  Training samples: {metadata.get('num_train_samples', 'unknown')}")
        print(f"  Test samples (out-of-domain): {metadata.get('num_test_samples_out_of_domain', 'unknown')}")
        if 'num_test_samples_in_domain' in metadata:
            print(f"  Test samples (in-domain): {metadata.get('num_test_samples_in_domain', 'unknown')}")


def compare_features(model1_dir, model2_dir):
    """Compare features between two models."""
    print("\n" + "=" * 80)
    print("COMPARING FEATURES BETWEEN MODELS")
    print("=" * 80)
    
    # Load both models
    params1, metadata1 = load_model_features(model1_dir)
    params2, metadata2 = load_model_features(model2_dir)
    
    # Extract feature info
    features1 = set(params1.get("selected_feature_indices", []))
    features2 = set(params2.get("selected_feature_indices", []))
    
    if not features1 and not features2:
        print("\nNeither model uses feature selection.")
        return
    
    if not features1:
        print(f"\nModel 1 does not use feature selection (all features)")
        print(f"Model 2 uses {len(features2)} selected features")
        return
    
    if not features2:
        print(f"\nModel 1 uses {len(features1)} selected features")
        print(f"Model 2 does not use feature selection (all features)")
        return
    
    # Compare
    common = features1.intersection(features2)
    only_in_1 = features1 - features2
    only_in_2 = features2 - features1
    
    print(f"\nModel 1: {os.path.basename(model1_dir)}")
    print(f"  Selected features: {len(features1)}")
    
    print(f"\nModel 2: {os.path.basename(model2_dir)}")
    print(f"  Selected features: {len(features2)}")
    
    print(f"\nComparison:")
    print(f"  Features in common: {len(common)} ({100*len(common)/max(len(features1), len(features2)):.1f}% overlap)")
    print(f"  Only in Model 1: {len(only_in_1)}")
    print(f"  Only in Model 2: {len(only_in_2)}")
    
    # Check if features are identical
    if features1 == features2:
        print(f"\n✅ IDENTICAL FEATURE SELECTION")
        print(f"   Both models use exactly the same {len(features1)} features.")
        print(f"   You can safely compare their performance directly.")
    else:
        print(f"\n⚠️  DIFFERENT FEATURE SELECTIONS")
        print(f"   Models use different feature subsets.")
        print(f"   Performance comparison may be affected by feature differences.")
    
    # Show feature names if available
    if "selected_feature_names" in params1 and "selected_feature_names" in params2:
        names1 = set(params1["selected_feature_names"])
        names2 = set(params2["selected_feature_names"])
        common_names = names1.intersection(names2)
        
        if len(common_names) < len(common):
            print(f"\n  Common features:")
            for name in sorted(common_names):
                print(f"    - {name}")


def main():
    parser = argparse.ArgumentParser(
        description="Inspect features used by a saved model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    parser.add_argument(
        "model_dir",
        help="Path to the saved model directory"
    )
    parser.add_argument(
        "--compare", "-c",
        dest="compare_dir",
        help="Path to another model directory to compare features with"
    )
    
    args = parser.parse_args()
    
    # Check if model directory exists
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory not found: {args.model_dir}")
        sys.exit(1)
    
    # Load and print features
    try:
        training_params, metadata = load_model_features(args.model_dir)
        print_feature_info(args.model_dir, training_params, metadata)
        
        # Compare if requested
        if args.compare_dir:
            if not os.path.exists(args.compare_dir):
                print(f"\nError: Comparison model directory not found: {args.compare_dir}")
                sys.exit(1)
            compare_features(args.model_dir, args.compare_dir)
        
        # Print usage hint for feature reuse
        if "selected_feature_indices" in training_params:
            print("\n" + "=" * 80)
            print("REUSING THESE FEATURES IN ANOTHER MODEL")
            print("=" * 80)
            print(f"\nTo train a different model type with these same {len(training_params['selected_feature_indices'])} features,")
            print(f"set this parameter in train_10_datasets_evaluate_TEST.py:")
            print(f"\n  load_features_from_model = \"{args.model_dir}\"")
            print(f"\nThen run the script with your desired model_type (e.g., 'logistic_regression' or 'svm').")
            print(f"The new model will use the exact same feature subset as this model.")
        
    except Exception as e:
        print(f"\nError: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

