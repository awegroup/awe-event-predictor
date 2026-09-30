"""
Script to analyze feature importance using multiple methods.

Can be run in two ways:
1. From saved model directory
2. From pipeline results (if run interactively)

Usage:
    python analyze_feature_importance.py <model_dir> [--top-n 20] [--save <output_path>]
    
    Or import and use:
        from functions.ml.feature_importance import *
        results = analyze_feature_importance_from_model(model_dir, top_n=20)
"""

import sys
import argparse
import numpy as np
from typing import Dict, List, Optional
import warnings
warnings.filterwarnings('ignore')

from functions.ml.model_persistence import load_model
from functions.ml.feature_importance import (
    compute_permutation_importance,
    compute_mutual_information,
    compute_shap_importance,
    plot_feature_importance_comparison,
    compare_importance_methods
)


def analyze_feature_importance_from_model(
    model_dir: str,
    top_n: int = 20,
    use_shap: bool = True,
    n_repeats: int = 10,
    shap_n_samples: int = 100,
    save_plot: Optional[str] = None
) -> Dict:
    """
    Analyze feature importance from a saved model.
    
    Parameters
    ----------
    model_dir : str
        Path to saved model directory.
    top_n : int
        Number of top features to analyze.
    use_shap : bool
        Whether to compute SHAP importance (can be slow).
    n_repeats : int
        Number of permutations for permutation importance.
    shap_n_samples : int
        Number of samples to use for SHAP computation.
    save_plot : str, optional
        Path to save the comparison plot.
        
    Returns
    -------
    dict
        Dictionary with all importance results and comparison data.
    """
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    # Load model and data
    print(f"\nLoading model from: {model_dir}")
    model_data = load_model(model_dir)
    
    # Support both new generic 'model' key and legacy 'svm_model' key
    model = model_data.get('model') or model_data.get('svm_model')
    if model is None:
        raise ValueError("Model not found in saved data. Expected 'model' or 'svm_model' key.")
    
    model_type = model_data.get('model_type', 'svm')  # Default to 'svm' for backward compatibility
    scaler = model_data['scaler']
    X_test = model_data.get('X_test')  # May not be saved
    y_test = model_data['y_test']
    feature_names = model_data.get('feature_names', [])
    
    # If X_test not saved, we cannot compute importance
    if X_test is None:
        print("  ✗ Error: X_test not found in saved model.")
        print("  Feature importance analysis requires test features (X_test).")
        print("  Please re-save the model with X_test included, or run analysis")
        print("  directly from pipeline results.")
        return None
    
    # Scale features if needed (should already be scaled, but just in case)
    # Actually, if loaded from saved model, X_test should already be scaled
    # But let's check if scaler is available and apply if needed
    
    print(f"\nAnalyzing {len(feature_names)} features...")
    print(f"  Test set size: {len(X_test)} samples")
    
    # Compute importance using different methods
    importance_results = []
    
    # 1. Permutation Importance
    print("\n" + "-"*70)
    # Ensure NaN-free data
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    
    perm_result = compute_permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring='balanced_accuracy'
    )
    importance_results.append(perm_result)
    
    # 2. Mutual Information
    print("\n" + "-"*70)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    mi_result = compute_mutual_information(X_test, y_test)
    importance_results.append(mi_result)
    
    # 3. SHAP Importance (optional, can be slow)
    if use_shap:
        print("\n" + "-"*70)
        try:
            shap_result = compute_shap_importance(
                model, X_test, feature_names,
                n_samples=shap_n_samples
            )
            importance_results.append(shap_result)
        except ImportError:
            print("  SHAP not available, skipping...")
        except Exception as e:
            print(f"  Error computing SHAP: {e}")
            print("  Continuing with other methods...")
    
    # Compare methods
    print("\n" + "="*70)
    print("COMPARING METHODS")
    print("="*70)
    
    comparison_df, consensus_df = compare_importance_methods(
        importance_results, feature_names, top_n=top_n
    )
    
    # Print top features
    print(f"\nTop {top_n} Features (Consensus Ranking):")
    print("-"*70)
    for i, (_, row) in enumerate(consensus_df.head(top_n).iterrows(), 1):
        print(f"{i:2d}. {row['signal']:30s} {row['feature_type']:25s} "
              f"(score: {row['score']:.4f}, avg_rank: {row['rank']:.1f})")
    
    # Plot comparison
    if len(importance_results) > 0:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        plot_feature_importance_comparison(
            importance_results, feature_names,
            top_n=top_n, save_path=save_plot
        )
    
    return {
        'importance_results': importance_results,
        'comparison_df': comparison_df,
        'consensus_df': consensus_df,
        'feature_names': feature_names
    }


def analyze_feature_importance_from_pipeline(
    model,
    scaler,
    X_test: np.ndarray,
    y_test: np.ndarray,
    feature_names: List[str],
    top_n: int = 20,
    use_shap: bool = True,
    n_repeats: int = 10,
    shap_n_samples: int = 100,
    save_plot: Optional[str] = None
) -> Dict:
    """
    Analyze feature importance from pipeline results.
    
    Parameters
    ----------
    model
        Trained model (SVM, Logistic Regression, etc.).
    scaler
        Fitted scaler.
    X_test : np.ndarray
        Scaled test features.
    y_test : np.ndarray
        Test labels.
    feature_names : list
        List of feature names.
    top_n : int
        Number of top features to analyze.
    use_shap : bool
        Whether to compute SHAP importance.
    n_repeats : int
        Number of permutations for permutation importance.
    shap_n_samples : int
        Number of samples for SHAP computation.
    save_plot : str, optional
        Path to save the comparison plot.
        
    Returns
    -------
    dict
        Dictionary with all importance results and comparison data.
    """
    print("="*70)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*70)
    
    print(f"\nAnalyzing {len(feature_names)} features...")
    print(f"  Test set size: {len(X_test)} samples")
    
    # Compute importance using different methods
    importance_results = []
    
    # 1. Permutation Importance
    print("\n" + "-"*70)
    perm_result = compute_permutation_importance(
        model, X_test, y_test,
        n_repeats=n_repeats,
        scoring='balanced_accuracy'
    )
    importance_results.append(perm_result)
    
    # 2. Mutual Information
    print("\n" + "-"*70)
    mi_result = compute_mutual_information(X_test, y_test)
    importance_results.append(mi_result)
    
    # 3. SHAP Importance (optional)
    if use_shap:
        print("\n" + "-"*70)
        try:
            shap_result = compute_shap_importance(
                model, X_test, feature_names,
                n_samples=shap_n_samples
            )
            importance_results.append(shap_result)
        except ImportError:
            print("  SHAP not available, skipping...")
        except Exception as e:
            print(f"  Error computing SHAP: {e}")
            print("  Continuing with other methods...")
    
    # Compare methods
    print("\n" + "="*70)
    print("COMPARING METHODS")
    print("="*70)
    
    comparison_df, consensus_df = compare_importance_methods(
        importance_results, feature_names, top_n=top_n
    )
    
    # Print top features
    print(f"\nTop {top_n} Features (Consensus Ranking):")
    print("-"*70)
    for i, (_, row) in enumerate(consensus_df.head(top_n).iterrows(), 1):
        print(f"{i:2d}. {row['signal']:30s} {row['feature_type']:25s} "
              f"(score: {row['score']:.4f}, avg_rank: {row['rank']:.1f})")
    
    # Plot comparison
    if len(importance_results) > 0:
        print("\n" + "="*70)
        print("GENERATING PLOTS")
        print("="*70)
        plot_feature_importance_comparison(
            importance_results, feature_names,
            top_n=top_n, save_path=save_plot
        )
    
    return {
        'importance_results': importance_results,
        'comparison_df': comparison_df,
        'consensus_df': consensus_df,
        'feature_names': feature_names
    }


def main():
    parser = argparse.ArgumentParser(
        description='Analyze feature importance using multiple methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'model_dir',
        type=str,
        help='Path to directory containing saved model'
    )
    
    parser.add_argument(
        '--top-n',
        type=int,
        default=10,
        help='Number of top features to analyze (default: 10)'
    )
    
    parser.add_argument(
        '--no-shap',
        action='store_true',
        help='Skip SHAP computation (faster)'
    )
    
    parser.add_argument(
        '--n-repeats',
        type=int,
        default=10,
        help='Number of permutations for permutation importance (default: 10)'
    )
    
    parser.add_argument(
        '--shap-samples',
        type=int,
        default=100,
        help='Number of samples for SHAP computation (default: 100)'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the comparison plot'
    )
    
    args = parser.parse_args()
    
    try:
        results = analyze_feature_importance_from_model(
            args.model_dir,
            top_n=args.top_n,
            use_shap=not args.no_shap,
            n_repeats=args.n_repeats,
            shap_n_samples=args.shap_samples,
            save_plot=args.save
        )
        
        if results:
            print("\n✓ Feature importance analysis complete!")
        else:
            print("\n✗ Analysis failed - check error messages above")
            sys.exit(1)
            
    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

