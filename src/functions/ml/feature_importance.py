from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

from sklearn.inspection import permutation_importance
from sklearn.feature_selection import mutual_info_classif
import matplotlib.pyplot as plt
import seaborn as sns


# Feature type mapping (based on feature_extraction.py)
FEATURE_NAMES = {
    0: "mean",
    1: "std",
    2: "min",
    3: "max",
    4: "median",
    5: "skewness",
    6: "kurtosis",
    7: "Q1",
    8: "Q3",
    9: "IQR",
    10: "range",
    11: "RMS",
    12: "variance",
    13: "MAD",
    14: "zero_crossing_rate",
    15: "energy",
    16: "peak_to_peak",
    17: "crest_factor",
    18: "dominant_freq_mag",
    19: "mean_freq_mag",
    20: "spectral_centroid",
    21: "spectral_spread",
    22: "spectral_rolloff",
    23: "spectral_flux",
    24: "spectral_entropy"
}


def parse_feature_name(feature_name: str) -> Tuple[str, str]:
    """
    Parse feature name to extract signal name and feature type.
    
    Parameters
    ----------
    feature_name : str
        Feature name in format "{signal_name}_feat_{feat_idx}"
        
    Returns
    -------
    signal_name : str
        Name of the signal
    feature_type : str
        Type of feature (e.g., "mean", "std", etc.)
    """
    parts = feature_name.split("_feat_")
    if len(parts) != 2:
        return feature_name, "unknown"
    
    signal_name = parts[0]
    feat_idx = int(parts[1])
    feature_type = FEATURE_NAMES.get(feat_idx, f"feat_{feat_idx}")
    
    return signal_name, feature_type


def compute_permutation_importance(
    model,
    X: np.ndarray,
    y: np.ndarray,
    n_repeats: int = 10,
    scoring: str = 'balanced_accuracy',
    random_state: int = 42,
    n_jobs: int = -1
) -> Dict:
    """
    Compute permutation importance for features.
    
    Permutation importance measures the decrease in model performance
    when a feature is randomly shuffled. Higher decrease = more important.
    
    Parameters
    ----------
    model
        Trained model (must have predict or predict_proba method).
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    n_repeats : int
        Number of times to permute each feature.
    scoring : str
        Scoring metric to use.
    random_state : int
        Random seed.
    n_jobs : int
        Number of parallel jobs.
        
    Returns
    -------
    dict
        Dictionary with importance scores, mean, std, and rankings.
    """
    print("\nComputing Permutation Importance...")
    print(f"  Using {n_repeats} permutations per feature...")
    
    result = permutation_importance(
        model, X, y,
        n_repeats=n_repeats,
        scoring=scoring,
        random_state=random_state,
        n_jobs=n_jobs
    )
    
    importances_mean = result.importances_mean
    importances_std = result.importances_std
    
    # Rank features by importance (descending)
    ranked_indices = np.argsort(importances_mean)[::-1]
    
    return {
        'method': 'permutation_importance',
        'scores': importances_mean,
        'std': importances_std,
        'ranked_indices': ranked_indices,
        'raw_result': result
    }


def compute_mutual_information(
    X: np.ndarray,
    y: np.ndarray,
    random_state: int = 42,
    discrete_features: bool = False
) -> Dict:
    """
    Compute mutual information between features and target.
    
    Mutual information measures the statistical dependency between
    features and target. Higher MI = more informative feature.
    
    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Target labels.
    random_state : int
        Random seed.
    discrete_features : bool
        Whether features are discrete.
        
    Returns
    -------
    dict
        Dictionary with MI scores and rankings.
    """
    print("\nComputing Mutual Information...")
    
    mi_scores = mutual_info_classif(
        X, y,
        random_state=random_state,
        discrete_features=discrete_features
    )
    
    # Rank features by MI (descending)
    ranked_indices = np.argsort(mi_scores)[::-1]
    
    return {
        'method': 'mutual_information',
        'scores': mi_scores,
        'ranked_indices': ranked_indices
    }


def compute_shap_importance(
    model,
    X: np.ndarray,
    feature_names: Optional[List[str]] = None,
    n_samples: int = 100,
    random_state: int = 42
) -> Dict:
    """
    Compute SHAP (SHapley Additive exPlanations) values for feature importance.
    
    SHAP values provide a unified measure of feature importance based on
    game theory. Uses KernelSHAP for model-agnostic explanation.
    
    Parameters
    ----------
    model
        Trained model (must have predict_proba method).
    X : np.ndarray
        Feature matrix (will sample n_samples for efficiency).
    feature_names : list, optional
        Names of features.
    n_samples : int
        Number of samples to use for SHAP computation (for efficiency).
    random_state : int
        Random seed.
        
    Returns
    -------
    dict
        Dictionary with SHAP importance scores and rankings.
    """
    try:
        import shap
    except ImportError:
        raise ImportError(
            "SHAP library not installed. Install with: pip install shap\n"
            "Skipping SHAP importance computation."
        )
    
    print("\nComputing SHAP Importance...")
    print(f"  Using {n_samples} samples for efficiency...")
    
    # Sample data for efficiency (SHAP can be slow on large datasets)
    np.random.seed(random_state)
    if len(X) > n_samples:
        sample_indices = np.random.choice(len(X), n_samples, replace=False)
        X_sample = X[sample_indices]
    else:
        X_sample = X
    
    # Use KernelExplainer for model-agnostic SHAP values
    explainer = shap.KernelExplainer(model.predict_proba, X_sample)
    shap_values = explainer.shap_values(X_sample, nsamples=100)
    
    # For binary classification, use the positive class (class 1)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Use class 1 (faulty)
    elif shap_values.ndim == 3:
        shap_values = shap_values[:, :, 1]  # shap >= 0.45: (samples, features, classes)
    
    # Compute mean absolute SHAP value per feature
    shap_importance = np.mean(np.abs(shap_values), axis=0)
    
    # Rank features by importance (descending)
    ranked_indices = np.argsort(shap_importance)[::-1]
    
    return {
        'method': 'shap',
        'scores': shap_importance,
        'ranked_indices': ranked_indices,
        'shap_values': shap_values,
        'explainer': explainer
    }


def compare_importance_methods(
    importance_results: List[Dict],
    feature_names: List[str],
    top_n: int = 20
) -> pd.DataFrame:
    """
    Compare and analyze consistency between different importance methods.
    
    Parameters
    ----------
    importance_results : list
        List of importance result dictionaries from different methods.
    feature_names : list
        Names of features.
    top_n : int
        Number of top features to analyze.
        
    Returns
    -------
    pd.DataFrame
        Comparison dataframe with rankings from all methods.
    """
    print(f"\nComparing importance methods (top {top_n} features)...")
    
    # Create comparison dataframe
    comparison_data = []
    
    for result in importance_results:
        method = result['method']
        ranked_indices = result['ranked_indices']
        scores = result['scores']
        
        for rank, idx in enumerate(ranked_indices[:top_n]):
            # Convert numpy scalar/array to Python int
            if isinstance(idx, np.ndarray):
                idx = int(idx.item())
            else:
                idx = int(idx)
            signal_name, feature_type = parse_feature_name(feature_names[idx])
            
            comparison_data.append({
                'method': method,
                'rank': rank + 1,
                'feature_index': idx,
                'feature_name': feature_names[idx],
                'signal': signal_name,
                'feature_type': feature_type,
                'score': float(scores[idx])  # Also convert score to float
            })
    
    df = pd.DataFrame(comparison_data)
    
    # Compute consensus ranking (average rank across methods)
    consensus = df.groupby('feature_index').agg({
        'rank': 'mean',
        'feature_name': 'first',
        'signal': 'first',
        'feature_type': 'first',
        'score': 'mean'
    }).sort_values('rank').reset_index()
    
    consensus['consensus_rank'] = range(1, len(consensus) + 1)
    
    return df, consensus


def plot_feature_importance_comparison(
    importance_results: List[Dict],
    feature_names: List[str],
    top_n: int = 20,
    save_path: Optional[str] = None,
    figsize: tuple = (16, 12)
):
    """
    Plot comprehensive comparison of feature importance methods.
    
    Parameters
    ----------
    importance_results : list
        List of importance result dictionaries.
    feature_names : list
        Names of features.
    top_n : int
        Number of top features to display.
    save_path : str, optional
        Path to save the figure.
    figsize : tuple
        Figure size.
    """
    # Get comparison data
    comparison_df, consensus_df = compare_importance_methods(
        importance_results, feature_names, top_n=top_n
    )
    
    num_methods = len(importance_results)
    
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(3, max(2, num_methods), hspace=0.3, wspace=0.3)
    
    # ============================================================
    # 1. Top features by each method (bar charts)
    # ============================================================
    for i, result in enumerate(importance_results):
        ax = fig.add_subplot(gs[0, i])
        
        method = result['method']
        ranked_indices = result['ranked_indices'][:top_n]
        scores = result['scores']
        
        # Get feature labels (signal + feature type)
        labels = []
        for idx in ranked_indices:
            # Convert numpy scalar/array to Python int
            if isinstance(idx, np.ndarray):
                idx = int(idx.item())
            else:
                idx = int(idx)
            signal, feat_type = parse_feature_name(feature_names[idx])
            labels.append(f"{signal}\n{feat_type}")
        
        y_pos = np.arange(len(ranked_indices))
        bars = ax.barh(y_pos, scores[ranked_indices], alpha=0.8)
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_xlabel('Importance Score', fontweight='bold')
        ax.set_title(f'{method.replace("_", " ").title()}\n(Top {top_n})', 
                    fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(axis='x', alpha=0.3)
    
    # ============================================================
    # 2. Consensus ranking (average across methods)
    # ============================================================
    ax = fig.add_subplot(gs[1, :])
    
    top_consensus = consensus_df.head(top_n)
    
    labels = []
    for _, row in top_consensus.iterrows():
        labels.append(f"{row['signal']}\n{row['feature_type']}")
    
    y_pos = np.arange(len(top_consensus))
    bars = ax.barh(y_pos, top_consensus['score'], alpha=0.8, color='#2ecc71')
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Average Importance Score', fontweight='bold')
    ax.set_title(f'Consensus Ranking (Average across {num_methods} methods)\nTop {top_n} Features', 
                fontweight='bold', fontsize=14)
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    
    # ============================================================
    # 3. Ranking correlation heatmap
    # ============================================================
    ax = fig.add_subplot(gs[2, 0])
    
    # Create ranking matrix
    ranking_matrix = np.zeros((len(feature_names), num_methods))
    for i, result in enumerate(importance_results):
        ranked_indices = result['ranked_indices']
        for rank, idx in enumerate(ranked_indices):
            # Convert numpy scalar/array to Python int
            if isinstance(idx, np.ndarray):
                idx = int(idx.item())
            else:
                idx = int(idx)
            ranking_matrix[idx, i] = rank + 1
    
    # Compute correlation between rankings
    ranking_df = pd.DataFrame(ranking_matrix, columns=[r['method'] for r in importance_results])
    corr_matrix = ranking_df.corr()
    
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        cbar_kws={'label': 'Correlation'},
        ax=ax
    )
    ax.set_title('Ranking Correlation Between Methods', fontweight='bold', fontsize=12)
    
    # ============================================================
    # 4. Feature overlap (Venn-like analysis)
    # ============================================================
    ax = fig.add_subplot(gs[2, 1])
    
    # Get top N features from each method
    top_features_by_method = {}
    for result in importance_results:
        method = result['method']
        # Convert numpy array indices to Python ints for set creation
        top_indices_list = []
        for idx in result['ranked_indices'][:top_n]:
            if isinstance(idx, np.ndarray):
                top_indices_list.append(int(idx.item()))
            else:
                top_indices_list.append(int(idx))
        top_features_by_method[method] = set(top_indices_list)
    
    # Count overlaps
    all_features = set(range(len(feature_names)))
    overlap_counts = []
    method_names = list(top_features_by_method.keys())
    
    for i, method1 in enumerate(method_names):
        for j, method2 in enumerate(method_names):
            if i < j:
                overlap = len(top_features_by_method[method1] & top_features_by_method[method2])
                overlap_counts.append({
                    'method1': method1,
                    'method2': method2,
                    'overlap': overlap,
                    'overlap_pct': overlap / top_n * 100
                })
    
    if overlap_counts:
        overlap_df = pd.DataFrame(overlap_counts)
        
        # Create bar chart
        labels = [f"{row['method1']}\nvs\n{row['method2']}" for _, row in overlap_df.iterrows()]
        bars = ax.bar(range(len(overlap_df)), overlap_df['overlap'], alpha=0.8, color='#3498db')
        
        ax.set_xticks(range(len(overlap_df)))
        ax.set_xticklabels(labels, fontsize=9, rotation=0)
        ax.set_ylabel('Number of Overlapping Features', fontweight='bold')
        ax.set_title(f'Top {top_n} Feature Overlap Between Methods', fontweight='bold', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Add percentage labels
        for i, (bar, pct) in enumerate(zip(bars, overlap_df['overlap_pct'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}\n({pct:.1f}%)',
                   ha='center', va='bottom', fontsize=9)
    
    fig.suptitle('Feature Importance Analysis: Multi-Method Comparison', 
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.tight_layout(rect=[0, 0, 1, 0.99])
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\nFeature importance plot saved to: {save_path}")
    else:
        plt.show()
    
    return comparison_df, consensus_df

