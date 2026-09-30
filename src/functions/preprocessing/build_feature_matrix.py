from __future__ import annotations
import numpy as np
import pandas as pd
from typing import List, Dict, Optional

from functions.preprocessing.feature_extraction import FeatureExtractor, segment_all_signals


def build_feature_matrix(
    data: pd.DataFrame,
    signal_names: List[str],
    start_indices: np.ndarray,
    end_indices: np.ndarray,
    labels: np.ndarray,
    num_sub_windows: Optional[int] = None,
    sub_window_mode: str = "concatenate"
) -> Dict[str, np.ndarray]:
    """
    Build feature matrix X_tot and label vector Y_tot from segmented signals.

    Parameters
    ----------
    data : pd.DataFrame
        Full dataset with all signals as columns.
    signal_names : List[str]
        List of signal column names to process.
    start_indices : np.ndarray
        Start indices for each frame.
    end_indices : np.ndarray
        End indices for each frame.
    labels : np.ndarray
        Labels for each frame (1 = faulty, 0 = non-faulty).
    num_sub_windows : int, optional
        Number of sub-windows to split each segment into (e.g., 5 for 5 sub-windows).
        If None, features are extracted from the entire segment (default behavior).
        The segment will be divided into exactly this many equal parts, ensuring
        consistent feature counts across all datasets.
    sub_window_mode : str, optional
        How to combine sub-window features:
        - "concatenate": Concatenate all sub-window features, preserving temporal order.
                        This preserves the most information and allows models to learn temporal patterns.
        - "aggregate": Compute statistics (mean, std, min, max) across sub-windows.
                      Loses temporal ordering but provides summary statistics.
        - "temporal": Base features from entire segment + temporal variation features (slopes, trends).
                      Recommended mode for fault prediction.

    Returns
    -------
    Dict[str, np.ndarray]
        Dictionary containing:
        - 'X_tot': Feature matrix (num_frames, num_features)
        - 'Y_tot': Label vector (num_frames,)
        - 'feature_names': List of feature names for each column
    """
    num_frames = len(start_indices)
    
    if num_frames != len(end_indices) or num_frames != len(labels):
        raise ValueError("start_indices, end_indices, and labels must have the same length")
    
    # Initialize feature extractor (no truncation - use all computed features)
    feature_extractor = FeatureExtractor(
        features_per_signal=None,  # None means use all computed features
        num_sub_windows=num_sub_windows,
        sub_window_mode=sub_window_mode
    )
    
    # Check which signals are available in the data
    available_signals = [sig for sig in signal_names if sig in data.columns]
    missing_signals = [sig for sig in signal_names if sig not in data.columns]
    
    if missing_signals:
        print(f"Warning: The following signals are not in the data: {missing_signals}")
    
    if len(available_signals) == 0:
        raise ValueError("No valid signals found in the data")
    
    sub_window_info = ""
    if num_sub_windows is not None and num_sub_windows > 1:
        sub_window_info = f" with {num_sub_windows} sub-windows ({sub_window_mode} mode)"
    print(f"Processing {len(available_signals)} signals (using all computed features){sub_window_info}...")
    
    # Extract features for each signal
    all_feature_matrices = []
    feature_names_list = []
    
    for signal_name in available_signals:
        # Get signal data
        signal_data = data[signal_name].astype(float).to_numpy()
        
        # Segment signal using start/end indices
        signal_segments = segment_all_signals(
            data=signal_data,
            start_indices=start_indices,
            end_indices=end_indices
        )
        
        # Extract features for all segments
        feature_matrix = feature_extractor.extract_features_batch(signal_segments)
        
        # Store feature matrix
        all_feature_matrices.append(feature_matrix)
        
        # Generate feature names for this signal based on actual number of features
        num_features = feature_matrix.shape[1]
        for feat_idx in range(num_features):
            feature_names_list.append(f"{signal_name}_feat_{feat_idx}")
    
    # Concatenate all feature matrices horizontally
    X_tot = np.hstack(all_feature_matrices)
    
    # Y_tot is just the labels
    Y_tot = labels.copy()
    
    print(f"Built feature matrix: {X_tot.shape[0]} frames × {X_tot.shape[1]} features")
    print(f"Label distribution: {np.sum(Y_tot == 1)} faulty, {np.sum(Y_tot == 0)} non-faulty")
    
    return {
        'X_tot': X_tot,
        'Y_tot': Y_tot,
        'feature_names': feature_names_list,
        'signal_names': available_signals
    }

