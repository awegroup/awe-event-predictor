from __future__ import annotations
import numpy as np
from typing import Tuple, List
from sklearn.model_selection import train_test_split


def time_based_train_test_split(
    X: np.ndarray,
    y: np.ndarray,
    frame_times: List[np.ndarray],
    test_size: float = 0.2,
    stratify: bool = True,
    min_test_samples: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split data into train and test sets based on time, ensuring no temporal leakage.
    Preserves class distribution as much as possible.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (num_frames, num_features).
    y : np.ndarray
        Labels (num_frames,).
    frame_times : List[np.ndarray]
        List of time arrays, one per frame. We use the start time of each frame.
    test_size : float
        Proportion of data to use for testing (default: 0.2).
    stratify : bool
        If True, attempts to preserve class distribution in train/test split.
    min_test_samples : int
        Minimum number of samples required in test set for each class.

    Returns
    -------
    X_train, X_test, y_train, y_test, train_indices, test_indices
        Split data and indices for reference.
    """
    num_frames = len(X)
    
    if len(y) != num_frames:
        raise ValueError("X and y must have the same number of samples")
    
    if len(frame_times) != num_frames:
        raise ValueError("frame_times must have the same length as X")

    # Extract start time for each frame (use first element of each frame's time array)
    frame_start_times = np.array([ft[0] if len(ft) > 0 else 0.0 for ft in frame_times])
    
    # Sort by time to ensure temporal order
    sorted_indices = np.argsort(frame_start_times)
    sorted_times = frame_start_times[sorted_indices]
    sorted_X = X[sorted_indices]
    sorted_y = y[sorted_indices]
    
    # Find split point that preserves class distribution
    num_test = int(np.ceil(num_frames * test_size))
    num_train = num_frames - num_test
    
    if stratify:
        # Count classes in sorted data
        unique_classes, class_counts = np.unique(sorted_y, return_counts=True)
        num_classes = len(unique_classes)
        
        # Calculate desired test samples per class
        test_samples_per_class = {}
        train_samples_per_class = {}
        
        for cls, count in zip(unique_classes, class_counts):
            test_samples_per_class[cls] = max(
                min_test_samples,
                int(np.ceil(count * test_size))
            )
            train_samples_per_class[cls] = count - test_samples_per_class[cls]
        
        # Find split point that best preserves distribution
        # Strategy: Take the last N samples as test, but ensure we have enough of each class
        best_split_idx = num_train
        
        # Check if we can get enough samples of each class in test set
        test_y = sorted_y[-num_test:]
        test_class_counts = {cls: np.sum(test_y == cls) for cls in unique_classes}
        
        # If any class is underrepresented, adjust split point
        for cls in unique_classes:
            if test_class_counts[cls] < min_test_samples:
                # Need to find a split that includes more of this class
                # Find indices where this class appears
                class_indices = np.where(sorted_y == cls)[0]
                
                # Ensure we have at least min_test_samples in test
                if len(class_indices) >= min_test_samples:
                    # Find the earliest index that gives us enough samples
                    required_test_start = class_indices[-min_test_samples]
                    # Adjust split to ensure this class is well represented
                    if required_test_start < best_split_idx:
                        # We need to move split earlier to include more of this class
                        # But we want to keep test_size approximately correct
                        # So we'll use the split that gives us at least min_test_samples
                        pass  # Keep current split, will handle below
        
        # Final check: ensure minimum samples per class in test
        test_y_final = sorted_y[best_split_idx:]
        for cls in unique_classes:
            if np.sum(test_y_final == cls) < min_test_samples:
                # Find where to split to get enough of this class
                class_indices = np.where(sorted_y == cls)[0]
                if len(class_indices) >= min_test_samples:
                    # Need at least min_test_samples of this class in test
                    earliest_test_start = class_indices[-min_test_samples]
                    if earliest_test_start < best_split_idx:
                        best_split_idx = earliest_test_start
        
        # Ensure we don't make test set too large
        if num_frames - best_split_idx > num_test * 1.5:
            best_split_idx = num_frames - num_test
    else:
        # Simple time-based split: last test_size% goes to test
        best_split_idx = num_train
    
    # Perform the split (using sorted indices)
    train_sorted_indices = sorted_indices[:best_split_idx]
    test_sorted_indices = sorted_indices[best_split_idx:]
    
    X_train = sorted_X[:best_split_idx]
    X_test = sorted_X[best_split_idx:]
    y_train = sorted_y[:best_split_idx]
    y_test = sorted_y[best_split_idx:]
    
    # Return original indices (before sorting)
    train_indices = train_sorted_indices
    test_indices = test_sorted_indices
    
    # Print statistics
    print(f"\nTime-based Train-Test Split:")
    print(f"  Total frames: {num_frames}")
    print(f"  Train frames: {len(X_train)} ({len(X_train)/num_frames*100:.1f}%)")
    print(f"  Test frames: {len(X_test)} ({len(X_test)/num_frames*100:.1f}%)")
    print(f"  Train time range: [{sorted_times[0]:.2f}, {sorted_times[best_split_idx-1]:.2f}] seconds")
    print(f"  Test time range: [{sorted_times[best_split_idx]:.2f}, {sorted_times[-1]:.2f}] seconds")
    
    print(f"\n  Class distribution in train:")
    unique_train, counts_train = np.unique(y_train, return_counts=True)
    for cls, count in zip(unique_train, counts_train):
        print(f"    Class {cls}: {count} ({count/len(y_train)*100:.1f}%)")
    
    print(f"\n  Class distribution in test:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for cls, count in zip(unique_test, counts_test):
        print(f"    Class {cls}: {count} ({count/len(y_test)*100:.1f}%)")
    
    return X_train, X_test, y_train, y_test, train_indices, test_indices

