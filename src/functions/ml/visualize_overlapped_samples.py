#!/usr/bin/env python3
"""
Script to visualize samples grouped by predicted probability ranges and prediction categories,
with all samples for each signal plotted in an overlapped manner.

This script creates a hierarchical directory structure:
- Probability range folders (e.g., prob_0.0_to_0.4)
  - Prediction category folders (TP, TN, FP, FN)
    - PNG plots for each signal (all samples overlapped)
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Dict, List
import pickle
import pandas as pd
from collections import defaultdict

from functions.ml.model_persistence import load_model


def get_prediction_category(y_true: int, y_pred: int) -> str:
    """
    Determine prediction category from true and predicted labels.
    
    Parameters
    ----------
    y_true : int
        True label (0 or 1).
    y_pred : int
        Predicted label (0 or 1).
    
    Returns
    -------
    str
        Prediction category: 'TP', 'TN', 'FP', or 'FN'.
    """
    if y_true == 1 and y_pred == 1:
        return "TP"
    elif y_true == 0 and y_pred == 0:
        return "TN"
    elif y_true == 0 and y_pred == 1:
        return "FP"
    else:  # y_true == 1 and y_pred == 0
        return "FN"


def plot_overlapped_signals(
    signal_name: str,
    frames_data: List[Dict],
    window_length: float,
    pre_peak_time: float,
    post_peak_time: float,
    threshold_on_T: Optional[float] = None,
    figsize: tuple = (12, 8),
    title_suffix: str = ""
) -> None:
    """
    Plot multiple frames for a signal in an overlapped manner.
    
    Parameters
    ----------
    signal_name : str
        Name of the signal to plot.
    frames_data : List[Dict]
        List of dictionaries, each containing:
        - 'signal': numpy array of signal values for the frame
        - 'time': numpy array of time values (normalized, starting from 0)
    window_length : float
        Frame duration in seconds (historical window).
    pre_peak_time : float
        Time before the peak in seconds.
    post_peak_time : float
        Time after the peak in seconds.
    threshold_on_T : Optional[float]
        Threshold value to plot as horizontal line (for ground_tether_force).
    figsize : tuple
        Figure size.
    title_suffix : str
        Additional text to add to the title.
    """
    if len(frames_data) == 0:
        return None, None
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot each frame
    for frame_data in frames_data:
        signal = frame_data['signal']
        time = frame_data['time']
        
        if len(signal) == 0:
            continue
        
        ax.plot(time, signal, alpha=0.5, linewidth=1.5, color='blue')
    
    # Add vertical markers for different time segments
    # Marker 1: End of historical window (start of pre_peak_time)
    ax.axvline(window_length, color='green', linestyle=':', linewidth=2, 
               label=f'Window end ({window_length:.1f}s)', alpha=0.7)
    
    # Marker 2: Peak location (end of pre_peak_time, start of post_peak_time)
    peak_time = window_length + pre_peak_time
    ax.axvline(peak_time, color='orange', linestyle='--', linewidth=2,
               label=f'Peak (~{peak_time:.1f}s)', alpha=0.7)
    
    # Add threshold line if applicable
    if signal_name == "ground_tether_force" and threshold_on_T is not None:
        ax.axhline(threshold_on_T, color='red', linestyle='--', linewidth=2,
                  label=f'Threshold = {threshold_on_T:.1f}')
    
    # Labels and title
    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
    ax.set_ylabel(signal_name, fontsize=12, fontweight="bold")
    total_window = window_length + pre_peak_time + post_peak_time
    title = f"Overlapped {signal_name} ({len(frames_data)} samples, {total_window:.1f}s window)"
    if title_suffix:
        title += f"\n{title_suffix}"
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3)
    
    ax.legend(loc='best', fontsize=10)
    
    plt.tight_layout()
    
    return fig, ax


def visualize_overlapped_samples_by_probability_ranges(
    model_dir: str,
    output_base_dir: str,
    prob_ranges: Optional[list] = None,
    threshold: Optional[float] = None,
    figsize: tuple = (12, 8),
    max_samples: Optional[int] = None
) -> None:
    """
    Visualize samples grouped by predicted probability confidence ranges and prediction categories,
    with all samples for each signal plotted in an overlapped manner.
    
    Creates a hierarchical directory structure:
    - Probability range folders (e.g., prob_0.0_to_0.4)
      - Prediction category folders (TP, TN, FP, FN)
        - PNG plots for each signal (all samples overlapped)
    
    Parameters
    ----------
    model_dir : str
        Path to directory containing saved model (used by functions.ml.model_persistence.load_model).
    output_base_dir : str
        Base directory where probability range folders will be created.
    prob_ranges : Optional[list]
        List of tuples defining probability ranges. Default: [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)]
        Each tuple is (min_prob, max_prob). For all ranges except the last, max_prob is exclusive.
        For the last range, max_prob is inclusive.
    threshold : Optional[float]
        Decision threshold for converting probabilities to predictions. If None, uses threshold
        from saved model data, or defaults to 0.5.
    figsize : tuple
        Figure size for signal plots.
    max_samples : Optional[int]
        Maximum number of samples to process per category. If None, processes all samples.
        Useful for limiting processing time when there are many samples.
    
    Returns
    -------
    None
        Creates directory structure and saves PNG files.
    """
    # Default probability ranges
    if prob_ranges is None:
        prob_ranges = [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)]
    
    # Load saved model data
    print(f"Loading model from: {model_dir}")
    model_data = load_model(model_dir)
    
    y_test = model_data["y_test"]
    y_proba = model_data["y_proba"]
    
    if y_proba is None:
        raise ValueError("Predicted probabilities (y_proba) not found in saved model data.")
    
    # Get threshold
    if threshold is None:
        if "threshold" in model_data:
            threshold = float(model_data["threshold"])
        elif "training_params" in model_data and "optimal_threshold" in model_data["training_params"]:
            threshold = float(model_data["training_params"]["optimal_threshold"])
        else:
            threshold = 0.5
            print(f"Warning: No threshold found in model data, using default: {threshold}")
    
    print(f"Using decision threshold: {threshold:.4f}")
    
    # Ensure y_proba is 1D probabilities for positive class
    if hasattr(y_proba, "ndim") and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = np.ravel(y_proba)
    
    # Compute predictions from probabilities using threshold
    y_pred = (y_proba_pos >= threshold).astype(int)
    
    # Load test processing data
    test_processing_path = os.path.join(model_dir, "test_processing_data.pkl")
    if not os.path.exists(test_processing_path):
        raise FileNotFoundError(
            "Test processing data not found. Probability range visualization requires "
            "test_processing_data.pkl to be saved with the model."
        )
    
    with open(test_processing_path, 'rb') as f:
        test_processing_data = pickle.load(f)
    
    test_processing_results = test_processing_data.get("test_processing_results")
    test_frame_mapping = test_processing_data.get("test_frame_mapping")
    test_file_paths = test_processing_data.get("test_file_paths", [])
    window_length = test_processing_data.get("window_length", 5.0)
    pre_peak_time = test_processing_data.get("pre_peak_time", 0.5)
    post_peak_time = test_processing_data.get("post_peak_time", 3.0)
    threshold_on_T = test_processing_data.get("threshold_on_T", 2500)
    
    if test_processing_results is None or test_frame_mapping is None:
        raise ValueError("Test processing results or frame mapping not found in saved data")
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\nOutput directory: {output_base_dir}")
    
    # Process each probability range
    for range_idx, (min_prob, max_prob) in enumerate(prob_ranges):
        # Create range folder name
        range_name = f"prob_{min_prob:.1f}_to_{max_prob:.1f}"
        range_dir = os.path.join(output_base_dir, range_name)
        os.makedirs(range_dir, exist_ok=True)
        
        # Find samples in this probability range
        if range_idx == len(prob_ranges) - 1:
            # Last range: include max_prob
            mask = (y_proba_pos >= min_prob) & (y_proba_pos <= max_prob)
        else:
            # Other ranges: exclude max_prob
            mask = (y_proba_pos >= min_prob) & (y_proba_pos < max_prob)
        
        sample_indices = np.where(mask)[0]
        num_samples = len(sample_indices)
        
        print(f"\n{'='*70}")
        print(f"Probability Range [{min_prob:.2f}, {max_prob:.2f}]: {num_samples} samples")
        print(f"{'='*70}")
        
        if num_samples == 0:
            print(f"No samples found in range [{min_prob:.2f}, {max_prob:.2f}] - skipping.")
            continue
        
        # Group samples by prediction category
        categories = {
            "TP": [],
            "TN": [],
            "FP": [],
            "FN": []
        }
        
        for idx in sample_indices:
            if idx >= len(test_frame_mapping):
                continue
            true_label = y_test[idx]
            pred_label = y_pred[idx]
            category = get_prediction_category(true_label, pred_label)
            categories[category].append(idx)
        
        # Print category counts
        print(f"  Category distribution:")
        for cat, indices in categories.items():
            print(f"    {cat}: {len(indices)} samples")
        
        # Process each category
        for category_name, category_indices in categories.items():
            if len(category_indices) == 0:
                continue
            
            # Limit number of samples if max_samples is specified
            original_count = len(category_indices)
            if max_samples is not None and len(category_indices) > max_samples:
                category_indices = category_indices[:max_samples]
                print(f"\n  Processing {category_name} ({len(category_indices)}/{original_count} samples, capped at {max_samples})...")
            else:
                print(f"\n  Processing {category_name} ({len(category_indices)} samples)...")
            
            # Create category folder
            category_dir = os.path.join(range_dir, category_name)
            os.makedirs(category_dir, exist_ok=True)
            
            # Collect all signal data for this category
            # Structure: signal_name -> list of frame data
            signals_data: Dict[str, List[Dict]] = defaultdict(list)
            
            # Get all available signal columns from first dataset (assuming consistent across datasets)
            all_signal_columns = set()
            for dataset_idx in set(test_frame_mapping[idx][0] for idx in category_indices):
                if dataset_idx < len(test_processing_results):
                    results = test_processing_results[dataset_idx]
                    signal_cols = [col for col in results["filtered_data"].columns 
                                 if col not in ["time", "timestamp"] and 
                                 results["filtered_data"][col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                    all_signal_columns.update(signal_cols)
            
            # Process each sample to collect frame data
            for global_idx in category_indices:
                if global_idx >= len(test_frame_mapping):
                    continue
                
                dataset_idx, frame_idx = test_frame_mapping[global_idx]
                
                if dataset_idx >= len(test_processing_results):
                    continue
                
                results = test_processing_results[dataset_idx]
                
                # Get frame indices
                if frame_idx >= len(results["frame_start_indices"]):
                    continue
                
                frame_start_idx = results["frame_start_indices"][frame_idx]
                frame_end_idx = results["frame_end_indices"][frame_idx]
                
                # Get time vector (do NOT assume uniform sampling rate)
                filtered_data = results["filtered_data"]
                if "time" in filtered_data.columns:
                    time = pd.to_datetime(filtered_data["time"], unit="s")
                    time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()
                else:
                    # Fallback: assume 100 Hz
                    dt = 0.01
                    time_s = np.arange(len(filtered_data)) * dt
                
                # Determine if this frame has a corresponding peak (true label = 1)
                true_label = y_test[global_idx]
                
                # Calculate extended frame end index using actual time values
                # This handles non-uniform or varying sampling rates correctly
                if true_label == 1:
                    # Faulty frame: has a corresponding peak
                    peak_indices = results.get("peak_indices", np.array([]))
                    
                    if len(peak_indices) > 0:
                        # Find the closest peak to this frame using time-based search
                        # Expected peak time: frame_end_time + pre_peak_time
                        frame_end_time = time_s[frame_end_idx]
                        expected_peak_time = frame_end_time + pre_peak_time
                        
                        # Find closest peak by comparing peak times
                        peak_times = time_s[peak_indices.astype(int)]
                        closest_peak_idx = int(peak_indices[np.argmin(np.abs(peak_times - expected_peak_time))])
                        
                        # Get the actual peak time
                        peak_time = time_s[closest_peak_idx]
                        
                        # Target end time: peak_time + post_peak_time
                        target_end_time = peak_time + post_peak_time
                        
                        # Find the index closest to target_end_time using the actual time vector
                        extended_end_idx = int(np.argmin(np.abs(time_s - target_end_time)))
                        extended_end_idx = min(extended_end_idx, len(filtered_data) - 1)
                    else:
                        # Fallback: estimate peak location using time
                        frame_end_time = time_s[frame_end_idx]
                        target_end_time = frame_end_time + pre_peak_time + post_peak_time
                        extended_end_idx = int(np.argmin(np.abs(time_s - target_end_time)))
                        extended_end_idx = min(extended_end_idx, len(filtered_data) - 1)
                else:
                    # Non-faulty frame: no corresponding peak
                    # Still extend by pre_peak + post_peak for consistency
                    frame_end_time = time_s[frame_end_idx]
                    target_end_time = frame_end_time + pre_peak_time + post_peak_time
                    extended_end_idx = int(np.argmin(np.abs(time_s - target_end_time)))
                    extended_end_idx = min(extended_end_idx, len(filtered_data) - 1)
                
                # Extract frame data for each signal
                for signal_name in all_signal_columns:
                    if signal_name not in filtered_data.columns:
                        continue
                    
                    # Extract signal
                    full_signal = filtered_data[signal_name].to_numpy()
                    
                    # Safety check
                    if frame_start_idx < 0 or extended_end_idx >= len(full_signal) or frame_start_idx > extended_end_idx:
                        continue
                    
                    # Extract EXTENDED frame signal (includes window_length + pre_peak + post_peak)
                    frame_signal = full_signal[frame_start_idx:extended_end_idx+1]
                    frame_time = time_s[frame_start_idx:extended_end_idx+1]
                    
                    # Normalize time: start from 0
                    frame_start_time = time_s[frame_start_idx]
                    if len(frame_time) > 0:
                        t_frame = frame_time - frame_start_time
                    else:
                        total_window = window_length + pre_peak_time + post_peak_time
                        t_frame = np.linspace(0, total_window, len(frame_signal))
                    
                    # Store frame data
                    signals_data[signal_name].append({
                        'signal': frame_signal,
                        'time': t_frame
                    })
            
            # Plot each signal with all frames overlapped
            for signal_name in sorted(all_signal_columns):
                if signal_name not in signals_data or len(signals_data[signal_name]) == 0:
                    continue
                
                # Create overlapped plot
                fig, ax = plot_overlapped_signals(
                    signal_name=signal_name,
                    frames_data=signals_data[signal_name],
                    window_length=window_length,
                    pre_peak_time=pre_peak_time,
                    post_peak_time=post_peak_time,
                    threshold_on_T=threshold_on_T if signal_name == "ground_tether_force" else None,
                    figsize=figsize,
                    title_suffix=f"{category_name} | Range [{min_prob:.2f}, {max_prob:.2f}]"
                )
                
                # Save figure
                safe_signal_name = signal_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                save_path = os.path.join(category_dir, f"{safe_signal_name}.png")
                plt.savefig(save_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                
                print(f"    Saved {signal_name}: {len(signals_data[signal_name])} samples")
            
            print(f"    Completed {category_name}: {len(all_signal_columns)} signals saved to {category_dir}")
        
        print(f"\n  Completed range [{min_prob:.2f}, {max_prob:.2f}]: {num_samples} samples total")
    
    print(f"\n{'='*70}")
    print(f"All overlapped probability range visualizations completed!")
    print(f"Results saved to: {output_base_dir}")
    print(f"{'='*70}\n")


def parse_prob_ranges(s: str) -> list:
    """
    Parse probability ranges from string format.
    
    Expected format: "0.0,0.4;0.4,0.6;0.6,1.0"
    Each range is min,max separated by semicolons.
    
    Parameters
    ----------
    s : str
        String representation of probability ranges.
    
    Returns
    -------
    list
        List of tuples (min_prob, max_prob).
    """
    ranges = []
    for range_str in s.split(';'):
        parts = range_str.split(',')
        if len(parts) != 2:
            raise ValueError(f"Invalid range format: {range_str}. Expected 'min,max'")
        try:
            min_prob = float(parts[0].strip())
            max_prob = float(parts[1].strip())
            ranges.append((min_prob, max_prob))
        except ValueError as e:
            raise ValueError(f"Invalid number in range: {range_str}") from e
    return ranges


def parse_figsize(s: str) -> tuple:
    """
    Parse figure size from string format.
    
    Expected format: "10,6" or "10 6"
    
    Parameters
    ----------
    s : str
        String representation of figure size.
    
    Returns
    -------
    tuple
        Tuple (width, height).
    """
    # Try comma separator first
    if ',' in s:
        parts = s.split(',')
    else:
        parts = s.split()
    
    if len(parts) != 2:
        raise ValueError(f"Invalid figsize format: {s}. Expected 'width,height' or 'width height'")
    
    try:
        width = float(parts[0].strip())
        height = float(parts[1].strip())
        return (width, height)
    except ValueError as e:
        raise ValueError(f"Invalid number in figsize: {s}") from e


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Visualize samples grouped by predicted probability ranges and prediction categories, with all samples for each signal plotted overlapped.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default probability ranges
  python visualize_overlapped_samples.py --model_dir /path/to/model --output_dir /path/to/output
  
  # Custom probability ranges
  python visualize_overlapped_samples.py --model_dir /path/to/model --output_dir /path/to/output \\
      --prob_ranges "0.0,0.3;0.3,0.7;0.7,1.0"
  
  # Custom threshold and figure size
  python visualize_overlapped_samples.py --model_dir /path/to/model --output_dir /path/to/output \\
      --threshold 0.6 --figsize "14,10"
  
  # Limit to 100 samples per category (faster processing)
  python visualize_overlapped_samples.py --model_dir /path/to/model --output_dir /path/to/output \\
      --max_samples 100
        """
    )
    
    parser.add_argument(
        "--model_dir",
        type=str,
        required=True,
        help="Path to directory containing saved model"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Base directory where probability range folders will be created"
    )
    
    parser.add_argument(
        "--prob_ranges",
        type=str,
        default=None,
        help="Probability ranges in format 'min1,max1;min2,max2;...' (default: '0.0,0.4;0.4,0.6;0.6,1.0')"
    )
    
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Decision threshold for converting probabilities to predictions (default: uses saved threshold or 0.5)"
    )
    
    parser.add_argument(
        "--figsize",
        type=str,
        default="12,8",
        help="Figure size in format 'width,height' (default: '12,8')"
    )
    
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to process per category (default: None, processes all samples)"
    )
    
    args = parser.parse_args()
    
    # Parse probability ranges
    if args.prob_ranges is None:
        prob_ranges = None  # Will use default in function
    else:
        try:
            prob_ranges = parse_prob_ranges(args.prob_ranges)
        except ValueError as e:
            print(f"Error parsing probability ranges: {e}", file=sys.stderr)
            sys.exit(1)
    
    # Parse figure size
    try:
        figsize = parse_figsize(args.figsize)
    except ValueError as e:
        print(f"Error parsing figure size: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Validate paths
    if not os.path.exists(args.model_dir):
        print(f"Error: Model directory does not exist: {args.model_dir}", file=sys.stderr)
        sys.exit(1)
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run the visualization
    try:
        visualize_overlapped_samples_by_probability_ranges(
            model_dir=args.model_dir,
            output_base_dir=args.output_dir,
            prob_ranges=prob_ranges,
            threshold=args.threshold,
            figsize=figsize,
            max_samples=args.max_samples
        )
    except Exception as e:
        print(f"Error during visualization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

