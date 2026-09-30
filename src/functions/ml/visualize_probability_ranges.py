#!/usr/bin/env python3
"""
Script to visualize samples grouped by predicted probability ranges and prediction categories.

This script creates a hierarchical directory structure:
- Probability range folders (e.g., prob_0.0_to_0.4)
  - Prediction category folders (TP, TN, FP, FN)
    - Individual sample folders
      - PNG plots for each signal
"""

from __future__ import annotations
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
import pickle
import pandas as pd

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


def visualize_samples_by_probability_ranges(
    model_dir: str,
    output_base_dir: str,
    prob_ranges: Optional[list] = None,
    threshold: Optional[float] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Visualize samples grouped by predicted probability confidence ranges and prediction categories.
    
    Creates a hierarchical directory structure:
    - Probability range folders (e.g., prob_0.0_to_0.4)
      - Prediction category folders (TP, TN, FP, FN)
        - Individual sample folders
          - PNG plots for each signal
    
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
        Figure size for individual signal plots.
    
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
            
            # Create category folder
            category_dir = os.path.join(range_dir, category_name)
            os.makedirs(category_dir, exist_ok=True)
            
            print(f"\n  Processing {category_name} ({len(category_indices)} samples)...")
            
            # Group samples by dataset for efficient processing
            samples_by_dataset = {}
            for idx in category_indices:
                dataset_idx, frame_idx = test_frame_mapping[idx]
                if dataset_idx not in samples_by_dataset:
                    samples_by_dataset[dataset_idx] = []
                samples_by_dataset[dataset_idx].append((idx, frame_idx))
            
            # Process each sample
            sample_counter = 0
            for dataset_idx, sample_list in samples_by_dataset.items():
                if dataset_idx >= len(test_processing_results):
                    continue
                
                results = test_processing_results[dataset_idx]
                dataset_name = os.path.basename(test_file_paths[dataset_idx]) if dataset_idx < len(test_file_paths) else f"dataset_{dataset_idx}"
                
                # Get all available signal columns (exclude non-signal columns)
                signal_columns = [col for col in results["filtered_data"].columns 
                                if col not in ["time", "timestamp"] and 
                                results["filtered_data"][col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                
                if len(signal_columns) == 0:
                    print(f"    Warning: No signal columns found in dataset {dataset_name}")
                    continue
                
                # Process each sample from this dataset
                for global_idx, frame_idx in sample_list:
                    sample_counter += 1
                    prob_value = y_proba_pos[global_idx]
                    true_label = y_test[global_idx]
                    pred_label = y_pred[global_idx]
                    
                    # Create sample folder
                    sample_folder_name = f"sample_{sample_counter:05d}_prob_{prob_value:.4f}_true_{int(true_label)}_pred_{int(pred_label)}"
                    sample_dir = os.path.join(category_dir, sample_folder_name)
                    os.makedirs(sample_dir, exist_ok=True)
                    
                    # Get frame indices
                    if frame_idx >= len(results["frame_start_indices"]):
                        continue
                    
                    frame_start_idx = results["frame_start_indices"][frame_idx]
                    frame_end_idx = results["frame_end_indices"][frame_idx]
                    frame_label = results["frame_labels"][frame_idx] if frame_idx < len(results["frame_labels"]) else -1
                    
                    # Get time vector and calculate sampling rate
                    filtered_data = results["filtered_data"]
                    if "time" in filtered_data.columns:
                        time = pd.to_datetime(filtered_data["time"], unit="s")
                        time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()
                        dt = np.mean(np.diff(time_s)) if len(time_s) > 1 else 0.01
                    else:
                        dt = 0.01  # Default 100 Hz
                        time_s = np.arange(len(filtered_data)) * dt
                    
                    fs = 1.0 / dt  # Sampling frequency
                    
                    # Calculate extended frame end index (including pre_peak and post_peak time)
                    pre_peak_samples = int(round(pre_peak_time * fs))
                    post_peak_samples = int(round(post_peak_time * fs))
                    
                    # Determine if this frame has a corresponding peak (true label = 1)
                    if true_label == 1:
                        # Faulty frame: has a corresponding peak
                        peak_indices = results.get("peak_indices", np.array([]))
                        expected_peak_idx = frame_end_idx + pre_peak_samples
                        
                        if len(peak_indices) > 0:
                            # Find closest peak
                            closest_peak_idx = peak_indices[np.argmin(np.abs(peak_indices - expected_peak_idx))]
                            extended_end_idx = min(int(closest_peak_idx) + post_peak_samples, len(filtered_data) - 1)
                        else:
                            # Fallback: use estimated peak location
                            extended_end_idx = min(frame_end_idx + pre_peak_samples + post_peak_samples, len(filtered_data) - 1)
                    else:
                        # Non-faulty frame: no corresponding peak
                        extended_end_idx = min(frame_end_idx + pre_peak_samples + post_peak_samples, len(filtered_data) - 1)
                    
                    # Plot each signal individually
                    for signal_name in signal_columns:
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
                        
                        # Create figure
                        fig, ax = plt.subplots(figsize=figsize)
                        
                        # Plot signal
                        ax.plot(t_frame, frame_signal, linewidth=2, color='blue', label='Signal')
                        
                        # Add vertical markers for different time segments
                        ax.axvline(window_length, color='green', linestyle=':', linewidth=1.5, 
                                   label=f'Window end ({window_length:.1f}s)', alpha=0.7)
                        peak_time = window_length + pre_peak_time
                        ax.axvline(peak_time, color='orange', linestyle='--', linewidth=1.5,
                                   label=f'Peak (~{peak_time:.1f}s)', alpha=0.7)
                        
                        # Add threshold line if applicable and signal is ground_tether_force
                        if signal_name == "ground_tether_force" and threshold_on_T is not None:
                            ax.axhline(threshold_on_T, color='red', linestyle='--', linewidth=1.5,
                                      label=f'Threshold = {threshold_on_T:.1f}')
                        
                        # Labels and title
                        total_window = window_length + pre_peak_time + post_peak_time
                        ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
                        ax.set_ylabel(signal_name, fontsize=12, fontweight="bold")
                        title = (f"Signal: {signal_name} | {category_name} | {total_window:.1f}s window\n"
                                 f"Probability: {prob_value:.4f} | True: {int(true_label)} | "
                                 f"Pred: {int(pred_label)} | Frame Label: {int(frame_label)}")
                        ax.set_title(title, fontsize=11, fontweight="bold")
                        ax.grid(True, alpha=0.3)
                        ax.legend(loc='best', fontsize=9)
                        
                        plt.tight_layout()
                        
                        # Save figure
                        # Sanitize signal name for filename
                        safe_signal_name = signal_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                        save_path = os.path.join(sample_dir, f"{safe_signal_name}.png")
                        plt.savefig(save_path, dpi=150, bbox_inches="tight")
                        plt.close(fig)
                    
                    if sample_counter % 50 == 0:
                        print(f"    Processed {sample_counter}/{len(category_indices)} {category_name} samples...")
            
            print(f"    Completed {category_name}: {len(category_indices)} samples saved to {category_dir}")
        
        print(f"\n  Completed range [{min_prob:.2f}, {max_prob:.2f}]: {num_samples} samples total")
    
    print(f"\n{'='*70}")
    print(f"All probability range visualizations completed!")
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
        description="Visualize samples grouped by predicted probability ranges and prediction categories.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default probability ranges
  python visualize_probability_ranges.py --model_dir /path/to/model --output_dir /path/to/output
  
  # Custom probability ranges
  python visualize_probability_ranges.py --model_dir /path/to/model --output_dir /path/to/output \\
      --prob_ranges "0.0,0.3;0.3,0.7;0.7,1.0"
  
  # Custom threshold and figure size
  python visualize_probability_ranges.py --model_dir /path/to/model --output_dir /path/to/output \\
      --threshold 0.6 --figsize "12,8"
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
        default="10,6",
        help="Figure size in format 'width,height' (default: '10,6')"
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
        visualize_samples_by_probability_ranges(
            model_dir=args.model_dir,
            output_base_dir=args.output_dir,
            prob_ranges=prob_ranges,
            threshold=args.threshold,
            figsize=figsize
        )
    except Exception as e:
        print(f"Error during visualization: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

