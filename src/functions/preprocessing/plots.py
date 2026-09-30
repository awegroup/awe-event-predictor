import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def plot_overlapping_frames_for_signal(
    signal_name: str,
    raw_df: pd.DataFrame,
    processing_output: dict,
    window_length: float,
    label_value: int = 1,
    threshold: float = None,
    pre_peak_time: float = None,
    figsize=(10, 4),
    title_suffix: str = ""
):
    """
    Plot all frames for which frame_labels == label_value, for any signal column
    in the dataframe. Optionally extends the plot to include the continuation segment
    (signal evolution after frame end).

    Parameters
    ----------
    signal_name : str
        Column name from the dataframe, e.g. "ground_tether_force".
    raw_df : pd.DataFrame
        The dataframe to plot from. Should be the FILTERED dataframe from
        processing_output["filtered_data"] to match the frame indices.
    processing_output : dict
        Output of TractionDataProcessor.process().
    window_length : float
        Frame duration in seconds (same as used in segmentation).
    label_value : int
        Which label to plot (default: 1).
    threshold : float, optional
        Threshold value to plot as a horizontal line (e.g., threshold_on_T).
    pre_peak_time : float, optional
        Time duration in seconds to extend the plot after frame end.
        For label=1: extends to the corresponding peak (if available).
        For label=0: extends by this fixed duration.
        A vertical line will mark the frame end boundary.
    figsize : tuple
        Figure size.
    """

    # Extract processor outputs
    frame_start = processing_output["frame_start_indices"]
    frame_end = processing_output["frame_end_indices"]
    frame_labels = processing_output["frame_labels"]
    frame_times = processing_output.get("frame_times", [])
    peak_indices = processing_output.get("peak_indices", np.array([]))

    # Extract only desired frames
    selected_indices = np.where(frame_labels == label_value)[0]

    if len(selected_indices) == 0:
        print(f"No frames found with label={label_value}.")
        return

    # Pull the signal to plot
    if signal_name not in raw_df.columns:
        raise ValueError(f"signal_name '{signal_name}' not in dataframe columns.")

    full_signal = raw_df[signal_name].to_numpy()
    
    # Get actual time values from dataframe for proper continuity
    # Use the same calculation as in data_processing.py to ensure consistency
    if "time" in raw_df.columns:
        # Match the time calculation from TractionDataProcessor.process()
        time = pd.to_datetime(raw_df["time"], unit="s")
        time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()
    else:
        # Fallback: create time vector from sampling frequency
        dt = 0.01  # Default 100 Hz
        time_s = np.arange(len(full_signal)) * dt
    
    # Compute sampling frequency
    if len(time_s) > 1:
        dt = np.mean(np.diff(time_s))
        fs = 1.0 / dt if dt > 0 else 100.0
    else:
        fs = 100.0
    
    # Determine if we should extend plots
    extend_plot = (pre_peak_time is not None and pre_peak_time > 0)
    
    # Create mapping from frame indices to peak indices for faulty frames (label=1)
    frame_to_peak_map = {}
    if extend_plot and label_value == 1 and len(peak_indices) > 0:
        # Faulty frames are created from peaks in order
        # Map faulty frame indices to their corresponding peaks
        faulty_mask = frame_labels == 1
        faulty_frame_indices = np.where(faulty_mask)[0]
        
        # Faulty frames correspond to peaks in the same order
        # (faulty frames are created first, then non-faulty frames are added)
        for i, fidx in enumerate(faulty_frame_indices):
            if i < len(peak_indices):
                frame_e = frame_end[fidx]
                peak_idx = peak_indices[i]
                # Verify the relationship: frame_end + pre_peak_samples ≈ peak_idx
                pre_peak_samples = int(round(pre_peak_time * fs))
                expected_peak = frame_e + pre_peak_samples
                # Allow some tolerance (within 20% of pre_peak_samples)
                if abs(peak_idx - expected_peak) < max(0.2 * pre_peak_samples, 5):
                    frame_to_peak_map[fidx] = peak_idx

    plt.figure(figsize=figsize)

    for idx in selected_indices:
        s = frame_start[idx]
        e = frame_end[idx]

        # Safety check: ensure indices are within bounds
        if s < 0 or e >= len(full_signal) or s > e:
            print(f"Warning: Invalid frame indices [{s}, {e}] for frame {idx}, skipping.")
            continue

        # Extract frame signal (includes index e)
        frame = full_signal[s:e+1]
        
        # Get actual time values for the frame
        frame_time = time_s[s:e+1]
        
        # Normalize time: start from 0, relative to frame start
        frame_start_time = time_s[s]
        if len(frame_time) > 0:
            t_frame = frame_time - frame_start_time
        else:
            t_frame = np.linspace(0, window_length, len(frame))
        
        # Plot the frame
        plt.plot(t_frame, frame, alpha=0.5, linewidth=1.5)
        
        # If extending plot, add continuation segment
        if extend_plot:
            # Get the actual frame end time (normalized) - this is where continuation should start
            actual_frame_end_time = t_frame[-1] if len(t_frame) > 0 else window_length
            
            if label_value == 1 and idx in frame_to_peak_map:
                # For faulty frames: extend to the peak
                peak_idx = frame_to_peak_map[idx]
                # Extract continuation: from frame end (e) to peak (inclusive)
                # Include frame end point to ensure visual continuity
                cont_start = e
                cont_end = min(peak_idx + 1, len(full_signal))
            else:
                # For non-faulty frames or if no peak mapping: extend by fixed duration
                cont_start = e  # Include frame end point
                cont_samples = int(round(pre_peak_time * fs))
                cont_end = min(cont_start + cont_samples + 1, len(full_signal))
            
            if cont_start < len(full_signal) and cont_start < cont_end:
                # Extract continuation (includes frame end point for continuity)
                continuation = full_signal[cont_start:cont_end]
                
                if len(continuation) > 0:
                    # Get actual time values for continuation
                    cont_time = time_s[cont_start:cont_end]
                    
                    # Normalize time: use same reference as frame to ensure continuity
                    # Since we include the frame end point (index e) in continuation,
                    # t_cont[0] will equal t_frame[-1], ensuring perfect continuity
                    if len(cont_time) > 0:
                        t_cont = cont_time - frame_start_time
                    else:
                        # Fallback: start from frame end time
                        actual_duration = len(continuation) / fs
                        t_cont = np.linspace(actual_frame_end_time, 
                                            actual_frame_end_time + actual_duration, 
                                            len(continuation))
                    
                    plt.plot(t_cont, continuation, alpha=0.5, linewidth=1.5, 
                            color='orange', linestyle='--')

    # Add vertical line to demark frame end (where continuation begins)
    if extend_plot:
        # Use the actual frame end time (which should be close to window_length)
        # For consistency, use window_length as the demarcation
        plt.axvline(window_length, color='green', linestyle=':', linewidth=2,
                   label='Frame end (continuation start)', alpha=0.7)

    # Add threshold line if provided
    #if threshold is not None:
    #    plt.axhline(threshold, color='red', linestyle='--', linewidth=2,
    #                label=f'Threshold = {threshold:.1f}')

    # Update title and labels
    continuation_suffix = " (with continuation)" if extend_plot else ""
    custom_suffix = f" - {title_suffix}" if title_suffix else ""
    plt.title(f"Overlapped frames for signal '{signal_name}' (label={label_value}){continuation_suffix}{custom_suffix}")
    plt.xlabel("Time (s)")
    plt.ylabel(signal_name)
    
    # Create legend if we have any labels
    has_labels = threshold is not None or extend_plot
    if has_labels:
        plt.legend()
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
