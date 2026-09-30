from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple
import matplotlib.pyplot as plt
from typing import Dict, List, Optional


@dataclass
class SignalFrame:
    """Container for an extracted frame of the signal."""
    data: np.ndarray
    time: np.ndarray
    start_index: int
    end_index: int


class SignalSegmenter:
    """
    Segments a signal into fixed-length windows aligned to peak indices.

    Parameters
    ----------
    time : np.ndarray
        Time vector in seconds.
    signal : np.ndarray
        Signal values corresponding to the time vector.
    """

    def __init__(self, time: np.ndarray, signal: np.ndarray):
        if len(time) != len(signal):
            raise ValueError("Time and signal vectors must have the same length.")
        self.time = time
        self.signal = signal

        # Compute mean sampling frequency for reference (not used in time-based segmentation)
        dt = np.diff(time)
        self.fs = 1.0 / np.mean(dt)

    def segment(
        self,
        peak_indices: np.ndarray,
        window_length: float,
        pre_peak_time: float,
    ) -> List[SignalFrame]:
        """
        Extract fixed-length frames from the signal, ending pre_peak_time before each peak.
        
        Uses TIME-BASED segmentation for robustness across varying sampling rates.

        The frame captures the signal just before it crosses the threshold:
        1. Find the peak (where signal crosses threshold T)
        2. Go back pre_peak_time seconds from the peak (this is the END of the frame)
        3. Go further back window_length seconds (this is the START of the frame)
        
        This creates a window that ends pre_peak_time seconds before the peak,
        capturing the signal as it approaches the threshold.

        Parameters
        ----------
        peak_indices : np.ndarray
            Array of indices where peaks occur (where signal crosses threshold).
        window_length : float
            Length of each frame in seconds.
        pre_peak_time : float
            Time prior to each peak that marks the END of the frame (in seconds).
            The frame extends backward window_length seconds from this point.

        Returns
        -------
        List[SignalFrame]
            A list of SignalFrame objects containing each extracted window.
            
        Notes
        -----
        This method uses TIME-BASED indexing rather than sample-based indexing,
        making it robust to:
        - Non-uniform sampling rates
        - Different sampling rates across datasets
        - Time gaps or irregularities in the data
        """

        frames: List[SignalFrame] = []

        for peak_idx in peak_indices:
            peak_idx = int(peak_idx)
            
            # Ensure peak_idx is within bounds
            if peak_idx < 0 or peak_idx >= len(self.time):
                continue
            
            # Get the actual time at the peak
            peak_time = self.time[peak_idx]
            
            # Step 1: Calculate target end time (pre_peak_time BEFORE the peak)
            target_end_time = peak_time - pre_peak_time
            
            # Find the index closest to target_end_time using the actual time vector
            # This is robust to non-uniform sampling
            end_idx = int(np.argmin(np.abs(self.time - target_end_time)))
            
            # Ensure end_idx is not negative and not beyond peak
            end_idx = max(0, min(end_idx, peak_idx - 1))
            
            # Step 2: Calculate target start time (window_length BEFORE the end)
            target_start_time = target_end_time - window_length
            
            # Find the index closest to target_start_time
            start_idx = int(np.argmin(np.abs(self.time - target_start_time)))
            
            # Ensure start_idx is valid and before end_idx
            start_idx = max(0, min(start_idx, end_idx - 1))
            
            # Extract frame data
            frame_data = self.signal[start_idx:end_idx + 1]
            frame_time = self.time[start_idx:end_idx + 1]
            
            # Verify frame has reasonable duration (within tolerance)
            if len(frame_time) > 1:
                actual_duration = frame_time[-1] - frame_time[0]
                # Allow up to 20% deviation from expected duration
                expected_duration = window_length
                tolerance = 0.2 * expected_duration
                
                if abs(actual_duration - expected_duration) > tolerance:
                    # Frame duration is too far off, skip it
                    import warnings
                    warnings.warn(
                        f"Frame duration {actual_duration:.2f}s deviates significantly "
                        f"from expected {expected_duration:.2f}s (tolerance: {tolerance:.2f}s). "
                        f"Skipping frame at peak index {peak_idx}."
                    )
                    continue

            frames.append(
                SignalFrame(
                    data=frame_data,
                    time=frame_time,
                    start_index=start_idx,
                    end_index=end_idx,
                )
            )

        return frames

    def segment_as_arrays(
        self,
        peak_indices: np.ndarray,
        window_length: float,
        pre_peak_time: float,
    ) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
        """
        MATLAB-style output helper for users who want separate arrays instead
        of an object-oriented structure.

        Returns
        -------
        frames, frames_time, frameStartIdx, frameEndIdx
        """

        frames_obj = self.segment(peak_indices, window_length, pre_peak_time)

        frames = [f.data for f in frames_obj]
        frames_time = [f.time for f in frames_obj]
        frameStartIdx = np.array([f.start_index for f in frames_obj], dtype=int)
        frameEndIdx = np.array([f.end_index for f in frames_obj], dtype=int)

        return frames, frames_time, frameStartIdx, frameEndIdx

