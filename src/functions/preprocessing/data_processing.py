import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

from functions.preprocessing.firstAboveThreshold import ThresholdEventFinder
from functions.preprocessing.segmentSignal import SignalSegmenter
from functions.preprocessing.assignFrameLabels import FrameLabeler

class TractionDataProcessor:
    """
    High-level processor for kite tether traction data.
    """

    def __init__(
        self,
        min_height,
        threshold_on_T,
        min_seconds_between_peaks,
        window_length,
        pre_peak_time,
        plot=False
    ):

        self.min_height = min_height
        self.threshold_on_T = threshold_on_T
        self.min_seconds_between_peaks = min_seconds_between_peaks
        self.window_length = window_length
        self.pre_peak_time = pre_peak_time
        self.plot_enabled = plot

    # ------------------------------------------------------------------
    def process(self, data: pd.DataFrame):

        print("Incoming columns:", list(data.columns))

        # ----------------------------------------------------------
        # 1. Filter by kite height
        # ----------------------------------------------------------
        data = data[data["kite_height"] > self.min_height].copy()
        data.reset_index(drop=True, inplace=True)

        tether_T = data["ground_tether_force"].astype(float).to_numpy()

        # ----------------------------------------------------------
        # 2. Convert timestamps
        # ----------------------------------------------------------
        time = pd.to_datetime(data["time"], unit="s")
        time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()

        # ----------------------------------------------------------
        # 3. Sampling frequency
        # ----------------------------------------------------------
        dt = np.mean(np.diff(time_s))
        fs = 1.0 / dt

        # ----------------------------------------------------------
        # 4. Find peaks above threshold
        # ----------------------------------------------------------
        event_finder = ThresholdEventFinder(method="first") # method="first" or method="peak"
        peak_indices, peak_values = event_finder.find(
            signal=tether_T,
            threshold=self.threshold_on_T,
            min_peak_dist=self.min_seconds_between_peaks,   # interpreted as seconds
            time=time_s,
            plot=self.plot_enabled
        )

        # ----------------------------------------------------------
        # 5. Segment signal around peaks
        # ----------------------------------------------------------
        segmenter = SignalSegmenter(time=time_s, signal=tether_T)

        frames_faulty, frames_time_faulty, \
        frameStartIdx_faulty, frameEndIdx_faulty = segmenter.segment_as_arrays(
            peak_indices=peak_indices,
            window_length=self.window_length,
            pre_peak_time=self.pre_peak_time
        )

        num_faulty_frames = len(frames_faulty)

        # ----------------------------------------------------------
        # 6. Label frames
        # ----------------------------------------------------------
        labeler = FrameLabeler(signal=tether_T, time=time_s)

        labeled_frames = labeler.assign_labels(
            faulty_frames=frames_faulty,
            faulty_times=frames_time_faulty,
            faulty_start=frameStartIdx_faulty,
            faulty_end=frameEndIdx_faulty,
            threshold_on_T=self.threshold_on_T,
            window_length=self.window_length,
            peak_indices=peak_indices,
            min_seconds_between_peaks=self.min_seconds_between_peaks
        )

        # Unpack for MATLAB-style interface
        allFrames = [lf.data for lf in labeled_frames]
        allFrames_time = [lf.time for lf in labeled_frames]
        allStartIdx = np.array([lf.start_index for lf in labeled_frames])
        allEndIdx = np.array([lf.end_index for lf in labeled_frames])
        allLabels = np.array([lf.label for lf in labeled_frames])

        # ----------------------------------------------------------
        # Validation: Verify frame alignment
        # Note: Frames end pre_peak_time seconds BEFORE the peak.
        # They should show the signal approaching the threshold.
        # ----------------------------------------------------------
        faulty_mask = allLabels == 1
        if np.any(faulty_mask) and len(peak_indices) > 0:
            faulty_start = allStartIdx[faulty_mask]
            faulty_end = allEndIdx[faulty_mask]
            pre_peak_samples = int(round(self.pre_peak_time * fs))
            
            print(f"\nFrame validation (showing first 3 faulty frames):")
            print(f"  pre_peak_time: {self.pre_peak_time:.2f}s ({pre_peak_samples} samples)")
            print(f"  window_length: {self.window_length:.2f}s")
            
            num_to_show = min(3, len(faulty_start))
            for i, (s, e) in enumerate(zip(faulty_start[:num_to_show], faulty_end[:num_to_show])):
                frame_values = tether_T[s:e+1]
                max_val = np.max(frame_values)
                min_val = np.min(frame_values)
                final_val = frame_values[-1] if len(frame_values) > 0 else 0
                
                # Check what happens right after the frame (should approach/cross threshold)
                if e + 1 < len(tether_T):
                    next_val = tether_T[e + 1]
                    # Look for peak in the region after frame end
                    peak_search_end = min(e + 1 + pre_peak_samples + 20, len(tether_T))
                    post_frame_region = tether_T[e+1:peak_search_end]
                    if len(post_frame_region) > 0:
                        peak_val_after = np.max(post_frame_region)
                        print(f"  Frame {i}: end_idx={e}, frame_max={max_val:.1f}, "
                              f"frame_end_val={final_val:.1f}, next_val={next_val:.1f}, "
                              f"peak_after={peak_val_after:.1f} (threshold={self.threshold_on_T:.1f})")

        # ----------------------------------------------------------
        # Return results
        # ----------------------------------------------------------
        return {
            "filtered_data": data,
            "frame_data": allFrames,
            "frame_times": allFrames_time,
            "frame_start_indices": allStartIdx,
            "frame_end_indices": allEndIdx,
            "frame_labels": allLabels,
            "peak_indices": peak_indices,
            "peak_values": peak_values,
            "num_faulty_frames": num_faulty_frames,
            "threshold_on_T": self.threshold_on_T  # Include threshold for plotting
        }
