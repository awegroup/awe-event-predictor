from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import List


@dataclass
class LabeledFrame:
    """Container for a frame and its label."""
    data: np.ndarray
    time: np.ndarray
    start_index: int
    end_index: int
    label: int    # 1 = faulty, 0 = non-faulty


class FrameLabeler:
    """
    Creates frame-level labels and generates non-faulty frames based on
    thresholds and exclusion windows.
    """

    def __init__(self, signal: np.ndarray, time: np.ndarray):
        if len(signal) != len(time):
            raise ValueError("Signal and time vectors must match in length.")

        self.signal = signal
        self.time = time
        self.fs = 1.0 / np.mean(np.diff(time))

    def assign_labels(
        self,
        faulty_frames: List[np.ndarray],
        faulty_times: List[np.ndarray],
        faulty_start: np.ndarray,
        faulty_end: np.ndarray,
        threshold_on_T: float,
        window_length: float,
        peak_indices: np.ndarray = None,
        min_seconds_between_peaks: float = None,
    ) -> List[LabeledFrame]:
        """
        Combine faulty frames with newly generated non-faulty frames.

        Parameters
        ----------
        faulty_frames : list of np.ndarray
            Pre-computed faulty frame data.
        faulty_times : list of np.ndarray
            Time vectors associated with each faulty frame.
        faulty_start, faulty_end : np.ndarray
            Start/end indices for faulty frames.
        threshold_on_T : float
            Threshold defining fault regions.
        window_length : float
            Duration of a frame in seconds.
        peak_indices : np.ndarray, optional
            Indices of detected peaks. If provided along with min_seconds_between_peaks,
            regions within min_seconds_between_peaks after each peak will be excluded
            from both faulty and non-faulty frame generation.
        min_seconds_between_peaks : float, optional
            Minimum time between peaks. Used to define exclusion zones after each peak.

        Returns
        -------
        List[LabeledFrame]
            Combined list of faulty + non-faulty labeled frames.
        """

        window_samples = int(round(window_length * self.fs))
        num_faulty = len(faulty_frames)

        # -------------------------------
        # 1) Mark faulty samples in full signal
        # -------------------------------
        sample_labels = np.zeros(len(self.signal), dtype=int)

        for s, e in zip(faulty_start, faulty_end):
            if s <= e:
                sample_labels[s:e + 1] = 1

        # -------------------------------
        # 1.5) Mark exclusion zones after peaks (if parameters provided)
        # These regions should not be labeled as either faulty or non-faulty
        # -------------------------------
        exclusion_mask = np.zeros(len(self.signal), dtype=bool)
        
        if peak_indices is not None and min_seconds_between_peaks is not None:
            exclusion_samples = int(round(min_seconds_between_peaks * self.fs))
            
            for peak_idx in peak_indices:
                # Exclude region from peak to min_seconds_between_peaks after peak
                exclusion_start = int(peak_idx)
                exclusion_end = min(int(peak_idx) + exclusion_samples, len(self.signal))
                exclusion_mask[exclusion_start:exclusion_end] = True

        # -------------------------------
        # 2) Mask of non-faulty, below-threshold samples
        # Exclude both faulty regions and exclusion zones
        # -------------------------------
        candidate_mask = (sample_labels == 0) & (self.signal <= threshold_on_T) & (~exclusion_mask)

        # detect continuous segments
        d = np.diff(np.concatenate(([0], candidate_mask.astype(int), [0])))
        region_starts = np.where(d == 1)[0]
        region_ends = np.where(d == -1)[0] - 1

        # -------------------------------
        # 3) Extract non-faulty windows
        # -------------------------------
        non_faulty_frames: List[LabeledFrame] = []

        for rs, re in zip(region_starts, region_ends):
            region_len = re - rs + 1

            if region_len < window_samples:
                continue  # region too short

            # Number of non-overlapping windows
            num_windows = region_len // window_samples

            for k in range(num_windows):
                s = rs + k * window_samples
                e = s + window_samples - 1

                if e >= len(self.signal):
                    continue

                # safety check
                if np.any(self.signal[s:e + 1] > threshold_on_T):
                    continue

                non_faulty_frames.append(
                    LabeledFrame(
                        data=self.signal[s:e + 1],
                        time=self.time[s:e + 1],
                        start_index=s,
                        end_index=e,
                        label=0,
                    )
                )

        # -------------------------------
        # 4) Wrap faulty frames in LabeledFrame
        # -------------------------------
        faulty_labeled = [
            LabeledFrame(
                data=faulty_frames[i],
                time=faulty_times[i],
                start_index=int(faulty_start[i]),
                end_index=int(faulty_end[i]),
                label=1,
            )
            for i in range(num_faulty)
        ]

        # -------------------------------
        # 5) Combine and return
        # -------------------------------
        all_frames = faulty_labeled + non_faulty_frames

        print(f"Faulty frames (existing): {num_faulty}")
        print(f"Non-fault frames found:  {len(non_faulty_frames)}")
        if peak_indices is not None and min_seconds_between_peaks is not None:
            excluded_samples = np.sum(exclusion_mask)
            excluded_seconds = excluded_samples / self.fs
            print(f"Excluded region (after peaks): {excluded_samples} samples ({excluded_seconds:.1f}s)")
        print(f"Total frames:            {len(all_frames)}")

        return all_frames
