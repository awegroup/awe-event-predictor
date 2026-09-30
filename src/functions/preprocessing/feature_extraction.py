from __future__ import annotations

import numpy as np
from typing import List, Optional
from scipy.fft import fft
from scipy.stats import skew, kurtosis


class FeatureExtractor:
    """
    Feature extractor optimized for physical signal fault prediction.
    Focuses on amplitude, energy, variability, and temporal evolution.
    """

    def __init__(
        self,
        features_per_signal: Optional[int] = 40,
        num_sub_windows: Optional[int] = None,
        sub_window_mode: str = "temporal",
        low_freq_ratio: float = 0.1,
        sampling_rate: Optional[float] = None,
    ):
        """
        Parameters
        ----------
        features_per_signal : int, optional
            Output feature vector length. If None, all computed features are returned
            without truncation or padding.
        num_sub_windows : int, optional
            Number of sub-windows per segment.
        sub_window_mode : {"concatenate", "aggregate", "temporal"}
            How sub-window features are combined.
        low_freq_ratio : float
            Fraction of FFT bins considered "low frequency".
        sampling_rate : float, optional
            Sampling rate in Hz. Required for frequency-domain features above 1 Hz.
            If None, maximum amplitude above 1 Hz will be set to 0.
        """
        self.features_per_signal = features_per_signal
        self.num_sub_windows = num_sub_windows
        self.sub_window_mode = sub_window_mode
        self.low_freq_ratio = low_freq_ratio
        self.sampling_rate = sampling_rate

    # ------------------------------------------------------------------
    def _split_into_sub_windows(self, x: np.ndarray) -> List[np.ndarray]:
        if self.num_sub_windows is None or self.num_sub_windows == 1:
            return [x]

        if len(x) == 0:
            return [np.array([])] * self.num_sub_windows

        n = len(x)
        size = n // self.num_sub_windows

        return [
            x[i * size : (i + 1) * size if i < self.num_sub_windows - 1 else n]
            for i in range(self.num_sub_windows)
        ]

    # ------------------------------------------------------------------
    def _extract_base_features(self, x: np.ndarray) -> np.ndarray:
        """
        Base features designed for fault / overload detection.
        """
        if len(x) == 0:
            return np.zeros(25)

        x = np.asarray(x).flatten()
        n = len(x)

        mean = np.mean(x)
        std = np.std(x)
        rms = np.sqrt(np.mean(x ** 2))
        energy = np.sum(x ** 2)
        x_max = np.max(x)
        x_min = np.min(x)

        # Percentiles
        q75 = np.percentile(x, 75)
        q90 = np.percentile(x, 90)
        iqr = q75 - np.percentile(x, 25)
        median = np.median(x)

        # Zero-crossing rate (no mean subtraction – better for force signals)
        zcr = np.sum(np.diff(np.signbit(x))) / n if n > 0 else 0

        # Crest factor
        crest = x_max / rms if rms > 0 else 0

        # Time-to-maximum (normalized)
        t_max = np.argmax(x) / n if n > 0 else 0

        # Statistical features
        mad = np.mean(np.abs(x - mean))
        skewness_val = skew(x) if n > 2 else 0
        kurtosis_val = kurtosis(x) if n > 3 else 0

        # Range of cumulative sum
        cumsum_range = np.ptp(np.cumsum(x)) if n > 0 else 0

        # Time reversal asymmetry (third moment of differences)
        if n > 1:
            diff_x = np.diff(x)
            time_reversal_asymmetry = np.mean(diff_x ** 3) if len(diff_x) > 0 else 0
        else:
            time_reversal_asymmetry = 0

        # Maximum signal slope
        if n > 1:
            signal_slopes = np.diff(x)
            max_slope = np.max(np.abs(signal_slopes)) if len(signal_slopes) > 0 else 0
        else:
            max_slope = 0

        # Frequency-domain (minimal & stable)
        fft_vals = np.abs(fft(x))[: n // 2]
        fft_energy = np.sum(fft_vals)

        if fft_energy > 0:
            freqs = np.arange(len(fft_vals))
            centroid = np.sum(freqs * fft_vals) / fft_energy
            k = max(1, int(len(fft_vals) * self.low_freq_ratio))
            low_freq_ratio = np.sum(fft_vals[:k]) / fft_energy
            # FFT amplitude features
            fft_min = np.min(fft_vals)
            fft_max = np.max(fft_vals)
            # Maximum amplitude above 1 Hz
            if self.sampling_rate is not None and len(fft_vals) > 0:
                # Frequency resolution: sampling_rate / n
                freq_resolution = self.sampling_rate / n
                # Find bin index corresponding to 1 Hz
                bin_1hz = int(np.ceil(1.0 / freq_resolution)) if freq_resolution > 0 else 0
                if bin_1hz < len(fft_vals):
                    fft_max_above_1hz = np.max(fft_vals[bin_1hz:])
                else:
                    fft_max_above_1hz = 0
            else:
                fft_max_above_1hz = 0
        else:
            centroid = 0
            low_freq_ratio = 0
            fft_min = 0
            fft_max = 0
            fft_max_above_1hz = 0

        return np.array([
            mean,                      # 0
            std,                       # 1
            x_max,                     # 2 maximum
            rms,                       # 3
            energy,                    # 4
            q75,                       # 5
            q90,                       # 6
            iqr,                       # 7
            x_max - x_min,             # 8 peak-to-peak
            zcr,                       # 9
            crest,                     # 10
            t_max,                     # 11
            centroid,                  # 12
            low_freq_ratio,            # 13
            mad,                       # 14 MAD
            median,                    # 15 median
            x_min,                     # 16 minimum
            skewness_val,              # 17 skewness
            kurtosis_val,              # 18 kurtosis
            cumsum_range,              # 19 range of cumulative sum
            time_reversal_asymmetry,   # 20 time reversal asymmetry
            max_slope,                 # 21 maximum signal slope
            fft_min,                   # 22 FFT minimum amplitude
            fft_max,                   # 23 FFT maximum amplitude
            fft_max_above_1hz,         # 24 FFT maximum amplitude above 1 Hz
        ])

    # ------------------------------------------------------------------
    def extract_features(self, signal_segment: np.ndarray) -> np.ndarray:
        if len(signal_segment) == 0:
            # Return zeros with appropriate size based on mode
            if self.features_per_signal is not None:
                return np.zeros(self.features_per_signal)
            else:
                # Return base feature size if no truncation
                return np.zeros(25)

        signal_segment = np.asarray(signal_segment).flatten()

        # No sub-windowing
        if self.num_sub_windows is None or self.num_sub_windows == 1:
            features = self._extract_base_features(signal_segment)
        else:
            sub_windows = self._split_into_sub_windows(signal_segment)
            sub_feats = np.array([self._extract_base_features(sw) for sw in sub_windows])

            if self.sub_window_mode == "concatenate":
                features = sub_feats.flatten()
            elif self.sub_window_mode == "aggregate":
                features = np.hstack([
                    np.mean(sub_feats, axis=0),
                    np.std(sub_feats, axis=0),
                    np.min(sub_feats, axis=0),
                    np.max(sub_feats, axis=0),
                ])
            else:  # temporal (RECOMMENDED)
                base = self._extract_base_features(signal_segment)
                # Temporal statistics
                temporal_stats = np.hstack([
                    np.mean(sub_feats, axis=0),
                    np.std(sub_feats, axis=0),
                    np.max(sub_feats, axis=0),
                ])
                # Slopes of key indicators
                slopes = []
                key_idx = [0, 3, 4]  # mean, RMS, energy
                t = np.arange(len(sub_feats))
                for idx in key_idx:
                    slopes.append(np.polyfit(t, sub_feats[:, idx], 1)[0])
                features = np.concatenate([base, temporal_stats, slopes])

        # Enforce output size only if features_per_signal is specified
        if self.features_per_signal is not None:
            if len(features) > self.features_per_signal:
                features = features[: self.features_per_signal]
            else:
                features = np.pad(features, (0, self.features_per_signal - len(features)))

        return features

    # ------------------------------------------------------------------
    def extract_features_batch(self, signal_segments: List[np.ndarray]) -> np.ndarray:
        return np.array([self.extract_features(seg) for seg in signal_segments])


def segment_all_signals(
    data: np.ndarray,
    start_indices: np.ndarray,
    end_indices: np.ndarray
) -> List[np.ndarray]:
    """
    Segment a signal using pre-computed start/end indices.

    Parameters
    ----------
    data : np.ndarray
        Full signal data (1D array).
    start_indices : np.ndarray
        Start indices for each frame.
    end_indices : np.ndarray
        End indices for each frame.

    Returns
    -------
    List[np.ndarray]
        List of signal segments, one per frame.
    """
    segments = []
    for start_idx, end_idx in zip(start_indices, end_indices):
        # Ensure indices are valid
        start_idx = max(0, int(start_idx))
        end_idx = min(len(data) - 1, int(end_idx))
        
        if start_idx <= end_idx:
            segment = data[start_idx:end_idx + 1]
        else:
            segment = np.array([])
        
        segments.append(segment)
    
    return segments
