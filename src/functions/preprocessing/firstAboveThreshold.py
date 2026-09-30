from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from typing import Literal, Optional


class ThresholdEventFinder:
    """
    Detects events where a signal rises above a threshold, with optional clustering
    and optional interpretation of the minimum peak distance in seconds.
    Includes optional plotting of detected peaks.
    """

    def __init__(
        self,
        method: Literal["first", "peak"] = "first"
    ):
        self.method = method.lower()
        if self.method not in ("first", "peak"):
            raise ValueError("method must be 'first' or 'peak'")

    def find(
        self,
        signal: np.ndarray,
        threshold: float,
        min_peak_dist: float,
        time: Optional[np.ndarray] = None,
        plot: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Detect indices & values of signal samples above a threshold,
        clustered by a minimum peak distance (samples or seconds).

        Parameters
        ----------
        signal : np.ndarray
            Input signal (1-D).
        threshold : float
            Threshold for detection.
        min_peak_dist : float
            Minimum separation between events.
            - If time is None → interpreted as samples.
            - If time is provided → interpreted as seconds.
        time : np.ndarray, optional
            Time vector for interpreting min_peak_dist in seconds.
        plot : bool
            If True, displays a diagnostic plot highlighting peaks.

        Returns
        -------
        idxs : np.ndarray
            Indices of detected events.
        vals : np.ndarray
            Values of the signal at those indices.
        """

        signal = np.asarray(signal).reshape(-1)
        N = len(signal)

        if N == 0:
            return np.array([]), np.array([])

        if not np.isscalar(threshold):
            raise ValueError("threshold must be a scalar.")

        # -------------------------------
        # Convert min_peak_dist → samples if time vector provided
        # -------------------------------
        if time is not None:
            time = np.asarray(time).reshape(-1)
            if len(time) != N:
                raise ValueError("time vector must match signal length.")

            dt = np.diff(time)
            if np.any(dt <= 0) or not np.all(np.isfinite(dt)):
                raise ValueError("time must be strictly increasing.")

            fs_est = 1.0 / np.median(dt)
            min_peak_dist_samples = max(1, int(round(min_peak_dist * fs_est)))
        else:
            if min_peak_dist < 1:
                raise ValueError("min_peak_dist must be ≥ 1 when interpreted as samples.")
            min_peak_dist_samples = int(round(min_peak_dist))

        # -------------------------------
        # Find all points above threshold
        # -------------------------------
        above_idx = np.where(signal > threshold)[0]
        if len(above_idx) == 0:
            if plot:
                self._plot_results(time, signal, threshold, [], [])
            return np.array([]), np.array([])

        # -------------------------------
        # Cluster events by minimum distance
        # -------------------------------
        gaps = np.diff(above_idx)
        is_new_cluster = np.hstack(([True], gaps >= min_peak_dist_samples))
        cluster_ids = np.cumsum(is_new_cluster)
        num_clusters = cluster_ids[-1]

        # -------------------------------
        # Select representative index per cluster
        # -------------------------------
        idxs = np.zeros(num_clusters, dtype=int)

        if self.method == "first":
            first_positions = np.where(is_new_cluster)[0]
            idxs = above_idx[first_positions]

        elif self.method == "peak":
            for c in range(1, num_clusters + 1):
                members = above_idx[cluster_ids == c]
                max_idx = np.argmax(signal[members])
                idxs[c - 1] = members[max_idx]

        vals = signal[idxs]

        # -------------------------------
        # Optional plotting
        # -------------------------------
        if plot:
            self._plot_results(time, signal, threshold, idxs, vals)

        return idxs, vals

    # ------------------------------------------------------------------
    # Nice peak-visualization plot
    # ------------------------------------------------------------------
    def _plot_results(self, time, signal, threshold, idxs, vals):
        """Generate a clean diagnostic plot of the signal and detected peaks."""

        # If no time vector provided, use samples as x-axis
        x = time if time is not None else np.arange(len(signal))

        plt.figure(figsize=(12, 4))

        # Plot the signal
        plt.plot(x, signal, label="Signal", linewidth=1.5)

        # Horizontal threshold line
        plt.axhline(threshold, color="red", linestyle="--", linewidth=1.5,
                    label=f"Threshold = {threshold:.2f}")

        # Highlight peaks
        if len(idxs) > 0:
            plt.scatter(x[idxs], vals,
                        color="orange", edgecolors="black",
                        s=80, zorder=5, label="Detected Peaks")

            for xi in x[idxs]:
                plt.axvline(xi, color="orange", alpha=0.3, linewidth=1)

        plt.title("Threshold-Based Event Detection", fontsize=14)
        plt.xlabel("Time [s]" if time is not None else "Sample Index")
        plt.ylabel("Signal Value")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.show()
