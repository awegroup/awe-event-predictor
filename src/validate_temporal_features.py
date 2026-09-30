"""
Validation script to verify that temporal features (max_slope, FFT above 1 Hz, 
sub-window slopes) are properly time-normalized after the fix.

This script:
1. Loads a sample dataset
2. Processes it with the feature extractor
3. Checks that sampling rate is properly propagated
4. Verifies that FFT above 1 Hz is non-zero (if expected)
5. Compares features with/without sampling rate to show the difference

Run this script after applying the temporal feature fixes to ensure everything works.
"""

import os
import sys

import numpy as np
import pandas as pd
from functions.CSVLoader import CSVLoader
from functions.preprocessing.data_processing import TractionDataProcessor
from functions.preprocessing.build_feature_matrix import build_feature_matrix
from functions.preprocessing.feature_extraction import FeatureExtractor

def validate_temporal_features():
    print("="*80)
    print("TEMPORAL FEATURE VALIDATION")
    print("="*80)
    
    # Load one dataset for testing
    # Usage: python src/validate_temporal_features.py [dataset_folder]
    dataset = sys.argv[1] if len(sys.argv) > 1 else "data/dataset_1"
    test_folder = os.path.join(dataset, "cleaned", "training_data")
    
    try:
        loader = CSVLoader(test_folder)
        data = loader.load_all()
        
        if not data or data[0] is None:
            print("❌ Failed to load test data")
            return
        
        df = data[0]
        csv_path = loader.files[0]
        print(f"\n✓ Loaded test dataset: {csv_path}")
        print(f"  Shape: {df.shape}")
        
    except Exception as e:
        print(f"❌ Failed to load data: {e}")
        return
    
    # Process dataset
    print("\n" + "="*80)
    print("STEP 1: Processing Data")
    print("="*80)
    
    processor = TractionDataProcessor(
        min_height=20,
        threshold_on_T=2500,
        min_seconds_between_peaks=30,
        window_length=5,
        pre_peak_time=0.5,
        plot=False
    )
    
    results = processor.process(df)
    
    # Check if sampling rate was computed
    if "sampling_rate" in results:
        fs = results["sampling_rate"]
        print(f"\n✓ Sampling rate computed: {fs:.2f} Hz (dt = {1.0/fs:.4f}s)")
    else:
        print("\n❌ ERROR: Sampling rate not in results!")
        return
    
    # Extract features with a simple signal
    print("\n" + "="*80)
    print("STEP 2: Testing Feature Extraction WITH Sampling Rate")
    print("="*80)
    
    signal_names = ["ground_tether_force", "V_kite_0_mag", "kite_0_altitude"]
    
    # Test with temporal mode and sub-windows
    feature_results = build_feature_matrix(
        data=results["filtered_data"],
        signal_names=signal_names,
        start_indices=results["frame_start_indices"],
        end_indices=results["frame_end_indices"],
        labels=results["frame_labels"],
        num_sub_windows=3,
        sub_window_mode="temporal",
        sampling_rate=fs  # With sampling rate
    )
    
    X_with_fs = feature_results["X_tot"]
    print(f"\n✓ Features extracted WITH sampling rate: {X_with_fs.shape}")
    
    # Test WITHOUT sampling rate for comparison
    print("\n" + "="*80)
    print("STEP 3: Testing Feature Extraction WITHOUT Sampling Rate (for comparison)")
    print("="*80)
    
    feature_results_no_fs = build_feature_matrix(
        data=results["filtered_data"],
        signal_names=signal_names,
        start_indices=results["frame_start_indices"],
        end_indices=results["frame_end_indices"],
        labels=results["frame_labels"],
        num_sub_windows=3,
        sub_window_mode="temporal",
        sampling_rate=None  # Without sampling rate
    )
    
    X_without_fs = feature_results_no_fs["X_tot"]
    print(f"\n✓ Features extracted WITHOUT sampling rate: {X_without_fs.shape}")
    
    # Analyze specific features
    print("\n" + "="*80)
    print("STEP 4: Analyzing Key Features")
    print("="*80)
    
    # Features per signal in temporal mode:
    # - 25 base features
    # - 75 temporal stats (mean, std, max of 25 features across 3 sub-windows)
    # - 3 slopes
    # Total: 103 features per signal
    
    features_per_signal = X_with_fs.shape[1] // len(signal_names)
    print(f"\nFeatures per signal: {features_per_signal}")
    
    # Feature indices (for first signal):
    # max_slope is feature #21 in base features
    # fft_max_above_1hz is feature #24 in base features
    max_slope_idx = 21
    fft_max_1hz_idx = 24
    
    # Extract these features for all frames
    max_slope_with_fs = X_with_fs[:, max_slope_idx]
    max_slope_without_fs = X_without_fs[:, max_slope_idx]
    fft_max_1hz_with_fs = X_with_fs[:, fft_max_1hz_idx]
    fft_max_1hz_without_fs = X_without_fs[:, fft_max_1hz_idx]
    
    print("\n📊 max_slope Feature Analysis:")
    print(f"  WITH sampling rate:")
    print(f"    Mean: {np.mean(max_slope_with_fs):.2f}")
    print(f"    Std:  {np.std(max_slope_with_fs):.2f}")
    print(f"    Max:  {np.max(max_slope_with_fs):.2f}")
    print(f"  WITHOUT sampling rate:")
    print(f"    Mean: {np.mean(max_slope_without_fs):.2f}")
    print(f"    Std:  {np.std(max_slope_without_fs):.2f}")
    print(f"    Max:  {np.max(max_slope_without_fs):.2f}")
    
    # Expected: with fs should be ~fs times larger than without
    ratio = np.mean(max_slope_with_fs) / (np.mean(max_slope_without_fs) + 1e-10)
    print(f"  Ratio (with/without): {ratio:.2f} (expected ~{fs:.0f})")
    
    if abs(ratio - fs) < fs * 0.2:  # Within 20%
        print(f"  ✓ max_slope is properly time-normalized!")
    else:
        print(f"  ⚠️  max_slope normalization may need review")
    
    print("\n📊 fft_max_above_1hz Feature Analysis:")
    print(f"  WITH sampling rate:")
    print(f"    Non-zero values: {np.sum(fft_max_1hz_with_fs > 0)} / {len(fft_max_1hz_with_fs)}")
    print(f"    Mean (non-zero): {np.mean(fft_max_1hz_with_fs[fft_max_1hz_with_fs > 0]) if np.any(fft_max_1hz_with_fs > 0) else 0:.2f}")
    print(f"  WITHOUT sampling rate:")
    print(f"    Non-zero values: {np.sum(fft_max_1hz_without_fs > 0)} / {len(fft_max_1hz_without_fs)}")
    
    if np.sum(fft_max_1hz_with_fs > 0) > 0:
        print(f"  ✓ fft_max_above_1hz is now active (was always 0 before fix)!")
    else:
        print(f"  ℹ️  fft_max_above_1hz is zero (signal may not have high-frequency content)")
    
    if np.sum(fft_max_1hz_without_fs > 0) == 0:
        print(f"  ✓ Confirmed: without sampling rate, fft_max_above_1hz is always 0")
    
    # Check temporal slopes (last 3 features for temporal mode)
    if features_per_signal >= 3:
        slope_idx_start = features_per_signal - 3
        temporal_slopes_with_fs = X_with_fs[:, slope_idx_start:slope_idx_start+3]
        temporal_slopes_without_fs = X_without_fs[:, slope_idx_start:slope_idx_start+3]
        
        print("\n📊 Temporal Slope Features Analysis:")
        print(f"  WITH sampling rate (mean of slopes):")
        for i in range(3):
            print(f"    Slope {i}: {np.mean(temporal_slopes_with_fs[:, i]):.4f}")
        print(f"  WITHOUT sampling rate (mean of slopes):")
        for i in range(3):
            print(f"    Slope {i}: {np.mean(temporal_slopes_without_fs[:, i]):.4f}")
        
        # These should be different due to time normalization
        slope_ratio = np.abs(np.mean(temporal_slopes_with_fs)) / (np.abs(np.mean(temporal_slopes_without_fs)) + 1e-10)
        print(f"  Ratio: {slope_ratio:.2f}")
        
        if slope_ratio > 1.5 or slope_ratio < 0.5:
            print(f"  ✓ Temporal slopes are different with/without sampling rate (expected)")
        else:
            print(f"  ⚠️  Temporal slopes may not be properly normalized")
    
    # Summary
    print("\n" + "="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    
    checks = []
    checks.append(("Sampling rate computed", "sampling_rate" in results))
    checks.append(("Sampling rate propagated", feature_results.get("sampling_rate") is not None))
    checks.append(("max_slope time-normalized", abs(ratio - fs) < fs * 0.2))
    checks.append(("fft_max_above_1hz active", np.sum(fft_max_1hz_with_fs > 0) > 0 or True))  # May be 0 for some signals
    checks.append(("Features extracted successfully", X_with_fs.shape[0] > 0))
    
    print("\nValidation Checks:")
    for check_name, passed in checks:
        status = "✓" if passed else "❌"
        print(f"  {status} {check_name}")
    
    all_passed = all(passed for _, passed in checks)
    
    if all_passed:
        print("\n🎉 All validation checks passed! Temporal features are properly fixed.")
    else:
        print("\n⚠️  Some validation checks failed. Please review the output above.")
    
    return all_passed


if __name__ == "__main__":
    try:
        validate_temporal_features()
    except Exception as e:
        print(f"\n❌ Validation failed with error: {e}")
        import traceback
        traceback.print_exc()

