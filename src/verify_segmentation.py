#!/usr/bin/env python3
"""
Script to verify that frame segmentation is correctly applied to all signals.

This script:
1. Loads a sample dataset
2. Processes it to get frame indices
3. Verifies that all signals have the same length
4. Checks actual frame durations vs expected
5. Detects any NaN values in extracted frames
6. Compares sample-based vs time-based segmentation
"""

import numpy as np
import pandas as pd
import sys
import os
from functions.CSVLoader import CSVLoader
from functions.preprocessing.data_processing import TractionDataProcessor


def verify_signal_lengths(data: pd.DataFrame, signal_names: list) -> dict:
    """Check if all signals have the same length."""
    print("\n" + "="*70)
    print("SIGNAL LENGTH VERIFICATION")
    print("="*70)
    
    results = {}
    base_length = len(data)
    print(f"DataFrame length: {base_length} rows")
    
    all_same = True
    for sig in signal_names:
        if sig in data.columns:
            sig_length = len(data[sig])
            nan_count = data[sig].isna().sum()
            results[sig] = {
                'length': sig_length,
                'nan_count': nan_count,
                'non_nan_length': sig_length - nan_count
            }
            
            if sig_length != base_length:
                print(f"  ❌ {sig}: {sig_length} (MISMATCH!)")
                all_same = False
            else:
                print(f"  ✓ {sig}: {sig_length} rows, {nan_count} NaN")
    
    if all_same:
        print("\n✅ All signals have the same length")
    else:
        print("\n❌ Signal length mismatch detected!")
    
    return results


def check_sampling_rate_uniformity(time_s: np.ndarray) -> dict:
    """Check if sampling rate is uniform."""
    print("\n" + "="*70)
    print("SAMPLING RATE UNIFORMITY CHECK")
    print("="*70)
    
    dt = np.diff(time_s)
    mean_dt = np.mean(dt)
    std_dt = np.std(dt)
    min_dt = np.min(dt)
    max_dt = np.max(dt)
    
    fs_mean = 1.0 / mean_dt if mean_dt > 0 else 0
    
    print(f"  Mean dt: {mean_dt*1000:.3f} ms ({fs_mean:.2f} Hz)")
    print(f"  Std dt:  {std_dt*1000:.3f} ms")
    print(f"  Min dt:  {min_dt*1000:.3f} ms ({1.0/min_dt:.2f} Hz)")
    print(f"  Max dt:  {max_dt*1000:.3f} ms ({1.0/max_dt:.2f} Hz)")
    
    # Check uniformity: is std < 1% of mean?
    uniformity_pct = (std_dt / mean_dt) * 100 if mean_dt > 0 else 0
    print(f"  Variability: {uniformity_pct:.2f}% of mean")
    
    if uniformity_pct < 1.0:
        print("\n✅ Sampling rate is highly uniform (< 1% variation)")
        status = "uniform"
    elif uniformity_pct < 5.0:
        print("\n⚠️  Sampling rate has moderate variation (1-5%)")
        status = "moderate_variation"
    else:
        print("\n❌ Sampling rate has significant variation (> 5%)")
        status = "high_variation"
    
    return {
        'mean_dt': mean_dt,
        'std_dt': std_dt,
        'min_dt': min_dt,
        'max_dt': max_dt,
        'fs_mean': fs_mean,
        'uniformity_pct': uniformity_pct,
        'status': status
    }


def verify_frame_durations(
    data: pd.DataFrame,
    frame_start_indices: np.ndarray,
    frame_end_indices: np.ndarray,
    expected_duration: float
) -> dict:
    """Verify that extracted frames have the expected duration."""
    print("\n" + "="*70)
    print("FRAME DURATION VERIFICATION")
    print("="*70)
    
    time = pd.to_datetime(data["time"], unit="s")
    time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()
    
    durations = []
    duration_errors = []
    
    for i, (start_idx, end_idx) in enumerate(zip(frame_start_indices, frame_end_indices)):
        if start_idx < len(time_s) and end_idx < len(time_s):
            actual_duration = time_s[end_idx] - time_s[start_idx]
            error = abs(actual_duration - expected_duration)
            durations.append(actual_duration)
            duration_errors.append(error)
    
    durations = np.array(durations)
    duration_errors = np.array(duration_errors)
    
    print(f"  Expected duration: {expected_duration:.3f} s")
    print(f"  Mean actual: {np.mean(durations):.3f} s")
    print(f"  Std actual:  {np.std(durations):.3f} s")
    print(f"  Min actual:  {np.min(durations):.3f} s")
    print(f"  Max actual:  {np.max(durations):.3f} s")
    print(f"  Mean error:  {np.mean(duration_errors)*1000:.1f} ms")
    print(f"  Max error:   {np.max(duration_errors)*1000:.1f} ms")
    
    # Check if errors are acceptable (< 100ms)
    large_errors = np.sum(duration_errors > 0.1)
    if large_errors == 0:
        print(f"\n✅ All frame durations within 100ms tolerance")
        status = "good"
    else:
        print(f"\n⚠️  {large_errors} frames have duration errors > 100ms")
        status = "needs_attention"
    
    return {
        'mean_duration': np.mean(durations),
        'std_duration': np.std(durations),
        'mean_error': np.mean(duration_errors),
        'max_error': np.max(duration_errors),
        'large_errors': large_errors,
        'status': status
    }


def check_nan_in_frames(
    data: pd.DataFrame,
    signal_names: list,
    frame_start_indices: np.ndarray,
    frame_end_indices: np.ndarray
) -> dict:
    """Check for NaN values in extracted frames."""
    print("\n" + "="*70)
    print("NaN VALUES IN FRAMES")
    print("="*70)
    
    results = {}
    total_nan_frames = 0
    
    for sig in signal_names:
        if sig not in data.columns:
            continue
        
        signal_data = data[sig].astype(float).to_numpy()
        frames_with_nan = 0
        total_nan_count = 0
        
        for i, (start_idx, end_idx) in enumerate(zip(frame_start_indices, frame_end_indices)):
            if start_idx < len(signal_data) and end_idx < len(signal_data):
                frame = signal_data[start_idx:end_idx+1]
                nan_count = np.isnan(frame).sum()
                if nan_count > 0:
                    frames_with_nan += 1
                    total_nan_count += nan_count
        
        results[sig] = {
            'frames_with_nan': frames_with_nan,
            'total_nan_count': total_nan_count
        }
        
        if frames_with_nan > 0:
            print(f"  ⚠️  {sig}: {frames_with_nan} frames with NaN ({total_nan_count} total NaN values)")
            total_nan_frames += frames_with_nan
        else:
            print(f"  ✓ {sig}: No NaN in frames")
    
    if total_nan_frames == 0:
        print(f"\n✅ No NaN values found in any frames")
    else:
        print(f"\n⚠️  Found NaN values in {total_nan_frames} frame-signal pairs")
    
    return results


def compare_segmentation_methods(
    time_s: np.ndarray,
    peak_indices: np.ndarray,
    window_length: float,
    pre_peak_time: float
) -> dict:
    """Compare sample-based vs time-based segmentation."""
    print("\n" + "="*70)
    print("SEGMENTATION METHOD COMPARISON")
    print("="*70)
    
    # Method 1: Sample-based (current approach)
    dt = np.mean(np.diff(time_s))
    fs = 1.0 / dt
    window_samples = int(round(window_length * fs))
    pre_peak_samples = int(round(pre_peak_time * fs))
    
    sample_based_indices = []
    for peak_idx in peak_indices:
        end_idx = max(int(peak_idx) - pre_peak_samples, 0)
        start_idx = max(end_idx - window_samples + 1, 0)
        sample_based_indices.append((start_idx, end_idx))
    
    # Method 2: Time-based (recommended approach)
    time_based_indices = []
    for peak_idx in peak_indices:
        peak_time = time_s[int(peak_idx)]
        end_time = peak_time - pre_peak_time
        end_idx = int(np.argmin(np.abs(time_s - end_time)))
        start_time = end_time - window_length
        start_idx = int(np.argmin(np.abs(time_s - start_time)))
        time_based_indices.append((start_idx, end_idx))
    
    # Compare the two methods
    index_diffs_start = []
    index_diffs_end = []
    
    for (s1, e1), (s2, e2) in zip(sample_based_indices, time_based_indices):
        index_diffs_start.append(abs(s1 - s2))
        index_diffs_end.append(abs(e1 - e2))
    
    index_diffs_start = np.array(index_diffs_start)
    index_diffs_end = np.array(index_diffs_end)
    
    print(f"  Method 1 (sample-based): fs={fs:.2f} Hz, window={window_samples} samples")
    print(f"  Method 2 (time-based): uses actual time vector")
    print(f"\n  Index differences (sample-based vs time-based):")
    print(f"    Start indices - Mean diff: {np.mean(index_diffs_start):.2f} samples")
    print(f"                    Max diff:  {np.max(index_diffs_start)} samples")
    print(f"    End indices   - Mean diff: {np.mean(index_diffs_end):.2f} samples")
    print(f"                    Max diff:  {np.max(index_diffs_end)} samples")
    
    if np.max(index_diffs_start) == 0 and np.max(index_diffs_end) == 0:
        print("\n✅ Both methods produce identical results")
        status = "identical"
    elif np.max(index_diffs_start) <= 2 and np.max(index_diffs_end) <= 2:
        print("\n✅ Methods are nearly identical (< 2 samples difference)")
        status = "very_close"
    else:
        print("\n⚠️  Methods produce different results - consider time-based approach")
        status = "different"
    
    return {
        'mean_diff_start': np.mean(index_diffs_start),
        'max_diff_start': np.max(index_diffs_start),
        'mean_diff_end': np.mean(index_diffs_end),
        'max_diff_end': np.max(index_diffs_end),
        'status': status
    }


def main():
    # Parameters (should match train_10_datasets_evaluate_TEST.py)
    min_height = 20
    threshold_on_T = 2500
    min_seconds_between_peaks = 30
    window_length = 5.0
    pre_peak_time = 0.5
    
    signal_names = [
        "V_kite_0_mag",
        "kite_0_vx",
        "kite_0_vy",
        "kite_0_vz",
        "kite_0_altitude",
        "ground_tether_reelout_speed",
        "ground_tether_length",
        "ground_tether_force",
        "kite_pos_east",
        "kite_pos_north",
        "kite_height",
        "kite_elevation",
        "kite_azimuth",
        "kite_distance",
        "kite_heading",
        "kite_course",
        "kite_actual_steering",
        "kite_actual_depower",
        "ground_mech_power",
        "kite_turn_rate",
        "kite_course_rate",
    ]
    
    # Load ALL datasets for testing
    print("="*70)
    print("SEGMENTATION VERIFICATION SCRIPT - MULTI-DATASET CHECK")
    print("="*70)
    
    # Usage: python src/verify_segmentation.py [dataset_folder]
    dataset = sys.argv[1] if len(sys.argv) > 1 else "data/dataset_1"
    test_folder = os.path.join(dataset, "cleaned", "test_data")
    train_folder = os.path.join(dataset, "cleaned", "training_data")
    
    # Check both train and test datasets
    folders_to_check = [
        ("TEST", test_folder),
        ("TRAIN", train_folder)
    ]
    
    all_dataset_stats = []
    
    for folder_type, folder_path in folders_to_check:
        print(f"\n{'='*70}")
        print(f"CHECKING {folder_type} DATASETS: {folder_path}")
        print("="*70)
        
        try:
            loader = CSVLoader(folder_path)
            if len(loader.files) == 0:
                print(f"❌ No CSV files found in {folder_path}")
                continue
            
            print(f"\nFound {len(loader.files)} dataset(s)")
            all_data = loader.load_all()
            
            processor = TractionDataProcessor(
                min_height,
                threshold_on_T,
                min_seconds_between_peaks,
                window_length,
                pre_peak_time,
                plot=False
            )
            
            # Process each dataset and collect statistics
            for idx, (data, file_path) in enumerate(zip(all_data, loader.files)):
                if data is None:
                    print(f"\n⚠️  Dataset {idx+1}: Failed to load")
                    continue
                
                print(f"\n{'-'*70}")
                print(f"Dataset {idx+1}/{len(all_data)}: {os.path.basename(file_path)}")
                print(f"{'-'*70}")
                print(f"Raw data: {len(data)} rows, {len(data.columns)} columns")
                
                # Process data
                results = processor.process(data)
                
                filtered_data = results["filtered_data"]
                time = pd.to_datetime(filtered_data["time"], unit="s")
                time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()
                
                # Collect statistics
                dt = np.diff(time_s)
                mean_dt = np.mean(dt)
                fs_mean = 1.0 / mean_dt if mean_dt > 0 else 0
                
                # Check frame durations
                durations = []
                for start_idx, end_idx in zip(results["frame_start_indices"], results["frame_end_indices"]):
                    if start_idx < len(time_s) and end_idx < len(time_s):
                        actual_duration = time_s[end_idx] - time_s[start_idx]
                        durations.append(actual_duration)
                
                stats = {
                    'folder_type': folder_type,
                    'filename': os.path.basename(file_path),
                    'raw_rows': len(data),
                    'filtered_rows': len(filtered_data),
                    'num_columns': len(data.columns),
                    'fs_mean': fs_mean,
                    'num_peaks': len(results['peak_indices']),
                    'num_frames': len(results['frame_start_indices']),
                    'num_faulty': np.sum(results['frame_labels'] == 1),
                    'num_non_faulty': np.sum(results['frame_labels'] == 0),
                    'mean_frame_duration': np.mean(durations) if len(durations) > 0 else 0,
                }
                
                all_dataset_stats.append(stats)
                
                print(f"Filtered data: {len(filtered_data)} rows")
                print(f"Sampling rate: {fs_mean:.2f} Hz")
                print(f"Peaks: {stats['num_peaks']}")
                print(f"Frames: {stats['num_frames']} (Faulty: {stats['num_faulty']}, Non-faulty: {stats['num_non_faulty']})")
                print(f"Mean frame duration: {stats['mean_frame_duration']:.3f}s")
                
        except FileNotFoundError:
            print(f"❌ Folder not found: {folder_path}")
        except Exception as e:
            print(f"❌ Error processing {folder_path}: {e}")
            import traceback
            traceback.print_exc()
    
    # Cross-dataset comparison
    if len(all_dataset_stats) > 0:
        print(f"\n{'='*70}")
        print("CROSS-DATASET COMPARISON")
        print("="*70)
        
        # Group by folder type
        train_stats = [s for s in all_dataset_stats if s['folder_type'] == 'TRAIN']
        test_stats = [s for s in all_dataset_stats if s['folder_type'] == 'TEST']
        
        print(f"\nSampling Rate Consistency:")
        print(f"{'Dataset':<50} {'Folder':<10} {'Sampling Rate':<15}")
        print(f"{'-'*75}")
        all_fs = []
        for s in all_dataset_stats:
            print(f"{s['filename']:<50} {s['folder_type']:<10} {s['fs_mean']:.2f} Hz")
            all_fs.append(s['fs_mean'])
        
        all_fs = np.array(all_fs)
        if np.std(all_fs) < 0.01:
            print(f"\n✅ All datasets have consistent sampling rate ({np.mean(all_fs):.2f} Hz)")
        else:
            print(f"\n⚠️  Sampling rates vary across datasets (std: {np.std(all_fs):.3f} Hz)")
        
        print(f"\nFrame Duration Consistency:")
        print(f"{'Dataset':<50} {'Folder':<10} {'Mean Duration':<15} {'Error':<10}")
        print(f"{'-'*85}")
        all_durations = []
        for s in all_dataset_stats:
            error = abs(s['mean_frame_duration'] - window_length)
            print(f"{s['filename']:<50} {s['folder_type']:<10} {s['mean_frame_duration']:.3f}s      {error*1000:.1f}ms")
            all_durations.append(s['mean_frame_duration'])
        
        all_durations = np.array(all_durations)
        if np.std(all_durations) < 0.01:
            print(f"\n✅ Frame durations are consistent across datasets")
        else:
            print(f"\n⚠️  Frame durations vary across datasets (std: {np.std(all_durations)*1000:.1f}ms)")
        
        print(f"\nDataset Statistics Summary:")
        print(f"  Total datasets checked: {len(all_dataset_stats)}")
        print(f"    TRAIN: {len(train_stats)}")
        print(f"    TEST: {len(test_stats)}")
        print(f"  Mean sampling rate: {np.mean(all_fs):.2f} ± {np.std(all_fs):.3f} Hz")
        print(f"  Mean frame duration: {np.mean(all_durations):.3f} ± {np.std(all_durations)*1000:.1f}ms")
        print(f"  Expected frame duration: {window_length:.3f}s")
        print(f"  Mean duration error: {abs(np.mean(all_durations) - window_length)*1000:.1f}ms")
        
        # Final assessment
        print(f"\n{'='*70}")
        print("FINAL ASSESSMENT")
        print("="*70)
        
        issues = []
        
        # Check sampling rate consistency
        if np.std(all_fs) >= 0.01:
            issues.append(f"⚠️  Inconsistent sampling rates across datasets (std: {np.std(all_fs):.3f} Hz)")
        
        # Check frame duration consistency
        if np.std(all_durations) >= 0.01:
            issues.append(f"⚠️  Inconsistent frame durations across datasets (std: {np.std(all_durations)*1000:.1f}ms)")
        
        # Check frame duration accuracy
        mean_error = abs(np.mean(all_durations) - window_length)
        if mean_error > 0.05:  # > 50ms
            issues.append(f"⚠️  Frame durations differ from expected ({mean_error*1000:.1f}ms error)")
        
        if len(issues) == 0:
            print("✅ All cross-dataset checks passed!")
            print("   - Sampling rates are consistent")
            print("   - Frame durations are consistent")
            print("   - Segmentation is working correctly across all datasets")
        else:
            print("Issues found:")
            for issue in issues:
                print(f"  {issue}")
            print("\nRecommendation:")
            if np.std(all_fs) >= 0.01:
                print("  - Different datasets have different sampling rates")
                print("  - Time-based segmentation is STRONGLY recommended")
            else:
                print("  - Consider time-based segmentation for more accurate frame durations")
    else:
        print("\n❌ No datasets were successfully processed")


if __name__ == "__main__":
    main()

