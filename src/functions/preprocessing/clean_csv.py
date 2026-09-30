"""
Clean raw flight-log CSVs so they can be read by the training script.

For each file: drop the unnamed index column, forward-fill NaNs, add velocity
magnitudes, convert attitude angles from degrees to radians and unwrap all
angle columns. The result is written as a comma-separated CSV.
"""

from pathlib import Path

import numpy as np
import pandas as pd


def process_all_csvs(input_folder, output_folder=None, sep=' ', fill_nan=True, verbose=True):
    """
    Process all CSV files in a folder: drop index column, fill NaN values, 
    compute velocity magnitudes, convert yaw/roll/pitch to radians and unwrap, 
    and save to cleaned folder.
    
    Parameters:
    -----------
    input_folder : str or Path
        Path to the folder containing CSV files to process
    output_folder : str or Path, optional
        Path to the output folder. If None, creates a "cleaned" subfolder in input_folder.
        Default: None
    sep : str, optional
        Separator used in CSV files. Default: ' ' (space-separated)
    fill_nan : bool, optional
        Whether to fill NaN values with previous values. Default: True
    verbose : bool, optional
        Whether to print progress information. Default: True
    
    Returns:
    --------
    dict
        Dictionary with processing summary:
        - 'processed': list of successfully processed files
        - 'failed': list of files that failed to process
        - 'total': total number of files processed
        - 'angle_types': dict mapping filename to dict of angle column types (degrees/radians)
    """
    input_folder = Path(input_folder)
    
    if not input_folder.exists():
        raise ValueError(f"Input folder does not exist: {input_folder}")
    
    # Set up output folder
    if output_folder is None:
        output_folder = input_folder / "cleaned"
    else:
        output_folder = Path(output_folder)
    
    # Create output folder if it doesn't exist
    output_folder.mkdir(parents=True, exist_ok=True)
    
    if verbose:
        print(f"Input folder: {input_folder}")
        print(f"Output folder: {output_folder}")
        print("-" * 80)
    
    # Find all CSV files
    csv_files = sorted(input_folder.glob("*.csv"))
    
    if len(csv_files) == 0:
        print(f"No CSV files found in {input_folder}")
        return {'processed': [], 'failed': [], 'total': 0, 'angle_types': {}}
    
    if verbose:
        print(f"Found {len(csv_files)} CSV file(s) to process\n")
    
    processed_files = []
    failed_files = []
    angle_types = {}  # Track angle column types per file
    
    # Define yaw/roll/pitch columns to process
    yaw_columns = [
        'kite_0_roll',
        'kite_0_pitch',
        'kite_0_yaw',
        'kite_1_roll',
        'kite_1_pitch',
        'kite_1_yaw',
        'kite_heading',
        'kite_course'
    ]
    
    for i, csv_file in enumerate(csv_files, 1):
        if verbose:
            print(f"[{i}/{len(csv_files)}] Processing: {csv_file.name}")
        
        try:
            # Load the CSV file
            df = pd.read_csv(csv_file, sep=sep, low_memory=False)
            
            if verbose:
                print(f"  Loaded: {df.shape[0]} rows × {df.shape[1]} columns")
            
            # Check if the first column is an index column (unnamed)
            first_col = df.columns[0]
            is_index_col = (first_col.startswith('Unnamed:') or 
                          first_col == '' or 
                          (isinstance(first_col, str) and first_col.strip() == ''))
            
            if is_index_col:
                # Drop the first column (the index column)
                df = df.drop(df.columns[0], axis=1)
                if verbose:
                    print(f"  ✓ Dropped index column: '{first_col}'")
            
            # Fill NaN values if requested
            if fill_nan:
                nan_before = df.isna().sum().sum()
                df = df.ffill()  # Forward fill
                nan_after = df.isna().sum().sum()
                if verbose:
                    if nan_before > 0:
                        print(f"  ✓ Filled {nan_before - nan_after} NaN values")
                    else:
                        print(f"  ✓ No NaN values to fill")

            # Compute velocity magnitudes
            for kite_idx in (0, 1):
                cols = [f'kite_{kite_idx}_vx', f'kite_{kite_idx}_vy', f'kite_{kite_idx}_vz']
                missing = [c for c in cols if c not in df.columns]
                if missing:
                    if verbose:
                        print(f"  Skipping V_kite_{kite_idx}_mag: missing columns {missing}")
                    continue

                # Ensure numeric (coerce non-numeric -> NaN), then magnitude = sqrt(vx^2+vy^2+vz^2)
                arr = df[cols].apply(pd.to_numeric, errors='coerce')
                df[f'V_kite_{kite_idx}_mag'] = np.sqrt((arr ** 2).sum(axis=1))
                if verbose:
                    print(f"  ✓ Added column: V_kite_{kite_idx}_mag (from {cols})")
            
            # Process yaw/roll/pitch columns: convert deg to rad and unwrap
            file_angle_types = {}  # Track types for this file
            for col in yaw_columns:
                if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
                    # Check if values exist and are non-null
                    if df[col].notna().sum() == 0:
                        if verbose:
                            print(f"  Skipping {col}: all values are NaN")
                        continue
                    
                    # Heuristic: assume degrees if values exceed ±2π
                    max_abs_val = df[col].abs().max()
                    if max_abs_val > 2 * np.pi:
                        # Degrees → radians → unwrap
                        yaw_rad = np.deg2rad(df[col].values)
                        df[col] = np.unwrap(yaw_rad)
                        file_angle_types[col] = 'degrees'
                        if verbose:
                            print(f"  ✓ Converted {col} from degrees to radians and unwrapped (max abs value: {max_abs_val:.4f})")
                    else:
                        # Already radians, just unwrap
                        df[col] = np.unwrap(df[col].values)
                        file_angle_types[col] = 'radians'
                        if verbose:
                            print(f"  ✓ Unwrapped {col} (already radians, max abs value: {max_abs_val:.4f})")
            
            # Store angle types for this file
            if file_angle_types:
                angle_types[csv_file.name] = file_angle_types
            
            # Save to output folder
            output_path = output_folder / csv_file.name
            df.to_csv(output_path, index=False, sep=',')
            
            if verbose:
                print(f"  ✓ Saved to: {output_path}")
            
            processed_files.append(csv_file.name)
            
        except Exception as e:
            error_msg = f"  ✗ Error processing {csv_file.name}: {str(e)}"
            if verbose:
                print(error_msg)
            failed_files.append({'file': csv_file.name, 'error': str(e)})
        
        if verbose:
            print()
    
    # Summary
    if verbose:
        print("=" * 80)
        print("PROCESSING SUMMARY")
        print("=" * 80)
        print(f"Total files: {len(csv_files)}")
        print(f"Successfully processed: {len(processed_files)}")
        print(f"Failed: {len(failed_files)}")
        
        if failed_files:
            print("\nFailed files:")
            for item in failed_files:
                print(f"  - {item['file']}: {item['error']}")
        
        # Print angle type consistency report
        if angle_types:
            print("\n" + "=" * 80)
            print("ANGLE COLUMN TYPE CONSISTENCY REPORT")
            print("=" * 80)
            
            # Collect all columns and their types across files
            all_cols = set()
            for file_types in angle_types.values():
                all_cols.update(file_types.keys())
            
            if all_cols:
                print(f"\nChecking consistency for {len(all_cols)} angle column(s):")
                for col in sorted(all_cols):
                    col_types = {}
                    for filename, file_types in angle_types.items():
                        if col in file_types:
                            detected_type = file_types[col]
                            if detected_type not in col_types:
                                col_types[detected_type] = []
                            col_types[detected_type].append(filename)
                    
                    if len(col_types) == 1:
                        # Consistent across all files
                        detected_type = list(col_types.keys())[0]
                        print(f"  ✓ {col:30s}: {detected_type:8s} (consistent across {len(col_types[detected_type])} file(s))")
                    else:
                        # Inconsistent - report which files have which type
                        print(f"  ⚠ {col:30s}: INCONSISTENT")
                        for detected_type, files in col_types.items():
                            print(f"      - {detected_type:8s}: {len(files)} file(s) - {', '.join(files)}")
    
    return {
        'processed': processed_files,
        'failed': failed_files,
        'total': len(csv_files),
        'angle_types': angle_types
    }
