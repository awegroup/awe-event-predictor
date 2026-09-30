import os
import pandas as pd
from typing import List, Optional


class CSVLoader:
    """
    Loads multiple CSV files from a folder, enforcing consistent columns.
    """

    def __init__(self, folder: str):
        self.folder = folder
        self.files = self._discover_csv_files()
        self.expected_columns: Optional[List[str]] = None

    def _discover_csv_files(self) -> List[str]:
        """Return paths to all CSV files in the folder, sorted by name."""
        return [
            os.path.join(self.folder, f)
            for f in sorted(os.listdir(self.folder))
            if f.lower().endswith(".csv")
        ]

    def load_all(self) -> List[Optional[pd.DataFrame]]:
        """
        Load all CSV files and check column consistency.

        Returns
        -------
        List of DataFrames (None for failed files)
        """
        all_data: List[Optional[pd.DataFrame]] = []

        for idx, path in enumerate(self.files, start=1):
            print(f"Importing {idx}/{len(self.files)}: {os.path.basename(path)}")

            try:
                df = self._load_single_csv(path)

                # First file: record expected columns
                if self.expected_columns is None:
                    self.expected_columns = list(df.columns)
                else:
                    self._check_column_consistency(df, path)

                all_data.append(df)

            except Exception as e:
                print(f"⚠️ Failed to import {path}: {e}")
                all_data.append(None)

        return all_data

    def _load_single_csv(self, path: str) -> pd.DataFrame:
        """Load one CSV file using pandas."""
        return pd.read_csv(path)

    def _check_column_consistency(self, df: pd.DataFrame, path: str):
        """Warn if the column count or order differs."""
        cols = list(df.columns)

        if len(cols) != len(self.expected_columns):
            print(f"⚠️ Column count mismatch in {os.path.basename(path)}")

        if cols != self.expected_columns:
            print(f"⚠️ Column name/order mismatch in {os.path.basename(path)}")

            # Show the differences
            mismatch = set(cols).symmetric_difference(self.expected_columns)
            print("   differing columns:", mismatch)