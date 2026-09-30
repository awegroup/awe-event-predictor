"""
Clean the raw flight logs of a dataset.

Expects the layout

    <dataset>/training_data/*.csv   raw training flights
    <dataset>/test_data/*.csv       raw out-of-domain test flights

and writes the cleaned files to

    <dataset>/cleaned/training_data/*.csv
    <dataset>/cleaned/test_data/*.csv

Usage:
    python src/preprocess.py --dataset data/dataset_1
"""

import argparse
import sys
from pathlib import Path

from functions.preprocessing.clean_csv import process_all_csvs

SPLITS = ("training_data", "test_data")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", default="data/dataset_1", help="Dataset folder (default: data/dataset_1)")
    parser.add_argument("--sep", default=" ", help="Column separator of the raw CSVs (default: space)")
    parser.add_argument("--no-fill-nan", action="store_true", help="Do not forward-fill NaN values")
    parser.add_argument("--quiet", action="store_true", help="Only print errors")
    args = parser.parse_args()

    dataset = Path(args.dataset)
    failed = 0
    for split in SPLITS:
        input_folder = dataset / split
        if not input_folder.is_dir():
            sys.exit(f"Missing folder: {input_folder}")
        result = process_all_csvs(
            input_folder,
            output_folder=dataset / "cleaned" / split,
            sep=args.sep,
            fill_nan=not args.no_fill_nan,
            verbose=not args.quiet,
        )
        failed += len(result["failed"])
        for item in result["failed"]:
            print(f"✗ {split}/{item['file']}: {item['error']}", file=sys.stderr)

    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
