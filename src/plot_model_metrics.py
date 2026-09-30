"""
Standalone script to plot metrics from a saved model.

Usage:
    python plot_model_metrics.py <model_dir> [--save <output_path>]

Example:
    python src/plot_model_metrics.py saved_models/svm_model
    python src/plot_model_metrics.py saved_models/svm_model --save metrics_plot.png
"""

import sys
import argparse
from functions.ml.plot_metrics import plot_model_metrics


def main():
    parser = argparse.ArgumentParser(
        description='Plot metrics from a saved SVM model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        'model_dir',
        type=str,
        help='Path to directory containing saved model'
    )
    
    parser.add_argument(
        '--save',
        type=str,
        default=None,
        help='Path to save the plot (if not provided, displays the plot)'
    )
    
    args = parser.parse_args()
    
    #try:
    plot_model_metrics(args.model_dir, save_path=args.save)
    print("\n✓ Metrics plotted successfully!")
   # except Exception as e:
    #    print(f"\n✗ Error plotting metrics: {e}", file=sys.stderr)
    #    sys.exit(1)


if __name__ == "__main__":
    main()

