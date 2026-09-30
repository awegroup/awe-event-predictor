"""
Per-class and macro metrics from a confusion matrix.

Usage:
    python src/compute_metrics.py --tn 90 --fp 10 --fn 5 --tp 15
"""

def safe_div(num, den):
    return num / den if den != 0 else 0.0


def classification_metrics_from_confusion(TN, FP, FN, TP):
    # ----- Class 1 (positive class) -----
    precision_1 = safe_div(TP, TP + FP)
    recall_1    = safe_div(TP, TP + FN)
    f1_1        = safe_div(2 * TP, 2 * TP + FP + FN)

    # ----- Class 0 (treat class 0 as positive) -----
    precision_0 = safe_div(TN, TN + FN)
    recall_0    = safe_div(TN, TN + FP)
    f1_0        = safe_div(2 * TN, 2 * TN + FN + FP)

    # ----- Macro averages -----
    precision_macro = (precision_0 + precision_1) / 2
    recall_macro    = (recall_0 + recall_1) / 2
    f1_macro        = (f1_0 + f1_1) / 2

    return {
        "class_0": {
            "precision": precision_0,
            "recall": recall_0,
            "f1": f1_0,
        },
        "class_1": {
            "precision": precision_1,
            "recall": recall_1,
            "f1": f1_1,
        },
        "macro_avg": {
            "precision": precision_macro,
            "recall": recall_macro,
            "f1": f1_macro,
        }
    }


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Per-class and macro precision, recall and F1 from a binary confusion matrix."
    )
    parser.add_argument("--tn", type=int, required=True, help="True negatives")
    parser.add_argument("--fp", type=int, required=True, help="False positives")
    parser.add_argument("--fn", type=int, required=True, help="False negatives")
    parser.add_argument("--tp", type=int, required=True, help="True positives")
    args = parser.parse_args()

    metrics = classification_metrics_from_confusion(args.tn, args.fp, args.fn, args.tp)

    for k, v in metrics.items():
        print(k, v)


if __name__ == "__main__":
    main()
