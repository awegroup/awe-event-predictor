from __future__ import annotations
import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, Tuple
import seaborn as sns

from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_curve,
    auc,
    accuracy_score,
)

from functions.ml.model_persistence import load_model
import pickle
import pandas as pd


def plot_model_metrics(model_dir: str, save_path: Optional[str] = None, figsize: tuple = (16, 10)
                      ) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """
    Load model from directory and plot:
      - Main evaluation dashboard (uses saved y_pred with saved threshold from model)
      - Score Distribution Plot (by True Class) showing separability
      - Separate Precision–Recall curve figure (find best threshold by maximizing F1)
      - Print comparison of original vs optimal-threshold metrics
      - Plot Optimal-Threshold Confusion Matrix figure

    Parameters
    ----------
    model_dir : str
        Path to directory containing saved model (used by functions.ml.model_persistence.load_model).
    save_path : Optional[str]
        If provided, save the main figure to that path.
    figsize : tuple
        Size for the main figure.

    Returns
    -------
    (fig_main, fig_pr, fig_opt_cm)
        Matplotlib Figure objects for main dashboard, PR curve, and optimal-threshold confusion matrix.
    """

    # -------------------------
    # Load saved model data
    # -------------------------
    model_data = load_model(model_dir)

    y_test = model_data["y_test"]
    y_pred = model_data["y_pred"]
    y_proba = model_data["y_proba"]
    metrics = model_data.get("metrics", {})
    
    # Get threshold from saved model data (check multiple sources)
    threshold = 0.5  # Default fallback
    if "threshold" in model_data:
        threshold = float(model_data["threshold"])
    elif "training_params" in model_data and "optimal_threshold" in model_data["training_params"]:
        threshold = float(model_data["training_params"]["optimal_threshold"])

    # Ensure y_proba is 1D probabilities for positive class
    if hasattr(y_proba, "ndim") and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        # either already 1D or something else (attempt to flatten)
        y_proba_pos = np.ravel(y_proba)

    # -------------------------
    # MAIN FIGURE (original metrics, using saved threshold)
    # -------------------------
    fig_main = plt.figure(figsize=figsize)
    gs = fig_main.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

    # Per-class precision/recall/f1 (original predictions)
    precision_per_class = precision_score(y_test, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_test, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_test, y_pred, average=None, zero_division=0)

    classes = ["Non-Faulty (0)", "Faulty (1)"]
    x = np.arange(len(classes))
    width = 0.25

    ax1 = fig_main.add_subplot(gs[0, 0])
    bars1 = ax1.bar(x - width, precision_per_class, width, label="Precision", alpha=0.8)
    bars2 = ax1.bar(x, recall_per_class, width, label="Recall", alpha=0.8)
    bars3 = ax1.bar(x + width, f1_per_class, width, label="F1-Score", alpha=0.8)

    ax1.set_xlabel("Class", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax1.set_title(f"Per-Class Metrics: Precision, Recall, F1 (threshold={threshold:.3f})", fontsize=14, fontweight="bold")
    ax1.set_xticks(x)
    ax1.set_xticklabels(classes)
    ax1.set_ylim([0, 1.05])
    ax1.legend(loc="upper right", fontsize=9)
    ax1.grid(axis="y", alpha=0.25)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            ax1.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                     f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=9)

    # Accuracy & Balanced Accuracy
    ax2 = fig_main.add_subplot(gs[0, 1])
    bacc = balanced_accuracy_score(y_test, y_pred)
    accuracy = metrics.get("accuracy", accuracy_score(y_test, y_pred))

    bars = ax2.bar(["Accuracy", "Balanced\nAccuracy"], [accuracy, bacc], alpha=0.8)
    ax2.set_ylabel("Score", fontsize=12, fontweight="bold")
    ax2.set_title(f"Accuracy Metrics (threshold={threshold:.3f})", fontsize=14, fontweight="bold")
    ax2.set_ylim([0, 1.05])
    ax2.grid(axis="y", alpha=0.25)
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width() / 2.0, bar.get_height(),
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

    # Class distribution pie
    ax3 = fig_main.add_subplot(gs[0, 2])
    unique_classes, class_counts = np.unique(y_test, return_counts=True)
    # Build labels robustly in case classes aren't [0,1] ordering
    labels = []
    for cls in unique_classes:
        if cls == 0:
            labels.append(f"Non-Faulty (0)\n{(class_counts[unique_classes == cls])[0]} samples")
        else:
            labels.append(f"Faulty (1)\n{(class_counts[unique_classes == cls])[0]} samples")

    ax3.pie(class_counts, labels=labels, autopct="%1.1f%%", startangle=90,
            explode=(0.05,) * len(class_counts), shadow=True, textprops={"fontsize": 10})
    ax3.set_title("Test Set Class Distribution", fontsize=14, fontweight="bold")

    # Confusion matrix (original)
    ax4 = fig_main.add_subplot(gs[1, :2])
    cm_orig = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm_orig, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax4,
                annot_kws={"size": 12, "weight": "bold"})
    ax4.set_xlabel("Predicted Label", fontsize=12, fontweight="bold")
    ax4.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax4.set_title(f"Confusion Matrix (threshold={threshold:.3f})", fontsize=14, fontweight="bold")
    if cm_orig.shape == (2, 2):
        tn, fp, fn, tp = cm_orig.ravel()
        ax4.text(0.5, -0.12, f"TN={tn}  FP={fp}    FN={fn}  TP={tp}",
                 transform=ax4.transAxes, ha="center", fontsize=10,
                 bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    # Summary table (original)
    ax5 = fig_main.add_subplot(gs[1, 2])
    ax5.axis("off")
    summary_data = [
        ["Metric", "Value"],
        ["Overall Accuracy", f"{accuracy:.4f}"],
        ["Balanced Accuracy", f"{bacc:.4f}"],
        ["Macro F1", f"{metrics.get('macro_f1', (f1_score(y_test, y_pred, average='macro') if len(np.unique(y_test))>1 else 0)):.4f}"],
        ["", ""],
        ["Precision (Class 0)", f"{precision_per_class[0]:.4f}"],
        ["Recall (Class 0)", f"{recall_per_class[0]:.4f}"],
        ["F1 (Class 0)", f"{f1_per_class[0]:.4f}"],
        ["", ""],
        ["Precision (Class 1)", f"{precision_per_class[1]:.4f}"],
        ["Recall (Class 1)", f"{recall_per_class[1]:.4f}"],
        ["F1 (Class 1)", f"{f1_per_class[1]:.4f}"],
    ]

    if metrics.get("roc_auc") is not None:
        summary_data.append(["ROC-AUC", f"{metrics['roc_auc']:.4f}"])
    if metrics.get("pr_auc") is not None:
        summary_data.append(["PR-AUC", f"{metrics['pr_auc']:.4f}"])

    table = ax5.table(cellText=summary_data, cellLoc="left", loc="center", colWidths=[0.6, 0.4])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    for i in range(2):
        table[(0, i)].set_facecolor("#34495e")
        table[(0, i)].set_text_props(weight="bold", color="white")

    ax5.set_title(f"Summary Statistics (threshold={threshold:.3f})", fontsize=14, fontweight="bold", pad=12)

    fig_main.suptitle("SVM Model Evaluation Metrics (Original Predictions)", fontsize=16, fontweight="bold", y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Main metrics figure saved to: {save_path}")
    else:
        plt.show()

    # -------------------------
    # SCORE DISTRIBUTION PLOT (by True Class)
    # -------------------------
    # Use the threshold loaded from saved model data (already set above)
    
    # Separate probabilities by true class
    y_proba_class0 = y_proba_pos[y_test == 0]
    y_proba_class1 = y_proba_pos[y_test == 1]
    
    fig_dist = plt.figure(figsize=(12, 7))
    ax_dist = fig_dist.add_subplot(1, 1, 1)
    
    # Plot KDE curves
    sns.kdeplot(data=y_proba_class0, label="Non-Faulty (Class 0)", ax=ax_dist, 
                fill=True, alpha=0.5, color="blue", linewidth=2)
    sns.kdeplot(data=y_proba_class1, label="Faulty (Class 1)", ax=ax_dist, 
                fill=True, alpha=0.5, color="red", linewidth=2)
    
    # Overlay histograms for better visual clarity
    ax_dist.hist(y_proba_class0, bins=30, alpha=0.3, color="blue", 
                 density=True, label=None, edgecolor="black", linewidth=0.5)
    ax_dist.hist(y_proba_class1, bins=30, alpha=0.3, color="red", 
                 density=True, label=None, edgecolor="black", linewidth=0.5)
    
    # Add vertical line at decision threshold
    ax_dist.axvline(x=threshold, color="black", linestyle="--", linewidth=2, 
                    label=f"Decision Threshold = {threshold}")
    
    # Calculate TP/FP/FN/TN counts for annotation
    if cm_orig.shape == (2, 2):
        tn, fp, fn, tp = cm_orig.ravel()
        
        # Add text annotations showing TP/FP/FN/TN regions
        ax_dist.text(0.98, 0.95, 
                    f"TP (Class 1, ≥{threshold}): {tp}\n"
                    f"FN (Class 1, <{threshold}): {fn}\n"
                    f"FP (Class 0, ≥{threshold}): {fp}\n"
                    f"TN (Class 0, <{threshold}): {tn}",
                    transform=ax_dist.transAxes, 
                    fontsize=10, verticalalignment="top", horizontalalignment="right",
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.8),
                    family="monospace")
    
    # Shade regions to show TP/FP/FN/TN areas (optional visual aid)
    ylim = ax_dist.get_ylim()
    ax_dist.fill_between([threshold, 1.0], 0, ylim[1], alpha=0.1, color="green", 
                         label="Predicted Faulty (≥threshold)")
    ax_dist.fill_between([0.0, threshold], 0, ylim[1], alpha=0.1, color="orange", 
                         label="Predicted Non-Faulty (<threshold)")
    ax_dist.set_ylim(ylim)
    
    ax_dist.set_xlabel("Predicted Probability (Class 1: Faulty)", fontsize=12, fontweight="bold")
    ax_dist.set_ylabel("Density", fontsize=12, fontweight="bold")
    ax_dist.set_title("Score Distribution by True Class (shows separability & threshold effects)", 
                     fontsize=14, fontweight="bold")
    ax_dist.legend(loc="upper left", fontsize=10)
    ax_dist.grid(alpha=0.25, axis="y")
    ax_dist.set_xlim([-0.05, 1.05])
    
    plt.tight_layout()
    plt.show()

    # -------------------------
    # PR CURVE FIGURE (separate)
    # -------------------------
    precision_curve, recall_curve, pr_thresholds = precision_recall_curve(y_test, y_proba_pos)
    pr_auc = auc(recall_curve, precision_curve)

    # Compute F1 at each threshold explicitly (thresholds correspond to pr_thresholds)
    f1_scores = []
    precisions_at_thr = []
    recalls_at_thr = []
    for t in pr_thresholds:
        preds_t = (y_proba_pos >= t).astype(int)
        precisions_at_thr.append(precision_score(y_test, preds_t, zero_division=0))
        recalls_at_thr.append(recall_score(y_test, preds_t, zero_division=0))
        f1_scores.append(f1_score(y_test, preds_t, zero_division=0))

    f1_scores = np.array(f1_scores)
    precisions_at_thr = np.array(precisions_at_thr)
    recalls_at_thr = np.array(recalls_at_thr)

    if f1_scores.size > 0:
        best_idx = int(np.argmax(f1_scores))
        best_threshold = pr_thresholds[best_idx]
        best_f1 = f1_scores[best_idx]
        best_precision = precisions_at_thr[best_idx]
        best_recall = recalls_at_thr[best_idx]
    else:
        # Edge-case: no thresholds returned (e.g., constant scores) — fallback to 0.5
        best_threshold = 0.5
        best_precision = precision_score(y_test, (y_proba_pos >= best_threshold).astype(int), zero_division=0)
        best_recall = recall_score(y_test, (y_proba_pos >= best_threshold).astype(int), zero_division=0)
        best_f1 = f1_score(y_test, (y_proba_pos >= best_threshold).astype(int), zero_division=0)

    fig_pr = plt.figure(figsize=(9, 7))
    plt.plot(recall_curve, precision_curve, linewidth=2, label=f"PR Curve (AUC = {pr_auc:.3f})")
    # Annotate best threshold (we use the explicit best_precision/recall computed above)
    plt.scatter([best_recall], [best_precision], s=120, edgecolor="black",
                label=f"Best (max F1) = {best_f1:.3f}\nThreshold = {best_threshold:.3f}")
    plt.xlabel("Recall", fontsize=12, fontweight="bold")
    plt.ylabel("Precision", fontsize=12, fontweight="bold")
    plt.title(f"Precision–Recall Curve (PR-AUC = {pr_auc:.3f})", fontsize=14, fontweight="bold")
    plt.grid(alpha=0.25)
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.show()

    # -------------------------
    # Compute metrics at optimal threshold (do NOT overwrite original)
    # -------------------------
    y_pred_opt = (y_proba_pos >= best_threshold).astype(int)

    # Metrics under optimal threshold
    accuracy_opt = accuracy_score(y_test, y_pred_opt)
    bacc_opt = balanced_accuracy_score(y_test, y_pred_opt)
    precision_per_class_opt = precision_score(y_test, y_pred_opt, average=None, zero_division=0)
    recall_per_class_opt = recall_score(y_test, y_pred_opt, average=None, zero_division=0)
    f1_per_class_opt = f1_score(y_test, y_pred_opt, average=None, zero_division=0)
    macro_f1_opt = f1_score(y_test, y_pred_opt, average="macro", zero_division=0)
    cm_opt = confusion_matrix(y_test, y_pred_opt)

    # -------------------------
    # PRINT COMPARISON REPORT
    # -------------------------
    print("\n" + "=" * 70)
    print(f"Original metrics (threshold = {threshold:.4f})".ljust(40) + " | " + "Optimal-threshold metrics (max-F1)".rjust(25))
    print("=" * 70)
    print(f"Best threshold (max-F1): {best_threshold:.4f}")
    print("-" * 70)
    print(f"{'Metric':<30} {'Original':>10} {'Optimal':>14} {'Delta':>12}")
    print("-" * 70)

    def fmt(a): return f"{a:.4f}"

    # Accuracy
    print(f"{'Accuracy':<30} {fmt(accuracy):>10} {fmt(accuracy_opt):>14} {fmt(accuracy_opt - accuracy):>12}")
    # Balanced accuracy
    print(f"{'Balanced Accuracy':<30} {fmt(bacc):>10} {fmt(bacc_opt):>14} {fmt(bacc_opt - bacc):>12}")
    # Macro F1
    macro_f1_orig = metrics.get("macro_f1", (f1_score(y_test, y_pred, average="macro") if len(np.unique(y_test))>1 else 0.0))
    print(f"{'Macro F1':<30} {fmt(macro_f1_orig):>10} {fmt(macro_f1_opt):>14} {fmt(macro_f1_opt - macro_f1_orig):>12}")
    print("-" * 70)
    # Class-level breakdown
    print("Per-class metrics (class 0):")
    print(f"  Precision: {precision_per_class[0]:.4f} -> {precision_per_class_opt[0]:.4f}")
    print(f"  Recall:    {recall_per_class[0]:.4f} -> {recall_per_class_opt[0]:.4f}")
    print(f"  F1:        {f1_per_class[0]:.4f} -> {f1_per_class_opt[0]:.4f}")
    print("")
    print("Per-class metrics (class 1):")
    print(f"  Precision: {precision_per_class[1]:.4f} -> {precision_per_class_opt[1]:.4f}")
    print(f"  Recall:    {recall_per_class[1]:.4f} -> {recall_per_class_opt[1]:.4f}")
    print(f"  F1:        {f1_per_class[1]:.4f} -> {f1_per_class_opt[1]:.4f}")
    print("=" * 70 + "\n")

    # -------------------------
    # PLOT Optimal Threshold Confusion Matrix (new figure)
    # -------------------------
    fig_opt_cm = plt.figure(figsize=(7, 6))
    ax_opt = fig_opt_cm.add_subplot(1, 1, 1)
    sns.heatmap(cm_opt, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax_opt,
                annot_kws={"size": 12, "weight": "bold"})
    ax_opt.set_xlabel("Predicted Label (optimal threshold)", fontsize=12, fontweight="bold")
    ax_opt.set_ylabel("True Label", fontsize=12, fontweight="bold")
    ax_opt.set_title(f"Confusion Matrix @ threshold={best_threshold:.3f}", fontsize=14, fontweight="bold")

    if cm_opt.shape == (2, 2):
        tn_o, fp_o, fn_o, tp_o = cm_opt.ravel()
        ax_opt.text(0.5, -0.12, f"TN={tn_o}  FP={fp_o}    FN={fn_o}  TP={tp_o}",
                    transform=ax_opt.transAxes, ha="center", fontsize=10,
                    bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))

    plt.tight_layout()
    plt.show()

    # -------------------------
    # PLOT PREDICTION CATEGORIES (if data available)
    # -------------------------
    try:
        plot_prediction_categories(model_dir, y_test, y_pred)
    except Exception as e:
        print(f"\nNote: Prediction category plotting skipped: {e}")
        print("  (This is optional and requires test processing data to be saved)")

    # Return figures for further use if desired
    return fig_main, fig_pr, fig_opt_cm


def plot_prediction_categories(model_dir: str, y_test: np.ndarray, y_pred: np.ndarray) -> None:
    """
    Plot all prediction categories (TP, TN, FP, FN) from saved model.
    
    Parameters
    ----------
    model_dir : str
        Path to directory containing saved model.
    y_test : np.ndarray
        True test labels.
    y_pred : np.ndarray
        Predicted labels.
    """
    # Try to load test processing data
    test_processing_path = os.path.join(model_dir, "test_processing_data.pkl")
    if not os.path.exists(test_processing_path):
        raise FileNotFoundError(
            "Test processing data not found. Prediction category plotting requires "
            "test_processing_data.pkl to be saved with the model."
        )
    
    with open(test_processing_path, 'rb') as f:
        test_processing_data = pickle.load(f)
    
    test_processing_results = test_processing_data.get("test_processing_results")
    test_frame_mapping = test_processing_data.get("test_frame_mapping")
    test_file_paths = test_processing_data.get("test_file_paths", [])
    window_length = test_processing_data.get("window_length", 5.0)
    pre_peak_time = test_processing_data.get("pre_peak_time", 0.3)
    threshold_on_T = test_processing_data.get("threshold_on_T", 2500)
    
    if test_processing_results is None or test_frame_mapping is None:
        raise ValueError("Test processing results or frame mapping not found in saved data")
    
    from functions.preprocessing.plots import plot_overlapping_frames_for_signal
    
    # Select signals to plot (most informative ones)
    signals_to_plot = ["ground_tether_force"] #, "ground_tether_reelout_speed", "kite_0_yaw", "kite_elevation"]  # Can add more: "kite_1_pitch", "kite_azimuth", "kite_height"
    
    # Limit number of datasets to plot (to avoid too many plots)
    max_datasets_to_plot = 3
    
    # Define all prediction categories
    categories = {
        "True Positives (Correctly Predicted Faulty)": {
            "mask": (y_pred == 1) & (y_test == 1),
            "label_value": 1,
            "description": "predicted faulty, actually faulty"
        },
        "True Negatives (Correctly Predicted Non-Faulty)": {
            "mask": (y_pred == 0) & (y_test == 0),
            "label_value": 0,
            "description": "predicted non-faulty, actually non-faulty"
        },
        "False Positives (Incorrectly Predicted Faulty)": {
            "mask": (y_pred == 1) & (y_test == 0),
            "label_value": 0,
            "description": "predicted faulty, actually non-faulty"
        },
        "False Negatives (Incorrectly Predicted Non-Faulty)": {
            "mask": (y_pred == 0) & (y_test == 1),
            "label_value": 1,
            "description": "predicted non-faulty, actually faulty"
        }
    }
    
    # Plot each category
    for category_name, category_info in categories.items():
        print("\n" + "="*60)
        print(f"Analyzing {category_name}")
        print("="*60)
        
        indices = np.where(category_info["mask"])[0]
        num_samples = len(indices)
        print(f"Found {num_samples} samples ({category_info['description']})")
        
        if num_samples == 0:
            print(f"No {category_name.lower()} found - skipping plots.")
            continue
        
        # Group by dataset
        samples_by_dataset = {}
        for idx in indices:
            if idx >= len(test_frame_mapping):
                continue
            dataset_idx, frame_idx = test_frame_mapping[idx]
            if dataset_idx not in samples_by_dataset:
                samples_by_dataset[dataset_idx] = []
            samples_by_dataset[dataset_idx].append(frame_idx)
        
        print(f"Distributed across {len(samples_by_dataset)} test dataset(s)")
        
        datasets_plotted = 0
        for dataset_idx, frame_indices in samples_by_dataset.items():
            if datasets_plotted >= max_datasets_to_plot:
                print(f"\n(Plotting limited to first {max_datasets_to_plot} datasets)")
                break
            
            if dataset_idx >= len(test_processing_results):
                continue
            
            results = test_processing_results[dataset_idx]
            dataset_name = os.path.basename(test_file_paths[dataset_idx]) if dataset_idx < len(test_file_paths) else f"dataset_{dataset_idx}"
            
            print(f"\nPlotting {len(frame_indices)} samples from dataset: {dataset_name}")
            
            # Create processing output with only these frames
            processing_output = {
                "filtered_data": results["filtered_data"],
                "frame_start_indices": results["frame_start_indices"][frame_indices],
                "frame_end_indices": results["frame_end_indices"][frame_indices],
                "frame_labels": results["frame_labels"][frame_indices],
                "frame_times": [results["frame_times"][i] for i in frame_indices],
                "peak_indices": results.get("peak_indices", np.array([])),
                "threshold_on_T": results.get("threshold_on_T", threshold_on_T)
            }
            
            # Plot each signal
            for signal_name in signals_to_plot:
                if signal_name in results["filtered_data"].columns:
                    print(f"  Plotting {signal_name}...")
                    plot_overlapping_frames_for_signal(
                        signal_name=signal_name,
                        raw_df=results["filtered_data"],
                        processing_output=processing_output,
                        window_length=window_length,
                        label_value=category_info["label_value"],
                        threshold=results.get("threshold_on_T", threshold_on_T),
                        pre_peak_time=pre_peak_time,
                        figsize=(12, 6),
                        title_suffix=f" - {category_name}"  # Add category to title
                    )
            
            datasets_plotted += 1
        
        if num_samples > 0 and datasets_plotted == 0:
            print(f"(No datasets plotted - all samples are in datasets beyond the limit)")


def plot_samples_by_probability_ranges(
    model_dir: str,
    output_base_dir: str,
    prob_ranges: Optional[list] = None,
    figsize: tuple = (10, 6)
) -> None:
    """
    Visualize samples grouped by predicted probability confidence ranges.
    For each range, creates a folder structure with individual sample folders,
    each containing PNG plots of all raw signals for that sample.
    
    Parameters
    ----------
    model_dir : str
        Path to directory containing saved model (used by functions.ml.model_persistence.load_model).
    output_base_dir : str
        Base directory where probability range folders will be created.
    prob_ranges : Optional[list]
        List of tuples defining probability ranges. Default: [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)]
        Each tuple is (min_prob, max_prob). For all ranges except the last, max_prob is exclusive.
        For the last range, max_prob is inclusive.
    figsize : tuple
        Figure size for individual signal plots.
    
    Returns
    -------
    None
        Creates directory structure and saves PNG files.
    """
    # Default probability ranges
    if prob_ranges is None:
        prob_ranges = [(0.0, 0.4), (0.4, 0.6), (0.6, 1.0)]
    
    # Load saved model data
    print(f"Loading model from: {model_dir}")
    model_data = load_model(model_dir)
    
    y_test = model_data["y_test"]
    y_proba = model_data["y_proba"]
    
    if y_proba is None:
        raise ValueError("Predicted probabilities (y_proba) not found in saved model data.")
    
    # Ensure y_proba is 1D probabilities for positive class
    if hasattr(y_proba, "ndim") and y_proba.ndim == 2 and y_proba.shape[1] == 2:
        y_proba_pos = y_proba[:, 1]
    else:
        y_proba_pos = np.ravel(y_proba)
    
    # Load test processing data
    test_processing_path = os.path.join(model_dir, "test_processing_data.pkl")
    if not os.path.exists(test_processing_path):
        raise FileNotFoundError(
            "Test processing data not found. Probability range visualization requires "
            "test_processing_data.pkl to be saved with the model."
        )
    
    with open(test_processing_path, 'rb') as f:
        test_processing_data = pickle.load(f)
    
    test_processing_results = test_processing_data.get("test_processing_results")
    test_frame_mapping = test_processing_data.get("test_frame_mapping")
    test_file_paths = test_processing_data.get("test_file_paths", [])
    window_length = test_processing_data.get("window_length", 5.0)
    pre_peak_time = test_processing_data.get("pre_peak_time", 0.5)
    post_peak_time = test_processing_data.get("post_peak_time", 3.0)
    threshold_on_T = test_processing_data.get("threshold_on_T", 2500)
    
    if test_processing_results is None or test_frame_mapping is None:
        raise ValueError("Test processing results or frame mapping not found in saved data")
    
    # Create output base directory
    os.makedirs(output_base_dir, exist_ok=True)
    print(f"\nOutput directory: {output_base_dir}")
    
    # Process each probability range
    for range_idx, (min_prob, max_prob) in enumerate(prob_ranges):
        # Create range folder name
        range_name = f"prob_{min_prob:.1f}_to_{max_prob:.1f}"
        range_dir = os.path.join(output_base_dir, range_name)
        os.makedirs(range_dir, exist_ok=True)
        
        # Find samples in this probability range
        if range_idx == len(prob_ranges) - 1:
            # Last range: include max_prob
            mask = (y_proba_pos >= min_prob) & (y_proba_pos <= max_prob)
        else:
            # Other ranges: exclude max_prob
            mask = (y_proba_pos >= min_prob) & (y_proba_pos < max_prob)
        
        sample_indices = np.where(mask)[0]
        num_samples = len(sample_indices)
        
        print(f"\n{'='*70}")
        print(f"Probability Range [{min_prob:.2f}, {max_prob:.2f}]: {num_samples} samples")
        print(f"{'='*70}")
        
        if num_samples == 0:
            print(f"No samples found in range [{min_prob:.2f}, {max_prob:.2f}] - skipping.")
            continue
        
        # Group samples by dataset for efficient processing
        samples_by_dataset = {}
        for idx in sample_indices:
            if idx >= len(test_frame_mapping):
                continue
            dataset_idx, frame_idx = test_frame_mapping[idx]
            if dataset_idx not in samples_by_dataset:
                samples_by_dataset[dataset_idx] = []
            samples_by_dataset[dataset_idx].append((idx, frame_idx))
        
        # Process each sample
        sample_counter = 0
        for dataset_idx, sample_list in samples_by_dataset.items():
            if dataset_idx >= len(test_processing_results):
                continue
            
            results = test_processing_results[dataset_idx]
            dataset_name = os.path.basename(test_file_paths[dataset_idx]) if dataset_idx < len(test_file_paths) else f"dataset_{dataset_idx}"
            
            # Get all available signal columns (exclude non-signal columns)
            signal_columns = [col for col in results["filtered_data"].columns 
                            if col not in ["time", "timestamp"] and 
                            results["filtered_data"][col].dtype in [np.float64, np.float32, np.int64, np.int32]]
            
            if len(signal_columns) == 0:
                print(f"Warning: No signal columns found in dataset {dataset_name}")
                continue
            
            # Process each sample from this dataset
            for global_idx, frame_idx in sample_list:
                sample_counter += 1
                prob_value = y_proba_pos[global_idx]
                true_label = y_test[global_idx]
                
                # Create sample folder
                sample_folder_name = f"sample_{sample_counter:05d}_prob_{prob_value:.4f}_label_{int(true_label)}"
                sample_dir = os.path.join(range_dir, sample_folder_name)
                os.makedirs(sample_dir, exist_ok=True)
                
                # Get frame indices
                if frame_idx >= len(results["frame_start_indices"]):
                    continue
                
                frame_start_idx = results["frame_start_indices"][frame_idx]
                frame_end_idx = results["frame_end_indices"][frame_idx]
                frame_label = results["frame_labels"][frame_idx] if frame_idx < len(results["frame_labels"]) else -1
                
                # Get time vector and calculate sampling rate
                filtered_data = results["filtered_data"]
                if "time" in filtered_data.columns:
                    time = pd.to_datetime(filtered_data["time"], unit="s")
                    time_s = (time - time.iloc[0]).dt.total_seconds().to_numpy()
                    dt = np.mean(np.diff(time_s)) if len(time_s) > 1 else 0.01
                else:
                    dt = 0.01  # Default 100 Hz
                    time_s = np.arange(len(filtered_data)) * dt
                
                fs = 1.0 / dt  # Sampling frequency
                
                # Calculate extended frame end index (including pre_peak and post_peak time)
                pre_peak_samples = int(round(pre_peak_time * fs))
                post_peak_samples = int(round(post_peak_time * fs))
                
                # Determine if this frame has a corresponding peak (true label = 1)
                if y_test[idx] == 1:
                    # Faulty frame: has a corresponding peak
                    peak_indices = results.get("peak_indices", np.array([]))
                    expected_peak_idx = frame_end_idx + pre_peak_samples
                    
                    if len(peak_indices) > 0:
                        # Find closest peak
                        closest_peak_idx = peak_indices[np.argmin(np.abs(peak_indices - expected_peak_idx))]
                        extended_end_idx = min(int(closest_peak_idx) + post_peak_samples, len(filtered_data) - 1)
                    else:
                        # Fallback: use estimated peak location
                        extended_end_idx = min(frame_end_idx + pre_peak_samples + post_peak_samples, len(filtered_data) - 1)
                else:
                    # Non-faulty frame: no corresponding peak
                    extended_end_idx = min(frame_end_idx + pre_peak_samples + post_peak_samples, len(filtered_data) - 1)
                
                # Plot each signal individually
                for signal_name in signal_columns:
                    if signal_name not in filtered_data.columns:
                        continue
                    
                    # Extract signal
                    full_signal = filtered_data[signal_name].to_numpy()
                    
                    # Safety check
                    if frame_start_idx < 0 or extended_end_idx >= len(full_signal) or frame_start_idx > extended_end_idx:
                        continue
                    
                    # Extract EXTENDED frame signal (includes window_length + pre_peak + post_peak)
                    frame_signal = full_signal[frame_start_idx:extended_end_idx+1]
                    frame_time = time_s[frame_start_idx:extended_end_idx+1]
                    
                    # Normalize time: start from 0
                    frame_start_time = time_s[frame_start_idx]
                    if len(frame_time) > 0:
                        t_frame = frame_time - frame_start_time
                    else:
                        total_window = window_length + pre_peak_time + post_peak_time
                        t_frame = np.linspace(0, total_window, len(frame_signal))
                    
                    # Create figure
                    fig, ax = plt.subplots(figsize=figsize)
                    
                    # Plot signal
                    ax.plot(t_frame, frame_signal, linewidth=2, color='blue', label='Signal')
                    
                    # Add vertical markers for different time segments
                    ax.axvline(window_length, color='green', linestyle=':', linewidth=1.5, 
                               label=f'Window end ({window_length:.1f}s)', alpha=0.7)
                    peak_time = window_length + pre_peak_time
                    ax.axvline(peak_time, color='orange', linestyle='--', linewidth=1.5,
                               label=f'Peak (~{peak_time:.1f}s)', alpha=0.7)
                    
                    # Add threshold line if applicable and signal is ground_tether_force
                    if signal_name == "ground_tether_force" and threshold_on_T is not None:
                        ax.axhline(threshold_on_T, color='red', linestyle='--', linewidth=1.5,
                                  label=f'Threshold = {threshold_on_T:.1f}')
                    
                    # Labels and title
                    total_window = window_length + pre_peak_time + post_peak_time
                    ax.set_xlabel("Time (s)", fontsize=12, fontweight="bold")
                    ax.set_ylabel(signal_name, fontsize=12, fontweight="bold")
                    title = (f"Signal: {signal_name} | {total_window:.1f}s window\n"
                             f"Probability: {prob_value:.4f} | True Label: {int(true_label)} | "
                             f"Frame Label: {int(frame_label)}")
                    ax.set_title(title, fontsize=11, fontweight="bold")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc='best', fontsize=9)
                    
                    plt.tight_layout()
                    
                    # Save figure
                    # Sanitize signal name for filename
                    safe_signal_name = signal_name.replace("/", "_").replace("\\", "_").replace(" ", "_")
                    save_path = os.path.join(sample_dir, f"{safe_signal_name}.png")
                    plt.savefig(save_path, dpi=150, bbox_inches="tight")
                    plt.close(fig)
                
                if sample_counter % 100 == 0:
                    print(f"  Processed {sample_counter}/{num_samples} samples...")
        
        print(f"  Completed range [{min_prob:.2f}, {max_prob:.2f}]: {num_samples} samples saved to {range_dir}")
    
    print(f"\n{'='*70}")
    print(f"All probability range visualizations completed!")
    print(f"Results saved to: {output_base_dir}")
    print(f"{'='*70}\n")
