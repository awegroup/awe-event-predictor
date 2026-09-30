from __future__ import annotations
import numpy as np
from typing import Literal, Optional
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler, NearMiss, TomekLinks
from sklearn.utils import resample


def balance_classes(
    X: np.ndarray,
    y: np.ndarray,
    method: Literal["smote", "downsample", "upsample", "nearmiss", "nearmiss1", "nearmiss2", "nearmiss3", "tomek_links", "tomek_links_then_nearmiss", "none"] = "smote",
    random_state: int = 42,
    downsample_ratio: float = 1.0,
    nearmiss_version: int = 1,
    nearmiss_n_neighbors: int = 3
) -> tuple[np.ndarray, np.ndarray]:
    """
    Balance class distribution using various methods.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix (num_samples, num_features).
    y : np.ndarray
        Labels (num_samples,).
    method : str
        Balancing method:
        - 'smote': Synthetic Minority Oversampling
        - 'downsample': Random downsampling of majority class
        - 'upsample': Random upsampling of minority class
        - 'nearmiss', 'nearmiss1', 'nearmiss2', 'nearmiss3': NearMiss undersampling (preserves boundary samples)
        - 'tomek_links': Removes Tomek link pairs (noisy/boundary samples)
        - 'tomek_links_then_nearmiss': First removes Tomek links, then applies NearMiss
        - 'none': No balancing
    random_state : int
        Random seed for reproducibility.
    downsample_ratio : float
        Desired majority:minority ratio when method='downsample'.
        Example:
            1.0 -> 1:1 (classic downsampling)
            3.0 -> 3:1
            5.0 -> 5:1
    nearmiss_version : int
        Version of NearMiss to use (1, 2, or 3). Only used if method='nearmiss'.
        - 1: Selects majority samples closest to minority samples
        - 2: Selects majority samples closest to minority but farthest from other majority
        - 3: Selects a given number of majority samples for each minority sample
    nearmiss_n_neighbors : int
        Number of nearest neighbors for NearMiss. Only used for NearMiss-3.

    Returns
    -------
    X_balanced, y_balanced
    """
    unique_classes, class_counts = np.unique(y, return_counts=True)

    print(f"\nClass distribution before balancing:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")

    if method == "none":
        print("  No balancing applied.")
        return X, y

    if len(unique_classes) != 2:
        raise ValueError(f"Expected binary classification, got {len(unique_classes)} classes")

    minority_class = unique_classes[np.argmin(class_counts)]
    majority_class = unique_classes[np.argmax(class_counts)]
    minority_count = np.min(class_counts)
    majority_count = np.max(class_counts)

    print(f"\n  Minority class: {minority_class} ({minority_count} samples)")
    print(f"  Majority class: {majority_class} ({majority_count} samples)")
    print(f"  Original imbalance ratio: {majority_count/minority_count:.2f}:1")

    if method == "smote":
        print("\n  Applying SMOTE (Synthetic Minority Oversampling)...")
        if minority_count < 2:
            raise ValueError("SMOTE requires at least 2 minority samples")
        smote = SMOTE(
            random_state=random_state,
            k_neighbors=min(5, minority_count - 1)
        )
        X_balanced, y_balanced = smote.fit_resample(X, y)

    elif method == "downsample":
        # --------------------------------------------------
        # CONTROLLED DOWNSAMPLING
        # --------------------------------------------------
        target_majority = int(min(majority_count, downsample_ratio * minority_count))

        print(
            f"\n  Applying downsampling: "
            f"target ratio ≈ {target_majority/minority_count:.2f}:1 "
            f"({target_majority} majority samples)"
        )

        majority_indices = np.where(y == majority_class)[0]
        minority_indices = np.where(y == minority_class)[0]

        np.random.seed(random_state)
        majority_sampled = np.random.choice(
            majority_indices,
            size=target_majority,
            replace=False
        )

        balanced_indices = np.concatenate([minority_indices, majority_sampled])
        np.random.shuffle(balanced_indices)

        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]

    elif method == "upsample":
        print("\n  Applying random upsampling of minority class...")

        minority_indices = np.where(y == minority_class)[0]
        majority_indices = np.where(y == majority_class)[0]

        np.random.seed(random_state)
        minority_upsampled = np.random.choice(
            minority_indices,
            size=majority_count,
            replace=True
        )

        balanced_indices = np.concatenate([majority_indices, minority_upsampled])
        np.random.shuffle(balanced_indices)

        X_balanced = X[balanced_indices]
        y_balanced = y[balanced_indices]

    elif method in ["nearmiss", "nearmiss1", "nearmiss2", "nearmiss3"]:
        # --------------------------------------------------
        # NEARMISS UNDERSAMPLING
        # --------------------------------------------------
        # Determines version from method name or parameter
        if method == "nearmiss":
            version = nearmiss_version
        elif method == "nearmiss1":
            version = 1
        elif method == "nearmiss2":
            version = 2
        elif method == "nearmiss3":
            version = 3
        
        print(f"\n  Applying NearMiss-{version} undersampling...")
        print(f"    NearMiss preserves majority samples closest to minority samples")
        print(f"    (helps preserve boundary samples that are important for learning)")
        
        if version == 3:
            # NearMiss-3 requires n_neighbors_ver3 parameter
            nm = NearMiss(
                version=version,
                n_neighbors_ver3=nearmiss_n_neighbors
            )
        else:
            # NearMiss-1 and NearMiss-2 can optionally use n_neighbors
            nm = NearMiss(
                version=version,
                n_neighbors=nearmiss_n_neighbors
            )
        
        X_balanced, y_balanced = nm.fit_resample(X, y)
        
        # Calculate final ratio
        unique_after, counts_after = np.unique(y_balanced, return_counts=True)
        if len(unique_after) == 2:
            final_ratio = max(counts_after) / min(counts_after)
            print(f"    Final ratio: {final_ratio:.2f}:1")

    elif method == "tomek_links":
        # --------------------------------------------------
        # TOMEK LINKS CLEANING
        # --------------------------------------------------
        print("\n  Applying Tomek Links cleaning...")
        print(f"    Tomek Links removes pairs of samples from different classes")
        print(f"    that are each other's nearest neighbors (noisy/boundary samples)")
        
        tl = TomekLinks()
        X_balanced, y_balanced = tl.fit_resample(X, y)
        
        # Calculate how many samples were removed
        removed_count = len(X) - len(X_balanced)
        print(f"    Removed {removed_count} samples ({removed_count/len(X)*100:.1f}%)")
        
        # Calculate final ratio
        unique_after, counts_after = np.unique(y_balanced, return_counts=True)
        if len(unique_after) == 2:
            final_ratio = max(counts_after) / min(counts_after)
            print(f"    Final ratio: {final_ratio:.2f}:1 (may still be imbalanced)")

    elif method == "tomek_links_then_nearmiss":
        # --------------------------------------------------
        # COMBINED: TOMEK LINKS + NEARMISS
        # --------------------------------------------------
        print("\n  Applying Tomek Links cleaning followed by NearMiss undersampling...")
        print(f"    Step 1: Remove noisy/boundary pairs (Tomek Links)")
        print(f"    Step 2: Intelligently undersample majority class (NearMiss-{nearmiss_version})")
        
        # Step 1: Remove Tomek links
        tl = TomekLinks()
        X_cleaned, y_cleaned = tl.fit_resample(X, y)
        removed_tomek = len(X) - len(X_cleaned)
        print(f"    Removed {removed_tomek} samples via Tomek Links")
        
        # Step 2: Apply NearMiss
        if nearmiss_version == 3:
            nm = NearMiss(
                version=nearmiss_version,
                n_neighbors_ver3=nearmiss_n_neighbors
            )
        else:
            nm = NearMiss(
                version=nearmiss_version,
                n_neighbors=nearmiss_n_neighbors
            )
        
        X_balanced, y_balanced = nm.fit_resample(X_cleaned, y_cleaned)
        
        # Calculate final ratio
        unique_after, counts_after = np.unique(y_balanced, return_counts=True)
        if len(unique_after) == 2:
            final_ratio = max(counts_after) / min(counts_after)
            print(f"    Final ratio: {final_ratio:.2f}:1")

    else:
        raise ValueError(f"Unknown method: {method}")

    # --------------------------------------------------
    # FINAL DISTRIBUTION
    # --------------------------------------------------
    unique_balanced, counts_balanced = np.unique(y_balanced, return_counts=True)

    print(f"\nClass distribution after balancing:")
    for cls, count in zip(unique_balanced, counts_balanced):
        print(f"  Class {cls}: {count} samples ({count/len(y_balanced)*100:.1f}%)")

    return X_balanced, y_balanced
