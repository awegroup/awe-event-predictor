from __future__ import annotations
import os
import pickle
import json
import numpy as np
from typing import Dict, Optional
from datetime import datetime
import joblib


def save_model(
    model_dir: str,
    model,
    scaler,
    y_test: np.ndarray,
    y_pred: np.ndarray,
    y_proba: Optional[np.ndarray],
    metrics: Dict,
    model_type: str = "svm",
    feature_names: Optional[list] = None,
    training_params: Optional[Dict] = None,
    metadata: Optional[Dict] = None,
    X_test: Optional[np.ndarray] = None,
    test_processing_data: Optional[Dict] = None,
    threshold: Optional[float] = None,
    feature_importance: Optional[Dict] = None,
    in_domain_test_results: Optional[Dict] = None,
    per_dataset_metrics: Optional[Dict] = None,
    per_dataset_metrics_in_domain: Optional[Dict] = None
) -> str:
    """
    Save model, scaler, and all useful information to a directory.
    
    Parameters
    ----------
    model_dir : str
        Directory path where to save the model.
    model
        Trained model (SVM, Logistic Regression, etc.).
    scaler
        Fitted feature scaler.
    model_type : str
        Type of model: 'svm', 'logistic_regression', etc.
    y_test : np.ndarray
        Test set true labels.
    y_pred : np.ndarray
        Test set predictions.
    y_proba : np.ndarray, optional
        Test set predicted probabilities.
    metrics : dict
        Dictionary of evaluation metrics.
    feature_names : list, optional
        List of feature names.
    training_params : dict, optional
        Training parameters used.
    metadata : dict, optional
        Additional metadata.
    feature_importance : dict, optional
        Feature importance results (margin-based for SVM).
    in_domain_test_results : dict, optional
        In-domain test results containing y_test, y_pred, y_proba, and metrics.
    per_dataset_metrics : dict, optional
        Per-dataset metrics for out-of-domain test set (dataset_name -> metrics).
    per_dataset_metrics_in_domain : dict, optional
        Per-dataset metrics for in-domain test set (dataset_name -> metrics).
        
    Returns
    -------
    str
        Path to the saved model directory.
    """
    # Create directory if it doesn't exist
    os.makedirs(model_dir, exist_ok=True)
    
    print(f"\nSaving model to: {model_dir}")
    
    # Save model using joblib (more efficient for sklearn models)
    model_filename = f"{model_type}_model.joblib"
    model_path = os.path.join(model_dir, model_filename)
    joblib.dump(model, model_path)
    print(f"  ✓ Saved {model_type.upper().replace('_', ' ')} model: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  ✓ Saved scaler: {scaler_path}")
    
    # Save test data and predictions
    test_data_path = os.path.join(model_dir, "test_data.npz")
    save_dict = {
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba if y_proba is not None else np.array([])
    }
    if X_test is not None:
        save_dict['X_test'] = X_test
    if threshold is not None:
        save_dict['threshold'] = threshold
    np.savez(test_data_path, **save_dict)
    print(f"  ✓ Saved test data: {test_data_path}")
    if threshold is not None:
        print(f"  ✓ Saved decision threshold: {threshold:.4f}")
    
    # Save metrics as JSON (convert numpy types to native Python types)
    metrics_serializable = {}
    for key, value in metrics.items():
        if key == 'confusion_matrix':
            metrics_serializable[key] = value  # Already a list
        elif isinstance(value, (np.integer, np.floating)):
            metrics_serializable[key] = float(value)
        elif isinstance(value, np.ndarray):
            metrics_serializable[key] = value.tolist()
        elif value is None:
            metrics_serializable[key] = None
        else:
            metrics_serializable[key] = value
    
    metrics_path = os.path.join(model_dir, "metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics_serializable, f, indent=2)
    print(f"  ✓ Saved metrics: {metrics_path}")
    
    # Save feature names if provided
    if feature_names is not None:
        feature_names_path = os.path.join(model_dir, "feature_names.json")
        with open(feature_names_path, 'w') as f:
            json.dump(feature_names, f, indent=2)
        print(f"  ✓ Saved feature names: {feature_names_path}")
    
    # Save training parameters if provided
    if training_params is not None:
        training_params_path = os.path.join(model_dir, "training_params.json")
        # Convert numpy types to native Python types
        params_serializable = {}
        for key, value in training_params.items():
            if isinstance(value, (np.integer, np.floating)):
                params_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                params_serializable[key] = value.tolist()
            elif isinstance(value, (list, tuple)):
                params_serializable[key] = [float(v) if isinstance(v, (np.integer, np.floating)) else v for v in value]
            else:
                params_serializable[key] = value
        
        with open(training_params_path, 'w') as f:
            json.dump(params_serializable, f, indent=2)
        print(f"  ✓ Saved training parameters: {training_params_path}")
    
    # Save metadata if provided
    if metadata is not None:
        metadata_path = os.path.join(model_dir, "metadata.json")
        metadata_serializable = {}
        for key, value in metadata.items():
            if isinstance(value, (np.integer, np.floating)):
                metadata_serializable[key] = float(value)
            elif isinstance(value, np.ndarray):
                metadata_serializable[key] = value.tolist()
            else:
                metadata_serializable[key] = value
        
        with open(metadata_path, 'w') as f:
            json.dump(metadata_serializable, f, indent=2)
        print(f"  ✓ Saved metadata: {metadata_path}")
    
    # Save feature importance if provided
    if feature_importance is not None:
        feature_importance_path = os.path.join(model_dir, "feature_importance.json")
        # Convert numpy arrays to lists for JSON serialization
        importance_serializable = {}
        for key, value in feature_importance.items():
            if isinstance(value, np.ndarray):
                importance_serializable[key] = value.tolist()
            elif isinstance(value, (np.integer, np.floating)):
                importance_serializable[key] = float(value)
            elif isinstance(value, (int, float)):
                importance_serializable[key] = value
            else:
                importance_serializable[key] = value
        
        with open(feature_importance_path, 'w') as f:
            json.dump(importance_serializable, f, indent=2)
        print(f"  ✓ Saved feature importance: {feature_importance_path}")
    
    # Save timestamp
    timestamp = datetime.now().isoformat()
    info = {
        "saved_at": timestamp,
        "model_type": model_type.upper().replace('_', ' '),
        "files": {
            "model": model_filename,
            "scaler": "scaler.joblib",
            "test_data": "test_data.npz",
            "metrics": "metrics.json"
        }
    }
    if feature_names is not None:
        info["files"]["feature_names"] = "feature_names.json"
    if training_params is not None:
        info["files"]["training_params"] = "training_params.json"
    if metadata is not None:
        info["files"]["metadata"] = "metadata.json"
    if feature_importance is not None:
        info["files"]["feature_importance"] = "feature_importance.json"
    
    info_path = os.path.join(model_dir, "info.json")
    with open(info_path, 'w') as f:
        json.dump(info, f, indent=2)
    print(f"  ✓ Saved info: {info_path}")
    
    # Save test processing data if provided (for false positive plotting)
    if test_processing_data is not None:
        test_processing_path = os.path.join(model_dir, "test_processing_data.pkl")
        with open(test_processing_path, 'wb') as f:
            pickle.dump(test_processing_data, f)
        print(f"  ✓ Saved test processing data: {test_processing_path}")
        info["files"]["test_processing_data"] = "test_processing_data.pkl"
        # Update info.json with the new file
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    # Save in-domain test results if provided
    if in_domain_test_results is not None:
        print(f"\n  Saving in-domain test results...")
        
        # Save in-domain test data and predictions
        in_domain_test_path = os.path.join(model_dir, "in_domain_test_data.npz")
        in_domain_save_dict = {
            'y_test': in_domain_test_results.get('y_test'),
            'y_pred': in_domain_test_results.get('y_pred'),
            'y_proba': in_domain_test_results.get('y_proba') if in_domain_test_results.get('y_proba') is not None else np.array([])
        }
        np.savez(in_domain_test_path, **in_domain_save_dict)
        print(f"  ✓ Saved in-domain test data: {in_domain_test_path}")
        
        # Save in-domain metrics as JSON
        in_domain_metrics = in_domain_test_results.get('metrics', {})
        if in_domain_metrics:
            in_domain_metrics_serializable = {}
            for key, value in in_domain_metrics.items():
                if key == 'confusion_matrix':
                    in_domain_metrics_serializable[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    in_domain_metrics_serializable[key] = float(value)
                elif isinstance(value, np.ndarray):
                    in_domain_metrics_serializable[key] = value.tolist()
                elif value is None:
                    in_domain_metrics_serializable[key] = None
                else:
                    in_domain_metrics_serializable[key] = value
            
            in_domain_metrics_path = os.path.join(model_dir, "in_domain_metrics.json")
            with open(in_domain_metrics_path, 'w') as f:
                json.dump(in_domain_metrics_serializable, f, indent=2)
            print(f"  ✓ Saved in-domain metrics: {in_domain_metrics_path}")
            
            # Update info.json
            info["files"]["in_domain_test_data"] = "in_domain_test_data.npz"
            info["files"]["in_domain_metrics"] = "in_domain_metrics.json"
            with open(info_path, 'w') as f:
                json.dump(info, f, indent=2)
    
    # Save per-dataset metrics (out-of-domain) if provided
    if per_dataset_metrics is not None:
        print(f"\n  Saving per-dataset metrics (out-of-domain)...")
        per_dataset_metrics_serializable = {}
        
        for dataset_name, dataset_metrics in per_dataset_metrics.items():
            metrics_dict = {}
            for key, value in dataset_metrics.items():
                if key == 'confusion_matrix':
                    metrics_dict[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    metrics_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics_dict[key] = value.tolist()
                elif value is None:
                    metrics_dict[key] = None
                else:
                    metrics_dict[key] = value
            per_dataset_metrics_serializable[dataset_name] = metrics_dict
        
        per_dataset_metrics_path = os.path.join(model_dir, "per_dataset_metrics.json")
        with open(per_dataset_metrics_path, 'w') as f:
            json.dump(per_dataset_metrics_serializable, f, indent=2)
        print(f"  ✓ Saved per-dataset metrics (out-of-domain): {per_dataset_metrics_path}")
        
        # Update info.json
        info["files"]["per_dataset_metrics"] = "per_dataset_metrics.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    # Save per-dataset metrics (in-domain) if provided
    if per_dataset_metrics_in_domain is not None:
        print(f"\n  Saving per-dataset metrics (in-domain)...")
        per_dataset_metrics_in_domain_serializable = {}
        
        for dataset_name, dataset_metrics in per_dataset_metrics_in_domain.items():
            metrics_dict = {}
            for key, value in dataset_metrics.items():
                if key == 'confusion_matrix':
                    metrics_dict[key] = value
                elif isinstance(value, (np.integer, np.floating)):
                    metrics_dict[key] = float(value)
                elif isinstance(value, np.ndarray):
                    metrics_dict[key] = value.tolist()
                elif value is None:
                    metrics_dict[key] = None
                else:
                    metrics_dict[key] = value
            per_dataset_metrics_in_domain_serializable[dataset_name] = metrics_dict
        
        per_dataset_metrics_in_domain_path = os.path.join(model_dir, "per_dataset_metrics_in_domain.json")
        with open(per_dataset_metrics_in_domain_path, 'w') as f:
            json.dump(per_dataset_metrics_in_domain_serializable, f, indent=2)
        print(f"  ✓ Saved per-dataset metrics (in-domain): {per_dataset_metrics_in_domain_path}")
        
        # Update info.json
        info["files"]["per_dataset_metrics_in_domain"] = "per_dataset_metrics_in_domain.json"
        with open(info_path, 'w') as f:
            json.dump(info, f, indent=2)
    
    print(f"\nModel successfully saved to: {model_dir}")
    
    return model_dir


def load_model(model_dir: str) -> Dict:
    """
    Load model, scaler, and all saved information from a directory.
    
    Parameters
    ----------
    model_dir : str
        Directory path where the model is saved.
        
    Returns
    -------
    dict
        Dictionary containing:
        - 'model': Trained model (key name depends on model type)
        - 'model_type': Type of model loaded
        - 'scaler': Fitted scaler
        - 'y_test': Test set true labels
        - 'y_pred': Test set predictions
        - 'y_proba': Test set probabilities (if available)
        - 'metrics': Evaluation metrics
        - 'feature_names': Feature names (if available)
        - 'training_params': Training parameters (if available)
        - 'metadata': Additional metadata (if available)
    """
    if not os.path.exists(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    print(f"Loading model from: {model_dir}")
    
    results = {}
    
    # Try to determine model type from info.json if available
    info_path = os.path.join(model_dir, "info.json")
    model_type = "svm"  # default
    if os.path.exists(info_path):
        with open(info_path, 'r') as f:
            info = json.load(f)
            if "model_type" in info:
                model_type = info["model_type"].lower().replace(' ', '_')
            if "files" in info and "model" in info["files"]:
                model_filename = info["files"]["model"]
            else:
                model_filename = f"{model_type}_model.joblib"
    else:
        # Try common model filenames
        model_filename = None
        for possible_name in ["svm_model.joblib", "logistic_regression_model.joblib", "model.joblib"]:
            if os.path.exists(os.path.join(model_dir, possible_name)):
                model_filename = possible_name
                # Infer model type from filename
                if "svm" in possible_name:
                    model_type = "svm"
                elif "logistic" in possible_name:
                    model_type = "logistic_regression"
                break
        
        if model_filename is None:
            model_filename = "svm_model.joblib"  # fallback
    
    # Load model
    model_path = os.path.join(model_dir, model_filename)
    if os.path.exists(model_path):
        results['model'] = joblib.load(model_path)
        results['model_type'] = model_type
        # Also add legacy key for backward compatibility
        if model_type == "svm":
            results['svm_model'] = results['model']
        print(f"  ✓ Loaded {model_type.upper().replace('_', ' ')} model")
    else:
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    # Load scaler
    scaler_path = os.path.join(model_dir, "scaler.joblib")
    if os.path.exists(scaler_path):
        results['scaler'] = joblib.load(scaler_path)
        print(f"  ✓ Loaded scaler")
    else:
        raise FileNotFoundError(f"Scaler file not found: {scaler_path}")
    
    # Load test data
    test_data_path = os.path.join(model_dir, "test_data.npz")
    if os.path.exists(test_data_path):
        test_data = np.load(test_data_path)
        results['y_test'] = test_data['y_test']
        results['y_pred'] = test_data['y_pred']
        if 'y_proba' in test_data and len(test_data['y_proba']) > 0:
            results['y_proba'] = test_data['y_proba']
        else:
            results['y_proba'] = None
        if 'X_test' in test_data:
            results['X_test'] = test_data['X_test']
        if 'threshold' in test_data:
            results['threshold'] = float(test_data['threshold'])
        print(f"  ✓ Loaded test data")
    else:
        raise FileNotFoundError(f"Test data file not found: {test_data_path}")
    
    # Load metrics
    metrics_path = os.path.join(model_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            results['metrics'] = json.load(f)
        print(f"  ✓ Loaded metrics")
    else:
        raise FileNotFoundError(f"Metrics file not found: {metrics_path}")
    
    # Load feature names (optional)
    feature_names_path = os.path.join(model_dir, "feature_names.json")
    if os.path.exists(feature_names_path):
        with open(feature_names_path, 'r') as f:
            results['feature_names'] = json.load(f)
        print(f"  ✓ Loaded feature names")
    
    # Load training parameters (optional)
    training_params_path = os.path.join(model_dir, "training_params.json")
    if os.path.exists(training_params_path):
        with open(training_params_path, 'r') as f:
            results['training_params'] = json.load(f)
        print(f"  ✓ Loaded training parameters")
    
    # Load metadata (optional)
    metadata_path = os.path.join(model_dir, "metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path, 'r') as f:
            results['metadata'] = json.load(f)
        print(f"  ✓ Loaded metadata")
    
    print(f"\nModel successfully loaded from: {model_dir}")
    
    return results

