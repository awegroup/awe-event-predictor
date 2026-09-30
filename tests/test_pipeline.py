"""
End-to-end test on the synthetic dummy dataset:

    make_dummy_dataset -> preprocess -> train (SVM + feature selection)
    -> train (LR on SVM-selected features) -> plot_model_metrics
    -> analyze_feature_importance / compute_metrics

Each step is run as a subprocess, exactly as a user would run it.
"""

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"


def run(script, *args, cwd):
    env = {**os.environ, "MPLBACKEND": "Agg", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.run(
        [sys.executable, str(SRC / script), *map(str, args)],
        cwd=cwd, env=env, capture_output=True, text=True, encoding="utf-8",
    )
    assert proc.returncode == 0, f"{script} failed:\n{proc.stdout[-4000:]}\n{proc.stderr[-4000:]}"
    return proc.stdout


@pytest.fixture(scope="module")
def workdir(tmp_path_factory):
    return tmp_path_factory.mktemp("pipeline")


@pytest.fixture(scope="module")
def dataset(workdir):
    ds = workdir / "data" / "dummy_dataset"
    run("make_dummy_dataset.py", "--output", ds, cwd=workdir)
    run("preprocess.py", "--dataset", ds, "--quiet", cwd=workdir)
    return ds


def test_preprocess_output(dataset):
    for split, n in (("training_data", 4), ("test_data", 2)):
        files = sorted((dataset / "cleaned" / split).glob("*.csv"))
        assert len(files) == n
        df = pd.read_csv(files[0])
        assert not any(c.startswith("Unnamed") for c in df.columns)
        assert {"V_kite_0_mag", "ground_tether_force", "time"} <= set(df.columns)
        assert df.isna().sum().sum() == 0
        # yaw was in degrees and wrapped; now radians and unwrapped
        assert df["kite_0_yaw"].diff().abs().max() < 1.0


def test_train_evaluate(dataset, workdir):
    common = ["--dataset", dataset, "--n-iter", 3, "--cache-dir", workdir / "cache"]

    svm_dir = workdir / "svm_fs"
    run("train.py", "--model", "svm", "--feature-selection", "--pre-peak-time", 0.5,
        "--output", svm_dir, *common, cwd=workdir)

    params = json.loads((svm_dir / "training_params.json").read_text())
    assert params["n_selected_features"] > 0
    for name in ("metrics.json", "in_domain_metrics.json", "per_dataset_metrics.json",
                 "per_dataset_metrics_in_domain.json", "scaler.joblib"):
        assert (svm_dir / name).exists(), name

    # The dummy peaks have a clear precursor, so a working pipeline must find them
    metrics = json.loads((svm_dir / "metrics.json").read_text())
    assert metrics["pr_auc"] > 0.8, metrics

    lr_dir = workdir / "lr_from_svm"
    out = run("train.py", "--model", "logistic_regression", "--features-from", svm_dir,
              "--pre-peak-time", 0.5, "--output", lr_dir, *common, cwd=workdir)
    assert "Loaded cached features" in out
    lr_params = json.loads((lr_dir / "training_params.json").read_text())
    assert lr_params["selected_feature_indices"] == params["selected_feature_indices"]

    run("plot_model_metrics.py", svm_dir, "--save", workdir / "fig.png", cwd=workdir)
    assert (workdir / "fig.png").exists()

    run("analyze_feature_importance.py", svm_dir, "--top-n", 5, "--shap-samples", 30,
        "--save", workdir / "importance.png", cwd=workdir)
    assert (workdir / "importance.png").exists()


def test_compute_metrics(workdir):
    out = run("compute_metrics.py", "--tn", 90, "--fp", 10, "--fn", 5, "--tp", 15, cwd=workdir)
    assert "macro_avg" in out
