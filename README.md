# awe-event-predictor

Machine-learning prediction of critical tether-load events in airborne wind energy (AWE) systems.

This repository contains the code for the paper

> M. Pasquali, O. Cayon, S. Theodoulis, C. Farges, T.-B. Airimitoaie, P. Lanusse and R. Schmehl,
> *Machine learning-based prediction of critical load events in airborne wind energy systems*,
> Wind Energy Science Discussions (preprint), 2026.

If you use this code, please [cite the paper](#citation).

## What it does

From flight logs of an AWE kite, the pipeline:

1. **Detects critical load events.** An event is a tether-force peak above a threshold (default 2500 N). Peaks less than 30 s apart are merged into one event.
2. **Builds labelled windows.** A 5 s window that ends `pre_peak_time` seconds (0.2, 0.5 or 1.0 s) before an event is labelled 1. Non-overlapping 5 s windows that stay below the threshold are labelled 0. The 30 s after each event are excluded.
3. **Extracts features.** It computes statistical, temporal and spectral features for each window from 21 signals (tether force, reel-out speed, kite velocity, position, attitude, steering, depower, …).
4. **Trains a classifier.** The classifier is an SVM or logistic regression. Hyperparameters are tuned by randomized search with cross-validation on a class-balanced training set. The decision threshold is tuned on a validation set with the natural class distribution. An optional step selects features by SVM margin importance.
5. **Evaluates the model.** It is tested **in-domain** on a held-out 20 % of the training windows and **out-of-domain** on separate test flights.

Labels do not have to be supplied: they are derived from `ground_tether_force`.

## Installation

The code was developed with Python 3.12.

```bash
git clone <repository-url>
cd awe-event-predictor
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Run every command below from the repository root.

## Quick start with the dummy dataset

The flight data used in the paper are not part of this repository. To try the pipeline, generate a small synthetic dataset. Its flights are not physically accurate. They mimic the input format and contain load peaks that have a learnable precursor.

```bash
python src/make_dummy_dataset.py --output data/dummy_dataset
python src/preprocess.py --dataset data/dummy_dataset
python src/train.py --dataset data/dummy_dataset --model svm --feature-selection --pre-peak-time 0.5
python src/plot_model_metrics.py saved_models/svm_prob_FS20m20b_peak0p5 --save main_fig.png
```

The automated test runs the same steps in about 1–2 minutes:

```bash
pytest
```

## Using your own data

### Folder layout

Put each dataset in its own folder under `data/`:

```
data/
└── dataset_1/
    ├── training_data/   # raw flight logs used for training and in-domain testing
    │   ├── flight_a.csv
    │   └── ...
    └── test_data/       # raw flight logs used only for out-of-domain testing
        └── flight_x.csv
```

`data/` is git-ignored.

### Input format

- One CSV file per flight.
- Space-separated. If your files use another separator, pass it with `--sep`.
- The first column may be an unnamed index. It is dropped.
- `time` holds Unix time in seconds.

The following columns are required:

| Column | Description |
|---|---|
| `time` | Unix time [s] |
| `ground_tether_force` | Tether force at the ground station [N]. Used for labelling. |
| `kite_height` | Kite height [m]. Samples below 20 m are discarded. |
| `kite_0_vx`, `kite_0_vy`, `kite_0_vz` | Kite velocity components [m/s] |
| `kite_0_altitude` | Kite altitude [m] |
| `ground_tether_reelout_speed`, `ground_tether_length` | Reel-out speed [m/s] and tether length [m] |
| `kite_pos_east`, `kite_pos_north` | Kite position [m] |
| `kite_elevation`, `kite_azimuth`, `kite_distance` | Spherical kite position [rad, rad, m] |
| `kite_heading`, `kite_course` | Heading and course angle [rad or deg] |
| `kite_actual_steering`, `kite_actual_depower` | Steering and depower actuation [%] |
| `ground_mech_power` | Mechanical power at the ground station [W] |
| `kite_turn_rate`, `kite_course_rate` | Turn rate and course rate |

The signal list is set in `signal_names` at the top of [src/train.py](src/train.py). Edit that list to use other or fewer signals.

## Pipeline

### 1. Preprocess

```bash
python src/preprocess.py --dataset data/dataset_1
```

For each file in `training_data/` and `test_data/`, the script:

1. Drops the index column.
2. Forward-fills NaNs.
3. Adds the velocity magnitudes `V_kite_0_mag` and `V_kite_1_mag`, when the velocity columns exist.
4. Converts the roll, pitch and yaw columns from degrees to radians.
5. Unwraps all angle columns, including `kite_heading` and `kite_course`.

It writes comma-separated files to `data/dataset_1/cleaned/{training_data,test_data}/`.

[src/preprocess_csvs.ipynb](src/preprocess_csvs.ipynb) contains the same cleaning step, plus data-quality checks and plots for a single flight.

### 2. Train and evaluate

```bash
python src/train.py --dataset data/dataset_1 --model svm --pre-peak-time 0.5
```

| Option | Default | Description |
|---|---|---|
| `--dataset` | `data/dataset_1` | Dataset folder. The script reads `<dataset>/cleaned/`. |
| `--model` | `logistic_regression` | `svm` or `logistic_regression` |
| `--pre-peak-time` | `1.0` | Gap in seconds between the end of a window and the event |
| `--feature-selection` | off | SVM only. Keeps the top 20 margin-importance and top 20 boundary-importance features, then retrains. |
| `--features-from MODEL_DIR` | – | Reuses the features selected by a saved model |
| `--output MODEL_DIR` | auto | Output folder. The default is `saved_models/<model>_<mode>_<features>_peak<t>`. |
| `--n-iter` / `--cv-folds` | `50` / `3` | Settings of the randomized hyperparameter search |
| `--seed` | `42` | Random seed |
| `--cache-dir` | `feature_cache` | Feature cache folder |

Further settings, such as the load threshold, window length, balancing method and threshold-tuning method, are constants at the top of [src/train.py](src/train.py) and in its `__main__` block.

**Paper variants.** Run each variant with `--pre-peak-time` set to 0.2, 0.5 and 1.0:

| Variant | Command |
|---|---|
| SVM, all features | `python src/train.py --model svm` |
| SVM, feature selection (FS20m20b) | `python src/train.py --model svm --feature-selection` |
| LR, all features | `python src/train.py --model logistic_regression` |
| LR, SVM-selected features | `python src/train.py --model logistic_regression --features-from saved_models/svm_prob_FS20m20b_peak<t>` |

Train the SVM feature-selection model before the last variant, because that variant reuses its selected features.

Features are cached per input file in `feature_cache/`. The cache is keyed on the file contents and the preprocessing parameters. Delete the folder to force recomputation.

### 3. Outputs and metrics

The training run prints in-domain and out-of-domain metrics and per-flight tables. It saves the following files to the output folder:

| File | Contents |
|---|---|
| `metrics.json` | Out-of-domain metrics |
| `in_domain_metrics.json` | In-domain metrics |
| `per_dataset_metrics.json`, `per_dataset_metrics_in_domain.json` | Metrics per flight |
| `training_params.json` | Settings, tuned hyperparameters, decision threshold and selected features |
| `metadata.json` | Sample counts and class distributions |
| model, `scaler.joblib`, test data | Everything needed to reload and re-evaluate the model |

To plot the confusion matrix, PR curve and score distribution on the out-of-domain set, run:

```bash
python src/plot_model_metrics.py <model_dir> --save main_fig.png
```

To compare permutation importance, mutual information and SHAP, run:

```bash
python src/analyze_feature_importance.py <model_dir> --top-n 20 --save importance.png
```

To get per-class and macro precision, recall and F1 from a confusion matrix, run:

```bash
python src/compute_metrics.py --tn 90 --fp 10 --fn 5 --tp 15
```

### Diagnostics

The following scripts are optional:

- `python src/verify_segmentation.py data/dataset_1` checks the window lengths and alignment on all flights.
- `python src/inspect_model_features.py <model_dir>` lists the features a saved model uses.
- [src/visualize_peaks.ipynb](src/visualize_peaks.ipynb) plots the detected events and overlays the labelled windows.

## Repository structure

```
src/
├── make_dummy_dataset.py       # synthetic example data
├── preprocess.py               # step 1: clean raw CSVs
├── train.py                    # step 2: features, training, evaluation
├── plot_model_metrics.py       # step 3: figures
├── analyze_feature_importance.py
├── compute_metrics.py
└── functions/
    ├── CSVLoader.py
    ├── cache/                  # feature cache
    ├── preprocessing/          # cleaning, event detection, windowing, labelling, feature extraction
    └── ml/                     # balancing, SVM / LR training, threshold tuning, feature importance, persistence
tests/
└── test_pipeline.py            # end-to-end test on the dummy dataset
```

## Citation

If you use this code, please cite:

```bibtex
@article{pasquali2026critical,
  title   = {Machine learning-based prediction of critical load events in airborne wind energy systems},
  author  = {Pasquali, Monika and Cayon, Oriol and Theodoulis, Spilios and Farges, Christophe and
             Airimitoaie, Tudor-Bogdan and Lanusse, Patrick and Schmehl, Roland},
  journal = {Wind Energy Science Discussions},
  year    = {2026},
  note    = {Preprint}
}
```

The same metadata is in [CITATION.cff](CITATION.cff). GitHub uses that file to show a "Cite this repository" button.

## License

[MIT](LICENSE) © 2026 Monika Pasquali
