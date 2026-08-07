## Predicting Next-Hour San José Police Call Volume

This CS 171 machine-learning project predicts the number of San José police calls expected during the next hour. The workflow converts incident-level police-call records into hourly observations, creates leakage-safe time-series features, trains several regression models, and compares their performance on a held-out chronological test period.

## Project Overview

- **Task:** Regression
- **Target:** `NEXT_HOUR_CALLS`
- **Data:** City of San José Police Calls for Service for 2024, 2025, and partial 2026
- **Final hourly dataset:** 21,236 rows
- **Predictors:** 48 numerical features
- **Models:** Mean baseline, persistence baseline, linear regression, Elastic Net, decision tree, random forest, MLP, and LSTM
- **Primary metrics:** MAE, RMSE, and R²
- **Current best test model:** Tuned random forest

## Repository Structure

```text
cs171-police-call-forecasting/
├── data/
│   └── README.md
├── figures/
├── notebooks/
│   ├── 01_data_preparation.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing_and_split.ipynb
│   ├── 04_baseline_and_tree_models.ipynb
│   ├── 05_neural_networks.ipynb
│   └── 06_model_evaluation.ipynb
├── paper/
├── results/
├── src/
├── requirements.txt
└── README.md
```

The raw CSV files are not stored in this repository because of their size.

## Recommended Environment

The notebooks were developed in **Google Colab** and save files to Google Drive. A GPU runtime is recommended for Notebook 05 because it trains MLP and LSTM models.

### Python packages

```text
pandas
numpy
matplotlib
seaborn
scikit-learn
joblib
tensorflow
great-tables
```

In Google Colab, most packages are already installed. Install `great-tables` if needed:

```python
!pip install -q great-tables
```

## 1. Download and Name the Raw Data

Download the yearly **Police Calls for Service** CSV files from the City of San José Open Data Portal:

https://data.sanjoseca.gov/

Rename the files exactly as follows:

```text
police_calls_2024.csv
police_calls_2025.csv
police_calls_2026.csv
```

The notebooks expect the files to use the original City dataset columns, including fields such as `OFFENSE_DATE`, `OFFENSE_TIME`, `PRIORITY`, and `CALL_TYPE`.

## 2. Create the Google Drive Folder Structure

Create this folder in Google Drive:

```text
MyDrive/CS171_Police_Call_Project/
```

Place the raw files and create the model folders as shown below:

```text
CS171_Police_Call_Project/
├── data_raw/
│   ├── police_calls_2024.csv
│   ├── police_calls_2025.csv
│   └── police_calls_2026.csv
├── data_processed/
├── figures/
├── results/
├── Results_Graphs/
└── models/
    ├── LinearRegression/
    ├── ElasticNet (Regularization)/
    ├── DecisionTree/
    ├── RandomForest/
    ├── MLP/
    └── LSTM/
```

Notebook 04 currently saves some figures to a separate folder named `MyDrive/Results`. Create that folder as well, or change its `save_figure()` path to the project folder.

The following setup cell can create all required folders:

```python
from pathlib import Path

project_folder = Path("/content/drive/MyDrive/CS171_Police_Call_Project")

required_folders = [
    "data_raw",
    "data_processed",
    "figures",
    "results",
    "Results_Graphs",
    "models/LinearRegression",
    "models/ElasticNet (Regularization)",
    "models/DecisionTree",
    "models/RandomForest",
    "models/MLP",
    "models/LSTM",
]

for folder in required_folders:
    (project_folder / folder).mkdir(parents=True, exist_ok=True)

Path("/content/drive/MyDrive/Results").mkdir(parents=True, exist_ok=True)
```

## 3. Open the Notebooks in Google Colab

Open each notebook from the `notebooks/` folder. At the beginning of each notebook, mount Google Drive:

```python
from google.colab import drive
drive.mount("/content/drive")
```

Run the notebooks **in numerical order**. Later notebooks depend on files and models produced by earlier notebooks.

## 4. Run the Complete Pipeline

### Notebook 01 — Data Preparation

Run:

```text
notebooks/01_data_preparation.ipynb
```

This notebook:

- Loads and combines the 2024–2026 incident files.
- Removes exact duplicate records.
- Creates timestamps and sorts incidents chronologically.
- Aggregates incidents into hourly call counts.
- Creates the next-hour target.
- Engineers calendar, lag, rolling-average, priority, and call-type features.
- Creates chronological train, validation, and test labels.
- Removes rows without complete historical features.

Expected outputs in `data_processed/`:

```text
model_ready_hourly.csv
candidate_features.json
final_split_audit.csv
```

Expected final split sizes:

| Split | Rows | Prediction period |
|---|---:|---|
| Train | 14,830 | Jan. 8, 2024–Sep. 30, 2025 |
| Validation | 3,624 | Oct. 1, 2025–Feb. 28, 2026 |
| Test | 2,782 | Mar. 1, 2026–Jul. 8, 2026 |

### Notebook 02 — Exploratory Data Analysis

Run:

```text
notebooks/02_eda.ipynb
```

This notebook reads `model_ready_hourly.csv` and examines:

- The target distribution.
- Hour-of-day and day-of-week patterns.
- Yearly and monthly patterns.
- Priority and call-type distributions.
- Correlations between historical features and the target.
- High-volume and extreme-demand hours.
- Distribution differences among the chronological splits.

Figures are saved to:

```text
MyDrive/CS171_Police_Call_Project/figures/
```

### Notebook 03 — Preprocessing and Split Export

Run:

```text
notebooks/03_preprocessing_and_split.ipynb
```

This notebook:

- Uses the fixed chronological splits from Notebook 01.
- Separates the 48 predictors from `NEXT_HOUR_CALLS`.
- Fits median imputation and `StandardScaler` using only the training set.
- Applies the fitted preprocessing pipeline to validation and test data.
- Saves scaled and unscaled versions of each split.

Important outputs in `data_processed/`:

```text
train_scaled.csv
validation_scaled.csv
test_scaled.csv
train_unscaled.csv
validation_unscaled.csv
test_unscaled.csv
numerical_preprocessor.joblib
preprocessing_metadata.json
candidate_features.json
```

Additional split-audit files are saved in:

```text
MyDrive/CS171_Police_Call_Project/results/
```

### Notebook 04 — Baselines and Traditional Models

Run:

```text
notebooks/04_baseline_and_tree_models.ipynb
```

This notebook trains and validates:

- Mean baseline.
- Persistence baseline.
- Linear regression.
- Elastic Net.
- Decision tree regressor.
- Random forest regressor.

Hyperparameters are selected using the validation set. The fitted models are saved as:

```text
models/LinearRegression/linear_regression.joblib
models/ElasticNet (Regularization)/elastic_net.joblib
models/DecisionTree/decision_tree_.joblib
models/RandomForest/random_forest.joblib
```

Do not use the test set to tune these models.
Notebook runtime: Approximately 5-6 minutes. Restart kernel if notebook is frozen. 

### Notebook 05 — Neural Networks

Run:

```text
notebooks/05_neural_networks.ipynb
```

For faster training, select a GPU in Colab:

```text
Runtime → Change runtime type → T4 GPU
```

This notebook trains:

- An MLP for 100 epochs.
- An MLP with early stopping.
- LSTM models with lookback windows of 2, 4, 6, 12, 24, and 48 hours.

The LSTM lookback must be selected using validation performance only. Based on the latest validation results, use:

```python
best_ts = 6
```

The notebook saves:

```text
models/MLP/mlp_model_epoch100.keras
models/MLP/mlp_model_es.keras
models/MLP/mlp_history_100.csv
models/MLP/mlp_history_es.csv
models/LSTM/lstm_model_timestep_2.keras
models/LSTM/lstm_model_timestep_4.keras
models/LSTM/lstm_model_timestep_6.keras
models/LSTM/lstm_model_timestep_12.keras
models/LSTM/lstm_model_timestep_24.keras
models/LSTM/lstm_model_timestep_48.keras
models/LSTM/lstm_history_timestep_*.csv
```

### Notebook 06 — Final Model Evaluation

Run:

```text
notebooks/06_model_evaluation.ipynb
```

Before running, confirm that the selected LSTM lookback matches the validation-selected value:

```python
BEST_TIME_STEP = 6
```

Load the exact feature list to preserve the feature order used during training:

```python
import json

with open(
    os.path.join(processed_data_dir, "candidate_features.json"),
    "r"
) as file:
    feature_names = json.load(file)

X_test = test_scaled_df[feature_names]
y_test = test_scaled_df["NEXT_HOUR_CALLS"]
```

The final comparison should include both baselines:

```python
training_mean = train_scaled_df["NEXT_HOUR_CALLS"].mean()
mean_preds = np.full(len(y_test_trimmed), training_mean)
```

Add the mean baseline to the model comparison table:

```python
("Mean Baseline", *compute_metrics(y_test_trimmed, mean_preds))
```

Notebook 06 then:

- Loads all saved traditional and neural-network models.
- Aligns predictions using the six-hour LSTM lookback.
- Evaluates every model on the same held-out test timestamps.
- Reports MAE, MSE, RMSE, and R².
- Produces forecast, residual, feature-importance, and LSTM-validation figures.

Final figures are saved to:

```text
MyDrive/CS171_Police_Call_Project/Results_Graphs/
```

## 5. Correct Model-Selection Procedure

Use the datasets for the following purposes:

- **Training set:** Fit model parameters.
- **Validation set:** Select hyperparameters and the LSTM lookback window.
- **Test set:** Evaluate the already-selected models once.

Do not select the LSTM lookback because it performs best on the test set. This would leak test information into model selection and make the final evaluation less reliable.

## Expected Headline Result

In the current held-out test evaluation, the tuned random forest achieved the strongest performance:

| Metric | Result |
|---|---:|
| MAE | 5.616 calls |
| RMSE | 8.142 calls |
| R² | 0.711 |

These values may change slightly if package versions, saved models, notebook code, or source data are changed.

## Troubleshooting

### `FileNotFoundError`

Confirm that:

- Google Drive is mounted.
- The folder is named exactly `CS171_Police_Call_Project`.
- Raw CSV files use the exact required filenames.
- All previous notebooks were run successfully.
- Model subfolders exist before Notebook 04 or 05 attempts to save files.

### `ModuleNotFoundError: great_tables`

Run:

```python
!pip install -q great-tables
```

Then restart the Colab runtime if necessary.

### Feature-name or feature-count error

Load `candidate_features.json` and select columns using that exact ordered list rather than dropping columns manually.

### LSTM input-shape error

Confirm that:

- `BEST_TIME_STEP` matches the saved LSTM filename.
- Test sequences were created with the same lookback.
- All non-LSTM predictions and targets were trimmed by the same lookback length.

### Different results after rerunning

TensorFlow and package-version differences may cause small changes. The notebooks use seed 42 and deterministic TensorFlow settings where possible, but exact reproducibility can still depend on the runtime and hardware.

## Team

- Afrin Hossain
- Anushka Chandrashekar
- Sampiya Bhusal

## Data Source

City of San José, **Police Calls for Service**, City of San José Open Data Portal.

## Disclaimer

This project is for academic purposes. The forecasts estimate short-term citywide call demand and should not be treated as exact staffing, deployment, or public-safety decisions.
