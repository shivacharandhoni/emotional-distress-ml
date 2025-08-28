# Predicting Emotional Distress in Cancer Patients Using QoL Data and Machine Learning

This project predicts emotional distress in cancer patients and their family members using quality-of-life (QoL) data and machine learning models.  

## Workflow Overview

```text
+------------------+        +------------------+       +-------------------+
|                  |        |                  |       |                   |
|   Prepare Data   +------->|   Train Models   +------>|  Evaluate Models  |
| (data/qol_dataset.csv)   | (LogReg, RF, XGB, LGBM)|  | (Accuracy, ROC AUC)|
+------------------+        +------------------+       +-------------------+
                                         |
                                         v
                              +--------------------+
                              | Compare & Select   |
                              |   Best Model       |
                              +--------------------+
                                         |
                                         v
                              +--------------------+
                              |  Run Predictions   |
                              | (Confusion Matrix,|
                              | ROC, Report)       |
                              +--------------------+

## Overview
The workflow includes:
- Data preparation
- Model training (Logistic Regression, Random Forest, XGBoost, LightGBM)
- Predictions and performance evaluation
- Visualization of confusion matrix and ROC curves
- Model comparison

---

## Step 1: Clone Repository

git clone https://github.com/shivacharandhoni/emotional-distress-ml.git
cd emotional-distress-ml

Step 2: Set Up Python Environment

Create a virtual environment:

python -m venv .venv


Activate the environment:

Windows (Git Bash / PowerShell):

source .venv/Scripts/activate


Linux / MacOS:

source .venv/bin/activate


Install dependencies:

pip install -r requirements.txt

Step 3: Prepare Data

Place your QoL dataset CSV in the data/ folder. Example:
data/qol_dataset.csv

Ensure it contains a target column (distress) and identifier columns (subject_id, cohort_role).

Step 4: Train Models
python -m src.train \
  --data data/qol_dataset.csv \
  --target distress \
  --group_col cohort_role \
  --id_col subject_id \
  --outdir models


Trains: Logistic Regression, Random Forest, XGBoost, LightGBM

Outputs:

models/*.pkl → Trained models

models/reports/model_results.json → Model metrics

ROC plots per model


Step 5: Run Predictions
python -m src.predict \
  --model models/rf_model.pkl \
  --data data/qol_dataset.csv \
  --target distress \
  --outdir results


Outputs:

results/confusion_matrix.png

results/roc_curve.png

results/classification_report.txt

Console prints Accuracy, Precision, Recall, F1, ROC AUC

Step 6: Compare Models

Copy model_results.json to metadata.json:

copy .\models\reports\model_results.json .\models\metadata.json


Run comparison script:

python compare_models.py


Generates comparison plots and highlights the best model.

Step 7: Folder Structure
emotional-distress-ml/
│
├─ data/
│   └─ qol_dataset.csv
│
├─ models/
│   ├─ rf_model.pkl
│   ├─ log_reg_model.pkl
│   ├─ xgb_model.pkl
│   ├─ lgbm_model.pkl
│   └─ reports/
│       └─ model_results.json
│
├─ results/
│   ├─ confusion_matrix.png
│   ├─ roc_curve.png
│   └─ classification_report.txt
│
├─ src/
│   ├─ train.py
│   └─ predict.py
│
├─ compare_models.py
├─ README.md
└─ requirements.txt

Step 8: GitHub Upload
git add .
git commit -m "Add project files and instructions"
git push origin main

Notes:

Virtual environment must be active before running scripts.

All outdir folders (models, results) are auto-created if missing.

Accuracy may vary depending on the dataset used.

## Dependencies

Required Python libraries:

- pandas
- numpy
- scikit-learn
- xgboost
- lightgbm
- matplotlib
- seaborn

Install them with:

pip install -r requirements.txt
