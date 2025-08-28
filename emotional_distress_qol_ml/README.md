
# Predicting Emotional Distress in Cancer Patients & Family Members using QoL + ML

This repo trains and evaluates machine learning models to predict *emotional distress* (binary) from **Quality of Life (QoL)** questionnaires (e.g., EORTC QLQ-C30, FACT-G, PROMIS items), plus demographics/clinical covariates. It includes a synthetic data generator so you can run end‑to‑end without real PHI.

---

## Quickstart (local)

```bash
# 1) Create environment
python -m venv .venv
# or: conda create -n qol-ml python=3.11 -y && conda activate qol-ml
. .venv/bin/activate  # Windows: .venv\Scripts\activate

# 2) Install deps
pip install -r requirements.txt

# 3) (Optional) Generate synthetic data to test the pipeline
python -m src.make_synthetic --n 1500 --seed 42

# 4) Train
python -m src.train --data data/qol_dataset.csv --target distress --group_col cohort_role --id_col subject_id --outdir models

# 5) Evaluate on a held‑out test set (automatically done in train) or on an external CSV:
python -m src.evaluate --data data/qol_dataset.csv --model models/best_model.joblib --target distress

# 6) Inference on new data (no targets needed)
python -m src.infer --data data/new_qol.csv --model models/best_model.joblib --out predictions.csv
```

## Data expectations

- CSV file where each row is a **patient** or a **family member**.
- Include:
  - `subject_id`: unique identifier (string/int).
  - `cohort_role`: one of `patient` or `family` (any case).
  - QoL feature columns, e.g. `qlq_*`, `fact_*`, `promis_*` (floats 0-100 or Likert recoded).
  - Optional demographics/clinical covariates: `age`, `sex`, `stage`, `treatment`, etc.
  - `distress` (0/1) — binary target. For example, **Distress Thermometer ≥ 4** → 1; else 0.

> The code auto‑detects QoL columns by prefix (`qlq_`, `fact_`, `promis_`) plus any numeric covariates (excluding target/group/id).

## Models

The training script compares a few well‑performing baselines:

- **Logistic Regression** (with elastic net)
- **Random Forest**
- **XGBoost** (if installed; otherwise skipped gracefully)
- **LightGBM** (if installed; otherwise skipped gracefully)

All models run inside a **Pipeline** with preprocessing:
- Missing value imputation (median for numeric, most frequent for categorical)
- Standardization for linear models
- One‑hot encoding for categoricals
- **SMOTE** for class imbalance (train folds only)
- Probability **calibration** (Platt/Isotonic per CV)

Evaluation: ROC‑AUC, PR‑AUC, accuracy, balanced accuracy, F1, calibration Brier score. Plots saved under `reports/`.

## Ethics / Safety (brief)

- Use **de‑identified** data, follow IRB/IEC guidelines.
- Validate for **demographic fairness**; script outputs group‑wise metrics if `sex`/`age_group` exist.
- Models assist clinicians; they must not replace clinical judgment.

## Repository layout

```
data/                 # put CSVs here (synthetic created by make_synthetic)
models/               # trained artifacts (joblib + metadata JSON)
notebooks/            # optional EDA
reports/              # metrics & figures
src/
  evaluate.py
  infer.py
  make_synthetic.py
  train.py
  utils.py
requirements.txt
```

## Citation (example)

> Project: Predicting Emotional Distress in Cancer Patients and Their Family Members Using Quality of Life Data and Machine Learning.

