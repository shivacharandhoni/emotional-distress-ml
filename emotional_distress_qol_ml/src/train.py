from __future__ import annotations
import pandas as pd
import numpy as np
import argparse, os, json, warnings, joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    accuracy_score, balanced_accuracy_score, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

from .utils import detect_columns, standardize_column_names, summarize_target, save_json


# -------------------------------
# Optional Imports
# -------------------------------
def try_import_xgb():
    try:
        import xgboost as xgb  # noqa
        return True
    except Exception:
        return False


def try_import_lgbm():
    try:
        import lightgbm as lgb  # noqa
        return True
    except Exception:
        return False


# -------------------------------
# Preprocessor
# -------------------------------
def build_preprocessor(num_cols, cat_cols, scale_linear=False):
    transformers = []

    if len(num_cols) > 0:   # ✅ FIX: explicit length check
        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if scale_linear:
            num_steps.append(("scaler", StandardScaler()))
        transformers.append(("num", Pipeline(num_steps), num_cols))

    if len(cat_cols) > 0:   # ✅ FIX: explicit length check
        cat_steps = [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
        transformers.append(("cat", Pipeline(cat_steps), cat_cols))

    return ColumnTransformer(transformers)


# -------------------------------
# Train & Evaluate Models
# -------------------------------
def train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, outdir):
    results = {}
    models = {
        "log_reg": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "rf": RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42),
    }

    if try_import_xgb():
        from xgboost import XGBClassifier
        models["xgb"] = XGBClassifier(
            use_label_encoder=False, eval_metric="logloss", random_state=42
        )

    if try_import_lgbm():
        from lightgbm import LGBMClassifier
        models["lgbm"] = LGBMClassifier(random_state=42)

    for name, model in models.items():
        print(f"Training {name}...")
        pipe = ImbPipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("smote", SMOTE(random_state=42)),
                ("clf", model),
            ]
        )

        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        metrics = {
            "roc_auc": roc_auc_score(y_test, y_prob),
            "avg_precision": average_precision_score(y_test, y_prob),
            "f1": f1_score(y_test, y_pred),
            "accuracy": accuracy_score(y_test, y_pred),
            "balanced_acc": balanced_accuracy_score(y_test, y_pred),
            "brier": brier_score_loss(y_test, y_prob),
        }
        results[name] = metrics

        # Save model
        joblib.dump(pipe, os.path.join(outdir, f"{name}_model.pkl"))

        # Save ROC curve
        plt.figure()
        from sklearn.metrics import RocCurveDisplay
        RocCurveDisplay.from_estimator(pipe, X_test, y_test)
        plt.title(f"ROC Curve - {name}")
        plt.savefig(os.path.join(outdir, f"{name}_roc.png"))
        plt.close()

    return results


# -------------------------------
# Main Script
# -------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, required=True)
    ap.add_argument("--target", type=str, default="distress")
    ap.add_argument("--group_col", type=str, default="cohort_role")
    ap.add_argument("--id_col", type=str, default="subject_id")
    ap.add_argument("--outdir", type=str, default="models")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    reports_dir = os.path.join(args.outdir, "reports")
    os.makedirs(reports_dir, exist_ok=True)

    # -------------------------------
    # Load Data
    # -------------------------------
    df = pd.read_csv(args.data)
    df = standardize_column_names(df)
    assert args.target in df.columns, f"Target '{args.target}' not found."
    feature_cols, cat_cols = detect_columns(df, args.target, args.id_col, args.group_col)

    X = df[feature_cols]
    y = df[args.target]

    print("Target summary:")
    print(summarize_target(df, args.target))

    # -------------------------------
    # Split
    # -------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.seed
    )

    # -------------------------------
    # Preprocessor
    # -------------------------------
    preprocessor = build_preprocessor(
        X_train.select_dtypes(include=np.number).columns,
        X_train.select_dtypes(exclude=np.number).columns,
    )

    # -------------------------------
    # Train & Evaluate
    # -------------------------------
    results = train_and_evaluate(X_train, y_train, X_test, y_test, preprocessor, args.outdir)

    # -------------------------------
    # Save Results
    # -------------------------------
    save_json(results, os.path.join(reports_dir, "model_results.json"))
    print("Training completed. Results saved.")


if __name__ == "__main__":
    main()
