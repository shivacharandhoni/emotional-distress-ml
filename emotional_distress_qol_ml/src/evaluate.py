
import argparse, os, json, joblib
import numpy as np, pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score, brier_score_loss
from .utils import standardize_column_names, detect_columns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--target", default="distress")
    ap.add_argument("--group_col", default="cohort_role")
    ap.add_argument("--id_col", default="subject_id")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = standardize_column_names(df)
    feature_cols, cat_cols = detect_columns(df, args.target, args.id_col, args.group_col)

    model = joblib.load(args.model)
    X = df[feature_cols + cat_cols]
    y = df[args.target].astype(int)
    prob = model.predict_proba(X)[:,1]
    pred = (prob >= 0.5).astype(int)

    out = {
        "roc_auc": float(roc_auc_score(y, prob)),
        "pr_auc": float(average_precision_score(y, prob)),
        "brier": float(brier_score_loss(y, prob)),
    }
    print(json.dumps(out, indent=2))
    print("\nClassification report:\n")
    print(classification_report(y, pred, digits=3))

if __name__ == "__main__":
    main()
