
import argparse, os, json, joblib
import numpy as np, pandas as pd
from .utils import standardize_column_names, detect_columns

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="distress")
    ap.add_argument("--group_col", default="cohort_role")
    ap.add_argument("--id_col", default="subject_id")
    args = ap.parse_args()

    df = pd.read_csv(args.data)
    df = standardize_column_names(df)
    feature_cols, cat_cols = detect_columns(df, args.target, args.id_col, args.group_col)

    model = joblib.load(args.model)
    X = df[feature_cols + cat_cols]
    prob = model.predict_proba(X)[:,1]
    out_df = df.copy()
    out_df["distress_proba"] = prob
    if args.id_col in out_df.columns:
        cols = [args.id_col, "distress_proba"]
        out_df[cols].to_csv(args.out, index=False)
    else:
        out_df[["distress_proba"]].to_csv(args.out, index=False)
    print(f"Wrote predictions to {args.out}")

if __name__ == "__main__":
    main()
