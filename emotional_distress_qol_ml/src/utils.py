
from __future__ import annotations
import json, os, re, warnings
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd

QOL_PREFIXES = ("qlq_", "fact_", "promis_")

def detect_columns(df: pd.DataFrame,
                   target: str,
                   id_col: Optional[str] = None,
                   group_col: Optional[str] = None) -> Tuple[List[str], List[str]]:
    cols = list(df.columns)
    drop = {target}
    if id_col: drop.add(id_col)
    if group_col: drop.add(group_col)
    # QoL-like + numeric covariates
    feature_cols = []
    cat_cols = []
    for c in cols:
        if c in drop: 
            continue
        if any(c.lower().startswith(p) for p in QOL_PREFIXES):
            feature_cols.append(c)
        else:
            if pd.api.types.is_numeric_dtype(df[c]):
                feature_cols.append(c)
            else:
                cat_cols.append(c)
    return feature_cols, cat_cols

def stratify_series(y: pd.Series) -> pd.Series:
    # For rare positive rate, stratify by y (binary). If multi-class or continuous, return None.
    return y if y.nunique() <= 10 else None
def save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)
def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [re.sub(r"\s+", "_", c.strip().lower()) for c in df.columns]
    return df

def summarize_target(df: pd.DataFrame, target: str) -> Dict[str, Any]:
    vc = df[target].value_counts(dropna=False).to_dict()
    pos_rate = float(df[target].mean()) if pd.api.types.is_numeric_dtype(df[target]) else None
    return {"counts": vc, "pos_rate": pos_rate}
