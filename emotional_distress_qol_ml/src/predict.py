import os
import datetime
import argparse
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report, roc_curve
)

def load_model(model_path):
    return joblib.load(model_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Path to trained model file")
    parser.add_argument("--data", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--target", type=str, required=True, help="Target column name")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory for plots")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    # ================================
    # 🗂️ Handle output directory
    # ================================
    if args.outdir is None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        args.outdir = f"results_{timestamp}"
    os.makedirs(args.outdir, exist_ok=True)

    # Load model
    model = load_model(args.model)
    print(f"\nLoaded model: {args.model}\n")

    # Load dataset
    df = pd.read_csv(args.data)
    X = df.drop(columns=[args.target])
    y = df[args.target]

    # Predict
    preds = model.predict(X)
    print("Predictions (first 20):")
    print(preds[:20], "\n")

    # Metrics
    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds, zero_division=0)
    rec = recall_score(y, preds, zero_division=0)
    f1 = f1_score(y, preds, zero_division=0)

    try:
        probs = model.predict_proba(X)[:, 1]
        auc = roc_auc_score(y, probs)
    except:
        probs = None
        auc = None

    print("✅ Performance Metrics:")
    print(f"Accuracy:   {acc:.4f}")
    print(f"Precision:  {prec:.4f}")
    print(f"Recall:     {rec:.4f}")
    print(f"F1 Score:   {f1:.4f}")
    if auc is not None:
        print(f"ROC AUC:    {auc:.4f}")

    # Confusion matrix
    cm = confusion_matrix(y, preds)
    print("\nConfusion Matrix:")
    print(cm)

    # Class distribution
    unique, counts = np.unique(preds, return_counts=True)
    pred_counts = dict(zip(unique, counts))
    print("\nPrediction counts:", pred_counts)

    # Classification report
    print("\nClassification Report:")
    report = classification_report(y, preds, digits=4)
    print(report)

    # Save classification report
    with open(f"{args.outdir}/classification_report.txt", "w") as f:
        f.write("Classification Report\n")
        f.write("====================\n\n")
        f.write(report)

    print(f"\n📄 Classification report saved as: {args.outdir}/classification_report.txt")

    # ========================
    # 📊 Plot Confusion Matrix
    # ========================
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No Distress", "Distress"],
                yticklabels=["No Distress", "Distress"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{args.outdir}/confusion_matrix.png")
    plt.close()

    # ========================
    # 📊 Plot ROC Curve
    # ========================
    if probs is not None:
        fpr, tpr, _ = roc_curve(y, probs)
        plt.figure(figsize=(6, 5))
        plt.plot(fpr, tpr, label=f"ROC Curve (AUC={auc:.2f})")
        plt.plot([0, 1], [0, 1], "k--")
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(f"{args.outdir}/roc_curve.png")
        plt.close()

    print(f"\n📂 Plots saved in: {args.outdir}/ (confusion_matrix.png, roc_curve.png)")

if __name__ == "__main__":
    main()
