import json
import pandas as pd
import matplotlib.pyplot as plt
import os

# Make sure results folder exists
os.makedirs("results", exist_ok=True)

# Load your JSON results
with open("models/metadata.json") as f:
    results = json.load(f)

# Convert to DataFrame
df_results = pd.DataFrame(results).T  # models as rows

# Optional: sort by ROC AUC
df_results = df_results.sort_values("roc_auc", ascending=False)

# Save comparison table CSV
df_results.to_csv("results/model_comparison.csv", index=True)
print("Saved CSV: results/model_comparison.csv")
print(df_results)

# Plot metrics comparison
metrics = ["roc_auc", "accuracy", "f1"]

plt.figure(figsize=(8, 5))
df_results[metrics].plot(kind="bar")
plt.title("Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("results/model_comparison.png")
plt.show()
print("Saved plot: results/model_comparison.png")
