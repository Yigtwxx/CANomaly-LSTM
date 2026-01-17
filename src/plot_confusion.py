# plot_confusion.py (Seaborn version)
# ==============================================================
# 📌 FILE PURPOSE: Confusion Matrix and Classification Report
# - Read error and label values from recon_errors.csv
# - Automatically find threshold that maximizes F1 score
# - Elegant heatmap visualization with Seaborn
# - Outputs: confusion_matrix.png, confusion_report.txt
# ==============================================================

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns                    # Visualization library

from sklearn.metrics import confusion_matrix, classification_report, f1_score

# -------------------- SETTINGS --------------------
CSV_PATH = "recon_errors.csv"   # Input file (train_lstm_ae.py output)
USER_THRESHOLD = None           # Manual threshold (e.g., 0.7), None for automatic

# -------------------- HELPER FUNCTIONS --------------------
def find_column(df, candidates):
    """
    Find the appropriate column name from possible candidates in DataFrame.
    Provides compatibility with different data formats.
    """
    for c in candidates:
        if c in df.columns:
            return c
    return None

# -------------------- FILE CHECK --------------------
if not os.path.exists(CSV_PATH):
    print(f"❌ Error: '{CSV_PATH}' not found. Was train_lstm_ae.py executed?")
    sys.exit(1)

df = pd.read_csv(CSV_PATH)
print("📄 CSV columns:", list(df.columns))

# Detect error and label columns
# Alternative names are checked for different naming conventions
err_col = find_column(df, ['error', 'recon_error', 'reconstruction_error'])
lab_col = find_column(df, ['label', 'y_true', 'anomaly', 'target'])

if err_col is None or lab_col is None:
    print("⚠️ Expected columns not found. Please check the file.")
    sys.exit(1)

errors = df[err_col].values           # Reconstruction errors
y_true = df[lab_col].astype(int).values  # True labels (0=normal, 1=anomaly)

# -------------------- THRESHOLD DETERMINATION --------------------
if USER_THRESHOLD is not None:
    thr = float(USER_THRESHOLD)
    auto_info = "(manual threshold)"
else:
    # Automatic threshold search: maximize F1 score
    # Search range: between 50th and 99.9th percentile
    cand_thrs = np.linspace(np.percentile(errors, 50), np.percentile(errors, 99.9), 200)
    best_f1, best_thr = -1, cand_thrs[0]
    for t in cand_thrs:
        preds = (errors > t).astype(int)  # Error > threshold means anomaly
        f1 = f1_score(y_true, preds)
        if f1 > best_f1:
            best_f1, best_thr = f1, t
    thr = best_thr
    auto_info = f"(automatic, best F1={best_f1:.4f})"

print(f"\n🔹 Threshold used: {thr:.6f} {auto_info}")

# -------------------- PREDICTION AND METRICS --------------------
y_pred = (errors > thr).astype(int)  # Binary predictions
cm = confusion_matrix(y_true, y_pred)  # 2x2 confusion matrix

# Parse confusion matrix values
# TN (True Negative): Correctly predicted normal
# FP (False Positive): Normal incorrectly labeled as anomaly
# FN (False Negative): Missed anomaly
# TP (True Positive): Correctly predicted anomaly
tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)

print("\n📊 Confusion Matrix:")
print(cm)
print("\n📈 Classification Report:\n", classification_report(y_true, y_pred, digits=4))

# -------------------- SEABORN VISUALIZATION --------------------
plt.figure(figsize=(6,5))
sns.set_theme(style="darkgrid")  # Seaborn theme setting

# Draw confusion matrix as heatmap
ax = sns.heatmap(cm,
                 annot=True, fmt="d",      # Show numbers in cells
                 cmap="coolwarm",          # Color palette
                 cbar=True,                # Color scale
                 linewidths=0.6,           # Cell line thickness
                 linecolor='gray',
                 annot_kws={"size":14, "weight":"bold"})

# Title and labels
plt.title("Confusion Matrix (Seaborn Heatmap)", fontsize=15, weight="bold")
plt.xlabel(f"Predicted Label\nThreshold={thr:.6f}", fontsize=12)
plt.ylabel("True Label", fontsize=12)
plt.xticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'])
plt.yticks([0.5, 1.5], ['Normal (0)', 'Anomaly (1)'], rotation=0)

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=300)
plt.close()
print("🖼️ confusion_matrix.png file saved.")

# -------------------- TEXT REPORT --------------------
with open("confusion_report.txt", "w", encoding="utf-8") as f:
    f.write(f"Threshold value: {thr:.6f} {auto_info}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(np.array2string(cm))
    f.write("\n\n")
    f.write(classification_report(y_true, y_pred, digits=4))
print("🗒️ confusion_report.txt saved.")

# -------------------- SUMMARY --------------------
acc = (tp+tn)/(tp+tn+fp+fn)  # Calculate accuracy
print(f"\n✅ Accuracy: {acc:.4f}")
print(f"TP={tp}, FP={fp}, FN={fn}, TN={tn}")
print("Completed.")
