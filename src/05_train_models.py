"""
05_train_models.py

Purpose
-------
Train a Logistic Regression baseline and a Random Forest on the 90-point
training dataset, using a spatially-aware train/test split, and evaluate
both with metrics appropriate for a small, imbalanced multi-class problem.

Why a spatially-stratified split, not a random one
------------------------------------------------------
Randomly shuffling points into train/test can let geographically nearby
points end up on both sides of the split. Since nearby pixels tend to be
spectrally similar (spatial autocorrelation), the model could then get a
falsely high test score just by "recognizing the neighborhood" rather
than genuinely learning the spectral/phenology pattern of each class.

This script splits WITHIN each class by longitude (the first ~70% of
each class's points, sorted west-to-east, go to training; the rest go to
testing), so the model is evaluated on a genuinely different part of the
study area for every class — the honest way to test whether the model
generalizes, given our small dataset.

Why macro-averaged metrics
-----------------------------
Our classes are imbalanced (27 cropland vs. as few as 13
vegetation_forest points). Plain accuracy would let the model look good
just by nailing the biggest class. Macro-averaging computes each metric
per class and then averages, weighting every class equally regardless of
size — a fairer picture for this dataset.

Honest caveat about sample size
-----------------------------------
90 training points across 5 classes is a small dataset (as few as ~13
points for the smallest class, with only ~4 in that class's test set).
Results here should be read as a reasonable baseline demonstration, not
as a precise, generalizable accuracy figure — this is stated plainly in
the README's limitations section.
"""

import pandas as pd
import numpy as np
import os
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report,
)
import matplotlib.pyplot as plt

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")
FIGURES_DIR = os.path.join(BASE_DIR, "outputs", "figures")
os.makedirs(METRICS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

FEATURE_COLS = [
    "dryseason_blue", "dryseason_green", "dryseason_red", "dryseason_rededge",
    "dryseason_nir", "dryseason_swir1", "dryseason_swir2",
    "ndvi_s1_land_prep_mar_apr", "ndvi_s2_early_growth_may_jun",
    "ndvi_s3_peak_growth_jun_sep", "ndvi_s4_maturity_sep_oct",
    "ndvi_s5_postharvest_dec_jan", "ndvi_amplitude",
]
LABEL_COL = "final_label"
TRAIN_FRACTION = 0.7
RANDOM_SEED = 42

# ---------------------------------------------------------------------------
# 1. Load data and do the spatially-stratified split
# ---------------------------------------------------------------------------
df = pd.read_csv(os.path.join(PROCESSED_DIR, "training_data.csv"))
print(f"Loaded {len(df)} training points")
print(df[LABEL_COL].value_counts().to_string())

train_rows = []
test_rows = []

for label, group in df.groupby(LABEL_COL):
    group_sorted = group.sort_values("lon")
    n_train = max(1, int(round(len(group_sorted) * TRAIN_FRACTION)))
    train_rows.append(group_sorted.iloc[:n_train])
    test_rows.append(group_sorted.iloc[n_train:])

train_df = pd.concat(train_rows).reset_index(drop=True)
test_df = pd.concat(test_rows).reset_index(drop=True)

print(f"\nTrain set: {len(train_df)} points")
print(train_df[LABEL_COL].value_counts().to_string())
print(f"\nTest set: {len(test_df)} points")
print(test_df[LABEL_COL].value_counts().to_string())

X_train = train_df[FEATURE_COLS].values
y_train = train_df[LABEL_COL].values
X_test = test_df[FEATURE_COLS].values
y_test = test_df[LABEL_COL].values

# ---------------------------------------------------------------------------
# 2. Scale features (fit on train only, to avoid leaking test info)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

CLASS_LABELS = sorted(df[LABEL_COL].unique())

# ---------------------------------------------------------------------------
# 3. Train and evaluate both models
# ---------------------------------------------------------------------------
def evaluate(name, model, X_te, y_te):
    y_pred = model.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_te, y_pred, labels=CLASS_LABELS, average="macro", zero_division=0
    )
    cm = confusion_matrix(y_te, y_pred, labels=CLASS_LABELS)

    print(f"\n{'=' * 60}")
    print(f"{name}")
    print(f"{'=' * 60}")
    print(f"Accuracy:           {acc:.3f}")
    print(f"Macro precision:    {precision:.3f}")
    print(f"Macro recall:       {recall:.3f}")
    print(f"Macro F1:           {f1:.3f}")
    print("\nPer-class report:")
    print(classification_report(y_te, y_pred, labels=CLASS_LABELS, zero_division=0))

    return {
        "name": name,
        "accuracy": acc,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


print("\nTraining Logistic Regression...")
logreg = LogisticRegression(max_iter=2000, random_state=RANDOM_SEED)
logreg.fit(X_train_scaled, y_train)
logreg_results = evaluate("Logistic Regression", logreg, X_test_scaled, y_test)

print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_SEED)
rf.fit(X_train, y_train)  # tree-based models don't need scaled features
rf_results = evaluate("Random Forest", rf, X_test, y_test)

# ---------------------------------------------------------------------------
# 4. Save confusion matrices as figures
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, labels, title, out_path):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(title)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=130)
    plt.close()


plot_confusion_matrix(
    logreg_results["confusion_matrix"], CLASS_LABELS,
    "Logistic Regression — Confusion Matrix",
    os.path.join(FIGURES_DIR, "confusion_matrix_logreg.png"),
)
plot_confusion_matrix(
    rf_results["confusion_matrix"], CLASS_LABELS,
    "Random Forest — Confusion Matrix",
    os.path.join(FIGURES_DIR, "confusion_matrix_rf.png"),
)
print(f"\nSaved confusion matrix plots to {FIGURES_DIR}")

# ---------------------------------------------------------------------------
# 5. Feature importance (Random Forest only — Logistic Regression's
#    coefficients aren't directly comparable across differently-scaled
#    features in the same intuitive way)
# ---------------------------------------------------------------------------
importances = pd.Series(rf.feature_importances_, index=FEATURE_COLS).sort_values(ascending=False)
print("\nRandom Forest feature importances:")
print(importances.to_string())

fig, ax = plt.subplots(figsize=(8, 5))
importances.plot(kind="barh", ax=ax)
ax.invert_yaxis()
ax.set_xlabel("Importance")
ax.set_title("Random Forest Feature Importance")
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, "feature_importance_rf.png"), dpi=130)
plt.close()

# ---------------------------------------------------------------------------
# 6. Save metrics summary and trained models
# ---------------------------------------------------------------------------
summary = pd.DataFrame([
    {"model": "Logistic Regression", "accuracy": logreg_results["accuracy"],
     "macro_precision": logreg_results["macro_precision"],
     "macro_recall": logreg_results["macro_recall"], "macro_f1": logreg_results["macro_f1"]},
    {"model": "Random Forest", "accuracy": rf_results["accuracy"],
     "macro_precision": rf_results["macro_precision"],
     "macro_recall": rf_results["macro_recall"], "macro_f1": rf_results["macro_f1"]},
])
summary_path = os.path.join(METRICS_DIR, "model_comparison.csv")
summary.to_csv(summary_path, index=False)
print(f"\nSaved model comparison: {summary_path}")

joblib.dump(logreg, os.path.join(PROCESSED_DIR, "model_logreg.joblib"))
joblib.dump(rf, os.path.join(PROCESSED_DIR, "model_rf.joblib"))
joblib.dump(scaler, os.path.join(PROCESSED_DIR, "feature_scaler.joblib"))
print("Saved trained models and scaler to data/processed/")

print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(summary.to_string(index=False))