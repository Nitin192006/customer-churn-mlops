"""
evaluate.py
───────────
Standalone evaluation utilities: classification report, ROC curve,
feature importance plot, and SMOTE vs class_weight comparison table.
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, roc_curve, auc,
    ConfusionMatrixDisplay, confusion_matrix,
)

PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "models", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


def print_report(model, X_test, y_test, label="Test"):
    """Print a full classification report."""
    y_pred = model.predict(X_test)
    print(f"\n── Classification Report [{label}] ─────────────────")
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))


def plot_roc(model, X_test, y_test, label="Best Model", save=True):
    """Plot and optionally save ROC curve."""
    y_prob = model.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, lw=2, label=f"{label} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Churn Prediction")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "roc_curve.png")
        fig.savefig(path, dpi=120)
        print(f"[evaluate] ROC curve saved → {path}")
    plt.close(fig)
    return roc_auc


def plot_feature_importance(model, feature_names, top_n=20, save=True):
    """Bar chart of top-N feature importances."""
    try:
        importances = model.feature_importances_
    except AttributeError:
        print("[evaluate] Model has no feature_importances_ attribute.")
        return

    idx = np.argsort(importances)[::-1][:top_n]
    top_feats = [feature_names[i] for i in idx]
    top_vals = importances[idx]

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(top_feats[::-1], top_vals[::-1], color="#3B82F6")
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "feature_importance.png")
        fig.savefig(path, dpi=120)
        print(f"[evaluate] Feature importance plot saved → {path}")
    plt.close(fig)
    return dict(zip(top_feats, top_vals))


def plot_confusion_matrix(model, X_test, y_test, save=True):
    """Confusion matrix heatmap."""
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, model.predict(X_test))
    disp = ConfusionMatrixDisplay(cm, display_labels=["No Churn", "Churn"])
    disp.plot(ax=ax, colorbar=False, cmap="Blues")
    ax.set_title("Confusion Matrix — Test Set")
    plt.tight_layout()
    if save:
        path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=120)
        print(f"[evaluate] Confusion matrix saved → {path}")
    plt.close(fig)


def smote_vs_class_weight_comparison(summary_csv: str):
    """
    Load experiment summary and print a side-by-side comparison of
    SMOTE vs class_weight strategies per model type.
    """
    df = pd.read_csv(summary_csv)
    print("\n── SMOTE vs Class Weight Comparison ────────────────────────────")
    for model_type in ["RF", "XGBoost"]:
        subset = df[df["model_type"] == model_type]
        for strategy in ["smote", "class_weight", "none"]:
            s = subset[subset["imbalance_strategy"] == strategy]
            if s.empty:
                continue
            best = s.iloc[0]
            print(
                f"  {model_type:8s} | {strategy:12s} | "
                f"F1={best['test_f1']:.4f} | AUC={best['test_roc_auc']:.4f} | "
                f"Prec={best['test_precision']:.4f} | Rec={best['test_recall']:.4f}"
            )
    print()


if __name__ == "__main__":
    import sys, joblib
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import get_prepared_data

    MODELS_DIR_ROOT = os.path.join(os.path.dirname(__file__), "..", "models")
    meta = joblib.load(os.path.join(MODELS_DIR_ROOT, "best_model.pkl"))
    model = meta["model"]
    feature_names = meta["feature_names"]

    data = get_prepared_data(use_smote=False)  # load test set without SMOTE
    print_report(model, data["X_test"], data["y_test"])
    plot_roc(model, data["X_test"], data["y_test"])
    plot_feature_importance(model, feature_names)
    plot_confusion_matrix(model, data["X_test"], data["y_test"])
    smote_vs_class_weight_comparison(os.path.join(MODELS_DIR_ROOT, "experiment_summary.csv"))
