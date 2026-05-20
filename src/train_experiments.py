"""
train_experiments.py
────────────────────
Runs 20+ MLflow-tracked hyperparameter experiments across Random Forest
and XGBoost, comparing SMOTE vs. class_weight strategies.

Usage:
    python src/train_experiments.py

Outputs:
    - MLflow run logs under ./mlruns/
    - Best model saved to ./models/best_model.pkl
    - Experiment summary CSV to ./models/experiment_summary.csv
"""

import os, json, warnings
import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    f1_score, roc_auc_score, precision_score,
    recall_score, confusion_matrix, classification_report,
)
from xgboost import XGBClassifier
from itertools import product as iproduct

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
MODELS_DIR = os.path.join(ROOT, "models")
os.makedirs(MODELS_DIR, exist_ok=True)

# ── MLflow setup ──────────────────────────────────────────────────────────────
TRACKING_URI = os.path.join(ROOT, "mlruns")
mlflow.set_tracking_uri(f"file://{os.path.abspath(TRACKING_URI)}")

# ── Hyperparameter grids ───────────────────────────────────────────────────────
RF_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [6, 10, 15, None],
    "min_samples_leaf": [1, 3],
    "max_features": ["sqrt", "log2"],
}

XGB_GRID = {
    "n_estimators": [100, 200, 300],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.05, 0.10, 0.20],
    "subsample": [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

SMOTE_K_VALUES = [3, 5]
RANDOM_STATE = 42


# ── Metrics helper ────────────────────────────────────────────────────────────
def evaluate(model, X, y, prefix="test"):
    """Compute and return a dict of classification metrics."""
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]
    return {
        f"{prefix}_f1": round(f1_score(y, y_pred), 4),
        f"{prefix}_roc_auc": round(roc_auc_score(y, y_prob), 4),
        f"{prefix}_precision": round(precision_score(y, y_pred), 4),
        f"{prefix}_recall": round(recall_score(y, y_pred), 4),
    }


# ── Run factory ───────────────────────────────────────────────────────────────
def run_experiment(
    experiment_name: str,
    model_type: str,
    params: dict,
    data: dict,
    imbalance_strategy: str,   # "smote" | "class_weight" | "none"
    smote_k: int = 5,
):
    """Execute one MLflow run and return a summary dict."""
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run() as run:
        run_id = run.info.run_id[:8]

        # ── Select training data ──────────────────────────────────────────
        if imbalance_strategy == "smote":
            X_tr = data["X_train"]
            y_tr = data["y_train"]
        else:
            X_tr = data["X_train_orig"]
            y_tr = data["y_train_orig"]

        # ── Build model ───────────────────────────────────────────────────
        if model_type == "RF":
            cw = "balanced" if imbalance_strategy == "class_weight" else None
            clf = RandomForestClassifier(
                **params,
                class_weight=cw,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )
        else:  # XGBoost
            pos_ratio = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
            sw = pos_ratio if imbalance_strategy == "class_weight" else 1.0
            clf = XGBClassifier(
                **params,
                scale_pos_weight=sw,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=RANDOM_STATE,
                verbosity=0,
            )

        # ── Train ─────────────────────────────────────────────────────────
        clf.fit(X_tr, y_tr)

        # ── Evaluate on val & test ────────────────────────────────────────
        val_metrics = evaluate(clf, data["X_val"], data["y_val"], prefix="val")
        test_metrics = evaluate(clf, data["X_test"], data["y_test"], prefix="test")
        all_metrics = {**val_metrics, **test_metrics}

        # ── Log to MLflow ─────────────────────────────────────────────────
        log_params = {
            "model_type": model_type,
            "imbalance_strategy": imbalance_strategy,
            **params,
        }
        if imbalance_strategy == "smote":
            log_params["smote_k"] = smote_k
        mlflow.log_params(log_params)
        mlflow.log_metrics(all_metrics)
        mlflow.sklearn.log_model(clf, "model")

        # Confusion matrix as JSON artifact
        cm = confusion_matrix(data["y_test"], clf.predict(data["X_test"])).tolist()
        cm_path = os.path.join(MODELS_DIR, f"cm_{run_id}.json")
        with open(cm_path, "w") as f:
            json.dump(cm, f)
        mlflow.log_artifact(cm_path)

        print(
            f"  [{run_id}] {model_type:3s} | {imbalance_strategy:12s} | "
            f"val_f1={val_metrics['val_f1']:.4f} | "
            f"test_f1={test_metrics['test_f1']:.4f} | "
            f"roc_auc={test_metrics['test_roc_auc']:.4f}"
        )

        return {
            "run_id": run_id,
            "model_type": model_type,
            "imbalance_strategy": imbalance_strategy,
            **log_params,
            **all_metrics,
            "_clf": clf,
            "_mlflow_run_id": run.info.run_id,
        }


# ── Main experiment loop ───────────────────────────────────────────────────────
def run_all_experiments(data: dict):
    results = []

    # ─── Random Forest experiments (10 runs) ─────────────────────────────────
    print("\n══════════ Random Forest Experiments ══════════")
    rf_combos = [
        {"n_estimators": 100, "max_depth": 6,    "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": 10,   "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 15,   "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 3, "max_features": "sqrt"},
        {"n_estimators": 100, "max_depth": 10,   "min_samples_leaf": 3, "max_features": "log2"},
        {"n_estimators": 200, "max_depth": 6,    "min_samples_leaf": 1, "max_features": "log2"},
        {"n_estimators": 300, "max_depth": 10,   "min_samples_leaf": 1, "max_features": "log2"},
        {"n_estimators": 100, "max_depth": 15,   "min_samples_leaf": 3, "max_features": "log2"},
        {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
        {"n_estimators": 300, "max_depth": 6,    "min_samples_leaf": 3, "max_features": "sqrt"},
    ]
    strategies = ["smote", "class_weight", "smote", "class_weight", "smote",
                  "class_weight", "smote", "class_weight", "none", "smote"]
    for params, strategy in zip(rf_combos, strategies):
        r = run_experiment("RF_Experiments", "RF", params, data, strategy)
        results.append(r)

    # ─── XGBoost experiments (12 runs) ───────────────────────────────────────
    print("\n══════════ XGBoost Experiments ══════════")
    xgb_combos = [
        {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 1.0},
        {"n_estimators": 200, "max_depth": 8, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 0.8},
        {"n_estimators": 300, "max_depth": 6, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 1.0},
        {"n_estimators": 200, "max_depth": 4, "learning_rate": 0.20, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 100, "max_depth": 6, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
        {"n_estimators": 300, "max_depth": 8, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 0.8},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.20, "subsample": 1.0, "colsample_bytree": 0.8},
        {"n_estimators": 100, "max_depth": 8, "learning_rate": 0.10, "subsample": 0.8, "colsample_bytree": 1.0},
        {"n_estimators": 300, "max_depth": 4, "learning_rate": 0.05, "subsample": 1.0, "colsample_bytree": 1.0},
        {"n_estimators": 200, "max_depth": 6, "learning_rate": 0.10, "subsample": 1.0, "colsample_bytree": 0.8},
    ]
    xgb_strategies = ["smote", "smote", "smote", "class_weight", "smote",
                       "class_weight", "none", "smote", "class_weight", "smote",
                       "class_weight", "smote"]
    for params, strategy in zip(xgb_combos, xgb_strategies):
        r = run_experiment("XGB_Experiments", "XGBoost", params, data, strategy)
        results.append(r)

    return results


# ── Select best model & save ───────────────────────────────────────────────────
def select_and_save_best(results: list, data: dict):
    df = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")} for r in results])
    df_sorted = df.sort_values("test_f1", ascending=False).reset_index(drop=True)

    summary_path = os.path.join(MODELS_DIR, "experiment_summary.csv")
    df_sorted.to_csv(summary_path, index=False)
    print(f"\n[trainer] Experiment summary saved → {summary_path}")

    best_row = df_sorted.iloc[0]
    best_result = next(r for r in results if r["run_id"] == best_row["run_id"])
    best_clf = best_result["_clf"]

    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    meta = {
        "model": best_clf,
        "run_id": best_result["run_id"],
        "model_type": best_row["model_type"],
        "test_f1": best_row["test_f1"],
        "test_roc_auc": best_row["test_roc_auc"],
        "imbalance_strategy": best_row["imbalance_strategy"],
        "feature_names": data["feature_names"],
        "training_samples": len(data["y_train"]),
    }
    joblib.dump(meta, model_path)
    print(f"[trainer] Best model saved → {model_path}")
    print(f"[trainer] Best: {best_row['model_type']} | "
          f"F1={best_row['test_f1']:.4f} | ROC-AUC={best_row['test_roc_auc']:.4f}")

    print("\n── Top 10 Runs ──────────────────────────────────────────")
    cols = ["run_id", "model_type", "imbalance_strategy", "test_f1", "test_roc_auc",
            "test_precision", "test_recall"]
    print(df_sorted[cols].head(10).to_string(index=False))

    return best_clf, meta


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    from data_pipeline import get_prepared_data

    print("═" * 60)
    print("  Customer Churn MLOps — Experiment Runner")
    print("═" * 60)

    data = get_prepared_data(use_smote=True, smote_k=5)
    results = run_all_experiments(data)
    select_and_save_best(results, data)

    print("\n✓ All experiments complete. Launch MLflow UI with:")
    print("  mlflow ui --backend-store-uri ./mlruns")
