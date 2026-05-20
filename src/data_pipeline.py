"""
data_pipeline.py
────────────────
End-to-end data preparation for the Telco Customer Churn dataset.
Handles loading, cleaning, encoding, scaling, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib
import os

# ── Constants ────────────────────────────────────────────────────────────────
TARGET = "Churn"
RANDOM_STATE = 42
DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "WA_Fn-UseC_-Telco-Customer-Churn.csv")
SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "models", "scaler.pkl")

NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORICAL_COLS = [
    "gender", "Partner", "Dependents", "PhoneService", "MultipleLines",
    "InternetService", "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod",
]
DROP_COLS = ["customerID"]


def load_raw(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the Telco CSV. Falls back to synthetic data if file is missing."""
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"[data_pipeline] Loaded real dataset: {df.shape}")
    else:
        print("[data_pipeline] Dataset not found — generating synthetic data for demo.")
        df = _generate_synthetic(n=7043)
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleaning strategy:
    - Drop customerID (identifier, no predictive signal).
    - TotalCharges has ~11 whitespace entries → coerce to NaN, then impute
      with median (safe because they are brand-new customers with tenure=0).
    - Binarise target: 'Yes' → 1, 'No' → 0.
    """
    df = df.copy()

    # Drop ID column
    df.drop(columns=DROP_COLS, errors="ignore", inplace=True)

    # Fix TotalCharges whitespace → NaN → median impute
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    median_tc = df["TotalCharges"].median()
    n_missing = df["TotalCharges"].isna().sum()
    df["TotalCharges"].fillna(median_tc, inplace=True)
    print(f"[data_pipeline] Imputed {n_missing} missing TotalCharges values with median={median_tc:.2f}")

    # Encode target
    df[TARGET] = (df[TARGET] == "Yes").astype(int)

    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering decisions:
    - charges_per_tenure: monthly charges normalised by tenure (1-indexed to
      avoid div-by-zero). Captures pricing intensity relative to loyalty.
    - long_contract: binary flag for 1-year or 2-year contracts vs. month-to-month.
      Month-to-month customers churn far more frequently.
    - senior_no_support: interaction flag for SeniorCitizen without TechSupport.
      Seniors lacking support correlate with higher churn.
    """
    df = df.copy()

    df["charges_per_tenure"] = df["MonthlyCharges"] / (df["tenure"] + 1)
    df["long_contract"] = (df["Contract"].isin(["One year", "Two year"])).astype(int)
    df["senior_no_support"] = (
        (df["SeniorCitizen"] == 1) & (df["TechSupport"] == "No")
    ).astype(int)

    print("[data_pipeline] Derived features: charges_per_tenure, long_contract, senior_no_support")
    return df


def encode_and_scale(df: pd.DataFrame, fit_scaler: bool = True):
    """
    One-hot encode categoricals; StandardScale numericals.
    Returns (X, y, feature_names, scaler).
    """
    df = df.copy()
    y = df.pop(TARGET)

    # One-hot encode
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS, drop_first=False)

    # Scale numericals + derived numerical features
    num_cols = [c for c in df.columns if c in NUMERICAL_COLS + ["charges_per_tenure"]]
    scaler = StandardScaler()
    if fit_scaler:
        df[num_cols] = scaler.fit_transform(df[num_cols])
        os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
        joblib.dump(scaler, SCALER_PATH)
        print(f"[data_pipeline] Scaler saved → {SCALER_PATH}")
    else:
        scaler = joblib.load(SCALER_PATH)
        df[num_cols] = scaler.transform(df[num_cols])

    feature_names = list(df.columns)
    print(f"[data_pipeline] Feature matrix shape: {df.shape}")
    return df.values, y.values, feature_names, scaler


def split(X, y, val_size: float = 0.10, test_size: float = 0.15):
    """
    Stratified split: 75% train / 10% val / 15% test.
    Stratification preserves the ~26% churn class ratio.
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=RANDOM_STATE, stratify=y_temp
    )
    print(
        f"[data_pipeline] Split → train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
    )
    churn_rate = y_train.mean()
    print(f"[data_pipeline] Train churn rate: {churn_rate:.2%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def apply_smote(X_train, y_train, k_neighbors: int = 5):
    """Apply SMOTE to training data only. Returns resampled arrays."""
    sm = SMOTE(k_neighbors=k_neighbors, random_state=RANDOM_STATE)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    print(
        f"[data_pipeline] SMOTE applied: {len(y_train)} → {len(y_res)} samples "
        f"(k_neighbors={k_neighbors})"
    )
    return X_res, y_res


def get_prepared_data(use_smote: bool = True, smote_k: int = 5):
    """Convenience wrapper: full pipeline end-to-end."""
    df = load_raw()
    df = clean(df)
    df = engineer_features(df)
    X, y, feature_names, scaler = encode_and_scale(df, fit_scaler=True)
    X_train, X_val, X_test, y_train, y_val, y_test = split(X, y)

    if use_smote:
        X_train_res, y_train_res = apply_smote(X_train, y_train, k_neighbors=smote_k)
    else:
        X_train_res, y_train_res = X_train, y_train

    return {
        "X_train": X_train_res,
        "y_train": y_train_res,
        "X_train_orig": X_train,
        "y_train_orig": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "X_test": X_test,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
    }


# ── Synthetic data generator (demo fallback) ─────────────────────────────────
def _generate_synthetic(n: int = 7043, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    contracts = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.25, 0.20])
    tenure = rng.integers(0, 73, n)
    monthly = rng.uniform(18, 119, n).round(2)
    total = (monthly * (tenure + 1) * rng.uniform(0.9, 1.1, n)).round(2)

    # Churn more likely for month-to-month + high charges + low tenure
    churn_prob = (
        0.05
        + 0.25 * (contracts == "Month-to-month")
        + 0.15 * (monthly > 70)
        - 0.10 * (tenure > 24)
    )
    churn_prob = np.clip(churn_prob, 0.02, 0.90)
    churn = (rng.random(n) < churn_prob).astype(int)

    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    yes_no = lambda p: rng.choice(["Yes", "No"], n, p=[p, 1 - p])

    df = pd.DataFrame({
        "customerID": [f"C{i:05d}" for i in range(n)],
        "gender": rng.choice(["Male", "Female"], n),
        "SeniorCitizen": rng.choice([0, 1], n, p=[0.84, 0.16]),
        "Partner": yes_no(0.48),
        "Dependents": yes_no(0.30),
        "tenure": tenure,
        "PhoneService": yes_no(0.90),
        "MultipleLines": rng.choice(["Yes", "No", "No phone service"], n, p=[0.42, 0.48, 0.10]),
        "InternetService": internet,
        "OnlineSecurity": rng.choice(["Yes", "No", "No internet service"], n, p=[0.28, 0.50, 0.22]),
        "OnlineBackup": rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "DeviceProtection": rng.choice(["Yes", "No", "No internet service"], n, p=[0.34, 0.44, 0.22]),
        "TechSupport": rng.choice(["Yes", "No", "No internet service"], n, p=[0.29, 0.49, 0.22]),
        "StreamingTV": rng.choice(["Yes", "No", "No internet service"], n, p=[0.38, 0.40, 0.22]),
        "StreamingMovies": rng.choice(["Yes", "No", "No internet service"], n, p=[0.39, 0.39, 0.22]),
        "Contract": contracts,
        "PaperlessBilling": yes_no(0.59),
        "PaymentMethod": rng.choice(
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
            n, p=[0.34, 0.23, 0.22, 0.21],
        ),
        "MonthlyCharges": monthly,
        "TotalCharges": total.astype(str),
        "Churn": np.where(churn == 1, "Yes", "No"),
    })
    return df


if __name__ == "__main__":
    data = get_prepared_data(use_smote=True)
    print("Pipeline complete. Feature count:", len(data["feature_names"]))
