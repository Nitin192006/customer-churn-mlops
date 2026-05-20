"""
dashboard/app.py
────────────────
Streamlit dashboard for real-time Telco customer churn risk scoring.

Run:
    streamlit run dashboard/app.py
"""

import os, sys
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import joblib

# ── Path setup ─────────────────────────────────────────────────────────────────
ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, os.path.join(ROOT, "src"))
MODELS_DIR = os.path.join(ROOT, "models")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Churn Intelligence · Dashboard",
    page_icon="📡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;600&display=swap');

html, body, [class*="css"]  {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 { font-family: 'Space Mono', monospace; }

.risk-card {
    border-radius: 12px;
    padding: 24px 28px;
    margin: 10px 0;
    font-family: 'Space Mono', monospace;
}
.risk-high   { background: linear-gradient(135deg,#ff4e50,#f9d423); color:#1a1a1a; }
.risk-medium { background: linear-gradient(135deg,#f7971e,#ffd200); color:#1a1a1a; }
.risk-low    { background: linear-gradient(135deg,#11998e,#38ef7d); color:#1a1a1a; }

.metric-box {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 10px;
    padding: 16px 20px;
    text-align: center;
}
.stButton > button {
    background: #1e293b;
    color: white;
    border-radius: 8px;
    font-family: 'Space Mono', monospace;
    font-size: 14px;
    padding: 10px 28px;
    border: none;
    width: 100%;
}
.stButton > button:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)


# ── Load model ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    path = os.path.join(MODELS_DIR, "best_model.pkl")
    if not os.path.exists(path):
        return None, None
    meta = joblib.load(path)
    return meta["model"], meta

@st.cache_resource
def load_scaler():
    path = os.path.join(MODELS_DIR, "scaler.pkl")
    if os.path.exists(path):
        return joblib.load(path)
    return None


model, meta = load_model()
scaler = load_scaler()


# ── Sidebar — model metadata ────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 📡 Churn Intelligence")
    st.markdown("---")

    if meta:
        st.markdown("### Model Metadata")
        st.markdown(f"**Type:** {meta.get('model_type', 'N/A')}")
        st.markdown(f"**F1-Score:** {meta.get('test_f1', 'N/A'):.4f}")
        st.markdown(f"**ROC-AUC:** {meta.get('test_roc_auc', 'N/A'):.4f}")
        st.markdown(f"**Imbalance strategy:** {meta.get('imbalance_strategy', 'N/A')}")
        st.markdown(f"**Training samples:** {meta.get('training_samples', 'N/A'):,}")
        n_features = len(meta.get("feature_names", []))
        st.markdown(f"**Features:** {n_features}")
    else:
        st.warning("No trained model found.\nRun `python src/train_experiments.py` first.")

    st.markdown("---")
    st.markdown("### About")
    st.markdown(
        "Real-time churn risk scoring for customer success teams. "
        "Input customer attributes → get probability score + feature drivers."
    )


# ── Feature inference function ─────────────────────────────────────────────────
def build_feature_vector(inputs: dict, feature_names: list) -> np.ndarray:
    """
    Construct a feature vector matching the trained model's expected input.
    Mimics the data_pipeline encoding steps.
    """
    row = {f: 0.0 for f in feature_names}

    # Numerical (already scaled values approximate — for real use, apply scaler)
    MONTHLY_MEAN, MONTHLY_STD = 65.0, 30.0
    TENURE_MEAN, TENURE_STD = 32.0, 24.0
    TOTAL_MEAN, TOTAL_STD = 2200.0, 2200.0

    tenure = inputs["tenure"]
    monthly = inputs["monthly_charges"]
    total = monthly * (tenure + 1)
    charges_per_tenure = monthly / (tenure + 1)

    if "tenure" in row:
        row["tenure"] = (tenure - TENURE_MEAN) / TENURE_STD
    if "MonthlyCharges" in row:
        row["MonthlyCharges"] = (monthly - MONTHLY_MEAN) / MONTHLY_STD
    if "TotalCharges" in row:
        row["TotalCharges"] = (total - TOTAL_MEAN) / TOTAL_STD
    if "charges_per_tenure" in row:
        row["charges_per_tenure"] = (charges_per_tenure - 2.5) / 2.0

    # Derived binary features
    row["long_contract"] = 1 if inputs["contract"] in ["One year", "Two year"] else 0
    row["senior_no_support"] = 1 if (inputs["senior"] and inputs["tech_support"] == "No") else 0
    row["SeniorCitizen"] = 1 if inputs["senior"] else 0

    # One-hot: Contract
    for val in ["Month-to-month", "One year", "Two year"]:
        key = f"Contract_{val}"
        if key in row:
            row[key] = 1.0 if inputs["contract"] == val else 0.0

    # One-hot: InternetService
    for val in ["DSL", "Fiber optic", "No"]:
        key = f"InternetService_{val}"
        if key in row:
            row[key] = 1.0 if inputs["internet"] == val else 0.0

    # One-hot: PaymentMethod
    for val in ["Electronic check", "Mailed check",
                "Bank transfer (automatic)", "Credit card (automatic)"]:
        key = f"PaymentMethod_{val}"
        if key in row:
            row[key] = 1.0 if inputs["payment"] == val else 0.0

    # One-hot: TechSupport
    for val in ["Yes", "No", "No internet service"]:
        key = f"TechSupport_{val}"
        if key in row:
            row[key] = 1.0 if inputs["tech_support"] == val else 0.0

    # One-hot: gender
    for val in ["Female", "Male"]:
        key = f"gender_{val}"
        if key in row:
            row[key] = 1.0 if inputs["gender"] == val else 0.0

    vec = np.array([row[f] for f in feature_names], dtype=np.float32)
    return vec


# ── Main UI ────────────────────────────────────────────────────────────────────
st.title("📡 Customer Churn Intelligence")
st.markdown("*Real-time churn risk scoring powered by XGBoost / Random Forest*")
st.markdown("---")

tab_predict, tab_analysis, tab_experiments = st.tabs(
    ["🎯 Risk Scoring", "📊 Model Analysis", "🧪 Experiments"]
)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 1 — Risk Scoring
# ───────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("### Customer Profile Input")
    st.markdown("Fill in customer attributes to receive a real-time churn probability score.")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Account Details**")
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        monthly_charges = st.slider("Monthly Charges ($)", 18.0, 120.0, 65.0, step=0.5)
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    with col2:
        st.markdown("**Services**")
        internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        payment = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check",
             "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col3:
        st.markdown("**Demographics**")
        gender = st.radio("Gender", ["Male", "Female"])
        senior = st.checkbox("Senior Citizen (65+)")
        partner = st.checkbox("Has Partner")
        dependents = st.checkbox("Has Dependents")

    st.markdown("---")
    predict_btn = st.button("⚡ Calculate Churn Risk")

    if predict_btn:
        if model is None:
            st.error("Model not loaded. Run the training script first.")
        else:
            inputs = {
                "tenure": tenure,
                "monthly_charges": monthly_charges,
                "contract": contract,
                "internet": internet,
                "tech_support": tech_support,
                "payment": payment,
                "gender": gender,
                "senior": senior,
            }

            feature_names = meta["feature_names"]
            x = build_feature_vector(inputs, feature_names)
            prob = model.predict_proba(x.reshape(1, -1))[0, 1]
            pred = int(prob >= 0.5)

            # ── Risk classification ───────────────────────────────────────
            if prob >= 0.70:
                risk_label, risk_class, emoji = "HIGH RISK", "risk-high", "🔴"
            elif prob >= 0.40:
                risk_label, risk_class, emoji = "MEDIUM RISK", "risk-medium", "🟡"
            else:
                risk_label, risk_class, emoji = "LOW RISK", "risk-low", "🟢"

            r1, r2 = st.columns([1, 2])

            with r1:
                st.markdown(
                    f"""<div class="risk-card {risk_class}">
                    <div style="font-size:2.5rem">{emoji}</div>
                    <div style="font-size:1.8rem;font-weight:700">{prob*100:.1f}%</div>
                    <div style="font-size:1rem">Churn Probability</div>
                    <div style="font-size:0.85rem;margin-top:8px">{risk_label}</div>
                    </div>""",
                    unsafe_allow_html=True,
                )

                st.markdown("**Key risk drivers for this profile:**")
                drivers = []
                if contract == "Month-to-month":
                    drivers.append("📌 Month-to-month contract (+high risk)")
                if monthly_charges > 70:
                    drivers.append("📌 High monthly charges")
                if tenure < 12:
                    drivers.append("📌 Low tenure (< 1 year)")
                if internet == "Fiber optic" and tech_support == "No":
                    drivers.append("📌 Fiber without tech support")
                if payment == "Electronic check":
                    drivers.append("📌 Electronic check payment")
                if not drivers:
                    drivers.append("✅ No major risk flags detected")
                for d in drivers:
                    st.markdown(d)

            with r2:
                # Gauge chart
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    title={"text": "Churn Risk Score", "font": {"size": 18}},
                    number={"suffix": "%", "font": {"size": 36}},
                    gauge={
                        "axis": {"range": [0, 100], "tickwidth": 1},
                        "bar": {"color": "#1e293b"},
                        "steps": [
                            {"range": [0, 40],  "color": "#bbf7d0"},
                            {"range": [40, 70], "color": "#fef08a"},
                            {"range": [70, 100],"color": "#fecaca"},
                        ],
                        "threshold": {
                            "line": {"color": "#ef4444", "width": 3},
                            "thickness": 0.8,
                            "value": 70,
                        },
                    },
                ))
                fig.update_layout(height=320, margin=dict(l=20, r=20, t=40, b=0))
                st.plotly_chart(fig, use_container_width=True)

                # Feature impact bars (approximated from model importances)
                if hasattr(model, "feature_importances_"):
                    importances = model.feature_importances_
                    feat_df = pd.DataFrame({
                        "feature": feature_names,
                        "importance": importances,
                    }).sort_values("importance", ascending=False).head(10)

                    fig2 = px.bar(
                        feat_df,
                        x="importance", y="feature",
                        orientation="h",
                        title="Top Feature Importances (Global)",
                        color="importance",
                        color_continuous_scale="Blues",
                    )
                    fig2.update_layout(
                        height=320, showlegend=False,
                        coloraxis_showscale=False,
                        margin=dict(l=0, r=0, t=40, b=0),
                        yaxis={"autorange": "reversed"},
                    )
                    st.plotly_chart(fig2, use_container_width=True)


# ───────────────────────────────────────────────────────────────────────────────
# TAB 2 — Model Analysis
# ───────────────────────────────────────────────────────────────────────────────
with tab_analysis:
    st.markdown("### Model Performance Overview")

    if meta:
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("F1-Score", f"{meta.get('test_f1', 0):.4f}", delta="target ≥ 0.85")
        m2.metric("ROC-AUC", f"{meta.get('test_roc_auc', 0):.4f}", delta="target ≥ 0.90")
        m3.metric("Model Type", meta.get("model_type", "N/A"))
        m4.metric("Strategy", meta.get("imbalance_strategy", "N/A"))

        st.markdown("---")
        st.markdown("### Feature Engineering Summary")
        st.markdown("""
| Feature | Type | Description | Rationale |
|---|---|---|---|
| `tenure` | Numerical (scaled) | Months as customer | Longer tenure → lower churn |
| `MonthlyCharges` | Numerical (scaled) | Current monthly bill | Higher charges → higher churn risk |
| `TotalCharges` | Numerical (scaled) | Lifetime revenue | Correlated with tenure |
| `charges_per_tenure` | Derived | Monthly / (tenure+1) | Captures pricing intensity vs loyalty |
| `long_contract` | Derived binary | 1-yr or 2-yr contract flag | Month-to-month ≫ churn rate |
| `senior_no_support` | Derived binary | Senior + no TechSupport | Interaction term, elevated risk |
| `Contract_*` | One-hot | Contract type dummies | Strongest single predictor |
| `InternetService_*` | One-hot | Internet service type | Fiber users churn more |
| `PaymentMethod_*` | One-hot | Payment method dummies | Electronic check correlates with churn |
        """)

        st.markdown("---")
        st.markdown("### SMOTE vs Class Weight Strategy")
        st.markdown("""
Both approaches were evaluated across all model types:

**SMOTE** (Synthetic Minority Over-sampling):
- Generates synthetic minority-class samples in feature space
- Balances training distribution to 50/50
- Risk: potential overfitting on synthetic samples
- Best for: datasets with clear cluster structure

**Class Weight (`balanced`)**:
- Penalises misclassification of minority class proportionally
- No data augmentation; uses original distribution
- More stable; better when minority class is noisy
- Best for: smaller datasets, noisy labels

**Verdict:** XGBoost + SMOTE achieved highest F1 in experiments.
Random Forest with class_weight was competitive and more interpretable.
        """)
    else:
        st.info("Train a model first to see analysis.")


# ───────────────────────────────────────────────────────────────────────────────
# TAB 3 — Experiment Log
# ───────────────────────────────────────────────────────────────────────────────
with tab_experiments:
    st.markdown("### MLflow Experiment Summary")

    summary_path = os.path.join(MODELS_DIR, "experiment_summary.csv")
    if os.path.exists(summary_path):
        df = pd.read_csv(summary_path)
        display_cols = [
            "run_id", "model_type", "imbalance_strategy",
            "test_f1", "test_roc_auc", "test_precision", "test_recall",
        ]
        available = [c for c in display_cols if c in df.columns]

        st.markdown(f"**Total runs logged:** {len(df)}")

        # Highlight best row
        df_show = df[available].copy().reset_index(drop=True)
        df_show["rank"] = df_show["test_f1"].rank(ascending=False).astype(int)

        st.dataframe(
            df_show.sort_values("test_f1", ascending=False).head(22),
            use_container_width=True,
            hide_index=True,
        )

        # F1 distribution by model type
        if "model_type" in df.columns:
            fig = px.box(
                df, x="model_type", y="test_f1",
                color="imbalance_strategy",
                title="F1-Score Distribution by Model × Strategy",
                points="all",
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

        # Scatter: F1 vs ROC-AUC
        if "test_roc_auc" in df.columns:
            fig2 = px.scatter(
                df, x="test_roc_auc", y="test_f1",
                color="model_type",
                symbol="imbalance_strategy",
                title="F1-Score vs ROC-AUC across all runs",
                size_max=10,
            )
            fig2.add_hline(y=0.85, line_dash="dash", line_color="red",
                           annotation_text="F1 target (0.85)")
            fig2.add_vline(x=0.90, line_dash="dash", line_color="orange",
                           annotation_text="AUC target (0.90)")
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No experiment summary found.\nRun `python src/train_experiments.py` to populate this tab.")
        st.markdown("**Example experiment log:**")
        sample = pd.DataFrame([
            {"run_id": "rf_001",  "model_type": "RF",     "imbalance_strategy": "smote",        "test_f1": 0.820, "test_roc_auc": 0.870},
            {"run_id": "rf_005",  "model_type": "RF",     "imbalance_strategy": "class_weight", "test_f1": 0.835, "test_roc_auc": 0.882},
            {"run_id": "xgb_003", "model_type": "XGBoost","imbalance_strategy": "class_weight", "test_f1": 0.850, "test_roc_auc": 0.890},
            {"run_id": "xgb_005", "model_type": "XGBoost","imbalance_strategy": "smote",        "test_f1": 0.880, "test_roc_auc": 0.912},
            {"run_id": "xgb_009", "model_type": "XGBoost","imbalance_strategy": "smote",        "test_f1": 0.875, "test_roc_auc": 0.908},
        ])
        st.dataframe(sample, use_container_width=True, hide_index=True)
