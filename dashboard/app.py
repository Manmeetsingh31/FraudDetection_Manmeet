"""
Real-Time Fraud Detection Dashboard
====================================
Multi-page Streamlit application for the Fraud Detection capstone project.

Pages:
    1. Overview        — KPIs and summary statistics
    2. Transaction Explorer — searchable/filterable table with live risk scores
    3. SHAP Explainer  — per-transaction waterfall plot + plain-English explanation

Run locally:
    cd FraudDetection
    streamlit run dashboard/app.py

Deploy on Streamlit Community Cloud:
    1. Push entire FraudDetection/ folder to a GitHub repo
    2. Go to share.streamlit.io → New app → connect repo
    3. Set main file path to: dashboard/app.py
"""

import os
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore")

# ── Path setup — works whether run from project root or dashboard/ ─────────────
HERE      = os.path.dirname(os.path.abspath(__file__))
ROOT      = os.path.dirname(HERE)
MODEL_PKG = os.path.join(HERE, "model.pkl")
DATA_TX   = os.path.join(ROOT, "data", "train_transaction.csv")
DATA_ID   = os.path.join(ROOT, "data", "train_identity.csv")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Fraud Detection Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #1E1E2E;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        border-left: 4px solid #5C6BC0;
    }
    .metric-value { font-size: 2.2rem; font-weight: 700; color: #E8EAF6; }
    .metric-label { font-size: 0.85rem; color: #9E9E9E; margin-top: 4px; }
    .risk-high   { color: #EF5350; font-weight: 700; }
    .risk-mid    { color: #FFA726; font-weight: 700; }
    .risk-low    { color: #66BB6A; font-weight: 700; }
    .explain-box {
        background: #F3F4F6;
        border-radius: 8px;
        padding: 14px 18px;
        border-left: 4px solid #5C6BC0;
        margin-bottom: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# Data & model loading — cached so they only run once
# ══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner="Loading data…")
def load_data():
    """Merge transaction + identity tables and compute basic features."""
    tx  = pd.read_csv(DATA_TX)
    idf = pd.read_csv(DATA_ID)
    df  = tx.merge(idf, on="TransactionID", how="left")

    # Engineered features (must match notebook)
    mean_amt = df["TransactionAmt"].mean()
    df["AmtToMeanRatio"] = (df["TransactionAmt"] / mean_amt).round(4)
    df["HourOfDay"]      = ((df["TransactionDT"] % 86_400) // 3600).astype(int)

    if "DeviceType" in df.columns and "id_01" in df.columns:
        df["DeviceRisk"] = (
            (df["DeviceType"] == "mobile") &
            (df["id_01"].fillna(0) < 0)
        ).astype(int)
    else:
        df["DeviceRisk"] = 0

    return df


@st.cache_resource(show_spinner="Loading model…")
def load_model():
    """Load pickled model package."""
    if not os.path.exists(MODEL_PKG):
        return None
    with open(MODEL_PKG, "rb") as f:
        return pickle.load(f)


def preprocess_row(row: pd.Series, pkg: dict) -> pd.DataFrame:
    """
    Apply the same preprocessing pipeline used during training so the
    model receives features in the exact format it was trained on.
    """
    features   = pkg["features"]
    scale_cols = pkg["scale_cols"]
    le_encoders= pkg["le_encoders"]
    scaler     = pkg["scaler"]

    x = row.reindex(features)

    # Impute numerics with 0, categoricals with 'Unknown'
    for col in features:
        if pd.isna(x[col]):
            if col in le_encoders:
                x[col] = 0
            else:
                x[col] = 0.0

    # Label-encode categoricals
    for col in le_encoders:
        if col in x.index:
            try:
                x[col] = le_encoders[col].transform([str(x[col])])[0]
            except ValueError:
                x[col] = 0

    df_row = pd.DataFrame([x.values], columns=features)

    # Scale
    valid_scale = [c for c in scale_cols if c in df_row.columns]
    if valid_scale:
        df_row[valid_scale] = scaler.transform(df_row[valid_scale])

    return df_row


def risk_tier(p: float) -> str:
    if p >= 0.75:   return "🔴 Critical Risk"
    elif p >= 0.40: return "🟡 Suspicious"
    else:           return "🟢 Clear"


def tier_color(tier: str) -> str:
    return {"🔴 Critical Risk": "#EF5350",
            "🟡 Suspicious":    "#FFA726",
            "🟢 Clear":         "#66BB6A"}.get(tier, "#9E9E9E")


# ══════════════════════════════════════════════════════════════════════════════
# Load everything
# ══════════════════════════════════════════════════════════════════════════════
df  = load_data()
pkg = load_model()

model_ready = pkg is not None

# Score all transactions if model is available
if model_ready:
    @st.cache_data(show_spinner="Scoring transactions…")
    def score_all(df):
        features   = pkg["features"]
        scale_cols = pkg["scale_cols"]
        le_encoders= pkg["le_encoders"]
        scaler     = pkg["scaler"]
        model      = pkg["model"]
        threshold  = pkg["threshold"]

        # Drop missing > 50% cols
        missing = df.isnull().mean()
        drop    = missing[missing > 0.5].index.tolist()
        drop    = [c for c in drop if c not in ["isFraud", "TransactionID"]]
        scored  = df.drop(columns=drop, errors="ignore").copy()

        for col in le_encoders:
            if col in scored.columns:
                scored[col] = scored[col].fillna("Unknown").astype(str)
                known = set(le_encoders[col].classes_)
                scored[col] = scored[col].apply(lambda v: v if v in known else le_encoders[col].classes_[0])
                scored[col] = le_encoders[col].transform(scored[col])

        num_cols = scored.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            scored[col].fillna(scored[col].median(), inplace=True)

        X = scored.reindex(columns=features, fill_value=0)
        valid_scale = [c for c in scale_cols if c in X.columns]
        if valid_scale:
            X[valid_scale] = scaler.transform(X[valid_scale])

        proba = model.predict_proba(X)[:, 1]
        return proba

    df["FraudProba"] = score_all(df)
    df["RiskTier"]   = df["FraudProba"].apply(risk_tier)
    df["FraudPred"]  = (df["FraudProba"] >= pkg["threshold"]).astype(int)


# ══════════════════════════════════════════════════════════════════════════════
# Sidebar navigation
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/fraud.png", width=64)
    st.title("Fraud Detection")
    st.caption("Capstone — Week 4")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["📊 Overview", "🔎 Transaction Explorer", "🧠 SHAP Explainer"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("**Filters**")

    if model_ready and "RiskTier" in df.columns:
        tier_filter = st.multiselect(
            "Risk Tier",
            options=["🔴 Critical Risk", "🟡 Suspicious", "🟢 Clear"],
            default=["🔴 Critical Risk", "🟡 Suspicious", "🟢 Clear"],
        )
        min_amt, max_amt = float(df["TransactionAmt"].min()), float(df["TransactionAmt"].quantile(0.99))
        amt_range = st.slider("Transaction Amount ($)", min_amt, max_amt, (min_amt, max_amt))
        hour_range = st.slider("Hour of Day", 0, 23, (0, 23))

        view_df = df[
            df["RiskTier"].isin(tier_filter) &
            df["TransactionAmt"].between(*amt_range) &
            df["HourOfDay"].between(*hour_range)
        ]
    else:
        view_df = df

    st.markdown("---")
    st.caption(f"Dataset: {len(df):,} transactions")
    if model_ready:
        st.caption(f"Threshold: {pkg['threshold']:.2f}")


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 1 — Overview
# ══════════════════════════════════════════════════════════════════════════════
if page == "📊 Overview":
    st.title("📊 Fraud Operations Overview")
    st.markdown("Real-time summary of transaction health across the portfolio.")
    st.markdown("---")

    total_tx   = len(df)
    fraud_col  = "FraudPred" if model_ready else "isFraud"
    total_fraud= int(df[fraud_col].sum()) if fraud_col in df.columns else int(df["isFraud"].sum())
    det_rate   = total_fraud / total_tx * 100
    avg_fraud_amt = df[df[fraud_col] == 1]["TransactionAmt"].mean() if fraud_col in df.columns else 0.0

    c1, c2, c3, c4 = st.columns(4)
    for col, val, label in [
        (c1, f"{total_tx:,}",        "Total Transactions"),
        (c2, f"{total_fraud:,}",     "Flagged as Fraud"),
        (c3, f"{det_rate:.2f}%",     "Detection Rate"),
        (c4, f"${avg_fraud_amt:.2f}","Avg Fraud Amount"),
    ]:
        col.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{val}</div>
            <div class="metric-label">{label}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Risk Tier Breakdown")
        if model_ready and "RiskTier" in df.columns:
            tier_counts = df["RiskTier"].value_counts().reindex(
                ["🔴 Critical Risk", "🟡 Suspicious", "🟢 Clear"]
            ).fillna(0)
            fig = go.Figure(go.Pie(
                labels=[t.split(" ", 1)[1] for t in tier_counts.index],
                values=tier_counts.values,
                hole=0.55,
                marker_colors=["#EF5350", "#FFA726", "#66BB6A"],
                textinfo="percent+label",
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>"
            ))
            fig.update_layout(
                showlegend=False,
                margin=dict(t=10, b=10, l=10, r=10),
                height=300,
                paper_bgcolor="rgba(0,0,0,0)"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Score transactions to see tier breakdown.")

    with col_right:
        st.subheader("Fraud Rate by Hour of Day")
        hourly = df.groupby("HourOfDay")["isFraud"].mean() * 100
        fig_h  = px.bar(
            x=hourly.index, y=hourly.values,
            labels={"x": "Hour of Day", "y": "Fraud Rate (%)"},
            color=hourly.values,
            color_continuous_scale="RdYlGn_r",
        )
        fig_h.update_layout(
            coloraxis_showscale=False,
            margin=dict(t=10, b=10, l=10, r=10),
            height=300,
            paper_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_h, use_container_width=True)

    st.subheader("Transaction Amount Distribution")
    fig_amt = px.histogram(
        df.sample(min(5000, len(df)), random_state=42),
        x="TransactionAmt",
        color="isFraud",
        barmode="overlay",
        nbins=60,
        log_x=True,
        opacity=0.65,
        color_discrete_map={0: "#2196F3", 1: "#F44336"},
        labels={"isFraud": "Fraud", "TransactionAmt": "Transaction Amount ($, log scale)"}
    )
    fig_amt.update_layout(margin=dict(t=10, b=10), height=300,
                          paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_amt, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 2 — Transaction Explorer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔎 Transaction Explorer":
    st.title("🔎 Transaction Explorer")
    st.markdown("Browse, filter, and look up live risk scores by TransactionID.")
    st.markdown("---")

    search_id = st.text_input("🔍 Search by TransactionID", placeholder="e.g. 2987001")

    if search_id.strip():
        try:
            tid = int(search_id.strip())
            row = df[df["TransactionID"] == tid]
            if len(row) == 0:
                st.warning(f"TransactionID {tid} not found.")
            else:
                r = row.iloc[0]
                cols = st.columns(4)
                cols[0].metric("TransactionID",  str(int(r["TransactionID"])))
                cols[1].metric("Amount",         f"${r['TransactionAmt']:.2f}")
                cols[2].metric("Hour of Day",    str(int(r.get("HourOfDay", 0))))
                cols[3].metric("True Label",     "Fraud" if r["isFraud"] == 1 else "Legitimate")

                if model_ready and "FraudProba" in df.columns:
                    proba = float(row["FraudProba"].values[0])
                    tier  = risk_tier(proba)
                    color = tier_color(tier)
                    st.markdown(
                        f'<h3 style="color:{color}">Risk Score: {proba:.3f} — {tier}</h3>',
                        unsafe_allow_html=True
                    )
                    st.progress(proba)
        except ValueError:
            st.error("Please enter a valid numeric TransactionID.")

    st.markdown("---")
    st.subheader(f"Showing {min(len(view_df), 1000):,} of {len(view_df):,} filtered transactions")

    display_cols = ["TransactionID", "TransactionAmt", "HourOfDay", "isFraud"]
    if model_ready and "FraudProba" in df.columns:
        display_cols += ["FraudProba", "RiskTier"]

    show_df = view_df[display_cols].head(1000).copy()
    if "FraudProba" in show_df.columns:
        show_df["FraudProba"] = show_df["FraudProba"].round(4)

    st.dataframe(show_df, use_container_width=True, height=420)

    if model_ready and "FraudProba" in df.columns:
        st.subheader("Interactive Risk Scatter")
        sample = view_df.sample(min(3000, len(view_df)), random_state=42)
        fig_sc = px.scatter(
            sample,
            x="HourOfDay",
            y="TransactionAmt",
            color="FraudProba",
            color_continuous_scale="RdYlGn_r",
            opacity=0.5,
            hover_data=["TransactionID", "isFraud", "RiskTier"],
            labels={"TransactionAmt": "Amount ($)", "HourOfDay": "Hour"},
        )
        fig_sc.update_layout(height=400, paper_bgcolor="rgba(0,0,0,0)",
                             margin=dict(t=10, b=10))
        st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PAGE 3 — SHAP Explainer
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠 SHAP Explainer":
    st.title("🧠 SHAP Transaction Explainer")
    st.markdown(
        "Enter a TransactionID to see *exactly* why the model flagged or cleared it — "
        "feature-by-feature, in plain English."
    )
    st.markdown("---")

    if not model_ready:
        st.error("Model not loaded. Run the notebook first to generate dashboard/model.pkl")
        st.stop()

    # Try to import SHAP — it may not be installed in the deployment env
    try:
        import shap
        import matplotlib.pyplot as plt
        shap_available = True
    except ImportError:
        shap_available = False
        st.warning("SHAP not installed in this environment. Showing feature contributions instead.")

    tx_input = st.text_input("TransactionID", placeholder="e.g. 2987001")

    if st.button("Explain Transaction") and tx_input.strip():
        try:
            tid = int(tx_input.strip())
            row = df[df["TransactionID"] == tid]

            if len(row) == 0:
                st.error(f"TransactionID {tid} not found.")
                st.stop()

            r     = row.iloc[0]
            x_row = preprocess_row(r, pkg)
            proba = float(pkg["model"].predict_proba(x_row)[:, 1][0])
            tier  = risk_tier(proba)
            color = tier_color(tier)

            # ── Summary banner ──────────────────────────────────────────────
            col1, col2, col3 = st.columns(3)
            col1.metric("TransactionID",  str(tid))
            col2.metric("Amount",         f"${r['TransactionAmt']:.2f}")
            col3.metric("True Label",     "Fraud" if r["isFraud"] == 1 else "Legitimate")

            st.markdown(
                f'<h2 style="color:{color}">Fraud Probability: {proba:.3f}  —  {tier}</h2>',
                unsafe_allow_html=True
            )
            st.progress(proba)
            st.markdown("---")

            # ── SHAP waterfall ──────────────────────────────────────────────
            if shap_available:
                with st.spinner("Computing SHAP values…"):
                    explainer   = shap.TreeExplainer(pkg["model"])
                    shap_values = explainer(x_row)

                    if hasattr(shap_values, "values") and isinstance(shap_values.values, list):
                        import shap as shap_lib
                        shap_exp = shap_lib.Explanation(
                            values      = shap_values.values[1],
                            base_values = shap_values.base_values[1]
                                          if hasattr(shap_values.base_values, "__len__")
                                          else shap_values.base_values,
                            data        = shap_values.data,
                            feature_names = shap_values.feature_names
                        )
                    else:
                        shap_exp = shap_values

                    fig, ax = plt.subplots(figsize=(10, 6))
                    shap.plots.waterfall(shap_exp[0], max_display=15, show=False)
                    plt.title(f"SHAP Waterfall — TransactionID {tid}", fontsize=12)
                    plt.tight_layout()
                    st.pyplot(fig)
                    plt.close(fig)

                # ── Plain-English explanation ────────────────────────────────
                st.subheader("Plain-English Explanation")

                sv   = shap_exp.values[0] if hasattr(shap_exp.values[0], "__len__") else shap_exp.values
                feats= pkg["features"]
                vals = x_row.iloc[0].values

                feat_df = pd.DataFrame({
                    "Feature": feats,
                    "SHAP":    sv,
                    "Value":   vals
                }).sort_values("SHAP", key=abs, ascending=False)

                top_pos = feat_df[feat_df["SHAP"] > 0].head(4)
                top_neg = feat_df[feat_df["SHAP"] < 0].head(4)

                if len(top_pos) > 0:
                    st.markdown("**🔺 Factors increasing fraud risk:**")
                    for _, frow in top_pos.iterrows():
                        st.markdown(
                            f'<div class="explain-box">📌 <b>{frow["Feature"]}</b> = '
                            f'{frow["Value"]:.3f}  →  pushes fraud probability '
                            f'<span class="risk-high">+{frow["SHAP"]:.4f}</span></div>',
                            unsafe_allow_html=True
                        )

                if len(top_neg) > 0:
                    st.markdown("**🔻 Factors decreasing fraud risk:**")
                    for _, frow in top_neg.iterrows():
                        st.markdown(
                            f'<div class="explain-box">📌 <b>{frow["Feature"]}</b> = '
                            f'{frow["Value"]:.3f}  →  lowers fraud probability '
                            f'<span class="risk-low">{frow["SHAP"]:.4f}</span></div>',
                            unsafe_allow_html=True
                        )

            else:
                # Fallback: show LightGBM feature importance as a proxy
                st.info("Showing model feature importance as a SHAP proxy.")
                fi = pd.Series(
                    pkg["model"].feature_importances_,
                    index=pkg["features"]
                ).sort_values(ascending=False).head(15)
                st.bar_chart(fi)

        except ValueError:
            st.error("Please enter a valid numeric TransactionID.")
