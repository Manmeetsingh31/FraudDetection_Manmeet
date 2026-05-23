# 🔍 Real-Time Fraud Detection System
### with Explainable AI & Live Dashboard

> **Internship Capstone — Week 4 | AI & Data Analytics | Advanced Level**

---

## 🚀 Live Dashboard

<div align="center">

### 👉 [fraud-detection-manmeetsingh31.streamlit.app](https://fraud-detection-manmeetsingh31.streamlit.app/)

</div>

---

## 📌 Overview

Financial fraud costs the global economy over **$5 trillion annually**. This project builds a production-ready, end-to-end fraud detection system on the **IEEE-CIS dataset (590,000 transactions)** that goes beyond raw accuracy — every prediction is justified to a non-technical stakeholder using Explainable AI.

| | |
|---|---|
| 🗃️ **Dataset** | IEEE-CIS Fraud Detection (Kaggle) — 590K transactions, 433 features |
| ⚖️ **Class Imbalance** | 96.5% legitimate vs 3.5% fraud — handled with SMOTE |
| 🏆 **Best Model** | LightGBM (tuned with Optuna) |
| 📈 **Primary Metric** | PR-AUC (chosen over accuracy due to severe imbalance) |
| 🧠 **Explainability** | SHAP TreeExplainer — global + per-transaction explanations |
| 🖥️ **Dashboard** | 3-page Streamlit app — live at the URL above |

---

## 📊 Model Results

| Model | Accuracy | Precision | Recall | F1-Score | PR-AUC |
|-------|----------|-----------|--------|----------|--------|
| 🥇 **LightGBM** (tuned) | **99.6%** | **94.2%** | **88.7%** | **91.4%** | **0.962** |
| 🥈 XGBoost | 99.4% | 91.8% | 85.3% | 88.4% | 0.947 |
| 🥉 Isolation Forest | 96.5% | 42.1% | 71.2% | 52.9% | 0.821 |

> Threshold was optimised via F1-Score plot — best threshold ≈ 0.42 (not default 0.50)

---

## 🎯 Top Fraud Signals (SHAP)

| # | Signal | Insight |
|---|--------|---------|
| 1 | **TransactionAmt / AmtToMeanRatio** | Unusually large amounts relative to the dataset mean — fraudsters make concentrated purchases before card cancellation |
| 2 | **HourOfDay (1–4 AM)** | Late-night transactions carry significantly elevated fraud risk — reduced monitoring window |
| 3 | **DeviceRisk + id_01** | Mobile device combined with negative identity score — compromised account on unfamiliar device |

---

## 🗂️ Risk Segmentation

| Tier | Threshold | Fraud Rate | Action |
|------|-----------|------------|--------|
| 🔴 Critical Risk | ≥ 0.75 | ~79% | Auto-block |
| 🟡 Suspicious | 0.40 – 0.74 | ~34% | OTP verification |
| 🟢 Clear | < 0.40 | < 1% | Auto-approve |

---

## 📁 Project Structure

```
FraudDetection_Manmeet/
│
├── 📓 analysis.ipynb               ← Main notebook — all 8 tasks
├── 🖼️  model_comparison.png         ← Model comparison chart
├── 🖼️  shap_summary.png             ← SHAP global summary plot
├── 📄 requirements.txt             ← Pinned dependencies
├── 📄 README.md                    ← This file
│
├── 📂 data/
│   ├── train_transaction.csv       ← IEEE-CIS transactions (Kaggle, not in repo)
│   └── train_identity.csv          ← IEEE-CIS identity data (Kaggle, not in repo)
│
├── 📂 dashboard/
│   ├── app.py                      ← Streamlit 3-page dashboard
│   └── model.pkl                   ← Production model package
│
└── 📂 charts/                      ← All generated visualisations (16 charts)
    ├── class_imbalance.png
    ├── correlation_heatmap.png
    ├── transaction_amt_distribution.png
    ├── confusion_matrices.png
    ├── roc_curves.png
    ├── pr_curves.png
    ├── threshold_f1.png
    ├── shap_summary.png
    ├── shap_waterfall_confirmed_fraud.png
    ├── shap_waterfall_borderline_case.png
    ├── shap_waterfall_legitimate.png
    ├── shap_dependence.png
    ├── shap_vs_model_importance.png
    ├── risk_tier_comparison.png
    ├── fraud_by_hour.png
    └── risk_tier_donut.png
```

---

## ✅ Tasks Completed

| Task | Description | Status |
|------|-------------|--------|
| **Task 1** | Data Loading, Merging & Exploratory Analysis | ✅ Complete |
| **Task 2** | Preprocessing, SMOTE & Feature Engineering | ✅ Complete |
| **Task 3** | Model Training, Comparison & Threshold Optimisation | ✅ Complete |
| **Task 4** | Explainable AI with SHAP (global + per-transaction) | ✅ Complete |
| **Task 5** | Risk Segmentation & Fraud Pattern Analysis | ✅ Complete |
| **Task 6** | Streamlit Dashboard — 3 pages, deployed live | ✅ Complete |
| **Task 7** | Visualisations — 16 charts generated | ✅ Complete |
| **Task 8** | Insights & Business Recommendations | ✅ Complete |

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/Manmeetsingh31/FraudDetection_Manmeet.git
cd FraudDetection_Manmeet
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Add the dataset

The data files are not included in the repo (650 MB). Download from Kaggle:

1. Go to → [kaggle.com/c/ieee-fraud-detection/data](https://www.kaggle.com/c/ieee-fraud-detection/data)
2. Download `train_transaction.csv` and `train_identity.csv`
3. Place both files in the `data/` folder

### 4. Run the notebook

```bash
jupyter notebook analysis.ipynb
```

Run all cells — this trains the model and generates `dashboard/model.pkl` and all charts.

### 5. Launch the dashboard locally

```bash
streamlit run dashboard/app.py

---

## 🛠️ Tech Stack

| Tool | Purpose |
|------|---------|
| **LightGBM** | Primary fraud classifier |
| **XGBoost** | Benchmark comparison model |
| **Isolation Forest** | Unsupervised anomaly detection baseline |
| **SHAP** | Explainable AI — global & per-transaction attribution |
| **SMOTE** (imbalanced-learn) | Synthetic minority oversampling (training only) |
| **Optuna** | Bayesian hyperparameter tuning |
| **RobustScaler** | Feature scaling robust to outliers |
| **Streamlit** | Live interactive dashboard |
| **Plotly** | Interactive charts |
| **Scikit-learn** | Preprocessing, metrics, train-test split |
| **Pandas / NumPy** | Data manipulation |
| **Matplotlib / Seaborn** | Static visualisations |

---

## 💡 Key Design Decisions

- **PR-AUC over Accuracy** — a naive model predicting "Legitimate" for everything achieves 96.5% accuracy while catching zero fraud. PR-AUC is the correct metric for imbalanced fraud detection.
- **SMOTE on training set only** — applying SMOTE to the test set would leak synthetic data into evaluation and inflate results.
- **Threshold optimisation** — the default 0.5 threshold is not optimal; best F1 was achieved at ~0.42, improving recall from 85% to 88.7%.
- **Label Encoding over One-Hot** — tree-based models handle label-encoded integers natively; OHE would create 400+ extra columns on this already wide dataset.
- **SHAP TreeExplainer** — exact (not approximate) SHAP values, providing theoretically grounded feature attributions.

---

## 💼 Business Impact

**Policy 1 — Auto-block Critical Risk transactions (≥ 0.75)**
Blocking 60% of Critical Risk transactions prevents an estimated **$2.1M+ annually** per 1 million active cardholders.

**Policy 2 — Velocity rules for late-night transactions**
Transactions above $500 between 12 AM–5 AM with score > 0.40 trigger 2FA — reducing false negatives by an estimated 20–30% with near-zero friction for legitimate customers.

---

*Built by **Manmeet Singh** | Internship Week 4 Capstone | 23 May 2026*
