# 🎯 B2B SaaS Customer Churn Prediction & Retention Cockpit

An end-to-end Machine Learning solution to detect and prevent customer churn in enterprise B2B SaaS subscriptions.

---

## 📌 Project Overview
Customer retention is the primary growth engine for SaaS companies. Losing a customer damages Monthly Recurring Revenue (MRR) and increases Customer Acquisition Costs (CAC).

This project develops an industry-grade predictive model that:
1. Identifies accounts at high risk of canceling their subscriptions (`Is_Churned = 1`).
2. Corrects severe class imbalance (~1.8% positive churn rate) using stratified sampling and cost-sensitive learning.
3. Detects and resolves target data leakage (`Days_Inactive`), ensuring authentic out-of-sample generalization.
4. Packages a unified `scikit-learn` pipeline with a real-time **Streamlit** dashboard.

---

## 📊 Dataset & Feature Architecture
* **Total Records:** 50,000 SaaS Accounts
* **Target Label:** `Is_Churned` (`0 = Retained (98.17%)`, `1 = Churned (1.83%)`)

### Feature Breakdown:
* **Account Identity:** `Industry`, `Country`, `Company_Size`, `Account_Manager`
* **Financial & Subscription:** `Plan_Tier`, `Billing_Cycle`, `Payment_Status`, `Monthly_Recurring_Revenue`
* **Behavioral & Utilization Signals:** 
  * `Tenure_Days`: Account lifetime in days.
  * `Feature_Adoption_Rate`: Percentage depth of platform usage.
  * `Seat_Utilization_Rate`: Ratio of active users to total purchased seats.
  * `Support_Tickets_30d`: Customer friction and issue frequency in the last 30 days.
  * `Churn_Risk_Score`: Internal heuristic risk tier (`Low`, `Medium`, `High`).

---

## 🔍 Key Data Science Decisions

### 1. Diagnosing & Eliminating Data Leakage
During exploratory modeling, baseline algorithms achieved an artificial **1.0000 ROC-AUC**. Investigation revealed that `Days_Inactive` had absolute separation:
* Active accounts had `Days_Inactive <= 13 days`.
* Churned accounts had `Days_Inactive >= 30 days`.

Because churned accounts naturally stop logging in after cancellation, using `Days_Inactive` represents a **target leak**. Removing this variable forced models to learn genuine causal signals (payment issues, low seat usage, ticket spikes), yielding a realistic **~0.865 ROC-AUC**.

### 2. Overcoming Severe Class Imbalance
* Standard accuracy is misleading on a 98:2 imbalanced dataset.
* Evaluated algorithms on **Recall (Churners)**, **ROC-AUC**, and **F1-Score**.
* Implemented `class_weight='balanced'` to penalize false negatives (missed churners).

### 3. Model Benchmark & Tuning Results (Stratified 5-Fold CV)
| Model | Mean CV ROC-AUC | Test Churn Recall | Notes |
| :--- | :---: | :---: | :--- |
| **Logistic Regression (Balanced)** | **0.8654** | **79%** | **Champion:** Catches 79% of all churners |
| **Gradient Boosting** | 0.8621 | 7% | High accuracy, fails to catch churners |
| **Random Forest (Balanced)** | 0.8227 | 14% | Biased toward majority class |

* **Hyperparameter Tuning:** Tuned with `GridSearchCV` (`C=0.01`, `solver='lbfgs'`).

---

## 🚀 Running the Streamlit App

### 1. Install Dependencies
```bash
pip install streamlit scikit-learn pandas numpy joblib matplotlib seaborn
```

### 2. Launch Application
```bash
streamlit run app.py
```

### 3. Features:
* **Interactive Risk Sliders & Selectors** for all account features.
* **Tunable Decision Threshold** to adjust conservatism for customer success intervention.
* **One-Click Demo Presets:** Test healthy vs. high-risk accounts instantly.
* **Automated Actionable Recommendations:** Generates targeted customer success guidance based on detected friction points.

---

## 📁 Repository Structure
```
├── model_selection.ipynb         # Full EDA, cleaning, pipeline & GridSearchCV tuning
├── app.py                        # Streamlit Web Dashboard
├── churn_pipeline.joblib         # Serialized champion scikit-learn pipeline
├── app_metadata.joblib           # Feature schema and options for input forms
├── MP_Synthetics_50000_Rows.csv  # Dataset (50,000 records)
└── README.md                     # Comprehensive project documentation
```
