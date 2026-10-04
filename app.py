import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Customer Churn Prediction Cockpit",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- LOAD MODEL AND METADATA ---
@st.cache_resource
def load_assets():
    model_path = "churn_pipeline.joblib"
    meta_path = "app_metadata.joblib"
    
    if not os.path.exists(model_path) or not os.path.exists(meta_path):
        st.error("Model assets not found. Please ensure 'churn_pipeline.joblib' and 'app_metadata.joblib' are in the project folder.")
        st.stop()
        
    pipeline = joblib.load(model_path)
    metadata = joblib.load(meta_path)
    return pipeline, metadata

pipeline, metadata = load_assets()
categorical_options = metadata['categorical_options']

# --- SIDEBAR: MODEL & DECISION CONTROLS ---
with st.sidebar:
    st.title("⚙️ Model Controls")
    st.markdown("""
    **Model Architecture:**  
    `Logistic Regression (Balanced Weights)`  
    * **CV ROC-AUC:** `0.8654`  
    * **Test Churn Recall:** `79%`
    """)
    st.divider()

    st.subheader("Decision Sensitivity")
    threshold = st.slider(
        "Risk Alert Threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.50,
        step=0.05,
        help="Probability threshold above which an account is flagged as Churn Risk. Lower values catch more churners earlier."
    )
    
    st.divider()
    st.subheader("⚡ Quick Test Presets")
    preset = st.radio(
        "Load demo scenario:",
        options=["Custom Input", "Healthy Account (Low Risk)", "Friction Account (High Risk)"],
        index=0
    )

# Preset defaults
defaults = {
    "Industry": "FinTech",
    "Country": "USA",
    "Company_Size": "51-200",
    "Account_Manager": "Sarah Jenkins",
    "Plan_Tier": "Pro",
    "Billing_Cycle": "Annual",
    "Payment_Status": "Active",
    "Churn_Risk_Score": "Low",
    "MRR": 499,
    "Active_Seats": 25,
    "Total_Seats": 30,
    "Tenure_Days": 450,
    "Feature_Adoption": 85.0,
    "Support_Tickets": 1
}

if preset == "Healthy Account (Low Risk)":
    defaults.update({
        "Payment_Status": "Active",
        "Churn_Risk_Score": "Low",
        "Active_Seats": 28,
        "Total_Seats": 30,
        "Feature_Adoption": 92.0,
        "Support_Tickets": 0
    })
elif preset == "Friction Account (High Risk)":
    defaults.update({
        "Payment_Status": "Failed",
        "Churn_Risk_Score": "High",
        "Active_Seats": 2,
        "Total_Seats": 20,
        "Feature_Adoption": 32.0,
        "Support_Tickets": 6
    })

# --- MAIN HEADER ---
st.title("🎯 SaaS Customer Churn Prediction Cockpit")
st.markdown("""
Predict customer retention risk for B2B SaaS accounts in real time using your trained scikit-learn machine learning pipeline.
""")

st.markdown("---")

# --- FORM: INPUT PARAMETERS ---
with st.form("churn_input_form"):
    st.subheader("1. Company & Subscription Details")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        industry = st.selectbox(
            "Industry", 
            options=categorical_options['Industry'], 
            index=categorical_options['Industry'].index(defaults["Industry"])
        )
        plan_tier = st.selectbox(
            "Plan Tier", 
            options=categorical_options['Plan_Tier'], 
            index=categorical_options['Plan_Tier'].index(defaults["Plan_Tier"])
        )

    with col2:
        country = st.selectbox(
            "Country", 
            options=categorical_options['Country'], 
            index=categorical_options['Country'].index(defaults["Country"])
        )
        billing_cycle = st.selectbox(
            "Billing Cycle", 
            options=categorical_options['Billing_Cycle'], 
            index=categorical_options['Billing_Cycle'].index(defaults["Billing_Cycle"])
        )

    with col3:
        company_size = st.selectbox(
            "Company Size", 
            options=categorical_options['Company_Size'], 
            index=categorical_options['Company_Size'].index(defaults["Company_Size"])
        )
        payment_status = st.selectbox(
            "Payment Status", 
            options=categorical_options['Payment_Status'], 
            index=categorical_options['Payment_Status'].index(defaults["Payment_Status"])
        )

    with col4:
        account_manager = st.selectbox(
            "Account Manager", 
            options=categorical_options['Account_Manager'], 
            index=categorical_options['Account_Manager'].index(defaults["Account_Manager"])
        )
        churn_risk_score = st.selectbox(
            "Internal Risk Tier", 
            options=categorical_options['Churn_Risk_Score'], 
            index=categorical_options['Churn_Risk_Score'].index(defaults["Churn_Risk_Score"])
        )

    st.subheader("2. Platform Usage & Engagement Metrics")
    col5, col6, col7, col8 = st.columns(4)

    with col5:
        mrr = st.number_input(
            "Monthly Recurring Revenue ($)", 
            min_value=10, 
            max_value=10000, 
            value=defaults["MRR"], 
            step=10
        )
        tenure_days = st.number_input(
            "Account Tenure (Days)", 
            min_value=1, 
            max_value=3000, 
            value=defaults["Tenure_Days"], 
            step=30
        )

    with col6:
        active_seats = st.number_input(
            "Active Seats Used", 
            min_value=1, 
            max_value=500, 
            value=defaults["Active_Seats"], 
            step=1
        )
        total_seats = st.number_input(
            "Total Seats Licensed", 
            min_value=1, 
            max_value=500, 
            value=defaults["Total_Seats"], 
            step=1
        )

    with col7:
        feature_adoption = st.slider(
            "Feature Adoption Rate (%)", 
            min_value=0.0, 
            max_value=100.0, 
            value=float(defaults["Feature_Adoption"]), 
            step=1.0
        )

    with col8:
        support_tickets = st.number_input(
            "Support Tickets (Last 30 Days)", 
            min_value=0, 
            max_value=30, 
            value=defaults["Support_Tickets"], 
            step=1
        )

    # Derived domain metric
    seat_utilization = min(1.0, active_seats / max(1, total_seats))
    st.caption(f"Calculated Seat Utilization Rate: **{seat_utilization*100:.1f}%**")

    submit_button = st.form_submit_button("🔍 Run Churn Prediction", use_container_width=True)

# --- PREDICTION & INSIGHTS DISPLAY ---
if submit_button:
    # Build payload matching exact training schema
    input_dict = {
        'Monthly_Recurring_Revenue': mrr,
        'Active_Seats_Used': active_seats,
        'Total_Seats_Allowed': total_seats,
        'Support_Tickets_30d': support_tickets,
        'Feature_Adoption_Rate': feature_adoption,
        'Tenure_Days': tenure_days,
        'Seat_Utilization_Rate': seat_utilization,
        'Industry': industry,
        'Country': country,
        'Company_Size': company_size,
        'Account_Manager': account_manager,
        'Plan_Tier': plan_tier,
        'Billing_Cycle': billing_cycle,
        'Payment_Status': payment_status,
        'Churn_Risk_Score': churn_risk_score
    }
    
    input_df = pd.DataFrame([input_dict])
    
    # Run prediction through full serialized pipeline
    churn_probability = pipeline.predict_proba(input_df)[0, 1]
    is_churn_alert = churn_probability >= threshold

    st.markdown("### 📊 Prediction Results")
    
    res_col1, res_col2, res_col3 = st.columns(3)
    
    with res_col1:
        st.metric(
            label="Churn Probability",
            value=f"{churn_probability * 100:.2f}%",
            delta=f"{(churn_probability - threshold) * 100:+.1f}% vs threshold"
        )
        
    with res_col2:
        if is_churn_alert:
            st.error("⚠️ HIGH RISK OF CHURN")
        else:
            st.success("✅ RETAINED (HEALTHY ACCOUNT)")
            
    with res_col3:
        st.metric(
            label="Active Decision Threshold",
            value=f"{threshold * 100:.1f}%"
        )

    # Actionable Customer Success Recommendations
    st.markdown("#### 🛠️ Recommended Customer Success Actions:")
    recommendations = []
    
    if payment_status in ["Failed", "Pending"]:
        recommendations.append("💳 **Billing Alert:** Urgent payment issue detected. Engage finance team to update payment method.")
    if seat_utilization < 0.4:
        recommendations.append("📉 **Under-utilization Warning:** License usage is below 40%. Schedule an onboarding session to drive adoption.")
    if support_tickets >= 4:
        recommendations.append("🎫 **High Friction:** Account logged 4+ support tickets in the last month. Assign a senior technical specialist.")
    if feature_adoption < 50:
        recommendations.append("🚀 **Feature Adoption Lag:** Product depth is under 50%. Send customized training materials and walkthrough webinars.")
    if not recommendations:
        recommendations.append("🌟 **Healthy Account:** Customer is engaged with stable metrics. Good candidate for annual renewal or plan upgrade.")

    for rec in recommendations:
        st.info(rec)
