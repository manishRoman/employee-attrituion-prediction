import streamlit as st
import pandas as pd
import numpy as np
import os
import traceback

# ==========================================
# 0. CATCH MISSING LIBRARIES
# ==========================================
try:
    import joblib
    import json
    import shap
    import matplotlib.pyplot as plt
except ImportError as e:
    st.error(f"🚨 Missing Library Error: {e}")
    st.stop()

# ==========================================
# 1. LOAD ASSETS
# ==========================================
@st.cache_resource
def load_assets():
    try:
        model = joblib.load('logistic_model.pkl')
        scaler = joblib.load('scaler.pkl')
        model_columns = joblib.load('model_columns.pkl')
        
        with open('baseline_employee.json', 'r') as file:
            baseline_profile = json.load(file)
            
        return model, scaler, model_columns, baseline_profile
    except Exception as e:
        return str(e)

assets = load_assets()

if isinstance(assets, str):
    st.error("🚨 File Loading Error! Ensure your .pkl files are in the same folder.")
    st.stop()
else:
    model, scaler, model_columns, baseline_profile = assets

# ==========================================
# 2. BUILD THE USER INTERFACE
# ==========================================
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")
st.title("📊Employee Attrition Predictor")

st.divider()

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=65, value=30)
    income = st.number_input("Monthly Income ($)", min_value=1000, max_value=20000, value=5000)
    distance = st.number_input("Distance From Home (Miles)", min_value=1, max_value=30, value=10)
    job_sat = st.slider("Job Satisfaction (1-Low, 4-High)", min_value=1, max_value=4, value=3)

with col2:
    overtime = st.selectbox("Works OverTime?", options=["No", "Yes"])
    marital_status = st.selectbox("Marital Status", options=["Single", "Married", "Divorced"])
    years_manager = st.number_input("Years With Current Manager", min_value=0, max_value=20, value=2)
    job_hop_index = st.slider("Job Hopping Index (0=Loyal, 1=Frequent Hopper)", min_value=0.0, max_value=1.0, value=0.2)

st.divider()

# ==========================================
# 3. THE PREDICTION ENGINE
# ==========================================
if st.button("Predict Flight Risk", type="primary", use_container_width=True):
    
    user_data = baseline_profile.copy()
    
    user_data['Age'] = age
    user_data['MonthlyIncome'] = income
    user_data['DistanceFromHome'] = distance
    user_data['JobSatisfaction'] = job_sat
    user_data['OverTime'] = overtime
    user_data['MaritalStatus'] = marital_status
    user_data['YearsWithCurrManager'] = years_manager
    user_data['Job_Hopping_Index'] = job_hop_index
    
    input_df = pd.DataFrame([user_data])
    input_encoded = pd.get_dummies(input_df)
    input_aligned = input_encoded.reindex(columns=model_columns, fill_value=0)
    input_scaled = scaler.transform(input_aligned)
    
    # We only need the probability of class 0 (Staying)
    prob_stay = model.predict_proba(input_scaled)[0][0] * 100
    
    # ==========================================
    # 4. DISPLAY RESULTS (SINGLE METRIC THRESHOLD)
    # ==========================================
    # You can change this 50.0 to anything! 
    # If you want to be extra cautious, you could set it to 60.0
    if prob_stay >= 50.0:
        st.success(f"✅ **SAFE** (Probability of staying: {prob_stay:.1f}%)")
        st.markdown("This employee is stable. Their likelihood of staying is high.")
    else:
        st.error(f"🚨 **HIGH ALERT** (Probability of staying: only {prob_stay:.1f}%)")
        st.markdown("This employee's 'staying power' has dropped to critical levels. Immediate retention strategies are recommended.")
        

  # ==========================================
    # 5. SIMPLE EXPLAINABILITY GRAPH
    # ==========================================
    st.divider()
    st.subheader("The Results")
    st.markdown("**Red** increases risk. **Green** increases loyalty.")
    
    # Extract the raw math from SHAP
    background_data = np.zeros((1, len(model_columns)))
    explainer = shap.LinearExplainer(model, background_data)
    shap_vals = explainer.shap_values(input_scaled)
    
    # Map the values to a Pandas DataFrame for easy plotting
    impacts = shap_vals[0]
    df_impact = pd.DataFrame({'Feature': model_columns, 'Impact': impacts})
    
    # --- THE FIX: FILTER OUT THE HIDDEN BACKGROUND COLUMNS ---
    # We only want to graph the 8 features the stakeholder can actually control
    ui_features = [
        'Age', 'MonthlyIncome', 'DistanceFromHome', 'JobSatisfaction', 
        'OverTime', 'MaritalStatus', 'YearsWithCurrManager', 'Job_Hopping_Index'
    ]
    
    # Keep only the rows where the Feature name matches our UI features
    # (We use str.contains so it catches dummy variables like 'MaritalStatus_Single')
    df_impact = df_impact[df_impact['Feature'].str.contains('|'.join(ui_features))]
    
    # Sort by the largest impact (absolute value)
    df_impact['Abs_Impact'] = df_impact['Impact'].abs()
    df_impact = df_impact.sort_values(by='Abs_Impact', ascending=True)
    
    # Draw a clean, standard Matplotlib chart
    fig, ax = plt.subplots(figsize=(8, 5))
    
    # Logic: Red if the number is positive (causing attrition), Green if negative (preventing attrition)
    colors = ['#ff4b4b' if x > 0 else '#00cc96' for x in df_impact['Impact']]
    
    ax.barh(df_impact['Feature'], df_impact['Impact'], color=colors)
    
    # Chart styling
    ax.axvline(0, color='black', linewidth=1.5) # Draw a strong center line
    ax.set_xlabel("Impact Score (Red = Quitting, Green = Staying)")
    ax.set_title("Input Factors Influencing This Employee")
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    plt.tight_layout()
    st.pyplot(fig)