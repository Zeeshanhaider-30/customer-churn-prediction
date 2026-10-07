import streamlit as st
import pandas as pd
import joblib
import json
import os

st.title("Customer Churn Prediction App")
st.write("Enter the customer details below to predict if they are likely to churn (leave) or stay.")

@st.cache_resource
def load_assets():
    if os.path.exists("best_model.joblib"):
        model = joblib.load("best_model.joblib")
        encoders = joblib.load("label_encoders.joblib")
        scaler = joblib.load("scaler.joblib")
        with open("feature_names.json", "r") as f:
            feature_names = json.load(f)
        return model, encoders, scaler, feature_names
    return None, None, None, None

model, encoders, scaler, feature_names = load_assets()

if model is None:
    st.error("Model assets not found. Please run the Jupyter Notebook first to train and save the model.")
else:
    # Form layout
    with st.form("churn_form"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen", ["Yes", "No"])
            partner = st.selectbox("Partner", ["Yes", "No"])
            dependents = st.selectbox("Dependents", ["Yes", "No"])
            tenure = st.number_input("Tenure Months", min_value=0, max_value=100, value=12)
            
        with col2:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiline = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
            security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
            backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
            protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
            
        with col3:
            support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
            tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
            
        col4, col5 = st.columns(2)
        with col4:
            monthly_charges = st.number_input("Monthly Charges", value=50.0)
        with col5:
            total_charges = st.number_input("Total Charges", value=500.0)
            
        submit = st.form_submit_button("Predict Churn")
        
    if submit:
        # Create input dictionary
        input_data = {
            'Gender': gender,
            'Senior Citizen': senior,
            'Partner': partner,
            'Dependents': dependents,
            'Tenure Months': tenure,
            'Phone Service': phone,
            'Multiple Lines': multiline,
            'Internet Service': internet,
            'Online Security': security,
            'Online Backup': backup,
            'Device Protection': protection,
            'Tech Support': support,
            'Streaming TV': tv,
            'Streaming Movies': movies,
            'Contract': contract,
            'Paperless Billing': paperless,
            'Payment Method': payment,
            'Monthly Charges': monthly_charges,
            'Total Charges': total_charges
        }
        
        df = pd.DataFrame([input_data])
        
        # Preprocess
        for col, le in encoders.items():
            if col in df.columns:
                # Handle unknown labels
                df[col] = df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
                df[col] = le.transform(df[col])
                
        num_cols = ['Tenure Months', 'Monthly Charges', 'Total Charges']
        df[num_cols] = scaler.transform(df[num_cols])
        
        df = df[feature_names]
        
        # Predict
        prediction = model.predict(df)[0]
        
        if prediction == 1:
            st.error("Prediction: The customer is likely to LEAVE (Churn).")
        else:
            st.success("Prediction: The customer is likely to STAY.")
