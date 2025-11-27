# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 16:25:58 2024

@author: prachet
"""
import pickle
import streamlit as st
from streamlit_option_menu import option_menu
import json
import pandas as pd

# Loading the saved model of diabetes prediction
with open("Preprocessing Files/ML-Project-2-Diabetes_Prediction_Pre_Processing_Files/columns.pkl", 'rb') as f:
    all_features_diabetes_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-2-Diabetes_Prediction_Pre_Processing_Files/scaler.pkl", 'rb') as f:
    scalers_diabetes_disease = pickle.load(f)
with open("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_svc.json", 'r') as file:
    best_features_svc_diabetes_disease = json.load(file)
with open("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_lr.json", 'r') as file:
    best_features_lr_diabetes_disease = json.load(file)
with open("Best Features/ML-Project-2-Diabetes_Prediction_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_diabetes_disease = json.load(file)
with open("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_svc_model.sav", 'rb') as f:
    loaded_model_svc_diabetes_disease = pickle.load(f)
with open("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_diabetes_disease = pickle.load(f)
with open("Models/ML-Project-2-Diabetes_Prediction_Models/diabetes_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_diabetes_disease = pickle.load(f)

# Loading the saved model of heart disease prediction
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/columns.pkl", 'rb') as f:
    all_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/cat_columns.pkl", 'rb') as f:
    cat_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/encoder.pkl", 'rb') as f:
    encoder_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/encoded_columns.pkl", 'rb') as f:
    encoded_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/training_columns.pkl", 'rb') as f:
    training_columns_heart_disease = pickle.load(f)
with open("Preprocessing Files/ML-Project-9-Heart_Disease_Prediction_Pre_Processing_Files/scaler.pkl", 'rb') as f:
    scaler_heart_disease = pickle.load(f)
with open("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_xgb.json", 'r') as file:
    best_features_xgb_heart_disease = json.load(file)
with open("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_rfc.json", 'r') as file:
    best_features_rfc_heart_disease = json.load(file)
with open("Best Features/ML-Project-9-Heart_Disease_Prediction_Best_Features/best_features_lr.json", 'r') as file:
    best_features_lr_heart_disease = json.load(file)
with open("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_xgb_model.sav", 'rb') as f:
    loaded_model_xgb_heart_disease = pickle.load(f)
with open("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_rfc_model.sav", 'rb') as f:
    loaded_model_rfc_heart_disease = pickle.load(f)
with open("Models/ML-Project-9-Heart_Disease_Prediction_Models/heart_disease_trained_lr_model.sav", 'rb') as f:
    loaded_model_lr_heart_disease = pickle.load(f)

def diabetes_prediction(input_data):
    df_diabetes_disease = pd.DataFrame([input_data], columns=all_features_diabetes_disease)
    df_diabetes_disease[all_features_diabetes_disease] = scalers_diabetes_disease.transform(df_diabetes_disease[all_features_diabetes_disease])
    
    df_best_features_svc_diabetes_disease = df_diabetes_disease[best_features_svc_diabetes_disease]
    df_best_features_lr_diabetes_disease = df_diabetes_disease[best_features_lr_diabetes_disease]
    df_best_features_rfc_diabetes_disease = df_diabetes_disease[best_features_rfc_diabetes_disease]
    
    prediction1_diabetes_disease = loaded_model_svc_diabetes_disease.predict(df_best_features_svc_diabetes_disease)
    prediction2_diabetes_disease = loaded_model_lr_diabetes_disease.predict(df_best_features_lr_diabetes_disease)
    prediction3_diabetes_disease = loaded_model_rfc_diabetes_disease.predict(df_best_features_rfc_diabetes_disease)
    
    return prediction1_diabetes_disease, prediction2_diabetes_disease, prediction3_diabetes_disease

def heart_disease_prediction(input_data):
    columns_heart_disease = all_columns_heart_disease
    df_heart_disease = pd.DataFrame([input_data], columns=columns_heart_disease)
    
    df_heart_disease[cat_columns_heart_disease] = df_heart_disease[cat_columns_heart_disease].astype('str')
    input_data_encoded_heart_disease = encoder_heart_disease.transform(df_heart_disease[cat_columns_heart_disease])
    input_data_encoded_df_heart_disease = pd.DataFrame(input_data_encoded_heart_disease, columns=encoded_columns_heart_disease)
    input_data_final_encoded_heart_disease = pd.concat([df_heart_disease.drop(cat_columns_heart_disease, axis=1).reset_index(drop=True), input_data_encoded_df_heart_disease], axis=1)
    input_data_scaled_heart_disease = scaler_heart_disease.transform(input_data_final_encoded_heart_disease)
    input_data_df_heart_disease = pd.DataFrame(input_data_scaled_heart_disease, columns=training_columns_heart_disease)

    df_best_features_xgb_heart_disease = input_data_df_heart_disease[best_features_xgb_heart_disease]
    df_best_features_rfc_heart_disease = input_data_df_heart_disease[best_features_rfc_heart_disease]
    df_best_features_lr_heart_disease = input_data_df_heart_disease[best_features_lr_heart_disease]

    prediction1_heart_disease = loaded_model_xgb_heart_disease.predict(df_best_features_xgb_heart_disease)
    prediction2_heart_disease = loaded_model_rfc_heart_disease.predict(df_best_features_rfc_heart_disease)
    prediction3_heart_disease = loaded_model_lr_heart_disease.predict(df_best_features_lr_heart_disease)
    
    return prediction1_heart_disease, prediction2_heart_disease, prediction3_heart_disease

def main():
    # Custom CSS for better UI
    st.markdown("""
    <style>
    .main {
        background-color: #f0f2f8;
    }
    .stButton>button {
        background-color: #4CAF30;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 8px;
    }
    .stButton>button:hover {
        background-color: #45b049;
    }
    .stSelectbox, .stNumberInput {
        margin-bottom: 10px;
    }
    .stTitle {
        color: #2E86AB;
        font-family: 'Arial', sans-serif;
    }
    .stSubheader {
        color: #A23A72;
    }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar for navigation
    with st.sidebar:
        st.markdown("## 🏥 Navigation")
        selected = option_menu('Multiple Disease Prediction System using ML',
                               ['Diabetes Prediction', 'Heart Disease Prediction'],
                               icons=['capsule', 'activity'],
                               default_index=0)
        st.markdown("---")
        st.info("Select a prediction type from the menu above.")

    # Diabetes Prediction Page
    if selected == 'Diabetes Prediction':
        st.markdown("<h1 style='text-align: center; color: #2E86AB;'>🩸 Diabetes Prediction using ML</h1>", unsafe_allow_html=True)
        st.markdown("### Enter the following details to predict diabetes risk:")
        
        col1, col2 = st.columns(2)
        with col1:
            Pregnancies = st.number_input("Number of Pregnancies", format="%.0f")
            Glucose = st.number_input("Glucose Level (mg/dL)", format="%.2f")
            BloodPressure = st.number_input("Blood Pressure (mm Hg)", format="%.2f")
            SkinThickness = st.number_input("Skin Thickness (mm)", format="%.2f")
        with col2:
            Insulin = st.number_input("Insulin Level (mu U/ml)", format="%.2f")
            BMI = st.number_input("BMI", format="%.2f")
            DiabetesPedigreeFunction = st.number_input("Diabetes Pedigree Function", format="%.3f")
            Age = st.number_input("Age (years)", format="%.0f")

        diabetes_diagnosis_svc, diabetes_diagnosis_lr, diabetes_diagnosis_rfc = diabetes_prediction([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age])
        
        if st.button("🔍 Predict Diabetes"):
            if diabetes_diagnosis_rfc[0] == 0:
                st.success("✅ The person is not diabetic.")
                st.balloons()
            else:
                st.warning("⚠️ The person is diabetic.")
                st.balloons()
        
        with st.expander("🔧 Show Advanced Options"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Predict with Random Forest"):
                    if diabetes_diagnosis_rfc[0] == 0:
                        st.success("✅ Not diabetic (RFC)")
                        st.balloons()
                    else:
                        st.warning("⚠️ Diabetic (RFC)")
                        st.balloons()
            with col2:
                if st.button("Predict with Logistic Regression"):
                    if diabetes_diagnosis_lr[0] == 0:
                        st.success("✅ Not diabetic (LR)")
                        st.balloons()
                    else:
                        st.warning("⚠️ Diabetic (LR)")
                        st.balloons()
            with col3:
                if st.button("Predict with SVC"):
                    if diabetes_diagnosis_svc[0] == 0:
                        st.success("✅ Not diabetic (SVC)")
                        st.balloons()
                    else:
                        st.warning("⚠️ Diabetic (SVC)")
                        st.balloons()

    # Heart Disease Prediction Page
    if selected == 'Heart Disease Prediction':
        st.markdown("<h1 style='text-align: center; color: #2E86AB;'>❤️ Heart Disease Prediction using ML</h1>", unsafe_allow_html=True)
        st.markdown("### Enter the following details to predict heart disease risk:")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            age = st.number_input("Age (years)", format="%.0f")
            resting_bp = st.number_input("Resting Blood Pressure (mm Hg)", format="%.0f")
            max_heart_achieved = st.number_input("Maximum Heart Rate Achieved", format="%.0f")
            oldpeak = st.number_input("Oldpeak", format="%.2f")
        with col2:
            option1 = st.selectbox('Gender', ('Male', 'Female'))
            sex = 0 if option1 == 'Female' else 1
            serum_cholestoral = st.number_input("Serum Cholesterol (mg/dl)", format="%.0f")
            option5 = st.selectbox('Exercise Induced Angina', ('Yes', 'No'))
            exercise_induced_angina = 0 if option5 == 'No' else 1
            option6 = st.selectbox('Slope of Peak Exercise ST Segment', ('0', '1', '2'))
            slope_of_peak_exercise = int(option6)
        with col3:
            option2 = st.selectbox('Chest Pain Type', ('0', '1', '2', '3'))
            chest_pain = int(option2)
            option3 = st.selectbox('Fasting Blood Sugar > 120 mg/dl', ('True', 'False'))
            fasting_blood_sugar = 0 if option3 == 'False' else 1
            option4 = st.selectbox('Resting ECG Results', ('0', '1', '2'))
            resting_ecg = int(option4)
            option7 = st.selectbox('Number of Major Vessels', ('0', '1', '2', '3'))
            number_of_major_vessels = int(option7)
            option8 = st.selectbox('Thal', ('None', 'Normal', 'Fixed defect', 'Reversible defect'))
            thal = {'None': 0, 'Normal': 1, 'Fixed defect': 2, 'Reversible defect': 3}[option8]

        heart_disease_diagnosis_xgb, heart_disease_diagnosis_rfc, heart_disease_diagnosis_lr = heart_disease_prediction([age, sex, chest_pain, resting_bp, serum_cholestoral, fasting_blood_sugar, resting_ecg, max_heart_achieved, exercise_induced_angina, oldpeak, slope_of_peak_exercise, number_of_major_vessels, thal])
        
        if st.button("🔍 Predict Heart Disease"):
            if heart_disease_diagnosis_xgb[0] == 0:
                st.success("✅ The person does not have heart disease.")
                st.balloons()
            else:
                st.warning("⚠️ The person has heart disease.")
                st.balloons()
        
        with st.expander("🔧 Show Advanced Options"):
            col1, col2, col3 = st.columns(3)
            with col1:
                if st.button("Predict with XGBoost"):
                    if heart_disease_diagnosis_xgb[0] == 0:
                        st.success("✅ No heart disease (XGB)")
                        st.balloons()
                    else:
                        st.warning("⚠️ Heart disease (XGB)")
                        st.balloons()
            with col2:
                if st.button("Predict with Random Forest"):
                    if heart_disease_diagnosis_rfc[0] == 0:
                        st.success("✅ No heart disease (RFC)")
                        st.balloons()
                    else:
                        st.warning("⚠️ Heart disease (RFC)")
                        st.balloons()
            with col3:
                if st.button("Predict with Logistic Regression"):
                    if heart_disease_diagnosis_lr[0] == 0:
                        st.success("✅ No heart disease (LR)")
                        st.balloons()
                    else:
                        st.warning("⚠️ Heart disease (LR)")
                        st.balloons()

if __name__ == '__main__':
    main()
