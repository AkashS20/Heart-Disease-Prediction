"""This module contains data about the prediction page"""

# Import necessary modules
import streamlit as st
import pandas as pd
import datetime
import sqlite3

# Import necessary functions from web_functions
from web_functions import (
    predict_decision_tree,
    predict_logistic_regression,
    predict_xgb_classifier,
    predict_random_forest
)

def create_table():
    """Create the predictions table if it doesn't exist"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS predictions (
            model TEXT,
            age INTEGER,
            sex INTEGER,
            cp INTEGER,
            trestbps INTEGER,
            chol INTEGER,
            fbs INTEGER,
            restecg INTEGER,
            thalach INTEGER,
            exang INTEGER,
            oldpeak REAL,
            slope INTEGER,
            ca INTEGER,
            thal INTEGER,
            prediction INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_prediction_to_db(model, features, prediction):
    """Log the prediction details to the SQLite database"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute('''
        INSERT INTO predictions (
            model, age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, prediction
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    ''', (model, *features, prediction))
    conn.commit()
    conn.close()

def app(df, X, y, scaler, continuous_val):
    """This function creates the prediction page"""
    create_table()
    
    # List of models
    models = ['Random Forest Classification', 'Logistic Regression Classification', 'XGBoost Classification', 'Decision Tree Classification']
    model = st.sidebar.selectbox('Which model would you like to use?', models)

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        f"""
            <p style="font-size:25px">
                This model uses <b style="color:green">{model}</b> for Cardiac Disease Prediction.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user
    sex_options = {"Female": 0, "Male": 1}
    cp_options = {
        "Typical Angina": 0,
        "Atypical Angina": 1,
        "Non-Anginal Pain": 2,
        "Asymptomatic": 3
    }
    fbs_options = {"False": 0, "True": 1}
    restecg_options = {
        "Normal": 0,
        "ST-T wave abnormality": 1,
        "Left ventricular hypertrophy": 2
    }
    exang_options = {"No": 0, "Yes": 1}
    slope_options = {
        "Upsloping": 0,
        "Flat": 1,
        "Downsloping": 2
    }
    ca_options = {"0": 0, "1": 1, "2": 2, "3": 3}
    thal_options = {
        "NULL": 0,
        "Normal": 1,
        "Fixed Defect": 2,
        "Reversible Defect": 3
    }

    # Nominal attribute select boxes
    sex = sex_options[st.selectbox('Sex:', list(sex_options.keys()))]
    cp = cp_options[st.selectbox('Type of chest pain experienced by patient:', list(cp_options.keys()))]
    fbs = fbs_options[st.selectbox('Blood sugar levels on fasting > 120 mg/dl:', list(fbs_options.keys()))]
    restecg = restecg_options[st.selectbox('Resting electrocardiographic results:', list(restecg_options.keys()))]
    exang = exang_options[st.selectbox('Exercise induced angina:', list(exang_options.keys()))]
    slope = slope_options[st.selectbox('The slope of the peak exercise ST segment:', list(slope_options.keys()))]
    ca = ca_options[st.selectbox('Number of major vessels:', list(ca_options.keys()))]
    thal = thal_options[st.selectbox('Thalassemia (Blood Disorder):', list(thal_options.keys()))]

    # Numeric attribute sliders
    age = st.slider("Age", 18, 80)
    trestbps = st.slider("Resting blood pressure", int(df["trestbps"].min()), int(df["trestbps"].max()))
    chol = st.slider("Serum cholesterol in mg/dl", int(df["chol"].min()), int(df["chol"].max()))
    thalach = st.slider('Maximum heart rate achieved', int(df["thalach"].min()), int(df["thalach"].max()))
    oldpeak = st.slider('Oldpeak', float(df["oldpeak"].min()), float(df["oldpeak"].max()))

    # Create a list to store all the features
    features = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    features_df = pd.DataFrame([features], columns=["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"])

    # Create a button to predict
    if st.button("Predict"):
        # Scale input features
        features_df[continuous_val] = scaler.transform(features_df[continuous_val])
        # st.write(features_df)
        if model == 'Random Forest Classification':
            prediction, score = predict_random_forest(X, y, features_df)
        elif model == 'Logistic Regression Classification':
            prediction, score = predict_logistic_regression(X, y, features_df)
        elif model == 'XGBoost Classification':
            prediction, score = predict_xgb_classifier(X, y, features_df)
        elif model == 'Decision Tree Classification':
            prediction, score = predict_decision_tree(X, y, features_df)
        
        st.info("Prediction Successful")

        # Print the output according to the prediction
        if prediction == 1:
            st.warning("The person is prone to getting a heart disease!!")
        else:
            st.success("The person is relatively safe from heart disease")

        # Print the score of the model
        st.write("This model has an accuracy of ", (score * 100), "%")

        # Log to file
        prediction = int(prediction[0])
        with open('logs.txt', 'a') as f:
            log_entry = f"{datetime.datetime.now()}, Model: {model}, Features: {features}, Prediction: {prediction}\n"
            f.write(log_entry)
        
        # Log to SQLite
        log_prediction_to_db(model, features, prediction)

