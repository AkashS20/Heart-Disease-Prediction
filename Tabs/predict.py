"""This module contains data about prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import (
    predict_decision_tree,
    predict_logistic_regression,
    predict_xgb_classifier,
    predict_random_forest
)

def app(df, X, y):
    """This function creates the prediction page"""

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
    cp = st.selectbox('Type of chest pain experienced by patient: 0 typical angina, 1 atypical angina, 2 non-anginal pain, 3 asymptomatic (Nominal)', (0, 1, 2, 3))
    sex = st.selectbox('Sex: 0-Female 1-Male', (0, 1))
    fbs = st.selectbox('Fasting blood sugar: Enter 0 or 1', (0, 1))
    restecg = st.selectbox('Resting electrocardiographic results: Enter 0, 1 or 2', (0, 1, 2))
    exang = st.selectbox('Exercise induced angina: ', (0, 1))
    slope = st.selectbox('The slope of the peak exercise ST segment: Enter 0, 1, 2, or 3', (0, 1, 2, 3))
    ca = st.selectbox('Number of major vessels: Enter 0, 1, 2, or 3', (0, 1, 2, 3))
    thal = st.selectbox('Thal: Enter 0, 1, or 2', (0, 1, 2))
    age = st.slider("Age", int(df["age"].min()), int(df["age"].max()))
    trestbps = st.slider("Resting blood pressure", int(df["trestbps"].min()), int(df["trestbps"].max()))
    chol = st.slider("Serum cholesterol in mg/dl", int(df["chol"].min()), int(df["chol"].max()))
    thalach = st.slider('Maximum heart rate achieved', int(df["thalach"].min()), int(df["thalach"].max()))
    oldpeak = st.slider('Oldpeak: Enter a value between 0 and 6.2', float(df["oldpeak"].min()), float(df["oldpeak"].max()))

    # Create a list to store all the features
    features = [cp, sex, fbs, restecg, exang, slope, ca, thal, age, trestbps, chol, thalach, oldpeak]

    # Create a button to predict
    if st.button("Predict"):
        if model == 'Random Forest Classification':
            prediction, score = predict_random_forest(X, y, features)
        elif model == 'Logistic Regression Classification':
            prediction, score = predict_logistic_regression(X, y, features)
        elif model == 'XGBoost Classification':
            prediction, score = predict_xgb_classifier(X, y, features)
        elif model == 'Decision Tree Classification':
            prediction, score = predict_decision_tree(X, y, features)
        
        st.info("Prediction Successful")

        # Print the output according to the prediction
        if prediction == 1:
            st.warning("The person is prone to getting a cardiac arrest!!")
        else:
            st.success("The person is relatively safe from cardiac arrest")

        # Print the score of the model
        st.write("The model used is trusted by doctors and has an accuracy of ", (score * 100), "%")
