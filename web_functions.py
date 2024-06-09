"""This module contains necessary function needed"""

# Import necessary modules
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import streamlit as st

def load_data():
    """This function returns the preprocessed data"""
    # Load the Diabetes dataset into DataFrame.
    df = pd.read_csv('dataset.csv')

    # Rename the column names in the DataFrame.
    
    # Perform feature and target split
    X = df[["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]]
    y = df['target']

    categorical_val = []
    continuous_val = []
    for column in df.columns:
        if len(df[column].unique()) <= 10:
            categorical_val.append(column)
        else:
            continuous_val.append(column)
    scaler = StandardScaler()
    X[continuous_val] = scaler.fit_transform(X[continuous_val])
    # X = X.values



    return df, X, y

# Decision Tree

def train_decision_tree(X, y):
    """This function trains the Decision Tree model and returns the model and model score"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = DecisionTreeClassifier(
        ccp_alpha=0.0, class_weight=None, criterion='entropy',
        max_depth=4, max_features=None, max_leaf_nodes=None,
        min_impurity_decrease=0.0, min_samples_leaf=1, 
        min_samples_split=2, min_weight_fraction_leaf=0.0,
        random_state=42, splitter='best'
    )
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score



def predict_decision_tree(X, y, features):
    """This function trains the Decision Tree model and makes predictions"""
    model, score = train_decision_tree(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score

# Logistic Regression

def train_logistic_regression(X, y):
    """This function trains the Logistic Regression model and returns the model and model score"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def predict_logistic_regression(X, y, features):
    """This function trains the Logistic Regression model and makes predictions"""
    model, score = train_logistic_regression(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score

# XGBoost Classifier

def train_xgb_classifier(X, y):
    """This function trains the XGBoost model and returns the model and model score"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = XGBClassifier(random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def predict_xgb_classifier(X, y, features):
    """This function trains the XGBoost model and makes predictions"""
    model, score = train_xgb_classifier(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score

# Random Forest Classifier

def train_random_forest(X, y):
    """This function trains the Random Forest model and returns the model and model score"""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    return model, score


def predict_random_forest(X, y, features):
    """This function trains the Random Forest model and makes predictions"""
    model, score = train_random_forest(X, y)
    prediction = model.predict(np.array(features).reshape(1, -1))
    return prediction, score
