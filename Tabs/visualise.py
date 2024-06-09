"""This modules contains data about visualisation page"""

# Import necessary modules
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
'''from sklearn.metrics import plot_confusion_matrix'''
from sklearn import tree
import streamlit as st
from web_functions import (
    train_decision_tree,
    train_logistic_regression,
    train_xgb_classifier,
    train_random_forest
)


# Import necessary functions from web_functions
from web_functions import train_decision_tree

def app(df, X, y):
    """This function create the visualisation page"""
    
    # Remove the warnings
    warnings.filterwarnings('ignore')
    st.set_option('deprecation.showPyplotGlobalUse', False)

    # Set the page title
    st.title("Visualise Heart Ailment Demographics")

    if st.checkbox("Show Feature Distribution Graph"):
        # st.subheader("Feature Distribution Graph")
        fig, ax = plt.subplots(figsize=(25, 25))
        # Generate the histogram within the figure  
        df.hist(ax=ax, grid=False)
        # Display the figure using Streamlit
        st.pyplot(fig)

    if st.checkbox("Show Sample Results"):
        st.subheader("Target Distribution")
        f,ax = plt.subplots(1,2,figsize=(15,7))
        df["target"].replace({0:"No Heart Disease",1:"Heart Disease"}).value_counts().plot(kind="pie",colors=["orange","blue"],ax=ax[0],explode=[0,0.1],autopct='%1.1f%%',shadow=True)
        ax[0].set_ylabel('')
        df["target"].replace({0:"Safe",1:"Prone"}).value_counts().plot(kind="bar", ax = ax[1],color=["orange","blue"])
        ax[1].set_ylabel('')
        ax[1].set_xlabel('')
        st.pyplot()

    # Create a checkbox to show correlation heatmap
    if st.checkbox("Show Correlation with Target Feature"):
        st.subheader("Correlation Heatmap")

        # fig = plt.figure(figsize = (8, 6))
        # ax = sns.heatmap(df.iloc[:, 1:].corr(), annot = True)   # Creating an object of seaborn axis and storing it in 'ax' variable
        # bottom, top = ax.get_ylim()                             # Getting the top and bottom margin limits.
        # ax.set_ylim(bottom + 0.5, top - 0.5)                    # Increasing the bottom and decreasing the top margins respectively.
        # st.pyplot(fig)
        sns.set_context('notebook', font_scale=2.3)

        # Create the bar plot for correlation with target
        fig1 = plt.figure(figsize=(20, 8))
        correlation_with_target = df.drop('target', axis=1).corrwith(df.target)
        correlation_with_target.plot(kind='bar', grid=True, title="Correlation with the target feature")
        st.pyplot(fig1)
        

    if st.checkbox("Plot Decision Tree"):
        model, score = train_decision_tree(X, y)
        # Export decision tree in dot format and store in 'dot_data' variable.
        dot_data = tree.export_graphviz(
            decision_tree=model, max_depth=3, out_file=None, filled=True, rounded=True,
            feature_names=X.columns, class_names=['0', '1']
        )
        # Plot the decision tree using the 'graphviz_chart' function of the 'streamlit' module.
        st.graphviz_chart(dot_data)

    if st.checkbox("Compare Model Accuracies"):
        models = ['Random Forest', 'Logistic Regression', 'XGBoost', 'Decision Tree']
    
        # Train each model and calculate accuracies
        accuracies = []
        for model_name in models:
            if model_name == 'Random Forest':
                model, accuracy = train_random_forest(X, y)
            elif model_name == 'Logistic Regression':
                model, accuracy = train_logistic_regression(X, y)
            elif model_name == 'XGBoost':
                model, accuracy = train_xgb_classifier(X, y)
            elif model_name == 'Decision Tree':
                model, accuracy = train_decision_tree(X, y)
            accuracies.append(accuracy * 100)
        
        # Create bar plot
        plt.figure(figsize=(12, 5))
        plt.bar(models, accuracies, color=['blue', 'orange', 'green', 'red'])
        plt.xlabel('Models')
        plt.ylabel('Accuracy (%)')
        plt.title('Model Accuracies')
        plt.ylim(0, 100)  # Setting y-axis limits
        plt.xticks(fontsize=14)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Display plot in Streamlit app
        st.pyplot()

