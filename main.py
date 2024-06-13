"""This is the main module to run the app"""

# Importing the necessary Python modules.
import streamlit as st

# Import necessary functions from web_functions
from web_functions import load_data

# Import pages
from Tabs import home, data, predict, visualise

from streamlit_navigation_bar import st_navbar


# Configure the app
st.set_page_config(
    page_title = 'Cardiac Disease Prediction',
    page_icon = 'machine learning',
    layout = 'wide',
    initial_sidebar_state = 'auto'
)
# Dictionary for pages
Tabs = {
    "Home": home,
    "Data Info": data,
    "Prediction": predict,
    "Visualisation": visualise
}



# Loading the dataset.
df, X, y, scaler, continuous_val = load_data()

pages = ["Home", "Prediction", "Data Info", "Visualisation"]

page = st_navbar(
    pages
    # options = {"show_sidebar" : False}
)


if page == "Prediction":
    Tabs[page].app(df, X, y, scaler, continuous_val)
elif (page == "Visualisation"):
    Tabs[page].app(df, X, y)
elif (page == "Data Info"):
    Tabs[page].app(df)
else:
    Tabs[page].app()

# Create a sidebar
# Add title to sidear
# st.sidebar.title("Navigation")

# # Create radio option to select the page
# page = st.sidebar.radio("Pages", list(Tabs.keys()))


# # Call the app funciton of selected page to run
# if page in ["Prediction", "Visualisation"]:
#     Tabs[page].app(df, X, y)
# elif (page == "Data Info"):
#     Tabs[page].app(df)
# else:
#     Tabs[page].app()
