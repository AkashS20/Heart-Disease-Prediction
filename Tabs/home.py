"""This modules contains data about home page"""

# Import necessary modules
import streamlit as st
from PIL import Image

def app():
    """This function create the home page"""
    
    st.title("Heart Disease Prediction Model")
    st.markdown(
    """<p style="font-size:20px;">  </p>
    """, unsafe_allow_html=True)
   
    col1, col2 = st.columns(2)


    with col1:
        image1 = Image.open("images/Heart1.png")
        resized_image1 = image1
        st.image(resized_image1, width = 250)
    with col2:
        image2 = Image.open("images/Heart2.jpg")
        resized_image2 = image2
        st.image(resized_image2, use_column_width=True)

    # Add brief describtion of your web app
    st.markdown(
    """<p style="font-size:20px;">
            Heart diseases remain a leading cause of morbidity and mortality worldwide, posing significant challenges to healthcare systems. These ailments encompass a range of conditions, from coronary artery disease and heart failure to arrhythmias and congenital heart defects. Early detection and precise prediction of heart diseases are critical to improving patient outcomes and reducing healthcare costs. Here, machine learning (ML) emerges as a transformative tool, leveraging vast datasets to identify patterns and risk factors that traditional methods might overlook. By analyzing medical records, imaging, genetic information, and lifestyle data, ML algorithms can predict the likelihood of heart disease with remarkable accuracy. These predictive models not only assist in early diagnosis but also personalize treatment plans, optimize resource allocation, and advance preventative care strategies. As ML continues to evolve, its integration into cardiology promises to enhance our understanding, management, and ultimately, the prevention of heart diseases, ushering in a new era of precision medicine.
        </p>
    """, unsafe_allow_html=True)