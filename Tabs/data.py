"""This module contains data about home page"""

# Import necessary modules
import streamlit as st
import pandas as pd
import sqlite3

def fetch_all_rows():
    """Fetch all rows from the SQLite database"""
    conn = sqlite3.connect('predictions.db')
    query = "SELECT * FROM predictions"
    df = pd.read_sql(query, conn)
    conn.close()
    return df

def clear_dataset():
    """Clear all rows from the SQLite database"""
    conn = sqlite3.connect('predictions.db')
    c = conn.cursor()
    c.execute("DELETE FROM predictions")
    conn.commit()
    conn.close()

def app(df):
    """This function create the Data Info page"""

    # Add title to the page
    st.title("Data Info page")

    # Add subheader for the section
    st.subheader("View Data")

    # Create an expansion option to check the data
    with st.expander("View data"):
        st.dataframe(df)

    # Create a section to columns values
    # Give subheader
    st.subheader("Columns Description:")

    # Create a checkbox to get the summary.
    if st.checkbox("View Summary"):
        st.dataframe(df.describe())

    # Create multiple check box in row
    col_name, col_dtype, col_data = st.columns(3)

    # Show name of all dataframe
    with col_name:
        if st.checkbox("Column Names"):
            st.dataframe(df.columns)

    # Show datatype of all columns 
    with col_dtype:
        if st.checkbox("Columns data types"):
            dtypes = df.dtypes.apply(lambda x: x.name)
            st.dataframe(dtypes)

    # Show data for each columns
    with col_data: 
        if st.checkbox("Columns Data"):
            col = st.selectbox("Column Name", list(df.columns))
            st.dataframe(df[col])

    # Add margin above the Fetch All Predictions button
    st.markdown("<br>", unsafe_allow_html=True)

    if st.button("Fetch Past Predictions"):
        predictions_df = fetch_all_rows()
        st.dataframe(predictions_df)

    if st.button("Clear Database"):
        clear_dataset()
        st.success("Database cleared successfully")
    
    

    # Add the link to you dataset
    st.markdown("""
                    <p style="font-size:24px">
                        <a 
                            href="https://raw.githubusercontent.com/AkashS20/Heart-Disease-Prediction/master/dataset.csv?token=GHSAT0AAAAAACSWJOE5NBCIUXFWY4ANQFXYZTKQBIQ"
                            target=_blank
                            style="text-decoration:none;"
                        >Get Dataset
                        </a> 
                    </p>
                """, unsafe_allow_html=True
    )