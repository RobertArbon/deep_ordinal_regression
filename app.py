"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd

    
def validate_input(df): 
    is_valid = (df.shape[1] >= 2) and \
               (df.select_dtypes('float').shape[1] >=1) and \
               (df.shape[0] >= 2)
    return is_valid

uploaded_file = st.file_uploader("Choose a file")
if uploaded_file is not None:
    raw_data = pd.read_csv(uploaded_file)

    input_is_valid = validate_input(raw_data)

    if input_is_valid: 
        st.write(f"Valid input")
        st.dataframe(raw_data)
    else: 
        st.write(f"Invalid input")




