"""
# My first app
Here's our first attempt at using data to create a table:
"""

import streamlit as st
import pandas as pd
import numpy as np


def digitize(df, col, limit) -> pd.Categorical:
    """
    Convert a continuous variable into a categorical variable.
    
    Parameters:
    -----------
    df : pd.DataFrame
        The dataframe containing the data
    col : str
        The column name of the variable to digitize
    limit : tuple
        The theoretical extent (lower_bound, upper_bound) of the variable
        
    Returns:
    --------
    pd.Categorical
        A categorical representation of the variable with bins defined by unique values
    """
    # Extract the values from the dataframe column
    values = df[col].dropna().unique()
    
    # Add the limit values if they're not already in the values
    lower_bound, upper_bound = limit
    
    all_values = np.sort(np.unique(np.append(values, [lower_bound, upper_bound])))
    
    # Create bins from the unique values
    bins = list(all_values) + [np.inf] # Captures max value
    
    # Digitize the data using the bins (closed='left' means bins are closed on the left and open on the right)
    return pd.cut(df[col].values, bins=bins, include_lowest=True, right=False)


def describe_digitized(y_digitized) -> pd.DataFrame:
    y_desc = y_digitized.describe()
    y_desc = y_desc.reset_index()
    y_desc.loc[~y_desc.categories.isna(), 'left'] = y_digitized.categories.left
    y_desc.loc[~y_desc.categories.isna(), 'right'] = y_digitized.categories.right
    return y_desc

def suggest_y_range(df, col): 
    buffer = .1
    y_vals = df[col].values
    y_vals_no_nan = y_vals[~np.isnan(y_vals)]
    y_min, y_max = y_vals_no_nan.min(), y_vals_no_nan.max()
    delta = buffer*(y_max-y_min)
    y_min_min = y_min - delta
    y_max_max = y_max + delta
    return y_min_min, y_max_max, (y_min, y_max)

    
def validate_input(df): 
    is_valid = (df.shape[1] >= 2) and \
               (df.select_dtypes('float').shape[1] >=1) and \
               (df.shape[0] >= 2)
    return is_valid

def run_model(df):
    y_col = st.selectbox('Select response variable', list(df.select_dtypes('float').columns),
                         key='select_response')

    y_range_suggestion = suggest_y_range(df, y_col)

    y_range = st.slider("Select Experiment Sensitivity", *y_range_suggestion, 
                        key='select_response_range')

    y_digitized = digitize(df, y_col, y_range)
    y_desc = describe_digitized(y_digitized)

    st.dataframe(y_desc.loc[:, ['left', 'right', 'counts', 'freqs']], key='y_description')


uploaded_file = st.file_uploader("Choose a file", )

if uploaded_file is not None:

    raw_data = pd.read_csv(uploaded_file)

    input_is_valid = validate_input(raw_data)

    if input_is_valid: 
        st.write(f"Valid input")
        st.dataframe(raw_data, key='raw_data')
    else: 
        st.write(f"Invalid input")

    run_model(raw_data)


