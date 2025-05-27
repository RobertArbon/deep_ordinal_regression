from io import BytesIO
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np
import pytest

"""
y_desc = y_digitized.describe()
y_desc = y_desc.reset_index()
y_desc.loc[~y_desc.categories.isna(), 'left'] = y_digitized.categories.left
y_desc.loc[~y_desc.categories.isna(), 'right'] = y_digitized.categories.right
st.dataframe(y_desc.loc[:, ['left', 'right', 'counts', 'freqs']])
"""

def test_digitize1(mocker):
    mock_uploader = mocker.patch('streamlit.file_uploader')
    with open('tests/data/to_digitize.csv', 'rb') as f:
        output = BytesIO(f.read())

    mock_uploader.return_value = output 
    at = AppTest.from_file('app.py')
    at.selectbox[0].set_value('y').run()
    at.slider[0].set_range((0, 4))
    at.run()
    df = at.dataframe[1]
    np.testing.assert_equal(df['counts'].values, 1)
    np.testing.assert_array_equal(df['left'].values, np.array([0, 1, 2, 3, 4] ))

def test_digitize2(mocker):
    mock_uploader = mocker.patch('streamlit.file_uploader')
    with open('tests/data/to_digitize.csv', 'rb') as f:
        output = BytesIO(f.read())

    mock_uploader.return_value = output 
    at = AppTest.from_file('app.py')
    at.selectbox[0].set_value('z').run()
    at.slider[0].set_range((-1, 3))
    at.run()
    df = at.dataframe[1]
    np.testing.assert_equal(df['counts'].values, 1)
    np.testing.assert_array_equal(df['left'].values, np.array([-1, 0, 2, 3, np.nan]))
