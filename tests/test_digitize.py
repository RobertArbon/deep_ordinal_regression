from io import BytesIO
from streamlit.testing.v1 import AppTest
import pandas as pd
import numpy as np
import pytest


def test_digitize1(mocker):
    mock_uploader = mocker.patch('streamlit.file_uploader')
    mock_uploader.return_value = 'tests/data/to_digitize.csv' 
    at = AppTest.from_file('app.py').run()
    at.selectbox[0].set_value('y').run()
    at.slider[0].set_range(0, 4).run()
    df = at.dataframe[1].value
    np.testing.assert_equal(df['counts'].values, 1)
    np.testing.assert_array_equal(df['left'].values, np.array([0, 1, 2, 3, 4] ))

def test_digitize2(mocker):
    mock_uploader = mocker.patch('streamlit.file_uploader')
    mock_uploader.return_value = 'tests/data/to_digitize.csv' 
    at = AppTest.from_file('app.py').run()
    at.selectbox(key='select_response').set_value('z').run()
    at.slider(key='select_response_range').set_range(-1, 3).run()
    df = at.dataframe[1].value

    np.testing.assert_equal(df['counts'].values, 1)
    np.testing.assert_array_equal(df['left'].values, np.array([-1, 0, 2, 3, np.nan]))
