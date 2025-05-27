from io import BytesIO
from streamlit.testing.v1 import AppTest
import pandas as pd
import pytest

def test_data_validation(mocker):
    mock_uploader = mocker.patch('streamlit.file_uploader')
    with open('tests/data/two_num_cols.csv', 'rb') as f:
        output = BytesIO(f.read())
    mock_uploader.return_value = output 
    at = AppTest.from_file('app.py')
    at.run()
    assert 'valid' in at.markdown[0].value.lower() 
