from io import BytesIO
from streamlit.testing.v1 import AppTest
import pandas as pd
import pytest

def test_data_validation(mocker):
    mock_uploader = mocker.patch('streamlit.file_uploader')
    with open('tests/data/two_num_cols.csv', 'rb') as f:
        output = BytesIO(f.read())
    mock_uploader.return_value = output 
    # print('OUTPUT!!  ', pd.read_csv(output))
    at = AppTest.from_file('app.py')
    at.run()
    assert at.dataframe[0]

    # df = pd.read_csv('tests/data/two_num_cols.csv')
    # at.session_state["raw_data"] = df
    # at.run()
    # assert at.dataframe[0].value.shape[1] == 2
    # assert   at.dataframe[0].value.shape[1] == 2
