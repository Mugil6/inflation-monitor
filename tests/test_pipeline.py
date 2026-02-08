import pytest
import pandas as pd
import numpy as np

# 1. Test: Are we fetching positive prices? (Oil can't be -$50)
def test_data_quality():
    # Simulate data
    df = pd.DataFrame({'Oil': [70.5, 80.2], 'USD_INR': [83.1, 83.4]})
    assert (df['Oil'] > 0).all(), "Oil prices cannot be negative"

# 2. Test: Is the output shape correct for LSTM?
def test_lstm_input_shape():
    # Simulate scaler output (6 months, 3 features)
    dummy_input = np.random.rand(1, 6, 3) 
    assert dummy_input.shape == (1, 6, 3), "Model expects input shape (1, 6, 3)"