from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import requests
import os
from io import StringIO

# --- CONSTANTS ---
MODEL_PATH = "/opt/airflow/model/rbi_lstm.keras"
SCALER_PATH = "/opt/airflow/model/scaler.pkl"

def get_market_data():
    """Fetches real-time market inputs"""
    print(" Fetching real-time market data...")
    
    # 1. USD/INR (Exchangerate API)
    try:
        res = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()
        latest_usd_inr = res['rates']['INR']
    except Exception as e:
        print(f" USD Fetch failed: {e}")
        latest_usd_inr = 83.10  # Fallback

    # 2. Crude Oil (Open Data GitHub - Same as Seed)
    try:
        oil_url = "https://raw.githubusercontent.com/datasets/oil-prices/master/data/wti-daily.csv"
        oil_res = requests.get(oil_url)
        oil_df = pd.read_csv(StringIO(oil_res.text))
        latest_oil = float(oil_df.iloc[-1]['Price'])
    except Exception as e:
        print(f" Oil Fetch failed: {e}")
        latest_oil = 76.50  # Fallback

    return latest_usd_inr, latest_oil

def post_to_supabase(oil, usd, pred):
    url_env = os.getenv('SUPABASE_URL')
    key_env = os.getenv('SUPABASE_KEY')

    target_url = f"{url_env}/rest/v1/macro_monitor"
    
    headers = {
        "apikey": key_env,
        "Authorization": f"Bearer {key_env}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    # PAYLOAD MATCHING 'macro_monitor' SCHEMA
    payload = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "oil_price": float(oil),
        "usd_inr": float(usd),
        "predicted_inflation": float(pred),
        "cpi_inflation": None,  # Unknown for future
        "is_forecast": True
    }
    
    try:
        response = requests.post(target_url, json=payload, headers=headers)
        print(f"Status: {response.status_code}")
    except Exception as e:
        print(f"Error: {e}")

def predict_inflation():
    print(" Starting Weekly Prediction Task...")
    
    latest_usd, latest_oil = get_market_data()
    print(f" Inputs -> Oil: {latest_oil}, USD/INR: {latest_usd}")
    
    # Prepare Dummy Input (Logic to handle 3-feature input for 1 prediction)
    # We create a window of zeros but fill the last row with current data
    # This acts as a 'Nowcast' proxy
    data_matrix = np.zeros((60, 3)) # Assuming lookback 60 or 10 depending on training
    data_matrix[-1, 0] = latest_oil
    data_matrix[-1, 1] = latest_usd
    data_matrix[-1, 2] = 0.0 
    
    print(" Loading Model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    # Scale & Predict
    scaled_data = scaler.transform(data_matrix)
    # Reshape for LSTM (1 sample, 60 timesteps, 3 features)
    # Note: Ensure this '60' matches your 'LOOKBACK_WINDOW' from training!
    model_input = scaled_data.reshape(1, 60, 3) 
    
    prediction = model.predict(model_input)
    
    # Inverse Scale Target Only
    cpi_min = scaler.data_min_[2]
    cpi_range = scaler.data_range_[2]
    final_result = (prediction[0][0] * cpi_range) + cpi_min
    
    print(f" FINAL PREDICTION: {final_result:.4f}")
    post_to_supabase(latest_oil, latest_usd, final_result)

# --- DAG ---
with DAG(
    'rbi_inflation_monitor',
    default_args={'owner': 'airflow', 'retries': 0},
    schedule_interval='0 0 * * 1', 
    start_date=datetime(2025, 1, 1), 
    catchup=False
) as dag:

    task_predict = PythonOperator(
        task_id='predict_inflation',
        python_callable=predict_inflation,
    )