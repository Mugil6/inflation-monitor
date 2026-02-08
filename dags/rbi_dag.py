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
    """Fetches real-time market inputs from public mirrors"""
    print(" Fetching real-time market data...")
    
    # 1. USD/INR
    try:
        res = requests.get("https://api.exchangerate-api.com/v4/latest/USD").json()
        latest_usd_inr = res['rates']['INR']
    except Exception as e:
        print(f" USD Fetch failed: {e}")
        latest_usd_inr = 83.10  # Fallback

    # 2. Crude Oil (WTI)
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
    """Syncs results to the 'macro_monitor' table based on your SQL schema"""
    url_env = os.getenv('SUPABASE_URL')
    key_env = os.getenv('SUPABASE_KEY')

    if not url_env or not key_env:
        print(" Supabase credentials missing. Skipping cloud sync.")
        return

    # Endpoint for your specific table
    target_url = f"{url_env}/rest/v1/macro_monitor"
    
    headers = {
        "apikey": key_env,
        "Authorization": f"Bearer {key_env}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }
    
    # Mapping to your EXACT SQL column names: 
    # date, oil_price, usd_inr, predicted_inflation, is_forecast
    payload = {
        "date": datetime.now().strftime('%Y-%m-%d'),
        "oil_price": float(oil),
        "usd_inr": float(usd),
        "predicted_inflation": float(pred),
        "is_forecast": True
    }
    
    try:
        response = requests.post(target_url, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            print(" Cloud Sync Complete: Entry added to macro_monitor.")
        else:
            # 409 Conflict usually means this date already exists (Primary Key constraint)
            print(f" Cloud Sync Error: {response.status_code} - {response.text}")
    except Exception as e:
        print(f" Failed to connect to Supabase: {e}")

def predict_inflation():
    print(" Starting Monthly Prediction Task...")
    
    # 1. Get Inputs
    latest_usd, latest_oil = get_market_data()
    print(f" Inputs -> Oil: {latest_oil}, USD/INR: {latest_usd}")
    
    # 2. Prepare Window (3-column requirement)
    data_matrix = np.zeros((60, 3))
    data_matrix[:, 0] = latest_oil
    data_matrix[:, 1] = latest_usd
    data_matrix[:, 2] = 0.0 
    
    # 3. Predict
    print(" Loading Model and Scaler...")
    model = tf.keras.models.load_model(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    
    scaled_data = scaler.transform(data_matrix)
    model_input = scaled_data.reshape(1, 60, 3) 
    
    prediction = model.predict(model_input)
    final_result = float(prediction[0][0])
    
    print(f" FINAL PREDICTION: {final_result:.4f}")

    # 4. Sync
    post_to_supabase(latest_oil, latest_usd, final_result)

# --- DAG DEFINITION ---
with DAG(
    'rbi_inflation_monitor',
    default_args={
        'owner': 'airflow',
        'retries': 0,
        'email_on_failure': False
    },
    schedule_interval='0 0 * * 1', 
    # Setting to today to ensure the UI shows the next run in the future
    start_date=datetime(2026, 2, 8), 
    catchup=False
) as dag:

    task_predict = PythonOperator(
        task_id='predict_inflation',
        python_callable=predict_inflation,
    )