from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import tensorflow as tf
import joblib
import requests
import os

# 1. CONFIGURATION
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

def predict_inflation():
    print("🚀 Starting Inflation Prediction Task...")
    
    # 2. FETCH LATEST DATA (FRED API)
    # Getting last 12 months to ensure we have enough for the 10-month lookback
    end = datetime.now()
    start = end - timedelta(days=365)
    
    try:
        # Crude Oil (MCOILWTICO) | USD/INR (DEXINUS) | India CPI (CPALTT01INM659N)
        df_raw = web.DataReader(['MCOILWTICO', 'DEXINUS', 'CPALTT01INM659N'], 'fred', start, end)
        df = df_raw.resample('MS').mean().ffill().dropna()
        
        # ENFORCE FEATURE ORDER: [Oil, USD, CPI]
        # This MUST match the train_model.py logic
        df = df[['MCOILWTICO', 'DEXINUS', 'CPALTT01INM659N']]
        current_features = df.tail(10).values # Get the last 10 months
        
        print(f"✅ Data Fetched. Latest Oil: {df.iloc[-1,0]:.2f}, USD: {df.iloc[-1,1]:.2f}")
    except Exception as e:
        print(f"❌ Data Fetch Error: {e}")
        return

    # 3. LOAD ARTIFACTS
    model = tf.keras.models.load_model('/opt/airflow/model/rbi_lstm.keras')
    scaler = joblib.load('/opt/airflow/model/scaler.pkl')

    # 4. PREPROCESS & PREDICT
    # Scale inputs using the same translator from training
    scaled_input = scaler.transform(current_features)
    scaled_input = np.expand_dims(scaled_input, axis=0) # Shape: (1, 10, 3)

    scaled_prediction = model.predict(scaled_input)
    
    # INVERSE TRANSFORM (The fix for the 1368% error)
    # We create a dummy row to reverse-scale only the 3rd column (CPI)
    dummy = np.zeros((1, 3))
    dummy[0, 2] = scaled_prediction[0, 0]
    prediction = scaler.inverse_transform(dummy)[0, 2]
    
    print(f"🎯 FINAL PREDICTION: {prediction:.2f}%")

    # 5. UPSERT TO SUPABASE (The fix for the 400 error)
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates" # <--- IMPORTANT: Overwrites if date exists
    }

    payload = {
        "date": datetime.now().strftime('%Y-%m-01'), # Floor to start of month
        "oil_price": float(df.iloc[-1, 0]),
        "usd_inr": float(df.iloc[-1, 1]),
        "predicted_inflation": float(prediction),
        "is_forecast": True
    }

    response = requests.post(
        f"{SUPABASE_URL}/rest/v1/macro_monitor",
        headers=headers,
        json=payload
    )
    
    print(f"📡 Supabase Status: {response.status_code}")
    if response.status_code >= 400:
        print(f"❌ Error Detail: {response.text}")

# 6. DAG DEFINITION
default_args = {
    'owner': 'mugil',
    'start_date': datetime(2026, 1, 1),
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

with DAG(
    'rbi_inflation_monitor',
    default_args=default_args,
    schedule_interval='@weekly', # Runs every Sunday night
    catchup=False
) as dag:

    task_predict = PythonOperator(
        task_id='predict_inflation',
        python_callable=predict_inflation
    )