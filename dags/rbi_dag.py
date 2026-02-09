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
    print("🚀 Starting V4 Delta-Inference Task...")
    
    # 2. FETCH LATEST DATA (FRED API)
    # Fetching ~14 months to ensure we have a solid window for diff() and 10-month lookback
    end = datetime.now()
    start = end - timedelta(days=420)
    
    try:
        # Features: Crude Oil (WTI) | USD/INR | India CPI
        df_raw = web.DataReader(['MCOILWTICO', 'DEXINUS', 'CPALTT01INM659N'], 'fred', start, end)
        
        # Preprocessing: Resample to Month Start, forward-fill missing market data
        df = df_raw.resample('MS').mean().ffill()
        
        # MLOPS V4 LOGIC: Calculate Delta (Stationarity)
        # This matches the train_model.py V4 logic
        df['cpi_delta'] = df['CPALTT01INM659N'].diff().fillna(0)
        
        # Anchor point: The last known actual CPI value reported by the government
        last_actual_cpi = df['CPALTT01INM659N'].dropna().iloc[-1]
        
        # ENFORCE FEATURE ORDER: [Oil, USD, CPI_Delta]
        features_df = df[['MCOILWTICO', 'DEXINUS', 'cpi_delta']].tail(10)
        current_features = features_df.values
        
        if len(current_features) < 10:
            raise ValueError(f"Insufficient data: Found {len(current_features)} months, need 10.")
            
        print(f"✅ Data Prepared. Anchor CPI: {last_actual_cpi:.2f}% | Latest Oil: {df.iloc[-1,0]:.2f}")
    except Exception as e:
        print(f"❌ Data Fetch/Prep Error: {e}")
        return

    # 3. LOAD PRODUCTION ARTIFACTS
    # Ensure these paths match your Docker volume mapping
    try:
        model = tf.keras.models.load_model('/opt/airflow/model/rbi_lstm.keras')
        scaler = joblib.load('/opt/airflow/model/scaler.pkl')
    except Exception as e:
        print(f"❌ Artifact Load Error: {e}")
        return

    # 4. PREPROCESS & PREDICT DELTA
    scaled_input = scaler.transform(current_features)
    scaled_input = np.expand_dims(scaled_input, axis=0) # Shape: (1, 10, 3)

    scaled_prediction = model.predict(scaled_input, verbose=0)
    
    # Inverse Transform only the Delta (3rd column)
    dummy = np.zeros((1, 3))
    dummy[0, 2] = scaled_prediction[0, 0]
    predicted_delta = scaler.inverse_transform(dummy)[0, 2]
    
    # 5. RECONSTRUCT & DEFLATION GUARD
    # Final Forecast = Last Actual + Model's Predicted Change
    final_prediction = last_actual_cpi + predicted_delta
    
    # Apply structural floor for the Indian economy (V4 fine-tuning)
    final_prediction = max(0.85, final_prediction)
    
    print(f"🎯 Predicted Δ: {predicted_delta:+.2f} | FINAL FORECAST: {final_prediction:.2f}%")

    # 6. UPSERT TO SUPABASE
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
        "Content-Type": "application/json",
        "Prefer": "resolution=merge-duplicates"
    }

    # Forecast date is the start of the current month
    payload = {
        "date": datetime.now().strftime('%Y-%m-01'),
        "oil_price": float(df.iloc[-1, 0]),
        "usd_inr": float(df.iloc[-1, 1]),
        "predicted_inflation": float(final_prediction),
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

# 7. DAG DEFINITION
default_args = {
    'owner': 'mugil',
    'depends_on_past': False,
    'start_date': datetime(2026, 1, 1),
    'email_on_failure': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=10),
}

with DAG(
    'rbi_inflation_monitor_v4',
    default_args=default_args,
    description='MLOps V4: Delta-based Inflation Forecasting',
    schedule_interval='@monthly', 
    catchup=False,
    tags=['mlops', 'inflation', 'india']
) as dag:

    task_predict = PythonOperator(
        task_id='predict_inflation_v4',
        python_callable=predict_inflation
    )