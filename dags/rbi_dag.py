from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import yfinance as yf
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from supabase import create_client, Client
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- CONFIGURATION ---
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
MODEL_PATH = "/opt/airflow/model/rbi_inflation_model.h5"
SCALER_PATH = "/opt/airflow/model/scaler.pkl"

# Define the DAG arguments
default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

def predict_inflation():
    print("Starting Prediction Task...")

    # 1. OPTIMIZATION: Download only the last 365 days (1 Year)
    # We need enough past data to create the "sequence" for the LSTM, 
    # but we don't need 20 years. 1 year is a safe buffer.
    end_date = datetime.now()
    start_date = (end_date - timedelta(days=365)).strftime('%Y-%m-%d')
    
    print(f" Fetching data from {start_date} to today...")
    
    tickers = ['INR=X', 'CL=F'] # USD/INR and Crude Oil
    try:
        data = yf.download(tickers, start=start_date, progress=False)
    except Exception as e:
        print(f" Yahoo Finance Error: {e}")
        raise

    # 2. DATA VALIDATION (Prevent IndexError)
    if data.empty:
        raise ValueError(" No data downloaded! Yahoo Finance blocked the request or tickers are wrong.")
    
    print(" Data Downloaded Successfully.")

    # 3. PREPROCESSING
    # We need to format the downloaded data exactly like the training data
    # yfinance returns a MultiIndex, we need to flatten it or select correctly
    try:
        # Handling yfinance structure (Adj Close is usually the standard)
        df_clean = data['Adj Close'].copy()
    except KeyError:
        # Fallback if 'Adj Close' isn't there (sometimes it's just 'Close')
        df_clean = data['Close'].copy()

    # Ensure we have the latest values
    latest_oil = df_clean['CL=F'].iloc[-1]
    latest_usd = df_clean['INR=X'].iloc[-1]
    
    print(f"📊 Latest Market Data -> Oil: ${latest_oil:.2f}, USD/INR: ₹{latest_usd:.2f}")

    # 4. LOAD MODEL & SCALER
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Model and Scaler loaded.")
    except Exception as e:
        print(f"Error loading model/scaler from {MODEL_PATH}: {e}")
        raise

    # 5. PREPARE INPUT FOR LSTM
    # LSTM expects a sequence (e.g., last 60 days). 
    # We scale the data using the SAME scaler from training.
    
    # Select the features in the correct order (Check your training code for order!)
    # Assuming order: [Oil, USD]

    input_data = df_clean[['CL=F', 'INR=X']].values
    
    # Scale the data
    scaled_data = scaler.transform(input_data)
    
    # Take the last 60 days (or whatever your window_size was during training)
    WINDOW_SIZE = 60 
    
    if len(scaled_data) < WINDOW_SIZE:
        raise ValueError(f"❌ Not enough data! Need {WINDOW_SIZE} days, got {len(scaled_data)}")

    # Reshape for LSTM: (1, 60, 2)
    last_sequence = scaled_data[-WINDOW_SIZE:]
    last_sequence = np.expand_dims(last_sequence, axis=0)

    # 6. PREDICT

    prediction = model.predict(last_sequence)
    # Inverse transform if the target was scaled (assuming scaler handled 3 columns or target scaler exists)
    # simpler approach: result is likely the scaled CPI value.
    
    # NOTE: If you scaled y (CPI) during training, you must inverse_transform this result.
    # For now, assuming raw output or handled externally:

    predicted_cpi = float(prediction[0][0])
    
    print(f"🔮 RAW Model Prediction: {predicted_cpi}")

    # 7. UPLOAD TO SUPABASE

    if SUPABASE_URL and SUPABASE_KEY:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        
        payload = {
            "date": datetime.now().strftime('%Y-%m-%d'),
            "oil_price": float(latest_oil),
            "usd_inr": float(latest_usd),
            "predicted_cpi": predicted_cpi
        }
        
        data, count = supabase.table('inflation_predictions').insert(payload).execute()
        print(f" Uploaded to Supabase: {data}")
    else:
        print(" Supabase credentials missing. Skipping upload.")

# Define the DAG

with DAG(
    'rbi_inflation_monitor',
    default_args=default_args,
    description='Fetch market data and predict Indian CPI Inflation',
    schedule_interval='@daily', # Runs once a day
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    task_predict = PythonOperator(
        task_id='predict_inflation',
        python_callable=predict_inflation,
    )

    task_predict