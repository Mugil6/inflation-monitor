import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import StandardScaler # StandardScaler is often better for deltas
from supabase import create_client
import joblib
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def train_v4_delta():
    print("📥 Fetching Data for Delta Training...")
    response = supabase.table("macro_monitor").select("*").order("date", desc=False).execute()
    df = pd.DataFrame(response.data).dropna(subset=['cpi_inflation_rate'])
    
    # CALCULATE DELTA (Stationarity)
    df['cpi_delta'] = df['cpi_inflation_rate'].diff().fillna(0)
    
    # Feature Set: [Oil, USD, CPI_Actual, CPI_Delta]
    features = df[['oil_price', 'usd_inr', 'cpi_delta']].values
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(features)

    X, y = [], []
    for i in range(10, len(scaled_data)):
        X.append(scaled_data[i-10:i])
        y.append(scaled_data[i, 2]) # We are predicting the DELTA
    
    X, y = np.array(X), np.array(y)

    # LIGHTWEIGHT ARCHITECTURE (Preventing Overfitting)
    model = Sequential([
        LSTM(32, input_shape=(10, 3)), # Fewer units = better generalization for small data
        Dropout(0.2),
        Dense(1) 
    ])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.005), loss='mae') # MAE is less sensitive to outliers
    
    print("🚀 Training Delta-Based Model...")
    model.fit(X, y, epochs=150, batch_size=2, verbose=0)

    model.save('model/rbi_lstm.keras')
    joblib.dump(scaler, 'model/scaler.pkl')
    print("✅ MLOps V4 Model Saved!")

if __name__ == "__main__":
    train_v4_delta()