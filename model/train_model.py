import os
import pandas as pd
import pandas_datareader.data as web
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

# 1. PRODUCTION STABILITY SETTINGS
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 2. FETCH DATA FROM FRED (Live Training Source)
print("🌐 Fetching historical data from FRED...")
start = '2015-01-01'
end = datetime.now().strftime('%Y-%m-%d')

try:
    # CPALTT01INM659N = India CPI | DEXINUS = USD to INR
    # MCOILWTICO = Crude Oil WTI
    df_raw = web.DataReader(['CPALTT01INM659N', 'DEXINUS', 'MCOILWTICO'], 'fred', start, end)
    
    # Clean and Resample to Monthly
    df = df_raw.resample('MS').mean().dropna()
    df.columns = ['Inflation_Rate', 'USD_INR', 'Crude_Oil']
    
    # Reorder to match your DAG/App logic: [Oil, USD, CPI]
    df = df[['Crude_Oil', 'USD_INR', 'Inflation_Rate']]
    print(f"✅ Successfully prepared {len(df)} months of training data.")
except Exception as e:
    print(f"❌ Failed to fetch data: {e}")
    exit()

# 3. SCALING (Ensures scikit-learn 1.3.2 compatibility)
features = df.values
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_features = scaler.fit_transform(features)

# 4. CREATE SEQUENCES
def create_sequences(data, lookback=10):
    X, y = [], []
    for i in range(len(data) - lookback):
        X.append(data[i:i + lookback])
        y.append(data[i + lookback, 2]) # Target: Inflation_Rate
    return np.array(X), np.array(y)

LOOKBACK = 10
X, y = create_sequences(scaled_features, LOOKBACK)

# 5. BUILD LSTM MODEL
model = Sequential([
    LSTM(64, activation='tanh', input_shape=(LOOKBACK, 3), return_sequences=True),
    Dropout(0.2),
    LSTM(32, activation='tanh'),
    Dropout(0.2),
    Dense(16, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# 6. TRAIN
print("🚀 Training Model...")
model.fit(X, y, epochs=60, batch_size=4, verbose=1)

# 7. EXPORT
os.makedirs('model', exist_ok=True)
model.save('model/rbi_lstm.keras')
joblib.dump(scaler, 'model/scaler.pkl')
print("✅ SUCCESS: Model and Scaler exported to /model folder.")