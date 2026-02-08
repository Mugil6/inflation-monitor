import pandas as pd
import numpy as np
import yfinance as yf
import pandas_datareader.data as web
import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler
import joblib
from datetime import datetime

# Disable OneDNN warnings (optional cleanup)
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


# 1. FETCH OFFICIAL DATA (INDIA)

print("Step 1: Fetching Official Data...")

start_date = '2015-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

# A. TARGET: India CPI (Consumer Price Index)
# New Series ID: CPALTT01INM659N (OECD Standard for India CPI)
try:
    cpi_df = web.DataReader('CPALTT01INM659N', 'fred', start_date, end_date)
    cpi_df.columns = ['CPI_Index']
    print(f"   - Successfully fetched CPI Data (Rows: {len(cpi_df)})")
except Exception as e:
    print(f"   - CRITICAL ERROR fetching FRED data: {e}")
    exit()

# B. FEATURES: High-Frequency Market Data
# CL=F: Brent Crude Oil
# INR=X: USD/INR Exchange Rate
print("   - Fetching Market Data (Yahoo Finance)...")
tickers = ['CL=F', 'INR=X']
market_data = yf.download(tickers, start=start_date, end=end_date)['Close']
market_data.columns = ['Oil', 'USD_INR']


# 2. DATA ALIGNMENT & ENGINEERING

print("Step 2: Processing & Aligning...")

# Resample daily market data to Monthly Averages to match CPI
market_monthly = market_data.resample('MS').mean()

# Align indices
cpi_df.index = pd.to_datetime(cpi_df.index)
market_monthly.index = pd.to_datetime(market_monthly.index)

# Merge
df = pd.concat([cpi_df, market_monthly], axis=1).dropna()

# CRITICAL: Calculate YoY Inflation Rate
# Formula: (Current_Index - Index_12_Months_Ago) / Index_12_Months_Ago * 100
df['Inflation_Rate'] = df['CPI_Index'].pct_change(periods=12) * 100
df = df.dropna()

print(f"   - Training Data Range: {df.index.min().date()} to {df.index.max().date()}")
print(f"   - Latest Official Inflation in Data: {df['Inflation_Rate'].iloc[-1]:.2f}%")

# Prepare Data for LSTM
data = df[['Oil', 'USD_INR', 'Inflation_Rate']].values


# 3. PREPROCESSING

scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data)

X, y = [], []
lookback = 6

for i in range(lookback, len(scaled_data)):
    X.append(scaled_data[i-lookback:i])
    y.append(scaled_data[i, 2]) # Predict Inflation Rate

X, y = np.array(X), np.array(y)


# 4. BUILD & TRAIN LSTM

print("Step 3: Training LSTM Model...")

model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(lookback, 3)),
    tf.keras.layers.LSTM(50, return_sequences=True),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.LSTM(50, return_sequences=False),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
model.fit(X, y, epochs=50, batch_size=16, verbose=1)


# 5. SAVE ARTIFACTS

print("Step 4: Saving Artifacts...")
model.save("model/rbi_lstm.keras")
joblib.dump(scaler, "model/scaler.pkl")

print("\n SUCCESS: Model trained on REAL data (Source: OECD/FRED).")
print("   - Ready for Docker Build.")