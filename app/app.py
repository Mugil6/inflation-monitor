import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from supabase import create_client, Client
from tensorflow.keras.models import load_model
import joblib
import os
from datetime import timedelta
from sklearn.metrics import mean_absolute_error

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="India Inflation Nowcaster", page_icon="🇮🇳", layout="wide")

# Load Secrets
url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]
supabase: Client = create_client(url, key)

# Constants
LOOKBACK_WINDOW = 10 
RBI_LOWER_BAND = 2.0
RBI_UPPER_BAND = 6.0

# Load Artifacts
@st.cache_resource
def load_artifacts():
    model = load_model('model/rbi_lstm.keras')
    scaler = joblib.load('model/scaler.pkl')
    return model, scaler

model, scaler = load_artifacts()

# --- 2. DATA INGESTION ---
def get_data():
    # Fetch from 'macro_monitor'
    response = supabase.table('macro_monitor').select("*").order('date', desc=False).execute()
    df = pd.DataFrame(response.data)
    df['date'] = pd.to_datetime(df['date'])
    
    # Filter for ACTUALS (History) to run the Backtest
    # We ignore the future forecast rows for the historical validation logic
    df_actuals = df[df['cpi_inflation'].notna()].copy()
    
    # Rename columns to match what LSTM expects
    df_actuals = df_actuals.rename(columns={
        'oil_price': 'crude_oil',     # Maps DB 'oil_price' -> Code 'crude_oil'
        'cpi_inflation': 'cpi'        # Maps DB 'cpi_inflation' -> Code 'cpi'
    })
    
    # Ensure numeric
    cols = ['crude_oil', 'usd_inr', 'cpi']
    for c in cols:
        df_actuals[c] = pd.to_numeric(df_actuals[c])
        
    return df_actuals

df = get_data()

# --- 3. ROLLING BACKTEST ENGINE ---
def generate_forecasts(df, model, scaler, lookback):
    dates = []
    actuals = []
    predictions = []
    
    # Extract Scaling Constants (Manual Decoupling Logic)
    cpi_min = scaler.data_min_[2]
    cpi_range = scaler.data_range_[2]
    
    if len(df) < (lookback + 12):
        st.error("Not enough data for 12-month backtest.")
        return None, None, None, None

    start_idx = len(df) - 12
    
    # Backtest Loop
    for i in range(start_idx, len(df)):
        raw_window = df.iloc[i-lookback:i][['crude_oil', 'usd_inr', 'cpi']].values
        scaled_window = scaler.transform(raw_window)
        scaled_window = scaled_window.reshape(1, lookback, 3)
        
        pred_scaled = model.predict(scaled_window, verbose=0)
        pred_final = (pred_scaled[0][0] * cpi_range) + cpi_min
        
        dates.append(df.iloc[i]['date'])
        actuals.append(df.iloc[i]['cpi'])
        predictions.append(pred_final)

    # Predict NEXT Month (Future)
    last_window_raw = df.iloc[-lookback:][['crude_oil', 'usd_inr', 'cpi']].values
    last_window_scaled = scaler.transform(last_window_raw).reshape(1, lookback, 3)
    
    future_pred_scaled = model.predict(last_window_scaled, verbose=0)
    future_pred = (future_pred_scaled[0][0] * cpi_range) + cpi_min
    
    next_month_date = df.iloc[-1]['date'] + pd.DateOffset(months=1)
    
    return dates, actuals, predictions, (next_month_date, future_pred)

# Run Engine
dates, actuals, preds, future_tuple = generate_forecasts(df, model, scaler, LOOKBACK_WINDOW)
next_date, next_val = future_tuple
mae = mean_absolute_error(actuals, preds)

# --- 4. DASHBOARD UI ---
st.title("🇮🇳 AI Inflation Nowcaster (Beta)")
st.markdown(f"**Live Forecast for {next_date.strftime('%B %Y')}:**")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("🤖 AI Prediction", f"{next_val:.2f}%", delta=f"{next_val - actuals[-1]:.2f}% vs Last Month")
with col2:
    st.metric("📉 Model Accuracy (MAE)", f"±{mae:.2f}%", help="Mean Absolute Error over last 12 months")
with col3:
    status = "✅ Within RBI Target" if 2.0 <= next_val <= 6.0 else "⚠️ Breach Detected"
    st.metric("🏛️ RBI Monitor", status)

# --- 5. VISUALIZATION ---
fig = go.Figure()

# RBI Band
fig.add_hrect(y0=RBI_LOWER_BAND, y1=RBI_UPPER_BAND, 
              fillcolor="green", opacity=0.1, line_width=0, 
              annotation_text="RBI Tolerance Band (2-6%)", annotation_position="top left")

# Actuals
fig.add_trace(go.Scatter(x=dates, y=actuals, mode='lines+markers', name='Actual CPI (Official)',
                         line=dict(color='#1f77b4', width=3)))

# Backtest
fig.add_trace(go.Scatter(x=dates, y=preds, mode='lines', name='AI Nowcast (Validation)',
                         line=dict(color='#ff7f0e', width=2, dash='dot')))

# Forecast Star
fig.add_trace(go.Scatter(x=[next_date], y=[next_val], mode='markers+text', name='Next Month Forecast',
                         marker=dict(color='red', size=15, symbol='star'),
                         text=[f"{next_val:.2f}%"], textposition="top center"))

# Confidence Interval
fig.add_trace(go.Scatter(x=[next_date, next_date], y=[next_val - mae, next_val + mae],
                         mode='lines', line=dict(color='red', width=4),
                         name=f'Confidence Interval (±{mae:.2f}%)'))

fig.update_layout(title="Real-Time Inflation Trajectory", xaxis_title="Date", yaxis_title="CPI (%)",
                  hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

st.plotly_chart(fig, use_container_width=True)

# --- 6. ARCHITECTURE ---
with st.expander("🛠️ How this works (System Architecture)"):
    # Ensure 'arch(1).jpg' is in your repo/app folder!
    st.image("arch(1).jpg", caption="Decoupled MLOps Architecture") 
    st.markdown("""
    * **Data Source:** Automated weekly ingestion from FRED API & Open Data.
    * **Model:** LSTM Neural Network trained on 10 years of macro-economic data.
    * **Sanity Rails:** Output is manually scaled to prevent outliers from breaking the forecast.
    """)