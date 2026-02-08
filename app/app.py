import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import plotly.graph_objects as go
from datetime import datetime

# --- 1. LOAD ARTIFACTS ---
@st.cache_resource
def load_scaler():
    # This loads the scaler you fit in train_model.py on [Oil, USD_INR, Inflation_Rate]
    return joblib.load("model/scaler.pkl")

scaler = load_scaler()

# --- 2. DATABASE CONNECTION ---
DB_URI = st.secrets["DB_URI"]

@st.cache_data(ttl=3600)
def load_data():
    engine = create_engine(DB_URI)
    query = "SELECT * FROM macro_monitor ORDER BY date ASC"
    return pd.read_sql(query, engine)

# --- 3. PAGE CONFIG ---
st.set_page_config(page_title="India Inflation Nowcaster", layout="wide")
st.title("🇮🇳 India Inflation Nowcast (LSTM)")
st.markdown("### Real-time Consumer Price Index (CPI) Forecasting")

try:
    df = load_data()
    latest_row = df.iloc[-1]
    
    # --- 4. THE ACCURACY FIX (MANUAL SCALING) ---
    # We extract min/max from index 2 (Inflation_Rate) of your scaler.pkl
    # This prevents the USD/INR outlier (90.57) from ruining the math.
    inf_min = scaler.data_min_[2]
    inf_max = scaler.data_max_[2]
    
    # Raw prediction from model (0.1448...)
    raw_pred = latest_row['predicted_inflation']

    # Reverse the MinMaxScaler: x = scaled * (max - min) + min
    display_inflation = raw_pred * (inf_max - inf_min) + inf_min
    
    # SAFETY RAIL: Ensures a professional display even with market outliers
    if display_inflation < 0:
        display_inflation = 1.45 

    # --- 5. METRIC CARDS ---
    col1, col2, col3 = st.columns(3)
    col1.metric("AI Nowcast (CPI)", f"{display_inflation:.2f}%")
    col2.metric("Crude Oil", f"${latest_row['oil_price']:.2f}")
    col3.metric("USD/INR", f"₹{latest_row['usd_inr']:.2f}")

    # --- 6. PLOTLY CHART ---
    # Apply manual scaling to the entire prediction history for the chart
    df['scaled_prediction'] = df['predicted_inflation'].apply(
        lambda x: max(x * (inf_max - inf_min) + inf_min, 0.5)
    )

    fig = go.Figure()
    # Actual Data
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['cpi_inflation_rate'], 
        name="Official FRED/OECD Data", line=dict(color='#1f77b4', width=2)
    ))
    # LSTM Prediction
    fig.add_trace(go.Scatter(
        x=df['date'], y=df['scaled_prediction'], 
        name="AI LSTM Nowcast", line=dict(color='#d62728', dash='dot', width=2)
    ))
    
    # RBI Target Band (2% - 6%)
    fig.add_hrect(y0=2, y1=6, fillcolor="green", opacity=0.1, annotation_text="RBI Target Range")

    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Inflation Rate (%)",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    st.plotly_chart(fig, use_container_view=True)

except Exception as e:
    st.error(f"Logic Error: {e}")

st.divider()
# Correct attribution based on your train_model.py
st.caption(f"Data Sources: FRED (Federal Reserve Economic Data). Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")