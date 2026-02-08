import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import plotly.graph_objects as go
from datetime import datetime

# --- 1. LOAD SCALER ---
# We load the scaler to get the exact training boundaries
scaler = joblib.load("model/scaler.pkl")

# --- 2. CONFIG ---
DB_URI = st.secrets["DB_URI"]

@st.cache_data(ttl=3600)
def load_data():
    engine = create_engine(DB_URI)
    query = "SELECT * FROM macro_monitor ORDER BY date ASC"
    return pd.read_sql(query, engine)

# --- 3. PAGE UI ---
st.set_page_config(page_title="India Inflation Nowcaster", layout="wide")
st.title("🇮🇳 India Inflation Nowcast (LSTM)")

try:
    df = load_data()
    latest_row = df.iloc[-1]
    
    # --- 4. THE ROBUST SCALING FIX ---
    # In train_model.py, your columns were: [Oil, USD_INR, Inflation_Rate]
    # Index 2 is your Inflation Rate.
    inf_min = scaler.data_min_[2]
    inf_max = scaler.data_max_[2]
    
    raw_pred = latest_row['predicted_inflation']

    # MATHEMATICAL REVERSAL:
    # If raw_pred is scaled (0-1), this formula restores the actual %
    # If raw_pred is already a %, this logic will result in a huge number,
    # so we add a safety check.
    
    if raw_pred <= 1.0:
        display_inflation = raw_pred * (inf_max - inf_min) + inf_min
    else:
        # If the value in DB is already 1.4 or 5.4, don't scale it!
        display_inflation = raw_pred

    # --- 5. DATA PREP FOR CHART ---
    # Apply the same scaling logic to the whole prediction column for the chart
    df['scaled_prediction'] = df['predicted_inflation'].apply(
        lambda x: x * (inf_max - inf_min) + inf_min if x <= 1.0 else x
    )

    # --- 6. DISPLAY ---
    col1, col2, col3 = st.columns(3)
    col1.metric("AI Nowcast (CPI)", f"{display_inflation:.2f}%")
    col2.metric("Crude Oil", f"${latest_row['oil_price']:.2f}")
    col3.metric("USD/INR", f"₹{latest_row['usd_inr']:.2f}")

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df['date'], y=df['cpi_inflation_rate'], name="Actual MoSPI Data"))
    fig.add_trace(go.Scatter(x=df['date'], y=df['scaled_prediction'], name="AI Prediction", line=dict(dash='dot')))
    
    # RBI Target Zone (4% +/- 2%)
    fig.add_hrect(y0=2, y1=6, fillcolor="green", opacity=0.1, annotation_text="RBI Target Range")
    
    st.plotly_chart(fig, use_container_view=True)

except Exception as e:
    st.error(f"Error: {e}")

st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")