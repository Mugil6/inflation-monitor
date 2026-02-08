import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sqlalchemy import create_engine
import plotly.graph_objects as go
from datetime import datetime

# --- 1. LOAD ARTIFACTS ---
# Loading the exact scaler created in train_model.py
scaler = joblib.load("model/scaler.pkl")

# --- 2. CONFIG & DB CONNECTION ---
DB_URI = st.secrets["DB_URI"]

@st.cache_data(ttl=3600)
def load_data():
    engine = create_engine(DB_URI)
    query = "SELECT * FROM macro_monitor ORDER BY date ASC"
    return pd.read_sql(query, engine)

# --- 3. UI SETUP ---
st.set_page_config(page_title="Inflation Nowcaster", layout="wide")
st.title("🇮🇳 India Inflation Nowcast (LSTM)")

try:
    df = load_data()
    
    # --- 4. THE INVERSE TRANSFORM LOGIC (Accuracy Fix) ---
    # We must provide 3 columns [Oil, USD, Pred] because the scaler was fit on 3
    # We apply this to the whole dataframe to fix the chart and the metric at once
    
    input_data = df[['oil_price', 'usd_inr', 'predicted_inflation']].values
    inv_data = scaler.inverse_transform(input_data)
    
    # Add the "True" predicted values back to the dataframe
    df['real_prediction'] = inv_data[:, 2]
    
    latest = df.iloc[-1]
    display_val = latest['real_prediction']

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("AI Nowcast (CPI)", f"{display_val:.2f}%")
    col2.metric("Crude Oil", f"${latest['oil_price']:.2f}")
    col3.metric("USD/INR", f"₹{latest['usd_inr']:.2f}")

    # --- 5. THE CHART ---
    fig = go.Figure()
    # Official Data
    fig.add_trace(go.Scatter(x=df['date'], y=df['cpi_inflation_rate'], 
                             name="Official MoSPI Data", line=dict(color='blue')))
    # AI Prediction (Correctly Scaled)
    fig.add_trace(go.Scatter(x=df['date'], y=df['real_prediction'], 
                             name="AI LSTM Nowcast", line=dict(color='red', dash='dot')))
    
    # RBI Target Band
    fig.add_hrect(y0=2, y1=6, fillcolor="green", opacity=0.1, annotation_text="RBI Target Band")
    
    st.plotly_chart(fig, use_container_view=True)

except Exception as e:
    st.error(f"App Error: {e}")

st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")