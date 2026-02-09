import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from supabase import create_client
import os

# --- INITIALIZATION ---
# Ensure these match your secrets.toml or environment variables
SUPABASE_URL = st.secrets["SUPABASE_URL"]
SUPABASE_KEY = st.secrets["SUPABASE_KEY"]
supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- DATA FETCHING ---
def get_data():
    # Fetching 12 months
    response = supabase.table("macro_monitor") \
        .select("*") \
        .order("date", desc=True) \
        .limit(12) \
        .execute()
    return pd.DataFrame(response.data).sort_values('date')

try:
    st.set_page_config(page_title="RBI Inflation Monitor", layout="wide")
    df = get_data()
    
    if not df.empty:
        # Get the absolute latest record for the metric cards
        latest_record = df.iloc[-1]
        
        st.title("📊 RBI Inflation Monitor & Forecast")
        
        # --- METRIC CARDS ---
        # Using iloc[-1] ensures we show the 1.84% forecast, not the 2.69% historical point
        col1, col2, col3 = st.columns(3)
        col1.metric("Next Month Forecast (Feb)", f"{latest_record['predicted_inflation']:.2f}%")
        col2.metric("Crude Oil (WTI)", f"${latest_record['oil_price']:.2f}")
        col3.metric("USD/INR Rate", f"₹{latest_record['usd_inr']:.2f}")

        # --- CHART LOGIC ---
        fig = go.Figure()

        # 1. Add RBI Target Band (4% to 6%)
        # This adds a subtle shaded region for the official target range
        fig.add_hrect(y0=4, y1=6, fillcolor="rgba(255, 255, 255, 0.05)", 
                      line_width=0, annotation_text="RBI Upper Tolerance (4-6%)", 
                      annotation_position="top left")
        
        # 2. Add Target Line at 4%
        fig.add_hline(y=4, line_dash="dash", line_color="gray", annotation_text="RBI Target 4%")

        # 3. Actual RBI CPI (Green Solid)
        df_actuals = df[df['cpi_inflation_rate'].notna()]
        fig.add_trace(go.Scatter(x=df_actuals['date'], y=df_actuals['cpi_inflation_rate'],
                                 mode='lines+markers', name='Actual RBI CPI',
                                 line=dict(color='#00d1b2', width=3)))

        # 4. LSTM Prediction (Red Dashed)
        # Using .notna() ensures the line connects all predictions
        df_forecasts = df[df['predicted_inflation'].notna()]
        fig.add_trace(go.Scatter(x=df_forecasts['date'], y=df_forecasts['predicted_inflation'],
                                 mode='lines+markers', name='LSTM Prediction',
                                 line=dict(color='#ff4b4b', width=2, dash='dot')))

        fig.update_layout(title="📈 Actual RBI CPI vs. LSTM Prediction",
                          template="plotly_dark",
                          xaxis_title="Date",
                          yaxis_title="Inflation Rate (%)",
                          hovermode="x unified",
                          height=600)

        st.plotly_chart(fig, use_container_width=True)

except Exception as e:
    st.error(f"Error loading dashboard: {e}")