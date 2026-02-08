import streamlit as st
import pandas as pd
from supabase import create_client
import plotly.graph_objects as go
import os

# 1. SETUP & THEME
st.set_page_config(page_title="RBI Inflation Monitor", layout="wide")
st.title("📈 RBI Inflation Monitor & Forecast")
st.markdown("---")

# Supabase Credentials
URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(URL, KEY)

# 2. DATA FETCHING
@st.cache_data(ttl=600)
def get_data():
    # Fetch all data from the last 12-14 months
    response = supabase.table("macro_monitor").select("*").order("date", ascending=True).execute()
    df = pd.DataFrame(response.data)
    df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df = get_data()

    # 3. METRIC CARDS (Top Row)
    latest_row = df.iloc[-1]
    
    # Check if the last row is a forecast or actual
    if latest_row['is_forecast']:
        current_val = latest_row['predicted_inflation']
        label = "Next Month Forecast"
    else:
        current_val = latest_row['cpi_inflation_rate']
        label = "Latest Actual CPI"

    col1, col2, col3 = st.columns(3)
    col1.metric(label, f"{current_val:.2f}%", delta_color="inverse")
    col2.metric("Oil Price (WTI)", f"${latest_row['oil_price']:.2f}")
    col3.metric("USD/INR", f"₹{latest_row['usd_inr']:.2f}")

    st.markdown("### 12-Month Performance: Actual vs. Forecast")

    # 4. DATA PROCESSING FOR COMPARISON
    # Filter for the last 12 months for the chart
    last_year_df = df.tail(12).copy()

    # 5. PLOTLY CHART
    fig = go.Figure()

    # Trace for Actual Inflation (cpi_inflation_rate)
    fig.add_trace(go.Scatter(
        x=last_year_df['date'], 
        y=last_year_df['cpi_inflation_rate'],
        mode='lines+markers',
        name='Actual CPI (%)',
        line=dict(color='#00CC96', width=3),
        connectgaps=True
    ))

    # Trace for Predicted Inflation (predicted_inflation)
    fig.add_trace(go.Scatter(
        x=last_year_df['date'], 
        y=last_year_df['predicted_inflation'],
        mode='lines+markers',
        name='Model Forecast (%)',
        line=dict(color='#EF553B', width=3, dash='dot'),
        connectgaps=True
    ))

    fig.update_layout(
        template="plotly_dark",
        height=500,
        xaxis_title="Date",
        yaxis_title="Inflation Rate (%)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

    # 6. RAW DATA TABLE
    with st.expander("View Raw Data Details"):
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"⚠️ Application Error: {e}")
    st.info("Check your Supabase column names: Ensure 'cpi_inflation_rate' and 'predicted_inflation' exist.")