import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go

# Connect to DB
# In Streamlit Cloud Secrets: [SUPABASE_URI]
DB_URI = st.secrets["SUPABASE_URI"]

st.set_page_config(page_title="Inflation Nowcast", layout="wide")

@st.cache_data(ttl=600)
def get_data():
    engine = create_engine(DB_URI)
    return pd.read_sql("SELECT * FROM macro_monitor ORDER BY date ASC", engine)

df = get_data()

st.title("🇮🇳 Independent Monetary Policy Monitor")
st.markdown("### AI-Driven Nowcasting for Inflation (CPI) & Macro Stability")

# Latest Prediction
latest = df[df['is_forecast'] == True].iloc[-1] if not df[df['is_forecast'] == True].empty else df.iloc[-1]

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Predicted Inflation (Next Month)", f"{latest['predicted_inflation']:.2f}%", 
              delta="vs Target (4%)", delta_color="inverse")
with col2:
    st.metric("Crude Oil Input", f"${latest['oil_price']:.2f}")
with col3:
    st.metric("USD/INR Input", f"₹{latest['usd_inr']:.2f}")

# Visualization
fig = go.Figure()

# Plot Official History (if you backfill DB with official data)
official = df[df['cpi_inflation_rate'].notnull()]
fig.add_trace(go.Scatter(x=official['date'], y=official['cpi_inflation_rate'], 
                         mode='lines', name='Official MoSPI Data', line=dict(color='blue')))

# Plot Forecast
forecast = df[df['is_forecast'] == True]
fig.add_trace(go.Scatter(x=forecast['date'], y=forecast['predicted_inflation'], 
                         mode='lines+markers', name='AI Nowcast', 
                         line=dict(color='red', dash='dot')))

# Add RBI Tolerance Band (4% +/- 2%)
fig.add_hrect(y0=2, y1=6, line_width=0, fillcolor="green", opacity=0.1, annotation_text="RBI Tolerance Band")

st.plotly_chart(fig, use_container_width=True)

st.caption(f"Data Sources: MOSPI (via FRED API), Yahoo Finance. Last Updated: {latest['date']}")