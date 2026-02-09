import streamlit as st
import pandas as pd
from supabase import create_client
import plotly.graph_objects as go
import os

# 1. PAGE SETUP
st.set_page_config(page_title="RBI Inflation Monitor", layout="wide")
st.title("📈 RBI Inflation Monitor & Forecast")
st.markdown("---")

URL = os.getenv("SUPABASE_URL")
KEY = os.getenv("SUPABASE_KEY")
supabase = create_client(URL, KEY)

@st.cache_data(ttl=60)
def get_data():
    # Use desc=False for the correct sorting syntax
    response = supabase.table("macro_monitor").select("*").order("date", desc=True).limit(15).execute()
    df = pd.DataFrame(response.data)
    if not df.empty:
        df['date'] = pd.to_datetime(df['date'])
    return df

try:
    df = get_data()

    if df.empty:
        st.warning("⚠️ Database empty. Run 'python seed_data.py' locally.")
    else:
        # Separate data for the chart
        df_actuals = df[df['cpi_inflation_rate'].notna()].copy()
        df_forecasts = df[df['is_forecast'] == True].copy()

        # 2. TOP METRICS
        latest_f = df_forecasts.iloc[-1]
        m1, m2, m3 = st.columns(3)
        m1.metric("Next Month Forecast (Feb)", f"{latest_f['predicted_inflation']:.2f}%")
        m2.metric("Crude Oil (WTI)", f"${latest_f['oil_price']:.2f}")
        m3.metric("USD/INR Rate", f"₹{latest_f['usd_inr']:.2f}")

        # 3. COMPARISON CHART
        st.markdown("### 📊 Actual RBI CPI vs. LSTM Prediction")
        
        fig = go.Figure()

        # Actuals Line (The History)
        fig.add_trace(go.Scatter(
            x=df_actuals['date'], y=df_actuals['cpi_inflation_rate'],
            name="Actual RBI CPI", line=dict(color='#00CC96', width=4)
        ))

        # Prediction Line (The Future)
        fig.add_trace(go.Scatter(
            x=df_forecasts['date'], y=df_forecasts['predicted_inflation'],
            name="LSTM Prediction", line=dict(color='#EF553B', width=3, dash='dot')
        ))

        # Add RBI Target Zones
        fig.add_hline(y=4.0, line_dash="dash", line_color="rgba(255,255,255,0.3)", annotation_text="Target")
        fig.add_hrect(y0=2.0, y1=6.0, fillcolor="white", opacity=0.05, layer="below", line_width=0)

        fig.update_layout(template="plotly_dark", height=500, hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True)

        # 4. RAW LOGS
        with st.expander("📝 View Raw Data Logs"):
            st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)

except Exception as e:
    st.error(f"Application Error: {e}")