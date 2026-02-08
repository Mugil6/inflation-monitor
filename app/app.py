import streamlit as st
import pandas as pd
import requests
import plotly.graph_objects as go

# --- CONFIGURATION ---
# In Streamlit Cloud: Go to Settings -> Secrets and add:
# SUPABASE_URL = "https://your-id.supabase.co"
# SUPABASE_KEY = "your-anon-key"

url = st.secrets["SUPABASE_URL"]
key = st.secrets["SUPABASE_KEY"]

st.set_page_config(page_title="RBI Inflation Nowcast", layout="wide", page_icon="📈")

@st.cache_data(ttl=3600)  # Cache for 1 hour to save API calls
def fetch_macro_data():
    endpoint = f"{url}/rest/v1/macro_monitor?select=*&order=date.asc"
    headers = {
        "apikey": key,
        "Authorization": f"Bearer {key}"
    }
    
    response = requests.get(endpoint, headers=headers)
    if response.status_code == 200:
        data = response.json()
        df = pd.DataFrame(data)
        df['date'] = pd.to_datetime(df['date'])
        return df
    else:
        st.error(f"Failed to fetch data: {response.status_code}")
        return pd.DataFrame()

# --- MAIN APP ---
st.title(" Inflation Monitor")
st.markdown("### AI-Driven Nowcasting for Inflation (CPI) & Macro Stability")
st.info("This system uses a Deep Learning (LSTM) model to forecast the next month's inflation based on Crude Oil and Forex volatility.")

df = fetch_macro_data()

if not df.empty:
    # 1. METRICS SECTION
    # Get the most recent entry
    latest = df.iloc[-1]
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Check if latest entry is a forecast or official
        label = "AI Nowcast (Next Month)" if latest['is_forecast'] else "Latest Official CPI"
        st.metric(label, f"{latest['predicted_inflation']:.2f}%", 
                  delta=f"{latest['predicted_inflation'] - 4.0:.2f}% vs Target", 
                  delta_color="inverse")
        
    with col2:
        st.metric("Crude Oil (WTI)", f"${latest['oil_price']:.2f}/bbl")
        
    with col3:
        st.metric("USD/INR Rate", f"₹{latest['usd_inr']:.2f}")

    # 2. VISUALIZATION SECTION
    fig = go.Figure()

    # RBI Tolerance Band (Shaded Area)
    fig.add_hrect(y0=2, y1=6, line_width=0, fillcolor="rgba(0,255,0,0.1)", 
                  annotation_text="RBI Target (4% ± 2%)", annotation_position="top left")

    # Plot Official Data
    official_df = df[df['cpi_inflation_rate'].notnull()]
    if not official_df.empty:
        fig.add_trace(go.Scatter(
            x=official_df['date'], 
            y=official_df['cpi_inflation_rate'],
            mode='lines+markers', 
            name='Official MoSPI Data',
            line=dict(color='#1f77b4', width=3)
        ))

    # Plot AI Forecasts
    forecast_df = df[df['is_forecast'] == True]
    if not forecast_df.empty:
        fig.add_trace(go.Scatter(
            x=forecast_df['date'], 
            y=forecast_df['predicted_inflation'],
            mode='lines+markers', 
            name='LSTM Nowcast',
            line=dict(color='#d62728', width=2, dash='dot'),
            marker=dict(size=8, symbol='diamond')
        ))

    fig.update_layout(
        title="Inflation Trend: Official vs. AI Forecast",
        xaxis_title="Date",
        yaxis_title="Inflation Rate (%)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3. DATA TABLE SECTION
    with st.expander("View Raw Macro Data"):
        st.dataframe(df.sort_values('date', ascending=False), use_container_width=True)

else:
    st.warning("No data found in Supabase. Ensure your Airflow DAG has completed at least one successful run.")

st.divider()
st.caption(f"Architecture: Airflow (Docker) -> LSTM (TensorFlow) -> Supabase -> Streamlit. Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")