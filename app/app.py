import streamlit as st
import pandas as pd
from sqlalchemy import create_engine
import plotly.graph_objects as go
from datetime import datetime  # Resolves NameError

# --- CONFIGURATION ---
DB_URI = st.secrets["DB_URI"]

@st.cache_data(ttl=3600)
def load_data():
    """Fetches data using SQLAlchemy for high-performance SQL execution."""
    engine = create_engine(DB_URI)
    query = "SELECT * FROM macro_monitor ORDER BY date ASC"
    return pd.read_sql(query, engine)

# --- UI SETUP ---
st.set_page_config(page_title="India Inflation Nowcaster", layout="wide")
st.title(" India Inflation Nowcast (LSTM)")

try:
    df = load_data()
    latest_row = df.iloc[-1]
    
    # ACCURACY FIX: If model output is 0.054, convert to 5.4%
    raw_pred = latest_row['predicted_inflation']
    display_pred = raw_pred * 100 if raw_pred < 1.0 else raw_pred

    # Display Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("AI Nowcast (CPI)", f"{display_pred:.2f}%")
    col2.metric("Crude Oil", f"${latest_row['oil_price']:.2f}")
    col3.metric("USD/INR", f"₹{latest_row['usd_inr']:.2f}")

    # Plotting
    fig = go.Figure()
    # Official Data
    fig.add_trace(go.Scatter(x=df['date'], y=df['cpi_inflation_rate'], name="Official Data"))
    # AI Forecast (with unit correction)
    chart_preds = df['predicted_inflation'].apply(lambda x: x*100 if x < 1.0 else x)
    fig.add_trace(go.Scatter(x=df['date'], y=chart_preds, name="AI Prediction", line=dict(dash='dot')))
    
    st.plotly_chart(fig, use_container_view=True)

except Exception as e:
    st.error(f"Error: {e}")

st.caption(f"Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")