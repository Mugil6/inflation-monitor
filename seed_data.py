import pandas as pd
import pandas_datareader.data as web
import requests
from io import StringIO
from datetime import datetime
from supabase import create_client
import os
from dotenv import load_dotenv

# 1. SETUP
load_dotenv() 
url = os.getenv("SUPABASE_URL")
key = os.getenv("SUPABASE_KEY")
supabase = create_client(url, key)

print("🚀 Starting Historical Data Backfill (Schema: macro_monitor)...")

start = '2015-01-01'
end = datetime.now().strftime('%Y-%m-%d')

# --- SOURCE A: FRED (For CPI & USD/INR) ---
print("   - Fetching CPI & USD/INR from FRED...")
try:
    # CPALTT01INM659N = India CPI, DEXINUS = USD/INR
    fred_data = web.DataReader(['CPALTT01INM659N', 'DEXINUS'], 'fred', start, end)
    
    # Process CPI (Already Monthly)
    cpi_df = fred_data[['CPALTT01INM659N']].rename(columns={'CPALTT01INM659N': 'cpi_inflation'})
    
    # Process USD/INR (Daily -> Monthly Average)
    usd_df = fred_data[['DEXINUS']].resample('MS').mean().rename(columns={'DEXINUS': 'usd_inr'})
    
except Exception as e:
    print(f"❌ Critical Error fetching FRED data: {e}")
    exit()

# --- SOURCE B: Open Data GitHub (For Crude Oil) ---
print("   - Fetching Crude Oil from Open Data...")
try:
    oil_url = "https://raw.githubusercontent.com/datasets/oil-prices/master/data/wti-daily.csv"
    response = requests.get(oil_url)
    oil_raw = pd.read_csv(StringIO(response.text))
    
    # Clean & Resample
    oil_raw['Date'] = pd.to_datetime(oil_raw['Date'])
    oil_raw.set_index('Date', inplace=True)
    oil_raw = oil_raw.sort_index()
    oil_raw = oil_raw[start:end]
    
    # Resample Daily Price -> Monthly Average
    oil_df = oil_raw.resample('MS').mean().rename(columns={'Price': 'oil_price'})
    
except Exception as e:
    print(f"❌ Critical Error fetching Oil data: {e}")
    exit()

# --- MERGE & ALIGN ---
print("   - Merging Datasets...")
df_final = pd.concat([cpi_df, usd_df, oil_df], axis=1).dropna()
df_final.index.name = 'date'
df_final.reset_index(inplace=True)
df_final['date'] = df_final['date'].dt.strftime('%Y-%m-%d')

# Add Schema Specific Columns
df_final['predicted_inflation'] = None
df_final['is_forecast'] = False

# --- UPLOAD TO SUPABASE ---
print(f"   - Uploading {len(df_final)} rows to 'macro_monitor'...")
data_list = df_final.to_dict(orient='records')

try:
    # Upsert in batches of 50
    for i in range(0, len(data_list), 50):
        batch = data_list[i:i+50]
        supabase.table('macro_monitor').upsert(batch).execute()
        print(f"     ... Uploaded batch {i//50 + 1}")
    print("✅ SUCCESS: History seeded successfully!")
except Exception as e:
    print(f"❌ Upload Error: {e}")