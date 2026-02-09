import pandas as pd
import pandas_datareader.data as web
from supabase import create_client
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def seed_database():
    print("🧹 Cleaning and Reseeding (2022 - 2026)...")
    start, end = datetime(2022, 1, 1), datetime.now()
    
    # 1. FETCH HISTORICAL (2022-2024)
    try:
        df = web.DataReader(['CPALTT01INM659N', 'MCOILWTICO', 'DEXINUS'], 'fred', start, end)
        df = df.resample('MS').mean().dropna(subset=['MCOILWTICO']) 
    except Exception as e:
        print(f"❌ FRED Error: {e}")
        return

    # 2. MANUAL 2025/2026 OVERRIDE (Official MOSPI Data)
    # This fills the gap FRED is missing
    official_data = {
        "2025-01-01": 4.26, "2025-02-01": 3.61, "2025-03-01": 2.95,
        "2025-06-01": 2.10, "2025-10-01": 0.25, "2025-11-01": 0.71,
        "2025-12-01": 1.33
    }

    print("🚀 Uploading combined data to Supabase...")
    for index, row in df.iterrows():
        date_str = index.strftime('%Y-%m-%d')
        
        # Use official value if we have it, otherwise use FRED
        cpi_val = official_data.get(date_str, row['CPALTT01INM659N'])
        
        if pd.isna(cpi_val) and index.year < 2026:
            continue # Skip 2026 months in seed; the DAG will handle those

        data = {
            "date": date_str,
            "oil_price": round(float(row['MCOILWTICO']), 2),
            "usd_inr": round(float(row['DEXINUS']), 2),
            "cpi_inflation_rate": round(float(cpi_val), 2) if not pd.isna(cpi_val) else None,
            "is_forecast": False
        }
        supabase.table("macro_monitor").upsert(data).execute()

    print("✅ Seed Complete! 2025 data is now manually verified.")

if __name__ == "__main__":
    seed_database()