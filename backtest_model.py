import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
from supabase import create_client
import os
from dotenv import load_dotenv

load_dotenv()
supabase = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))

def run_backtest():
    model = tf.keras.models.load_model('model/rbi_lstm.keras')
    scaler = joblib.load('model/scaler.pkl')

    response = supabase.table("macro_monitor").select("*").order("date", desc=False).execute()
    df = pd.DataFrame(response.data)
    
    # Create the delta column for the features
    df['cpi_delta'] = df['cpi_inflation_rate'].diff().fillna(0)
    features_raw = df[['oil_price', 'usd_inr', 'cpi_delta']].values

    print("🚀 Running Delta-Corrected Backtest...")
    for i in range(10, len(df)):
        target_date = df.iloc[i]['date']
        last_actual = df.iloc[i-1]['cpi_inflation_rate'] # Anchor
        
        input_window = features_raw[i-10:i]
        if np.isnan(input_window).any(): continue

        scaled_input = scaler.transform(input_window)
        scaled_input = np.expand_dims(scaled_input, axis=0)
        
        predicted_delta_scaled = model.predict(scaled_input, verbose=0)
        
        # Inverse Scale the Delta
        dummy = np.zeros((1, 3))
        dummy[0, 2] = predicted_delta_scaled[0, 0]
        predicted_delta = scaler.inverse_transform(dummy)[0, 2]

        # RECONSTRUCT: Final Prediction = Last Month + Predicted Change
        final_pred = last_actual + predicted_delta
        
        # Clamp to 0.5% (India's structural floor)
        final_pred = max(0.5, final_pred)

        supabase.table("macro_monitor").update({
            "predicted_inflation": round(float(final_pred), 2)
        }).eq("date", target_date).execute()
        
        print(f"✅ {target_date}: {final_pred:.2f}% (Delta: {predicted_delta:+.2f})")

if __name__ == "__main__":
    run_backtest()