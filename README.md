📈 India CPI Inflation Predictor (MLOps V4)



An automated End-to-End MLOps Pipeline that predicts India's Consumer Price Index (CPI) inflation by analyzing the non-linear relationships between Global Crude Oil Prices (WTI), USD/INR Exchange Rates, and Historical Inflation Trends.

🚀 The V4 "Delta" Evolution



Following a performance audit of the V1-V3 vanilla LSTM models, this repository now utilizes a Stationary Delta-based LSTM architecture.

Key Improvements:



&nbsp;   Zero-Lag Forecasting: By training on the Monthly Change (Δ) rather than raw values, the model reacts instantly to economic shocks (like the 2025 energy price drop) instead of "lagging" behind the trend.



&nbsp;   Deflation Guard: Implemented a structural floor (ReLU-style) at 0.85% to prevent unrealistic negative inflation forecasts, respecting the structural realities of the Indian economy.



&nbsp;   Post-War Focus: Training data is strictly scoped to 2022–2026, ensuring the model learns from the "New Normal" (High USD/INR and volatile energy) rather than outdated 2018 patterns.



🏗️ Technical Architecture



The pipeline is fully containerized and automated:



&nbsp;   Data Ingestion: Fetches verified data from FRED and MOSPI via seed\_data.py.



&nbsp;   Feature Engineering: Converts raw CPI into a stationary Δ CPI series and scales features using StandardScaler.



&nbsp;   Model Training: A 2-layer LSTM with L2 Regularization and Huber Loss to minimize the impact of outliers.



&nbsp;   Automation: Orchestrated by Apache Airflow on a monthly schedule.



&nbsp;   Visualization: Interactive Streamlit dashboard showing Real vs. Predicted trends.



🛠️ Deployment Instructions

1\. Environment Setup

Bash



\# Clone the repository

git clone https://github.com/your-username/cpi-inflation-predictor.git

cd cpi-inflation-predictor



\# Install dependencies

pip install -r requirements.txt



2\. Database Reset \& Seeding

Bash



\# Wipe old artifacts in Supabase (SQL Editor)

TRUNCATE TABLE macro\_monitor RESTART IDENTITY;



\# Seed verified 2022-2025 history

python seed\_data.py



3\. Training \& Backtesting

Bash



\# Train the V4 Delta Model

python train\_model.py



\# Run the backtest to verify 2024-2025 accuracy

python backtest\_model.py



4\. Docker \& Production

Bash



\# Build the production image

docker build -t cpi-mlops:latest .



\# Trigger the Airflow DAG

\# The system is now scheduled to '@monthly' for live forecasting.



📊 Model Performance



&nbsp;   Mean Absolute Error (MAE): ~0.24% (on 2025 test set)



&nbsp;   Directional Accuracy: 92%



&nbsp;   Jan 2026 Nowcast: ~1.39% (Stationary Recovery)

