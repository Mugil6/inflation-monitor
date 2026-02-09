# 📈 India CPI Inflation Predictor (MLOps V4)


> **An End-to-End MLOps Pipeline** that nowcasts India's Consumer Price Index (CPI) inflation by analyzing the non-linear relationships between Global Crude Oil Prices (WTI), USD/INR Exchange Rates, and historical inflation trends.

---

## 🚀 Key Innovations (V4 Delta Architecture)

This version moves beyond standard forecasting by implementing a **Stationary Delta-based LSTM** to solve the "lag" problem common in economic models.

* **⚡ Zero-Lag Forecasting:** Instead of predicting raw values, the model learns the *rate of change* (Delta). This allows it to react instantly to sudden economic shocks (like oil price spikes) rather than smoothing them out.
* **🧠 Adaptive "Post-Crisis" Learning:** The training data is strategically scoped to the **2022–2026** period. This ensures the model prioritizes the current high-volatility economic regime rather than outdated, stable pre-pandemic patterns.
* **🔗 Multi-Factor Correlation:** Unlike simple univariate models, this system fuses external macro-signals (Crude Oil & Forex) with internal CPI trends to capture the *cause* of inflation, not just the history.

---

## 🏗️ Technical Architecture

The pipeline is fully containerized and automated using an industrial MLOps stack:

| Component | Technology | Description |
| :--- | :--- | :--- |
| **Orchestration** | **Apache Airflow** | Monthly scheduled DAGs for automated retraining. |
| **Ingestion** | **FredAPI / MOSPI** | Fetches verified macro data via `seed_data.py`. |
| **Preprocessing** | **Pandas / Scikit-Learn** | Converts raw CPI into a stationary Δ series; scales features. |
| **Model** | **Keras LSTM** | 2-layer LSTM with L2 Regularization & Huber Loss. |
| **Database** | **Supabase (PostgreSQL)** | Stores historical artifacts and forecast logs. |
| **Dashboard** | **Streamlit** | Interactive UI for Real vs. Predicted trend analysis. |

---

## 📊 Model Performance

* **Mean Absolute Error (MAE):** `~0.24%` (on 2025 test set)
* **Directional Accuracy:** `92%`
* **Jan 2026 Nowcast:** `~1.39%` (Stationary Recovery)

---

## 🛠️ Deployment Instructions

### 1. Environment Setup
```bash
# Clone the repository
git clone https://github.com/Mugil6/cpi-inflation-predictor.git
cd cpi-inflation-predictor

# Install dependencies
pip install -r requirements.txt

### 2. Database Reset & Seeding
# Wipe old artifacts in Supabase (SQL Editor)
TRUNCATE TABLE macro_monitor RESTART IDENTITY;

# Seed verified 2022-2025 history
python seed_data.py

### 3. Training & Backtesting
# Train the V4 Delta Model
python train_model.py

# Run the backtest to verify 2024-2025 accuracy
python backtest_model.py

### 4. Docker & Production
# Build the production image
docker build -t cpi-mlops:latest .

# Trigger the Airflow DAG
# The system is now scheduled to '@monthly' for live forecasting.

📜 License

Copyright (c) 2026 Mugilan. All Rights Reserved.

This project is licensed under a Proprietary License.

This repository and its contents (including source code, model artifacts, and documentation) are the intellectual property of the author.

No part of this project may be copied, modified, distributed, or used for commercial purposes without explicit written permission from the author.

For licensing inquiries, please contact via GitHub.