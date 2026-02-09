\# 📈 India CPI Inflation Predictor (MLOps V4)



\[!\[Python 3.10](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/)

\[!\[Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B.svg)](https://streamlit.io/)

\[!\[Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

\[!\[License: Proprietary](https://img.shields.io/badge/License-Proprietary-red.svg)](#-license)



> \*\*An End-to-End MLOps Pipeline\*\* that nowcasts India's Consumer Price Index (CPI) inflation by analyzing the non-linear relationships between Global Crude Oil Prices (WTI), USD/INR Exchange Rates, and historical inflation trends.



---



\## 🚀 Key Innovations (V4 Delta Architecture)



This version moves beyond standard forecasting by implementing a \*\*Stationary Delta-based LSTM\*\* to solve the "lag" problem common in economic models.



\* \*\*⚡ Zero-Lag Forecasting:\*\* Instead of predicting raw values, the model learns the \*rate of change\* (Delta). This allows it to react instantly to sudden economic shocks (like oil price spikes) rather than smoothing them out.

\* \*\*🧠 Adaptive "Post-Crisis" Learning:\*\* The training data is strategically scoped to the \*\*2022–2026\*\* period. This ensures the model prioritizes the current high-volatility economic regime rather than outdated, stable pre-pandemic patterns.

\* \*\*🔗 Multi-Factor Correlation:\*\* Unlike simple univariate models, this system fuses external macro-signals (Crude Oil \& Forex) with internal CPI trends to capture the \*cause\* of inflation, not just the history.



---



\## 🏗️ Technical Architecture



The pipeline is fully containerized and automated using an industrial MLOps stack:



| Component | Technology | Description |

| :--- | :--- | :--- |

| \*\*Orchestration\*\* | \*\*Apache Airflow\*\* | Monthly scheduled DAGs for automated retraining. |

| \*\*Ingestion\*\* | \*\*FredAPI / MOSPI\*\* | Fetches verified macro data via `seed\_data.py`. |

| \*\*Preprocessing\*\* | \*\*Pandas / Scikit-Learn\*\* | Converts raw CPI into a stationary Δ series; scales features. |

| \*\*Model\*\* | \*\*Keras LSTM\*\* | 2-layer LSTM with L2 Regularization \& Huber Loss. |

| \*\*Database\*\* | \*\*Supabase (PostgreSQL)\*\* | Stores historical artifacts and forecast logs. |

| \*\*Dashboard\*\* | \*\*Streamlit\*\* | Interactive UI for Real vs. Predicted trend analysis. |



---



\## 📊 Model Performance



\* \*\*Mean Absolute Error (MAE):\*\* `~0.24%` (on 2025 test set)

\* \*\*Directional Accuracy:\*\* `92%`

\* \*\*Jan 2026 Nowcast:\*\* `~1.39%` (Stationary Recovery)



---



\## 🛠️ Deployment Instructions



\### 1. Environment Setup

```bash

\# Clone the repository

git clone \[https://github.com/your-username/cpi-inflation-predictor.git](https://github.com/your-username/cpi-inflation-predictor.git)

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



📜 License



Copyright (c) 2026 \[Your Name]. All Rights Reserved.



This project is licensed under a Proprietary License.



&nbsp;   The contents of this repository, including the code, model weights, and documentation, are the intellectual property of the author.



&nbsp;   Unauthorized copying, modification, distribution, or commercial use of this software is strictly prohibited.



&nbsp;   You may fork this repository for viewing purposes only.



For permission requests, please contact the author via GitHub.

