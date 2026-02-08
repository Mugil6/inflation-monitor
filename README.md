🇮🇳 India Inflation Nowcaster (LSTM-based MLOps Pipeline)

📌 Project Overview



Official Consumer Price Index (CPI) inflation data in India is a monthly lagging indicator. This project bridges that "information lag" by building a self-healing MLOps pipeline that treats inflation as a continuous nowcasting problem.



By ingesting high-frequency leading indicators like Crude Oil prices and USD/INR exchange rates, the system uses a Deep Learning (LSTM) model to provide weekly estimates of India’s inflation trajectory before official government releases.

🏗 System Architecture



The pipeline is designed for end-to-end automation and production stability:



&nbsp;   Data Ingestion: Automated fetching of OECD-standard CPI data (Series: CPALTT01INM659N) and market features via the FRED API.



&nbsp;   Orchestration: Apache Airflow (Dockerized) manages the lifecycle. It is scheduled with a weekly trigger (0 0 \* \* 1) to capture intramonth market volatility.



&nbsp;   Storage: Supabase (PostgreSQL) serves as the centralized cloud feature store for historical inputs and AI-generated predictions.



&nbsp;   Inference Engine: A Long Short-Term Memory (LSTM) neural network built with TensorFlow/Keras to model non-linear time-series dependencies.



&nbsp;   Visualization: A live Streamlit dashboard for real-time tracking, forecasting, and comparison against the RBI’s 2-6% target band.



🛠 Engineering Challenges \& Solutions

1\. The "Multivariate Outlier" Trap



Problem: In February 2026, market features encountered significant volatility (e.g., USD/INR spike to 90.57). Standard multidimensional inverse\_transform methods collapsed under these outliers, causing nonsensical model output.



Solution: I implemented Decoupled Manual Scaling. By extracting feature-specific min/max constants from the scaler.pkl for only the target variable (Inflation), I isolated the prediction from feature-level noise. This ensures the model remains technically robust even during "black swan" market events.

2\. Model Governance (Sanity Rails)



Problem: AI models can "hallucinate" extreme values when encountering unprecedented data points.



Solution: Integrated Model Sanity Rails within the deployment layer. These rails validate the LSTM output against historical benchmarks and current RBI projections (e.g., ~2.1% for FY26), ensuring the dashboard provides economically viable insights at all times.

🚀 DevOps \& CI/CD



&nbsp;   Containerization: The entire environment (Airflow, Python, and Dependencies) is containerized using Docker for environment parity.



&nbsp;   Automation: GitHub Actions workflows automate the testing and deployment cycles, ensuring that every code push is seamlessly reflected in the production dashboard.



&nbsp;   Security: Managed environment variables and database credentials through a hardened secrets management workflow to prevent exposure.



📈 How to Run Locally



&nbsp;   Clone the Repository:

&nbsp;   Bash



&nbsp;   git clone https://github.com/Mugil6/inflation-monitor.git



&nbsp;   Environment Variables: Create a .env file with your FRED\_API\_KEY and DB\_URI (Supabase connection string).



&nbsp;   Docker Compose:

&nbsp;   Bash



&nbsp;   docker-compose up -d



&nbsp;   Access Airflow: Open localhost:8080 to trigger the pipeline.



&nbsp;   Access Dashboard: Run streamlit run app/app.py.



📊 Data Attribution



&nbsp;   Inflation Index: FRED (Federal Reserve Economic Data) / OECD.



&nbsp;   Macro Features: Global Market Data (Crude Oil \& FX).



&nbsp;   Policy Benchmarks: Reserve Bank of India (RBI) Inflation Targeting Framework.

