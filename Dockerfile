# Use the official Airflow image with Python 3.11
FROM apache/airflow:2.10.0-python3.11

# Switch to root to install system dependencies
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user
USER airflow

# 1. Uninstall conflicting providers
RUN pip uninstall -y apache-airflow-providers-google

# 2. Install Dependencies
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "pandas==2.1.4" \
    "scikit-learn==1.3.2" \
    "tensorflow==2.16.1" \
    "yfinance==0.2.40" \
    "supabase" \
    "packaging" \
    "python-dotenv" \
    "pandas-datareader" \
    "requests"

# Copy code and model
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root model/ /opt/airflow/model/