# Use the official Airflow image with Python 3.11
FROM apache/airflow:2.10.0-python3.11

# Switch to root to install system dependencies
USER root
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Switch back to airflow user to install Python libraries
USER airflow

# 1. Uninstall conflicing providers (Google Cloud often conflicts with custom TF installs)
RUN pip uninstall -y apache-airflow-providers-google

# 2. Install the "Gold Standard" compatible library set
# We pin versions strictly to prevent "Dependency Hell"
RUN pip install --no-cache-dir \
    "numpy==1.26.4" \
    "pandas==2.1.4" \
    "scikit-learn==1.3.2" \
    "tensorflow==2.16.1" \
    "yfinance==0.2.40" \
    "supabase" \
    "packaging" \
    "python-dotenv"

# Copy your code and model into the container
COPY --chown=airflow:root dags/ /opt/airflow/dags/
COPY --chown=airflow:root model/ /opt/airflow/model/