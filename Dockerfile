# Use a slim Python 3.10 image to keep the container lightweight
FROM python:3.10-slim

# Set the working directory inside the container
WORKDIR /app

# 1. Install system dependencies
# build-essential is required for compiling packages like SHAP and Scikit-Learn
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 2. Handle Python dependencies
# We copy requirements first to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 3. Copy the project files
# This copies everything (src, mlflow.db, mlruns, etc.) into /app
COPY . .

# 4. Set Environment Variables
# PYTHONPATH ensures 'import src' works inside the container
ENV PYTHONPATH=/app
# This ensures MLflow defaults to the root DB we just standardized
ENV MLFLOW_TRACKING_URI=sqlite:////app/mlflow.db

# Expose the port FastAPI runs on
EXPOSE 8000

# 5. Start the application
# We use the updated module path: src.api.main
CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]