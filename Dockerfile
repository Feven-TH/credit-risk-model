FROM python:3.10-slim

WORKDIR /app

# Install system dependencies (needed for some ML packages)
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install (flexible versions solve clashes)
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Environment variables
ENV PYTHONPATH=/app
ENV MLFLOW_TRACKING_URI=sqlite:///src/mlflow.db

EXPOSE 8000

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]